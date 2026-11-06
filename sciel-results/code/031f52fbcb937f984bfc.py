"""
Strategy: Simulated Annealing on Set Membership with Maximality Penalty.
We treat the problem as minimizing |S| subject to S being Sidon and Maximal.
We allow intermediate states to be non-maximal, penalizing the "deficiency" (number of addable elements).
This allows the search to escape local minima of the "maximal-only" landscape by temporarily
increasing the size or allowing gaps, then "crystallizing" into a smaller maximal set.
"""
import time
import random
import math
import sys

N = 256

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # 1. Initial Construction: Greedy Max Blocking
    # This gives a valid, maximal Sidon set.
    def greedy_max_blocking():
        selected = []
        sums = set()
        available = set(range(1, N + 1))
        
        def is_valid(v):
            if 2 * v in sums:
                return False
            for x in selected:
                if (v + x) in sums:
                    return False
            return True

        while available:
            valid_candidates = [v for v in available if is_valid(v)]
            if not valid_candidates:
                break
            
            # Heuristic: Pick v that blocks the most others
            # Approximation: Pick v that creates the most new sums in the range [2, 2N]
            # Or simpler: Just pick randomly from valid to avoid bias? 
            # No, Max Blocking is better for initial compactness.
            # Cost: O(K * N) per step. K~20, N=256. 20*256*20 = 100k ops. Fast.
            
            best_v = None
            max_blocks = -1
            # Sample a subset of candidates if list is large to save time?
            # For N=256, full check is fine.
            for v in valid_candidates:
                # Count how many u in available become invalid if v is added
                # We can approximate by checking sums
                new_sums = set(sums)
                new_sums.add(2 * v)
                for x in selected:
                    new_sums.add(v + x)
                
                blocks = 0
                for u in available:
                    if u == v:
                        continue
                    # Check if u is invalid with new_sums
                    if 2 * u in new_sums:
                        continue # u is already blocked by itself? No, 2u is a sum. 
                    # Actually, if 2u is in new_sums, u is invalid.
                    # Wait, is_valid(u) checks if 2u in sums OR u+x in sums.
                    # So if 2u in new_sums, it's blocked.
                    # Or if any u+x in new_sums.
                    blocked = False
                    if 2 * u in new_sums:
                        blocked = True
                    else:
                        for x in selected:
                            if (u + x) in new_sums:
                                blocked = True
                                break
                    if blocked:
                        blocks += 1
                        # Early exit if blocks is already high?
                        if blocks > max_blocks:
                            max_blocks = blocks
                # Note: This inner loop is O(K*N). 
                # Total O(K^2 * N^2) roughly? 
                # K is small. N is 256.
                # 20 * 256 * 20 * 256 = 26M ops. Might be slow in Python.
                # Optimization: Precompute sums.
                
                # Let's simplify: Just pick the largest valid v? 
                # Or random?
                # Let's stick to a simpler heuristic for speed: Pick max v.
                # Large v's block large sums, leaving small sums open?
                # Actually, for Sidon, distribution matters.
                # Let's just pick randomly from valid_candidates for diversity in restarts?
                # No, we want a good initial solution.
                # Let's use a faster blocking metric:
                # Number of new sums added to the set.
                new_sum_count = 0
                for x in selected:
                    if (v + x) not in sums:
                        new_sum_count += 1
                if (2 * v) not in sums:
                    new_sum_count += 1
                
                if new_sum_count > max_blocks:
                    max_blocks = new_sum_count
                    best_v = v

            if best_v is None:
                best_v = random.choice(valid_candidates)
                
            selected.append(best_v)
            sums.add(2 * best_v)
            for x in selected:
                if x != best_v:
                    sums.add(best_v + x)
            available.remove(best_v)
            
        return selected, sums

    # Run a few greedy restarts to get a good initial best
    best_set = None
    best_size = 100
    
    for _ in range(10):
        s, sums = greedy_max_blocking()
        if len(s) < best_size:
            best_size = len(s)
            best_set = s
            
    # 2. Simulated Annealing
    # State: Set S, Set of Sums
    # Energy: |S| + Penalty * (N - |S| - Blocked_Count)
    # Actually, Maximality means Blocked_Count == N - |S|.
    # Deficiency = Number of elements z not in S such that S U {z} is Sidon.
    # We want Deficiency == 0.
    # Energy = |S| + 100 * Deficiency.
    
    def get_deficiency(S, sums):
        """Count how many elements can be added to S to keep it Sidon."""
        count = 0
        S_set = set(S)
        for z in range(1, N + 1):
            if z in S_set:
                continue
            # Check if S U {z} is Sidon
            # New sums: 2*z, and z+x for x in S
            valid = True
            if 2 * z in sums:
                valid = False
            else:
                for x in S:
                    if (z + x) in sums:
                        valid = False
                        break
            if valid:
                count += 1
        return count

    def is_sidon(S, sums):
        # Assumed valid if constructed correctly
        return True

    def is_maximal(S, sums):
        return get_deficiency(S, sums) == 0

    # Annealing Parameters
    T = 100.0
    T_min = 0.1
    alpha = 0.995
    
    current_S = list(best_set)
    current_sums = set()
    for i, x in enumerate(current_S):
        current_sums.add(2 * x)
        for j in range(i):
            current_sums.add(x + current_S[j])
            
    current_energy = len(current_S) + 100 * get_deficiency(current_S, current_sums)
    
    best_S = list(current_S)
    best_energy = current_energy
    
    step = 0
    while time.time() < deadline:
        step += 1
        
        # Generate Move
        move_type = random.random()
        
        new_S = list(current_S)
        new_sums = set(current_sums)
        
        if move_type < 0.3:
            # Remove
            if not current_S:
                continue
            x = random.choice(current_S)
            new_S.remove(x)
            # Recalculate sums for new_S is expensive. 
            # Better: Keep a count of sums?
            # For N=256, K~20, sums ~ 200.
            # Rebuilding sums from scratch: O(K^2) = 400 ops.
            # This is fast enough.
            new_sums = set()
            for i, a in enumerate(new_S):
                new_sums.add(2 * a)
                for b in new_S[i+1:]:
                    new_sums.add(a + b)
                    
        elif move_type < 0.6:
            # Add
            candidates = [z for z in range(1, N + 1) if z not in set(current_S)]
            if not candidates:
                continue
            z = random.choice(candidates)
            # Check if valid addition
            if 2 * z in current_sums:
                continue
            valid = True
            for x in current_S:
                if (z + x) in current_sums:
                    valid = False
                    break
            if not valid:
                continue
            new_S.append(z)
            new_sums = set(current_sums)
            new_sums.add(2 * z)
            for x in current_S:
                new_sums.add(z + x)
                
        else:
            # Swap
            if not current_S:
                continue
            x = random.choice(current_S)
            candidates = [z for z in range(1, N + 1) if z not in set(current_S)]
            if not candidates:
                continue
            z = random.choice(candidates)
            
            # Check if z is valid in S \ {x}
            temp_S = [v for v in current_S if v != x]
            temp_sums = set()
            for i, a in enumerate(temp_S):
                temp_sums.add(2 * a)
                for b in temp_S[i+1:]:
                    temp_sums.add(a + b)
            
            if 2 * z in temp_sums:
                continue
            valid = True
            for v in temp_S:
                if (z + v) in temp_sums:
                    valid = False
                    break
            if not valid:
                continue
            
            new_S = temp_S + [z]
            new_sums = temp_sums
            new_sums.add(2 * z)
            for v in temp_S:
                new_sums.add(z + v)
        
        # Calculate New Energy
        # Energy = |S| + 100 * Deficiency
        new_deficiency = get_deficiency(new_S, new_sums)
        new_energy = len(new_S) + 100 * new_deficiency
        
        delta_E = new_energy - current_energy
        
        if delta_E < 0:
            # Better
            current_S = new_S
            current_sums = new_sums
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_S = list(current_S)
                
        else:
            # Worse, accept with probability
            if T > T_min:
                prob = math.exp(-delta_E / T)
                if random.random() < prob:
                    current_S = new_S
                    current_sums = new_sums
                    current_energy = new_energy
        
        T *= alpha
        
        # Optional: Reheat if stuck?
        if step % 1000 == 0 and T < T_min * 2:
            T = 10.0

    # Ensure final result is maximal
    # If best_S is not maximal, we must repair it
    # Greedy repair: Add any valid element until maximal
    final_S = list(best_S)
    final_sums = set()
    for i, a in enumerate(final_S):
        final_sums.add(2 * a)
        for b in final_S[i+1:]:
            final_sums.add(a + b)
    
    def is_valid_add(v, S, sums):
        if 2 * v in sums:
            return False
        for x in S:
            if (v + x) in sums:
                return False
        return True

    while True:
        # Find any valid element
        found = False
        for v in range(1, N + 1):
            if v not in set(final_S):
                if is_valid_add(v, final_S, final_sums):
                    final_S.append(v)
                    final_sums.add(2 * v)
                    for x in final_S:
                        if x != v:
                            final_sums.add(v + x)
                    found = True
                    break
        if not found:
            break
            
    return final_S

# EVOLVE-BLOCK-END
