import time
import random
from collections import defaultdict

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s buffer

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        if not S:
            return True
        S_list = list(S)
        S_set = set(S_list)
        for i, a in enumerate(S_list):
            for j in range(i, len(S_list)):
                b = S_list[j]
                if a + b in S_set:
                    return False
        return True

    def solve_instance(vals, time_limit):
        n = len(vals)
        vals_set = set(vals)
        
        # Precompute conflict info for fast checks
        # For each x, store list of pairs (y, z) such that x+y=z
        # This helps in checking validity quickly
        # But for N=240, O(N^2) check is ~57k ops, which is fast.
        
        def can_add(S_set, x):
            # Check if x can be added to S_set
            if x in S_set:
                return True # Already there, but we usually check x not in S
            if 2*x in S_set:
                return False
            for y in S_set:
                if x + y in S_set:
                    return False
                if x - y in S_set:
                    return False
            return True

        def get_initial_candidates():
            candidates = []
            # 1. Odds
            candidates.append(set(x for x in vals if x % 2 != 0))
            # 2. Positives
            candidates.append(set(x for x in vals if x > 0))
            # 3. Negatives
            candidates.append(set(x for x in vals if x < 0))
            # 4. Top half by value
            sv = sorted(vals)
            candidates.append(set(sv[n//2:]))
            # 5. Bottom half by value
            candidates.append(set(sv[:n//2]))
            # 6. Top half by abs value
            sv_abs = sorted(vals, key=lambda x: abs(x), reverse=True)
            candidates.append(set(sv_abs[:n//2 + 1]))
            # 7. Bottom half by abs value
            candidates.append(set(sv_abs[n//2:]))
            
            # 8. Greedy with random order
            for _ in range(5):
                rng = random.Random()
                order = vals[:]
                rng.shuffle(order)
                S = set()
                for x in order:
                    if can_add(S, x):
                        S.add(x)
                candidates.append(S)
                
            # 9. Greedy with abs value order
            for order in [sorted(vals, key=lambda x: abs(x), reverse=True), 
                          sorted(vals, key=lambda x: abs(x))]:
                S = set()
                for x in order:
                    if can_add(S, x):
                        S.add(x)
                candidates.append(S)
                
            return candidates

        def local_search(S, time_limit):
            S = set(S)
            S_set = S
            last_improvement = time.time()
            rng = random.Random()
            
            while time.time() < time_limit:
                improved = False
                
                # 1. Try to add any element
                # Sort candidates by how many conflicts they have? 
                # Or just random?
                # To be fast, we can just iterate through all non-members
                # For N=240, iterating 240 items is fast.
                candidates_to_add = [x for x in vals if x not in S_set]
                rng.shuffle(candidates_to_add)
                for x in candidates_to_add:
                    if can_add(S_set, x):
                        S_set.add(x)
                        improved = True
                        last_improvement = time.time()
                        break
                
                if improved:
                    continue

                # 2. Try swaps: remove y, add x
                # We want to remove an element that is "blocking" many others
                # and add an element that is "easy"
                # Heuristic: Remove element with highest degree in conflict graph?
                # For now, random swap with lookahead
                
                # Try random swaps
                if S_set:
                    y = rng.choice(list(S_set))
                    S_test = S_set - {y}
                    
                    # Try to add as many as possible from remaining
                    # Specifically, try to add the one we removed? No.
                    # Try to add any element that was previously blocked by y
                    
                    # Find elements that conflict with y
                    # If we remove y, some elements might become addable
                    
                    # Simple strategy: Try to add a random x
                    candidates_to_add = [x for x in vals if x not in S_test]
                    rng.shuffle(candidates_to_add)
                    
                    best_x = None
                    best_gain = 0
                    # Check a few candidates
                    for x in candidates_to_add[:20]: # Limit to 20 for speed
                        if can_add(S_test, x):
                            # If we can add x, gain is 1 (remove y, add x -> net 0, but might unlock more)
                            # Actually, if we just swap, size stays same.
                            # We need to check if removing y allows adding MULTIPLE elements.
                            # Let's try to greedily add all possible elements after removing y
                            S_temp = S_test
                            count_added = 0
                            # Greedy add on S_temp
                            remaining = [z for z in vals if z not in S_temp]
                            rng.shuffle(remaining)
                            for z in remaining:
                                if can_add(S_temp, z):
                                    S_temp.add(z)
                                    count_added += 1
                            
                            if count_added > 0:
                                # Net gain = count_added (since we removed 1, added count_added)
                                # Wait, we removed y. So size change = count_added - 1
                                # We want count_added > 1
                                if count_added > 1:
                                    best_x = x # This logic is slightly off, we didn't specifically add x, we added a set
                                    # Let's just keep the best S_temp found
                                    if count_added - 1 > best_gain:
                                        best_gain = count_added - 1
                                        # Store S_temp? It's expensive to copy.
                                        # We'll just record that this y removal was good.
                                        # For simplicity, let's just apply the best y removal found.
                                        # We need to store the resulting set.
                                        # This is getting complex. Let's simplify.
                                        pass
                                    
                    # Simplified Swap: 
                    # Try removing y. Then greedily fill.
                    # If new size > old size, accept.
                    S_after_remove = S_set - {y}
                    S_greedy = set(S_after_remove)
                    remaining = [z for z in vals if z not in S_greedy]
                    rng.shuffle(remaining)
                    for z in remaining:
                        if can_add(S_greedy, z):
                            S_greedy.add(z)
                    
                    if len(S_greedy) > len(S_set):
                        S_set = S_greedy
                        improved = True
                        last_improvement = time.time()
                        break
                        
                # 3. If no improvement for a while, random restart from a different candidate?
                # Or just continue.
                
            return S_set

        candidates = get_initial_candidates()
        best_S = set()
        best_size = -1
        
        # Allocate time based on number of candidates
        # We have ~15 candidates.
        # Let's give each 1/3 of the time? No, we should prioritize larger initial candidates.
        
        candidates.sort(key=len, reverse=True)
        
        # Time management
        total_time = time_limit
        time_per_candidate = total_time / (len(candidates) + 5)
        
        for S in candidates:
            if time.time() > deadline:
                break
                
            if not is_sum_free(S):
                # Fix if invalid (should not happen from our constructors)
                S = set()
                for x in vals:
                    if can_add(S, x):
                        S.add(x)
            
            # Local search with limited time
            t_start = time.time()
            t_end = min(deadline - 5.0, t_start + time_per_candidate * 2) # Allow some extra time for good candidates
            S_improved = local_search(S, t_end)
            
            if len(S_improved) > best_size:
                best_size = len(S_improved)
                best_S = S_improved
        
        return best_S

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        
        # Allocate time
        if size == 60:
            t_lim = 20.0
        elif size == 120:
            t_lim = 40.0
        else:
            t_lim = 60.0
            
        S = solve_instance(vals, t_lim)
        
        if not is_sum_free(S):
            # Fallback
            S = set(x for x in vals if x % 2 != 0)
        
        results.append(sorted(S))
        
        if time.time() > deadline:
            break

    while len(results) < 3:
        results.append([])

    return results[:3]
