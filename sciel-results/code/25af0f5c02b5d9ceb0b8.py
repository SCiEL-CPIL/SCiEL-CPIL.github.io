"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = start_time + 87.0  # 90s wall-clock, leave 3s margin
    
    N = 256
    max_sum = 2 * N
    perfect_squares = set()
    for k in range(1, int(math.isqrt(max_sum)) + 2):
        sq = k * k
        if sq <= max_sum:
            perfect_squares.add(sq)
    
    # Self-loop check: 2a is a perfect square
    # 2a = k^2 => a = k^2/2, so k must be even, k=2m, a=2m^2
    self_loop_set = set()
    for m in range(1, 12):
        a = 2 * m * m
        if 1 <= a <= N:
            self_loop_set.add(a)
    
    # Remaining vertices
    remaining = [a for a in range(1, N + 1) if a not in self_loop_set]
    n_rem = len(remaining)
    
    # Map each remaining vertex to an index 0..n_rem-1
    vert_to_idx = {v: i for i, v in enumerate(remaining)}
    
    # Build adjacency as bitmasks for speed
    # adj_mask[i] is a bitmask of neighbors of vertex i
    adj_mask = [0] * n_rem
    for i in range(n_rem):
        a = remaining[i]
        mask = 0
        for b in remaining:
            if a != b and (a + b) in perfect_squares:
                j = vert_to_idx[b]
                mask |= (1 << j)
        adj_mask[i] = mask
    
    # Precompute degree for each vertex
    degrees = [bin(m).count('1') for m in adj_mask]
    
    def greedy_min_degree(rs=None):
        """Greedy: repeatedly pick vertex with smallest degree among remaining"""
        if rs is None:
            rs = random.Random()
        
        # Track which vertices are still available
        available = (1 << n_rem) - 1
        result_indices = []
        
        while available:
            # Find vertex with minimum degree in induced subgraph
            min_deg = n_rem + 1
            candidates = []
            avail_list = []
            temp = available
            while temp:
                lsb = temp & (-temp)
                i = lsb.bit_length() - 1
                avail_list.append(i)
                temp ^= lsb
            
            for i in avail_list:
                # Count neighbors still available
                deg = bin(adj_mask[i] & available).count('1')
                if deg < min_deg:
                    min_deg = deg
                    candidates = [i]
                elif deg == min_deg:
                    candidates.append(i)
            
            # Pick randomly from candidates
            v = rs.choice(candidates)
            result_indices.append(v)
            # Remove v and its neighbors from available
            available &= ~(1 << v)
            available &= ~adj_mask[v]
        
        return result_indices
    
    def local_improvement_fast(indices, rs, max_iters=2000):
        """Local improvement using bitmask operations for speed"""
        if not indices:
            return indices
        
        # Build bitmask of selected vertices
        selected_mask = 0
        for i in indices:
            selected_mask |= (1 << i)
        
        improved = True
        iters = 0
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # Try adding each non-selected vertex
            non_selected = ((1 << n_rem) - 1) & ~selected_mask
            non_sel_list = []
            temp = non_selected
            while temp:
                lsb = temp & (-temp)
                i = lsb.bit_length() - 1
                non_sel_list.append(i)
                temp ^= lsb
            
            rs.shuffle(non_sel_list)
            
            for v in non_sel_list:
                # Check conflicts: neighbors of v that are in selected
                conflicts = adj_mask[v] & selected_mask
                if conflicts == 0:
                    # Can add v directly
                    selected_mask |= (1 << v)
                    improved = True
                else:
                    # Try swap: remove all conflicts, add v
                    conflict_count = bin(conflicts).count('1')
                    net_gain = 1 - conflict_count
                    if net_gain > 0:
                        selected_mask = (selected_mask & ~conflicts) | (1 << v)
                        improved = True
                    elif net_gain == 0:
                        # Try 1-for-1 swap: for each conflict c, try removing c and adding v
                        # Only beneficial if c has more conflicts with other selected than v does
                        # Simplified: just try it if it might help
                        pass
        
        # Convert back to list
        result = []
        temp = selected_mask
        while temp:
            lsb = temp & (-temp)
            i = lsb.bit_length() - 1
            result.append(i)
            temp ^= lsb
        
        return result
    
    def iterated_local_search(rs, max_iters=100):
        """Iterated local search with perturbations"""
        # Start with greedy
        current = greedy_min_degree(rs)
        current = local_improvement_fast(current, rs, max_iters=1000)
        
        for _ in range(max_iters):
            if time.time() > DEADLINE:
                break
            
            # Perturb: remove k random elements
            k = rs.randint(2, 10)
            if k > len(current):
                k = len(current)
            to_remove = set(rs.sample(current, k))
            perturbed = [x for x in current if x not in to_remove]
            
            # Re-optimize
            perturbed = local_improvement_fast(perturbed, rs, max_iters=500)
            
            if len(perturbed) > len(current):
                current = perturbed
        
        return current
    
    def multi_start_local_search(rs, n_starts=10):
        """Multiple random starts with local improvement"""
        best = []
        for s in range(n_starts):
            if time.time() > DEADLINE:
                break
            
            # Random start
            current = []
            for i in range(n_rem):
                if rs.random() < 0.35:
                    current.append(i)
            
            # Clean up conflicts
            changed = True
            while changed:
                changed = False
                for i in current:
                    if adj_mask[i] & (sum((1 << x) for x in current if x != i)):
                        current.remove(i)
                        changed = True
                        break
            
            current = local_improvement_fast(current, rs, max_iters=1000)
            if len(current) > len(best):
                best = current
        
        return best
    
    # Run multiple strategies
    best_indices = []
    
    # Strategy 1: Deterministic greedy
    s1 = greedy_min_degree(random.Random(0))
    s1 = local_improvement_fast(s1, random.Random(1), max_iters=2000)
    if len(s1) > len(best_indices):
        best_indices = s1
    
    # Strategy 2: Multiple greedy with different random seeds
    for seed in range(100):
        if time.time() > DEADLINE:
            break
        s = greedy_min_degree(random.Random(seed + 100))
        s = local_improvement_fast(s, random.Random(seed + 200), max_iters=1500)
        if len(s) > len(best_indices):
            best_indices = s
    
    # Strategy 3: Iterated local search
    for seed in range(20):
        if time.time() > DEADLINE:
            break
        s = iterated_local_search(random.Random(seed + 1000), max_iters=50)
        if len(s) > len(best_indices):
            best_indices = s
    
    # Strategy 4: Multi-start random
    for seed in range(10):
        if time.time() > DEADLINE:
            break
        s = multi_start_local_search(random.Random(seed + 5000), n_starts=5)
        if len(s) > len(best_indices):
            best_indices = s
    
    # Final local improvement on best
    final_rs = random.Random(99999)
    best_indices = local_improvement_fast(best_indices, final_rs, max_iters=3000)
    
    # Convert to actual values
    result = sorted([remaining[i] for i in best_indices])
    
    # Validate
    valid = True
    for i in range(len(result)):
        a = result[i]
        if (2 * a) in perfect_squares:
            valid = False
            break
        for j in range(i + 1, len(result)):
            b = result[j]
            if (a + b) in perfect_squares:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback
        return [1]
    
    return result

# EVOLVE-BLOCK-END
