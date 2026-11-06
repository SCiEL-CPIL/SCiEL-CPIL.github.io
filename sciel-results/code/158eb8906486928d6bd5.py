import time
import math

def construct():
    start_time = time.time()
    deadline = 145.0 # Safety margin
    
    N = 512
    MAX_A = 511
    
    # 1. Precomputation
    # Prime factors
    prime_factors = [set() for _ in range(MAX_A + 1)]
    for p in range(2, MAX_A + 1):
        if not prime_factors[p]:
            for multiple in range(p, MAX_A + 1, p):
                prime_factors[multiple].add(p)
    
    # Weights
    weights = [0.0] * (MAX_A + 1)
    for a in range(1, MAX_A + 1):
        weights[a] = 1.0 / (N - a)
        
    # Conflict masks
    # conflict_mask[a] has bit b set if gcd(a, b) > 1
    conflict_mask = [0] * (MAX_A + 1)
    for a in range(1, MAX_A + 1):
        mask = 0
        for b in range(1, MAX_A + 1):
            if a == b:
                continue
            if prime_factors[a] & prime_factors[b]:
                mask |= (1 << b)
        conflict_mask[a] = mask
        
    # Initial Greedy Solution
    def greedy_init():
        chosen = []
        chosen_set = set()
        for a in range(MAX_A, 0, -1):
            if all(math.gcd(a, b) == 1 for b in chosen):
                chosen.append(a)
                chosen_set.add(a)
        return chosen, chosen_set
        
    chosen, chosen_set = greedy_init()
    best_score = sum(weights[a] for a in chosen)
    
    # B&B Variables
    # We represent the set of available vertices as a bitmask
    # Initially all vertices 1..511 are available
    # But we must ensure pairwise coprimality.
    # The B&B will handle the constraints.
    
    # To make B&B faster, we can start with the greedy solution as the incumbent.
    # And we can restrict the search space?
    # No, B&B explores all.
    
    # Helper: Get list of set bits
    def get_set_bits(mask):
        bits = []
        while mask:
            # Get lowest set bit
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            bits.append(idx)
            mask ^= lsb
        return bits
        
    # Helper: Greedy Upper Bound
    def greedy_bound(mask):
        if mask == 0:
            return 0.0
        total = 0.0
        m = mask
        # To speed up, we can just iterate through all bits and greedily pick
        # But we need to pick highest weight first.
        # Since weight is increasing with index, we pick highest index first.
        while m:
            # Find highest set bit
            # m.bit_length() - 1 gives index of highest set bit
            v = m.bit_length() - 1
            # Ensure v is valid (1..511)
            if v == 0: break
            total += weights[v]
            # Remove v and its conflicts
            m &= ~conflict_mask[v]
            m &= ~(1 << v)
        return total

    # Recursive B&B
    # To avoid deep recursion overhead, we might use an iterative stack,
    # but recursion is simpler and depth is limited.
    import sys
    sys.setrecursionlimit(10000)
    
    def branch_and_bound(available_mask, current_score):
        nonlocal best_score
        if time.time() > deadline:
            return
            
        if available_mask == 0:
            if current_score > best_score:
                best_score = current_score
            return
            
        # Calculate upper bound
        # Simple bound: sum of all weights in available_mask
        # This is loose. Let's use greedy bound for tighter pruning.
        # However, greedy_bound is expensive.
        # Let's try simple bound first, and if it's not pruning enough, switch.
        # Actually, let's just use the sum of weights of available vertices.
        # We can precompute a mapping from mask to sum? No, mask space is too large.
        
        # Let's calculate sum of weights for available_mask
        # This is O(k).
        ub_sum = 0.0
        m = available_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            ub_sum += weights[idx]
            m ^= lsb
            
        if current_score + ub_sum <= best_score:
            return
            
        # Pick a vertex to branch on.
        # Heuristic: Pick the vertex with highest weight (largest index)
        # This is because high weight vertices are more likely to be in the optimal solution.
        v = available_mask.bit_length() - 1
        
        # Branch 1: Include v
        # New available: available_mask without v and its neighbors
        new_mask_include = available_mask & ~conflict_mask[v] & ~(1 << v)
        branch_and_bound(new_mask_include, current_score + weights[v])
        
        # Branch 2: Exclude v
        new_mask_exclude = available_mask & ~(1 << v)
        branch_and_bound(new_mask_exclude, current_score)

    # Run B&B
    initial_mask = 0
    for a in range(1, MAX_A + 1):
        initial_mask |= (1 << a)
        
    # We can prune the initial mask?
    # No, just run it.
    
    # Optimization: 
    # The greedy solution is a lower bound.
    # We can also try to improve the greedy solution with local search before B&B
    # to get a tighter lower bound.
    
    def local_search(chosen, chosen_set):
        improved = True
        while improved:
            if time.time() > deadline - 10:
                break
            improved = False
            # 1-swap
            for i in range(len(chosen)):
                x = chosen[i]
                sx = weights[x]
                for y in range(MAX_A, 1, -1):
                    if y in chosen_set:
                        continue
                    if weights[y] <= sx:
                        break
                    # Check coprime
                    ok = True
                    if y != 1:
                        yf = prime_factors[y]
                        if yf:
                            for z in chosen:
                                if z == x:
                                    continue
                                if z != 1 and (yf & prime_factors[z]):
                                    ok = False
                                    break
                    if ok:
                        gain = weights[y] - sx
                        if gain > 0:
                            chosen_set.remove(x)
                            chosen_set.add(y)
                            chosen[i] = y
                            improved = True
                            break
                if improved:
                    break
        return chosen, chosen_set

    chosen, chosen_set = local_search(chosen, chosen_set)
    best_score = sum(weights[a] for a in chosen)
    
    # Now run B&B
    branch_and_bound(initial_mask, 0.0)
    
    # Return the best set found?
    # The B&B above only tracks score, not the set.
    # We need to track the set.
    # This complicates the B&B.
    # Alternative: Run B&B to find the optimal score, then reconstruct?
    # Or modify B&B to store the best set.
    
    # Let's modify B&B to store best_set.
    
    best_set = set(chosen)
    
    def branch_and_bound_v2(available_mask, current_score, current_set):
        nonlocal best_score, best_set
        if time.time() > deadline:
            return
            
        if available_mask == 0:
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
            return
            
        # Upper bound
        ub_sum = 0.0
        m = available_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            ub_sum += weights[idx]
            m ^= lsb
            
        if current_score + ub_sum <= best_score:
            return
            
        v = available_mask.bit_length() - 1
        
        # Branch 1: Include v
        new_mask_include = available_mask & ~conflict_mask[v] & ~(1 << v)
        current_set.add(v)
        branch_and_bound_v2(new_mask_include, current_score + weights[v], current_set)
        current_set.remove(v)
        
        # Branch 2: Exclude v
        new_mask_exclude = available_mask & ~(1 << v)
        branch_and_bound_v2(new_mask_exclude, current_score, current_set)

    branch_and_bound_v2(initial_mask, 0.0, set())
    
    return list(best_set)
