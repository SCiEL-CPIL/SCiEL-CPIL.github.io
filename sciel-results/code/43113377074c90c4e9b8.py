import time
from fractions import Fraction

def construct():
    deadline = time.time() + 55  # Safety margin before 60s limit
    
    denoms = list(range(2, 49))
    N = len(denoms)
    fracs = [Fraction(1, d) for d in denoms]
    
    # Work in ascending denominator order (smallest fractions first)
    # This makes pruning more effective
    order = list(range(N))  # index 0 -> denom 2, index 1 -> denom 3, etc.
    # Already in ascending order of denominators = ascending order of 1/d is descending
    # Actually we want to iterate from largest denominator (smallest fraction) to smallest
    # So reverse: start with 1/48, 1/47, ..., 1/2
    rev_order = list(range(N - 1, -1, -1))
    rev_fracs = [fracs[i] for i in rev_order]
    
    # Precompute suffix sums in reversed order
    suffix_sums = [Fraction(0)] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + rev_fracs[i]
    
    # Greedy initialization: take items in ascending denominator order (smallest fractions first)
    # This gives a reasonable starting point close to 1
    best_sum = Fraction(0)
    best_idxs = []
    current_sum = Fraction(0)
    current_idxs = []
    for i in range(N):
        # i is index in rev_order, so original index is rev_order[i]
        orig_idx = rev_order[i]
        if current_sum + fracs[orig_idx] < 1:
            current_sum += fracs[orig_idx]
            current_idxs.append(orig_idx)
    best_sum = current_sum
    best_idxs = current_idxs[:]
    
    # Local search: try 1-swap and 1-add/remove moves to improve
    def local_search(idxs, s):
        nonlocal best_sum, best_idxs
        improved = True
        idx_set = set(idxs)
        while improved and time.time() < deadline:
            improved = False
            
            # Try removing one element and adding another (1-swap)
            for i in list(idx_set):
                new_s = s - fracs[i]
                if new_s > best_sum:
                    # Try adding each non-selected item
                    for j in range(N):
                        if j not in idx_set:
                            candidate_s = new_s + fracs[j]
                            if candidate_s < 1 and candidate_s > best_sum:
                                best_sum = candidate_s
                                best_idxs = [k for k in idx_set if k != i] + [j]
                                idx_set = set(best_idxs)
                                s = candidate_s
                                improved = True
                                break
                if improved:
                    break
            if time.time() >= deadline:
                break
            
            # Try adding one element
            for j in range(N):
                if j not in idx_set and s + fracs[j] < 1:
                    candidate_s = s + fracs[j]
                    if candidate_s > best_sum:
                        best_sum = candidate_s
                        best_idxs = list(idx_set) + [j]
                        idx_set = set(best_idxs)
                        s = candidate_s
                        improved = True
                        break
            if time.time() >= deadline:
                break
            
            # Try removing one element (if it improves, though unlikely since we want max sum)
            # Skip pure removal since it decreases sum
        
        return best_sum, best_idxs
    
    best_sum, best_idxs = local_search(best_idxs, best_sum)
    
    # DFS with Branch and Bound in ascending denominator order (smallest fractions first)
    # Using iterative approach to avoid recursion depth issues
    def dfs_iterative():
        nonlocal best_sum, best_idxs
        # Stack-based DFS
        # Each entry: (i, s, idxs_tuple)
        # i is position in rev_order (0 = largest denom, N-1 = smallest denom)
        stack = [(0, Fraction(0), ())]
        
        while stack and time.time() < deadline:
            i, s, idxs = stack.pop()
            
            # Pruning: even if we take all remaining, can't beat best_sum
            if s + suffix_sums[i] <= best_sum:
                continue
            
            if i == N:
                if s > best_sum:
                    best_sum = s
                    best_idxs = list(idxs)
                continue
            
            # Push exclude first (so include is processed first due to LIFO)
            # Option: Exclude item i
            stack.append((i + 1, s, idxs))
            
            # Option: Include item i (if valid)
            if s + rev_fracs[i] < 1:
                new_idx = rev_order[i]  # Convert to original index
                stack.append((i + 1, s + rev_fracs[i], idxs + (new_idx,)))
        
        return best_sum, best_idxs
    
    best_sum, best_idxs = dfs_iterative()
    
    # Final check and return
    if not best_idxs:
        return [0]
        
    # Verify validity
    s_check = Fraction(0)
    for i in best_idxs:
        s_check += fracs[i]
    if s_check >= 1:
        # Fallback: use greedy result
        return current_idxs
        
    return best_idxs
