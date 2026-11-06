import time
from fractions import Fraction

def construct():
    deadline = time.time() + 55  # 55s deadline for safety (limit is 60s)
    
    denoms = list(range(2, 49))
    N = len(denoms)
    fracs = [Fraction(1, d) for d in denoms]
    
    # Reverse order: iterate from largest denominator (smallest fraction) to smallest
    # This makes pruning more effective as we accumulate small values first
    rev_fracs = list(reversed(fracs))
    rev_denoms = list(reversed(denoms))
    
    # Precompute suffix sums in reversed order
    suffix_sums = [Fraction(0)] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + rev_fracs[i]
        
    # Greedy initialization in descending denominator order (smallest fractions first)
    # This gives a reasonable starting point
    best_sum = Fraction(0)
    best_idxs = []
    current_sum = Fraction(0)
    current_idxs = []
    for i in range(N):
        if current_sum + rev_fracs[i] < 1:
            current_sum += rev_fracs[i]
            current_idxs.append(N - 1 - i)  # Convert back to original index
    best_sum = current_sum
    best_idxs = current_idxs[:]
    
    # Local search: try 1-swap and 1-add/remove moves to improve
    def local_search(idxs, s):
        nonlocal best_sum, best_idxs
        improved = True
        idx_set = set(idxs)
        while improved and time.time() < deadline:
            improved = False
            # Try removing one element
            for i in list(idx_set):
                new_s = s - fracs[i]
                if new_s > best_sum:
                    # Check if we can add something
                    for j in range(N):
                        if j not in idx_set and new_s + fracs[j] < 1:
                            candidate_s = new_s + fracs[j]
                            if candidate_s > best_sum:
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
            # Try swapping one element
            for i in list(idx_set):
                for j in range(N):
                    if j not in idx_set:
                        new_s = s - fracs[i] + fracs[j]
                        if 0 < new_s < 1 and new_s > best_sum:
                            best_sum = new_s
                            best_idxs = [k for k in idx_set if k != i] + [j]
                            idx_set = set(best_idxs)
                            s = new_s
                            improved = True
                            break
                if improved:
                    break
            if time.time() >= deadline:
                break
        return best_sum, best_idxs
    
    best_sum, best_idxs = local_search(best_idxs, best_sum)
    
    # DFS with Branch and Bound in descending denominator order
    # Using iterative approach to avoid recursion depth issues
    # State: (i, s, idxs) where i is position in rev_fracs
    def dfs_iterative():
        nonlocal best_sum, best_idxs
        # Stack-based DFS
        # Each entry: (i, s, idxs_tuple)
        stack = [(0, Fraction(0), ())]
        
        while stack and time.time() < deadline:
            i, s, idxs = stack.pop()
            
            # Pruning
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
                new_idx = N - 1 - i  # Convert to original index
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
    if s_check >= 1 or s_check <= 0:
        # Fallback
        return [0]
        
    return best_idxs
