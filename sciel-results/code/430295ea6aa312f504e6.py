import time
from fractions import Fraction

def construct():
    deadline = time.time() + 55  # Safety margin before 60s limit
    
    denoms = list(range(2, 49))
    N = len(denoms)
    fracs = [Fraction(1, d) for d in denoms]
    
    # Order: 1/2, 1/3, ..., 1/48 (descending value)
    # Indices are naturally 0..46 corresponding to denoms 2..48
    
    # Precompute suffix sums for pruning
    # suffix_sums[i] = sum(fracs[i] + ... + fracs[N-1])
    suffix_sums = [Fraction(0)] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + fracs[i]
        
    # 1. Greedy Initialization
    best_sum = Fraction(0)
    best_idxs = []
    current_sum = Fraction(0)
    current_idxs = []
    
    # Greedy: take item if it fits
    for i in range(N):
        if current_sum + fracs[i] < 1:
            current_sum += fracs[i]
            current_idxs.append(i)
            
    best_sum = current_sum
    best_idxs = current_idxs[:]
    
    # 2. Local Search Refinement
    # Try to improve the greedy solution by swaps and adds
    # We limit this to a few seconds to save time for DFS
    ls_deadline = time.time() + 5
    improved = True
    idx_set = set(best_idxs)
    
    while improved and time.time() < ls_deadline:
        improved = False
        
        # Try 1-swap: remove i, add j
        # To make this faster, we can just iterate through all pairs, but N=47 is small.
        # 47*47 = 2209 checks, very fast.
        for i in list(idx_set):
            for j in range(N):
                if j not in idx_set:
                    new_s = best_sum - fracs[i] + fracs[j]
                    if new_s < 1 and new_s > best_sum:
                        best_sum = new_s
                        idx_set.remove(i)
                        idx_set.add(j)
                        best_idxs = list(idx_set)
                        improved = True
                        break
            if improved:
                break
        if time.time() >= ls_deadline:
            break
        
        # Try 1-add: add j
        for j in range(N):
            if j not in idx_set:
                candidate_s = best_sum + fracs[j]
                if candidate_s < 1 and candidate_s > best_sum:
                    best_sum = candidate_s
                    idx_set.add(j)
                    best_idxs = list(idx_set)
                    improved = True
                    break
        if time.time() >= ls_deadline:
            break

    # 3. DFS with Branch and Bound
    # Use a mutable list for indices to avoid copying
    def dfs(i, s, idxs):
        nonlocal best_sum, best_idxs
        
        if time.time() > deadline:
            return
            
        # Pruning: Even if we take all remaining, we can't beat best_sum
        if s + suffix_sums[i] <= best_sum:
            return
            
        if i == N:
            if s > best_sum:
                best_sum = s
                best_idxs = idxs[:]
            return

        # Option 1: Include item i
        # Only if s + fracs[i] < 1
        if s + fracs[i] < 1:
            idxs.append(i)
            dfs(i + 1, s + fracs[i], idxs)
            idxs.pop()
            
        # Check time before exploring exclude branch
        if time.time() > deadline:
            return

        # Option 2: Exclude item i
        dfs(i + 1, s, idxs)

    dfs(0, Fraction(0), [])
    
    # Final check and return
    if not best_idxs:
        return [0]
        
    # Verify validity (should be guaranteed by construction, but safe)
    s_check = Fraction(0)
    for i in best_idxs:
        s_check += fracs[i]
    if s_check >= 1:
        # Fallback to greedy result if somehow invalid
        return current_idxs
        
    return best_idxs
