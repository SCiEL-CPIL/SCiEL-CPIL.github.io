import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 70.0  # Safety margin (90s limit, keep 20s buffer)
    
    max_target = 128
    
    # 1. Generate smooth numbers up to max_target
    smooth = []
    for n in range(1, max_target + 1):
        y = n
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth.append(n)
    
    n_smooth = len(smooth)
    
    # 2. Precompute divisibility relation for pairs of smooth numbers
    # divs[i][j] = True if smooth[i] divides smooth[j]
    divs = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i == j:
                divs[i][j] = True
            elif smooth[j] % smooth[i] == 0:
                divs[i][j] = True
    
    # For antichain check: no pair (i,j) with i!=j should have divs[i][j] or divs[j][i]
    # Precompute "conflict" matrix: conflict[i][j] = True if i and j cannot be in same antichain
    conflict = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if divs[i][j] or divs[j][i]:
                conflict[i][j] = True
                conflict[j][i] = True
    
    # 3. Find minimal representation for each sum 1..128
    best_sums = {}    # sum -> list of terms (actual values)
    best_counts = {}  # sum -> count
    
    targets_found = set()
    all_targets = set(range(1, 129))
    
    # Helper to check if a set of indices forms an antichain
    def is_antichain(indices):
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if conflict[indices[i]][indices[j]]:
                    return False
        return True
    
    # Iterate k from 1 upwards
    for k in range(1, 31):
        if time.time() > deadline:
            break
        
        # For k=1: every smooth number is a valid antichain of size 1
        if k == 1:
            for s in smooth:
                if s <= 128 and s not in best_counts:
                    best_sums[s] = [s]
                    best_counts[s] = 1
                    targets_found.add(s)
            continue
        
        # For k>=2: use combinations
        # Optimization: for k=2, we can iterate pairs more efficiently
        if k == 2:
            for i in range(n_smooth):
                si = smooth[i]
                for j in range(i + 1, n_smooth):
                    sj = smooth[j]
                    s = si + sj
                    if s > 128:
                        break
                    if s in best_counts:
                        continue
                    if not conflict[i][j]:
                        best_sums[s] = [si, sj]
                        best_counts[s] = 2
                        targets_found.add(s)
            continue
        
        # For k>=3: use combinations with pruning
        # We can prune: if the sum of the smallest k smooth numbers exceeds 128, stop
        # Also, we can break early in combinations if partial sum exceeds 128
        
        # Generate combinations of indices, then convert to values
        # To prune, we sort smooth (already sorted) and use the fact that
        # combinations are generated in lexicographic order on indices
        
        # For efficiency, precompute prefix sums? Not straightforward with combinations.
        # Instead, we'll use a recursive/backtracking approach for k>=3 to allow pruning.
        
        # Backtracking search for k terms summing to target, forming antichain
        # We'll search for each target individually? No, we want all targets.
        # Better: iterate all combinations of k indices, check sum and antichain.
        
        # Number of combinations: C(50, 3) = 19600, C(50,4) = 230300, C(50,5) = 2118760
        # C(50,6) = 15890700 - might be too slow for 60s if we need to go that high.
        # Let's estimate: most targets should be covered by k<=4 or 5.
        
        for indices in combinations(range(n_smooth), k):
            if time.time() > deadline:
                break
            
            s = sum(smooth[idx] for idx in indices)
            if s > 128:
                continue
            if s in best_counts:
                continue
            
            if is_antichain(indices):
                best_sums[s] = [smooth[idx] for idx in indices]
                best_counts[s] = k
                targets_found.add(s)
                
                if len(targets_found) == len(all_targets):
                    break
        
        if len(targets_found) == len(all_targets):
            break
        if time.time() > deadline:
            break
    
    # Check coverage
    missing = all_targets - targets_found
    if missing:
        # Fallback: try to find representations for missing targets with higher k
        # or use a different strategy. For now, try k up to 30 for missing only.
        for target in sorted(missing):
            if time.time() > deadline:
                break
            found = False
            for k in range(1, 31):
                if found:
                    break
                # Search for k terms summing to target
                # Use backtracking with pruning
                def backtrack(start_idx, remaining, current, k_left):
                    if k_left == 0:
                        return remaining == 0
                    if remaining <= 0:
                        return False
                    # Prune: minimum possible sum with k_left terms from start_idx onwards
                    if start_idx + k_left - 1 >= n_smooth:
                        return False
                    min_sum = sum(smooth[start_idx:start_idx + k_left]) if start_idx + k_left <= n_smooth else float('inf')
                    if min_sum > remaining:
                        return False
                    for idx in range(start_idx, n_smooth):
                        val = smooth[idx]
                        if val > remaining:
                            break
                        # Check conflict with all current terms
                        valid = True
                        for cidx in current:
                            if conflict[idx][cidx]:
                                valid = False
                                break
                        if not valid:
                            continue
                        current.append(idx)
                        if backtrack(idx + 1, remaining - val, current, k_left - 1):
                            return True
                        current.pop()
                    return False
                
                current = []
                if backtrack(0, target, current, k):
                    best_sums[target] = [smooth[idx] for idx in current]
                    best_counts[target] = k
                    targets_found.add(target)
                    found = True
    
    # Construct final answer
    answers = []
    for n in range(1, 129):
        if n in best_sums:
            answers.append(best_sums[n])
        else:
            # This should not happen. Add a placeholder.
            answers.append([])
    
    return answers
