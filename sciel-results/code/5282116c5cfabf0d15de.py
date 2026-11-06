import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    # Generate 2-3-5 smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth.append(x)
    
    smooth_set = set(smooth)
    smooth.sort()
    n_smooth = len(smooth)
    
    # Precompute compatibility (divisibility antichain check)
    compat = [[True] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(i+1, n_smooth):
            a, b = smooth[i], smooth[j]
            if a % b == 0 or b % a == 0:
                compat[i][j] = False
                compat[j][i] = False
    
    # Precompute index for quick lookup
    idx_map = {val: i for i, val in enumerate(smooth)}
    
    answers = [None] * 128

    # Level 1: 1 term
    for s in smooth:
        if s <= 128:
            answers[s - 1] = [s]

    # Level 2: 2 terms
    for i in range(n_smooth):
        a = smooth[i]
        for j in range(i + 1, n_smooth):
            b = smooth[j]
            s = a + b
            if s > 128:
                break
            if compat[i][j]:
                if answers[s - 1] is None:
                    answers[s - 1] = [a, b]

    # Check if all targets are covered
    if all(ans is not None for ans in answers):
        return answers

    # Level 3: 3 terms
    for i in range(n_smooth):
        a = smooth[i]
        for j in range(i + 1, n_smooth):
            b = smooth[j]
            if not compat[i][j]:
                continue
            for k in range(j + 1, n_smooth):
                c = smooth[k]
                s = a + b + c
                if s > 128:
                    break
                if compat[i][k] and compat[j][k]:
                    if answers[s - 1] is None:
                        answers[s - 1] = [a, b, c]

    # Check again
    if all(ans is not None for ans in answers):
        return answers

    # For remaining targets, use bitmask-based search for 4+ terms
    # Group smooth numbers into chains based on divisibility structure
    # A chain: numbers where each divides the next
    # Antichain constraint: can pick at most one from any chain
    
    # Build chains using Dilworth-like decomposition
    # For each number, find its "parents" (numbers that divide it)
    # Use greedy chain decomposition
    
    # Actually, let's use a more direct approach: for each target,
    # do a targeted search with pruning using the compatibility matrix
    
    for target_idx in range(128):
        if answers[target_idx] is not None:
            continue
        
        if time.time() > deadline:
            break
            
        target = target_idx + 1
        candidates = [(val, idx) for idx, val in enumerate(smooth) if val <= target]
        
        # Try counts from 4 to 8
        found = False
        for count in range(4, 9):
            if time.time() > deadline:
                break
            if len(candidates) < count:
                break
            
            # Backtracking with aggressive pruning
            def backtrack(start_idx, current_terms, remaining, count):
                if len(current_terms) == count:
                    if remaining == 0:
                        return [t[0] for t in current_terms]
                    return None
                
                if remaining < 0:
                    return None
                
                # Need enough candidates left
                if start_idx + (count - len(current_terms)) > len(candidates):
                    return None
                
                # Pruning: min possible sum from remaining candidates
                need = count - len(current_terms)
                if need > 0:
                    # Minimum sum: pick smallest 'need' candidates from start_idx onward
                    min_possible = sum(v for v, _ in candidates[start_idx:start_idx + need]) if start_idx + need <= len(candidates) else float('inf')
                    if min_possible > remaining:
                        return None
                    # Maximum sum: pick largest 'need' candidates
                    max_possible = sum(v for v, _ in candidates[-need:]) if need <= len(candidates) else float('-inf')
                    if max_possible < remaining:
                        return None
                
                for idx in range(start_idx, len(candidates)):
                    val, ci = candidates[idx]
                    if val > remaining:
                        continue
                    
                    # Check compatibility with all existing terms
                    ok = True
                    for _, existing_idx in current_terms:
                        if not compat[existing_idx][ci]:
                            ok = False
                            break
                    if not ok:
                        continue
                    
                    current_terms.append((val, ci))
                    result = backtrack(idx + 1, current_terms, remaining - val, count)
                    if result:
                        return result
                    current_terms.pop()
                
                return None

            result = backtrack(0, [], target, count)
            if result:
                answers[target_idx] = result
                found = True
                break
        
        if not found:
            # Fallback: try up to 15 terms with combinations
            for count in range(9, 16):
                if time.time() > deadline:
                    break
                if len(candidates) < count:
                    break
                for terms in combinations(candidates, count):
                    if sum(v for v, _ in terms) == target:
                        idxs = [ci for _, ci in terms]
                        ok = True
                        for ii in range(len(idxs)):
                            for jj in range(ii+1, len(idxs)):
                                if not compat[idxs[ii]][idxs[jj]]:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            answers[target_idx] = [v for v, _ in terms]
                            found = True
                            break
                if found:
                    break

    return answers
