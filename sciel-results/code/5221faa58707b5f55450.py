"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin below 90s limit
    
    # Generate all 5-smooth numbers up to 128
    smooth_numbers = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth_numbers.append(x)
    
    smooth_set = set(smooth_numbers)
    n_smooth = len(smooth_numbers)
    
    # Precompute divisibility: div[i][j] = True if smooth_numbers[i] divides smooth_numbers[j]
    div_matrix = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j:
                a, b = smooth_numbers[i], smooth_numbers[j]
                if b % a == 0 or a % b == 0:
                    div_matrix[i][j] = True
    
    # Map value -> index in smooth_numbers
    val_to_idx = {v: i for i, v in enumerate(smooth_numbers)}
    
    # For each target, find minimal-term antichain representation
    answers = []
    
    for n in range(1, 129):
        if time.time() - start_time > timeout:
            break
        
        found = None
        
        # Try k=1: single smooth number
        if n in smooth_set:
            found = [n]
        
        if found is None:
            # Try k=2: two smooth numbers, neither divides the other
            for i in range(n_smooth):
                a = smooth_numbers[i]
                if a >= n:
                    break
                b = n - a
                if b <= a:
                    continue
                if b in smooth_set:
                    j = val_to_idx[b]
                    if not div_matrix[i][j]:
                        found = [a, b]
                        break
        
        if found is None:
            # Try k=3: three smooth numbers forming antichain
            # Use targeted search: for each pair (i,j) that is antichain, check if n - a - b is smooth and forms antichain
            for i in range(n_smooth):
                a = smooth_numbers[i]
                if a >= n:
                    break
                for j in range(i + 1, n_smooth):
                    b = smooth_numbers[j]
                    if a + b >= n:
                        break
                    if div_matrix[i][j]:
                        continue
                    c = n - a - b
                    if c <= b:
                        continue
                    if c in smooth_set:
                        k_idx = val_to_idx[c]
                        if not div_matrix[i][k_idx] and not div_matrix[j][k_idx]:
                            found = [a, b, c]
                            break
                if found is not None:
                    break
        
        if found is None:
            # Try k=4: four smooth numbers forming antichain
            # Use targeted search with pruning
            for i in range(n_smooth):
                a = smooth_numbers[i]
                if a >= n:
                    break
                for j in range(i + 1, n_smooth):
                    b = smooth_numbers[j]
                    if a + b >= n:
                        break
                    if div_matrix[i][j]:
                        continue
                    for k_idx in range(j + 1, n_smooth):
                        c = smooth_numbers[k_idx]
                        if a + b + c >= n:
                            break
                        if div_matrix[i][k_idx] or div_matrix[j][k_idx]:
                            continue
                        d = n - a - b - c
                        if d <= c:
                            continue
                        if d in smooth_set:
                            l_idx = val_to_idx[d]
                            if (not div_matrix[i][l_idx] and not div_matrix[j][l_idx] 
                                and not div_matrix[k_idx][l_idx]):
                                found = [a, b, c, d]
                                break
                    if found is not None:
                        break
                if found is not None:
                    break
        
        if found is None:
            # Try k=5: five smooth numbers forming antichain
            for i in range(n_smooth):
                a = smooth_numbers[i]
                if a >= n:
                    break
                for j in range(i + 1, n_smooth):
                    b = smooth_numbers[j]
                    if a + b >= n:
                        break
                    if div_matrix[i][j]:
                        continue
                    for k_idx in range(j + 1, n_smooth):
                        c = smooth_numbers[k_idx]
                        if a + b + c >= n:
                            break
                        if div_matrix[i][k_idx] or div_matrix[j][k_idx]:
                            continue
                        for l_idx in range(k_idx + 1, n_smooth):
                            d = smooth_numbers[l_idx]
                            if a + b + c + d >= n:
                                break
                            if (div_matrix[i][l_idx] or div_matrix[j][l_idx] 
                                or div_matrix[k_idx][l_idx]):
                                continue
                            e = n - a - b - c - d
                            if e <= d:
                                continue
                            if e in smooth_set:
                                m_idx = val_to_idx[e]
                                if (not div_matrix[i][m_idx] and not div_matrix[j][m_idx] 
                                    and not div_matrix[k_idx][m_idx] and not div_matrix[l_idx][m_idx]):
                                    found = [a, b, c, d, e]
                                    break
                        if found is not None:
                            break
                    if found is not None:
                        break
                if found is not None:
                    break
        
        if found is None:
            # Fallback: use up to 30 terms via greedy + local search
            # This should rarely be needed
            candidates = [x for x in smooth_numbers if x <= n]
            found = _greedy_antichain(n, candidates, div_matrix, val_to_idx, smooth_set)
        
        answers.append(found)
    
    # If we timed out, fill remaining with fallback
    while len(answers) < 128:
        n = len(answers) + 1
        found = _greedy_antichain(n, [x for x in smooth_numbers if x <= n], div_matrix, val_to_idx, smooth_set)
        answers.append(found)
    
    return answers

def _greedy_antichain(n, candidates, div_matrix, val_to_idx, smooth_set):
    """Fallback: find an antichain representation using greedy approach with backtracking."""
    # Try to find any valid antichain of size 1-30
    # Use DFS with pruning
    import sys
    sys.setrecursionlimit(10000)
    
    candidates_sorted = sorted(candidates, reverse=True)
    best = None
    
    def dfs(remaining, chosen, idx, max_terms):
        nonlocal best
        if best is not None and len(chosen) >= len(best):
            return
        if remaining == 0:
            if best is None or len(chosen) < len(best):
                best = list(chosen)
            return
        if len(chosen) >= max_terms:
            return
        if idx >= len(candidates_sorted):
            return
        if remaining < 0:
            return
        
        # Try including candidates_sorted[idx]
        val = candidates_sorted[idx]
        if val <= remaining:
            # Check if it forms antichain with all chosen
            valid = True
            for c in chosen:
                if div_matrix[val_to_idx[val]][val_to_idx[c]]:
                    valid = False
                    break
            if valid:
                chosen.append(val)
                dfs(remaining - val, chosen, idx + 1, max_terms)
                chosen.pop()
        
        # Try skipping
        dfs(remaining, chosen, idx + 1, max_terms)
    
    dfs(n, [], 0, 30)
    if best is not None:
        return best
    
    # Absolute last resort: return [n] if smooth, else [1, n-1]
    if n in smooth_set:
        return [n]
    else:
        return [1, n - 1]
# EVOLVE-BLOCK-END
