# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    t0 = time.time()
    DEADLINE = 85.0

    # Precompute 2,3,5-smooth numbers up to 128
    smooth = []
    smooth_set = set()
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
            smooth_set.add(x)
    smooth.sort()
    n_smooth = len(smooth)
    
    # Create index lookup
    smooth_idx = {v: i for i, v in enumerate(smooth)}
    
    # Precompute divisibility: divs[i][j] = True if smooth[i] divides smooth[j]
    divs = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        si = smooth[i]
        for j in range(i + 1, n_smooth):
            if smooth[j] % si == 0:
                divs[i][j] = True
    
    # Precompute for each index, which later indices it divides
    divides_later = [set() for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if divs[i][j]:
                divides_later[i].add(j)
    
    # Precompute antichain compatibility: compat[i][j] = True if i and j can coexist in antichain
    # (neither divides the other)
    compat = [[True] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if divs[i][j]:
                compat[i][j] = False
                compat[j][i] = False
    
    # For each target, find minimum antichain representation
    answers = [None] * 129
    
    # Quick check: single term
    for n in range(1, 129):
        if n in smooth_set:
            answers[n] = [n]
    
    # For 2 terms: a + b = n, a < b, both smooth, a does not divide b
    for n in range(1, 129):
        if answers[n] is not None:
            continue
        for a in smooth:
            if a >= n:
                break
            b = n - a
            if b <= a:
                continue
            if b in smooth_set:
                ia = smooth_idx[a]
                ib = smooth_idx[b]
                if not divs[ia][ib]:  # a does not divide b
                    answers[n] = [a, b]
                    break
    
    # For 3 terms: a + b + c = n, a < b < c, all smooth, antichain
    for n in range(1, 129):
        if answers[n] is not None:
            continue
        found = False
        for i, a in enumerate(smooth):
            if a >= n:
                break
            for j in range(i + 1, n_smooth):
                b = smooth[j]
                if a + b >= n:
                    break
                c = n - a - b
                if c <= b:
                    continue
                if c in smooth_set:
                    k = smooth_idx[c]
                    # Check antichain: no pair divides another
                    if not divs[i][j] and not divs[i][k] and not divs[j][k]:
                        answers[n] = [a, b, c]
                        found = True
                        break
            if found:
                break
    
    # For 4+ terms: use DFS with pruning
    def dfs(remaining, min_idx, current, indices, max_terms):
        if remaining == 0:
            return current
        if len(current) >= max_terms:
            return None
        for idx in range(min_idx, n_smooth):
            val = smooth[idx]
            if val > remaining:
                break
            # Check antichain with existing terms
            valid = True
            for ti in indices:
                if divs[ti][idx] or divs[idx][ti]:
                    valid = False
                    break
            if not valid:
                continue
            res = dfs(remaining - val, idx + 1, current + [val], indices + [idx], max_terms)
            if res:
                return res
        return None
    
    for n in range(1, 129):
        if answers[n] is not None:
            continue
        # Try 4 terms first, then 5, etc.
        found = False
        for max_terms in range(4, 31):
            res = dfs(n, 0, [], [], max_terms)
            if res:
                answers[n] = res
                found = True
                break
        if not found:
            # Fallback: should not happen, but use [n] as placeholder
            answers[n] = [n]
    
    # Convert to list of lists (remove None at index 0)
    result = [answers[i] for i in range(1, 129)]
    return result
# EVOLVE-BLOCK-END
