import sys
import itertools

def construct():
    # 1. Generate smooth numbers <= 128 and their exponent vectors
    smooth = []
    exps = []
    for x in range(1, 129):
        y = x
        a, b, c = 0, 0, 0
        while y % 2 == 0:
            y //= 2
            a += 1
        while y % 3 == 0:
            y //= 3
            b += 1
        while y % 5 == 0:
            y //= 5
            c += 1
        if y == 1:
            smooth.append(x)
            exps.append((a, b, c))
    
    N = len(smooth)
    
    # 2. Precompute compatibility matrix
    # compat[i][j] is True if smooth[i] and smooth[j] are incomparable
    compat = [[False] * N for _ in range(N)]
    for i in range(N):
        ai, bi, ci = exps[i]
        for j in range(N):
            if i == j:
                continue
            aj, bj, cj = exps[j]
            div_ij = (ai <= aj and bi <= bj and ci <= cj)
            div_ji = (aj <= ai and bj <= bi and cj <= ci)
            if not div_ij and not div_ji:
                compat[i][j] = True

    best_k = [None] * 129
    best_sol = [None] * 129
    
    # Helper to check if a combination of indices is an antichain
    def is_antichain(comb):
        for u in range(len(comb)):
            i = comb[u]
            for v in range(u + 1, len(comb)):
                j = comb[v]
                if not compat[i][j]:
                    return False
        return True

    # Phase 1: Try small k (1 to 4) using itertools.combinations
    for k in range(1, 5):
        for comb in itertools.combinations(range(N), k):
            if is_antichain(comb):
                s = sum(smooth[i] for i in comb)
                if s <= 128 and best_k[s] is None:
                    best_k[s] = k
                    best_sol[s] = [smooth[i] for i in comb]
        
        # Check if all targets 1..128 are found
        if all(best_k[t] is not None for t in range(1, 129)):
            break

    # Phase 2: DFS for remaining targets
    # For each target not found, run a targeted DFS
    for t in range(1, 129):
        if best_k[t] is None:
            # Candidates are indices i where smooth[i] <= t
            cands = [i for i in range(N) if smooth[i] <= t]
            # Sort cands by value descending to help pruning
            cands.sort(key=lambda i: smooth[i], reverse=True)
            
            # Precompute prefix sums for cands values
            cs = [smooth[i] for i in cands]
            prefix = [0] * (len(cs) + 1)
            for i in range(len(cs)):
                prefix[i+1] = prefix[i] + cs[i]
            total_cs = prefix[len(cs)]
            
            current_set = []
            
            def dfs(start_pos, current_sum, k_rem):
                if k_rem == 0:
                    if current_sum == t:
                        return current_set[:]
                    return None
                
                if start_pos + k_rem > len(cands):
                    return None
                
                # Max add: largest k_rem in cs[start_pos:]
                max_add = prefix[start_pos + k_rem] - prefix[start_pos]
                if current_sum + max_add < t:
                    return None
                
                # Min add: smallest k_rem in cs[start_pos:]
                # These are the last k_rem elements of cs
                min_add = total_cs - prefix[len(cs) - k_rem]
                if current_sum + min_add > t:
                    return None

                for i in range(start_pos, len(cands) - k_rem + 1):
                    idx = cands[i]
                    val = cs[i]
                    
                    # Check compatibility
                    compatible = True
                    for prev_idx in current_set:
                        if not compat[idx][prev_idx]:
                            compatible = False
                            break
                    if not compatible:
                        continue
                    
                    if current_sum + val > t:
                        continue
                    
                    current_set.append(idx)
                    res = dfs(i + 1, current_sum + val, k_rem - 1)
                    if res is not None:
                        return res
                    current_set.pop()
                return None

            # Try k from 5 upwards (since 1-4 were tried globally)
            # But we can start from 1 to be safe, though it's redundant.
            # Starting from 5 is faster.
            found_sol = None
            for k in range(5, 31):
                sol = dfs(0, 0, k)
                if sol is not None:
                    found_sol = [smooth[i] for i in sol]
                    best_k[t] = k
                    best_sol[t] = found_sol
                    break
            
            if found_sol is None:
                # Fallback
                best_sol[t] = [t] if t in smooth else [1]

    answers = []
    for t in range(1, 129):
        answers.append(best_sol[t])
        
    return answers
