"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # 1. Generate all 2,3,5-smooth numbers up to 128
    smooth_set = set()
    for a in range(0, 8):  # 2^7 = 128
        val_a = 2 ** a
        for b in range(0, 5):  # 3^4 = 81, 3^5 = 243 > 128
            val_ab = val_a * (3 ** b)
            if val_ab > 128:
                break
            for c in range(0, 4):  # 5^3 = 125, 5^4 = 625 > 128
                val_abc = val_ab * (5 ** c)
                if val_abc > 128:
                    break
                smooth_set.add(val_abc)
    
    smooth = sorted(smooth_set)
    n_smooth = len(smooth)
    val_to_idx = {v: i for i, v in enumerate(smooth)}
    
    # 2. Precompute divisibility relations
    # divs[i][j] = True if smooth[i] divides smooth[j] (i != j)
    divs = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if smooth[j] % smooth[i] == 0:
                divs[i][j] = True
                divs[j][i] = True
    
    # 3. Precompute candidates for each target
    # candidates[n] = list of (idx, val) for smooth numbers <= n
    candidates = [[] for _ in range(129)]
    for n in range(1, 129):
        for idx, val in enumerate(smooth):
            if val <= n:
                candidates[n].append((idx, val))
            else:
                break
    
    # 4. Find minimal antichain representation for each target
    # Strategy: try k=1,2,3,... until found
    # For each k, use backtracking with pruning
    
    answers = [None] * 128
    
    # Helper: check if a set of indices forms an antichain
    def is_antichain(indices):
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if divs[indices[i]][indices[j]] or divs[indices[j]][indices[i]]:
                    return False
        return True
    
    # For each target, find the minimal k representation
    for n in range(1, 129):
        if time.time() > DEADLINE:
            # Fallback: fill remaining with valid representations
            # This should not happen for n<=128
            break
            
        cand = candidates[n]
        if not cand:
            continue
            
        best_answer = None
        
        # Try k=1
        for idx, val in cand:
            if val == n:
                best_answer = [val]
                break
        
        # Try k=2
        if best_answer is None and len(cand) >= 2:
            for i in range(len(cand)):
                if time.time() > DEADLINE:
                    break
                ii, vi = cand[i]
                for j in range(i + 1, len(cand)):
                    jj, vj = cand[j]
                    if vi + vj > n:
                        break
                    if vi + vj == n:
                        if not divs[ii][jj] and not divs[jj][ii]:
                            best_answer = [vi, vj]
                            break
                if best_answer is not None:
                    break
        
        # Try k=3
        if best_answer is None and len(cand) >= 3:
            found = False
            for i in range(len(cand)):
                if time.time() > DEADLINE:
                    break
                ii, vi = cand[i]
                for j in range(i + 1, len(cand)):
                    jj, vj = cand[j]
                    if vi + vj >= n:
                        break
                    if divs[ii][jj] or divs[jj][ii]:
                        continue
                    for k in range(j + 1, len(cand)):
                        kk, vk = cand[k]
                        if vi + vj + vk > n:
                            break
                        if vi + vj + vk == n:
                            if not divs[ii][kk] and not divs[kk][ii] and not divs[jj][kk] and not divs[kk][jj]:
                                best_answer = [vi, vj, vk]
                                found = True
                                break
                    if found:
                        break
                if found:
                    break
        
        # Try k=4
        if best_answer is None and len(cand) >= 4:
            found = False
            for i in range(len(cand)):
                if time.time() > DEADLINE:
                    break
                ii, vi = cand[i]
                for j in range(i + 1, len(cand)):
                    jj, vj = cand[j]
                    if vi + vj >= n:
                        break
                    if divs[ii][jj] or divs[jj][ii]:
                        continue
                    for k in range(j + 1, len(cand)):
                        kk, vk = cand[k]
                        if vi + vj + vk >= n:
                            break
                        if divs[ii][kk] or divs[kk][ii] or divs[jj][kk] or divs[kk][jj]:
                            continue
                        for l in range(k + 1, len(cand)):
                            ll, vl = cand[l]
                            if vi + vj + vk + vl > n:
                                break
                            if vi + vj + vk + vl == n:
                                if not divs[ii][ll] and not divs[ll][ii] and not divs[jj][ll] and not divs[ll][jj] and not divs[kk][ll] and not divs[ll][kk]:
                                    best_answer = [vi, vj, vk, vl]
                                    found = True
                                    break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
        
        # Try k=5 with backtracking
        if best_answer is None and len(cand) >= 5:
            found = False
            cand_vals = [v for _, v in cand]
            cand_idxs = [i for i, _ in cand]
            
            def backtrack(start, remaining, chosen_vals, chosen_idxs):
                nonlocal found
                if found:
                    return
                if len(chosen_vals) == 5:
                    if remaining == 0:
                        best_answer = list(chosen_vals)
                        found = True
                    return
                
                # Pruning: if remaining < 0, return
                if remaining < 0:
                    return
                
                # Try each candidate from start onwards
                for idx_pos in range(start, len(cand_vals)):
                    if found:
                        return
                    v = cand_vals[idx_pos]
                    if v > remaining:
                        break
                    
                    # Check divisibility against all chosen
                    compatible = True
                    idx = cand_idxs[idx_pos]
                    for c_idx in chosen_idxs:
                        if divs[idx][c_idx] or divs[c_idx][idx]:
                            compatible = False
                            break
                    
                    if compatible:
                        chosen_vals.append(v)
                        chosen_idxs.append(idx)
                        backtrack(idx_pos + 1, remaining - v, chosen_vals, chosen_idxs)
                        chosen_vals.pop()
                        chosen_idxs.pop()
            
            backtrack(0, n, [], [])
        
        # Fallback for k>=6: use combinations
        if best_answer is None:
            for k in range(6, 31):
                if time.time() > DEADLINE:
                    break
                if len(cand) < k:
                    break
                # Use combinations of values
                vals = [v for _, v in cand]
                idxs = [i for i, _ in cand]
                for combo in combinations(range(len(vals)), k):
                    if sum(vals[i] for i in combo) == n:
                        # Check antichain
                        if is_antichain([idxs[i] for i in combo]):
                            best_answer = [vals[i] for i in combo]
                            break
                if best_answer is not None:
                    break
        
        answers[n - 1] = best_answer
    
    # Verify all answers are valid
    for i in range(128):
        if answers[i] is None:
            # Should not happen, but fill with a valid representation
            n = i + 1
            # Try to find any valid representation
            for k in range(1, 31):
                if len(candidates[n]) < k:
                    continue
                vals = [v for _, v in candidates[n]]
                for combo in combinations(vals, k):
                    if sum(combo) == n:
                        # Check antichain
                        valid = True
                        for a, b in combinations(combo, 2):
                            if b % a == 0:
                                valid = False
                                break
                        if valid:
                            answers[i] = list(combo)
                            break
                if answers[i] is not None:
                    break
            if answers[i] is None:
                # Last resort: use n itself if smooth
                if n in val_to_idx:
                    answers[i] = [n]
                else:
                    answers[i] = []  # Should not happen
    
    return answers
# EVOLVE-BLOCK-END
