"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Generate smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
    
    n_smooth = len(smooth)
    smooth_set = set(smooth)
    
    # Precompute divisibility: divs[i] contains j if smooth[i] divides smooth[j] and i != j
    divs = [set() for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j and smooth[j] % smooth[i] == 0:
                divs[i].add(j)
    
    def is_antichain_idx(idx_tuple):
        for i in range(len(idx_tuple)):
            for j in range(i+1, len(idx_tuple)):
                a, b = idx_tuple[i], idx_tuple[j]
                if b in divs[a] or a in divs[b]:
                    return False
        return True
    
    # Pre-compute antichains by size and sum for small k
    max_k_precompute = 8
    antichains_by_size = [dict() for _ in range(max_k_precompute + 1)]
    
    for k in range(1, max_k_precompute + 1):
        for combo in combinations(range(n_smooth), k):
            if is_antichain_idx(combo):
                s = sum(smooth[i] for i in combo)
                if s <= 128:
                    if s not in antichains_by_size[k]:
                        antichains_by_size[k][s] = []
                    antichains_by_size[k][s].append(combo)
    
    answers = []
    for n in range(1, 129):
        if time.time() > deadline:
            # Fallback
            if n in smooth_set:
                answers.append([n])
            else:
                remaining = n
                chosen = []
                for s in reversed(smooth):
                    if s <= remaining:
                        valid = True
                        for c in chosen:
                            if c % s == 0 or s % c == 0:
                                valid = False
                                break
                        if valid:
                            chosen.append(s)
                            remaining -= s
                            if remaining == 0:
                                break
                if remaining == 0 and 1 <= len(chosen) <= 30:
                    answers.append(chosen)
                else:
                    answers.append([n] if n in smooth_set else [1])
            continue
        
        answer = None
        # Check precomputed antichains
        for k in range(1, max_k_precompute + 1):
            if n in antichains_by_size[k]:
                combo = antichains_by_size[k][n][0]
                answer = [smooth[i] for i in combo]
                break
        
        # If not found in precomputed, use DFS for larger k
        if answer is None:
            found = False
            for k in range(max_k_precompute + 1, 31):
                if time.time() > deadline:
                    break
                def dfs(start_idx, remaining, chosen_idx, chosen_vals):
                    nonlocal found
                    if found:
                        return
                    if len(chosen_idx) == k:
                        if remaining == 0:
                            found = True
                            answer = list(chosen_vals)
                        return
                    if remaining <= 0:
                        return
                    slots_left = k - len(chosen_idx)
                    if start_idx + slots_left > n_smooth:
                        return
                    for i in range(start_idx, n_smooth):
                        if found:
                            return
                        val = smooth[i]
                        if val > remaining:
                            break
                        valid = True
                        for ci in chosen_idx:
                            if i in divs[ci] or ci in divs[i]:
                                valid = False
                                break
                        if valid:
                            chosen_idx.append(i)
                            chosen_vals.append(val)
                            dfs(i + 1, remaining - val, chosen_idx, chosen_vals)
                            chosen_idx.pop()
                            chosen_vals.pop()
                
                dfs(0, n, [], [])
                if found:
                    break
        
        if answer is None:
            answer = [n] if n in smooth_set else [1]
        
        answers.append(answer)
    
    return answers
# EVOLVE-BLOCK-END
