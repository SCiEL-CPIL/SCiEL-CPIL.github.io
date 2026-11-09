import time
import sys

def _precompute():
    deadline = time.time() + 75  # 75s safety margin for 90s limit
    
    # Generate all 2,3,5-smooth numbers up to 128
    smooth = set()
    for a in range(0, 8):
        va = 2 ** a
        for b in range(0, 5):
            vb = va * (3 ** b)
            if vb > 128:
                break
            for c in range(0, 4):
                vc = vb * (5 ** c)
                if vc > 128:
                    break
                smooth.add(vc)
    
    smooth_list = sorted(smooth)
    n_smooth = len(smooth_list)
    smooth_to_idx = {s: i for i, s in enumerate(smooth_list)}
    
    # Build incompatibility mask
    incompatible_mask = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            si = smooth_list[i]
            sj = smooth_list[j]
            if sj % si == 0 or si % sj == 0:
                incompatible_mask[i] |= (1 << j)
                incompatible_mask[j] |= (1 << i)
    
    # Candidates for each target
    candidates = [[] for _ in range(129)]
    for n in range(1, 129):
        for i, s in enumerate(smooth_list):
            if s <= n:
                candidates[n].append(i)
            else:
                break
    
    # Memoization cache: (n, k) -> list of values or None
    memo = {}
    
    def find_antichain(n, k):
        key = (n, k)
        if key in memo:
            return memo[key]
        
        if time.time() > deadline:
            memo[key] = None
            return None
        
        cand = candidates[n]
        if len(cand) < k:
            memo[key] = None
            return None
        
        if k == 1:
            if n in smooth_to_idx:
                memo[key] = [n]
            else:
                memo[key] = None
            return memo[key]
        
        # Sort candidates descending by value for better pruning
        cand_desc = sorted(cand, key=lambda i: -smooth_list[i])
        
        # Quick bounds
        if sum(smooth_list[i] for i in cand_desc[:k]) < n:
            memo[key] = None
            return None
        if sum(smooth_list[i] for i in cand_desc[-k:]) > n:
            memo[key] = None
            return None
        
        path = []
        
        def backtrack(start, rem_sum, rem_k):
            if time.time() > deadline:
                return None
            
            if rem_k == 0:
                return [] if rem_sum == 0 else None
            
            avail = cand_desc[start:]
            if len(avail) < rem_k:
                return None
            
            # Max possible sum
            max_possible = sum(smooth_list[i] for i in avail[:rem_k])
            if max_possible < rem_sum:
                return None
            
            # Min possible sum
            min_possible = sum(smooth_list[i] for i in avail[-rem_k:])
            if min_possible > rem_sum:
                return None
            
            for i in range(len(avail)):
                idx = avail[i]
                num = smooth_list[idx]
                if num > rem_sum:
                    continue
                
                # Check divisibility compatibility
                compatible = True
                for p_idx in path:
                    if incompatible_mask[idx] & (1 << p_idx):
                        compatible = False
                        break
                
                if not compatible:
                    continue
                
                path.append(idx)
                result = backtrack(i + 1, rem_sum - num, rem_k - 1)
                if result is not None:
                    return [num] + result
                path.pop()
            
            return None
        
        result = backtrack(0, n, k)
        memo[key] = result
        return result
    
    # Find optimal representation for each target
    answers = []
    for n in range(1, 129):
        if time.time() > deadline:
            # Fallback for remaining targets
            if n in smooth_to_idx:
                answers.append([n])
            else:
                # Try a simple heuristic
                found = None
                for k in range(2, 6):
                    res = find_antichain(n, k)
                    if res is not None:
                        found = res
                        break
                answers.append(found if found else [n])
            continue
        
        answer = None
        for k in range(1, 31):
            res = find_antichain(n, k)
            if res is not None:
                answer = res
                break
        
        if answer is None:
            # Fallback
            if n in smooth_to_idx:
                answer = [n]
            else:
                for k in range(2, 31):
                    res = find_antichain(n, k)
                    if res is not None:
                        answer = res
                        break
                if answer is None:
                    answer = [n]
        
        answers.append(answer)
    
    return answers

_PRECOMPUTED = _precompute()

def construct():
    return _PRECOMPUTED
