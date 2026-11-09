import time
import random

def construct():
    deadline = time.time() + 75.0
    
    def is_valid_addition(S, x):
        n = len(S)
        if n < 2:
            return True
        
        max_sum_S = sum(S)
        dp_S = [0] * (n + 1)
        dp_S[0] = 1
        for y in S:
            for k in range(n - 1, -1, -1):
                if dp_S[k]:
                    dp_S[k+1] |= (dp_S[k] << y)
        
        for k in range(2, n + 1):
            target = k * x
            if target > max_sum_S:
                break
            if (dp_S[k] >> target) & 1:
                return False
        
        for i, a in enumerate(S):
            others = S[:i] + S[i+1:]
            m = len(others)
            if m == 0:
                continue
            
            max_sum_others = sum(others)
            dp_others = [0] * (m + 1)
            dp_others[0] = 1
            for y in others:
                for j in range(m - 1, -1, -1):
                    if dp_others[j]:
                        dp_others[j+1] |= (dp_others[j] << y)
            
            for K in range(2, m + 2):
                target_sum_U = K * a - x
                if target_sum_U <= 0:
                    continue
                if target_sum_U > max_sum_others:
                    break
                j = K - 1
                if (dp_others[j] >> target_sum_U) & 1:
                    return False
        
        return True

    best_set = []
    best_size = 0
    random.seed(42)
    
    for trial in range(50):
        if time.time() > deadline:
            break
        
        current_set = []
        remaining = list(range(1, 513))
        
        while len(current_set) < 48 and remaining:
            if current_set:
                scores = []
                for x in remaining:
                    min_dist = min(abs(x - y) for y in current_set)
                    scores.append((min_dist, x))
                scores.sort(reverse=True)
                candidates = [x for _, x in scores[:20]]
            else:
                candidates = remaining[:]
            
            random.shuffle(candidates)
            added = False
            for x in candidates:
                if is_valid_addition(current_set, x):
                    current_set.append(x)
                    remaining.remove(x)
                    added = True
                    break
            
            if not added:
                break
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
        if best_size == 48:
            return best_set

    if best_size < 48 and time.time() < deadline - 5:
        S = list(best_set)
        S_set = set(S)
        unused = [x for x in range(1, 513) if x not in S_set]
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            for u in unused:
                if len(S) >= 48:
                    break
                if is_valid_addition(S, u):
                    S.append(u)
                    S_set.add(u)
                    unused.remove(u)
                    improved = True
                    break
            
            if len(S) >= 48:
                break
                
            random.shuffle(unused)
            for u in unused[:20]:
                if time.time() > deadline - 2:
                    break
                for i, s in enumerate(S):
                    S_minus_s = S[:i] + S[i+1:]
                    if is_valid_addition(S_minus_s, u):
                        S[i] = u
                        S_set.remove(s)
                        S_set.add(u)
                        unused.remove(u)
                        unused.append(s)
                        improved = True
                        break
                if improved:
                    break

        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    return best_set
