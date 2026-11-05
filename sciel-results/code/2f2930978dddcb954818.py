import time
import sys
import random
import math
from itertools import permutations

def construct(n):
    if n <= 1:
        return "1" if n == 1 else ""
    
    def build_standard(m):
        if m == 1:
            return [1]
        S = [1]
        for k in range(2, m + 1):
            seen = set()
            perms_order = []
            for i in range(len(S) - (k - 1) + 1):
                window = tuple(S[i:i + k - 1])
                if len(set(window)) == k - 1 and window not in seen:
                    seen.add(window)
                    perms_order.append(window)
            
            new_S = []
            for P in perms_order:
                block = list(P) + [k] + list(P)
                max_possible_overlap = min(len(new_S), len(block) - 1)
                overlap = 0
                for k_ov in range(max_possible_overlap, -1, -1):
                    if k_ov == 0:
                        overlap = 0
                        break
                    if new_S[-k_ov:] == block[:k_ov]:
                        overlap = k_ov
                        break
                new_S.extend(block[overlap:])
            S = new_S
        return S

    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "".join(map(str, build_standard(4)))
    if n == 5:
        return "".join(map(str, build_standard(5)))
    if n == 6:
        return "".join(map(str, build_standard(6)))
    
    if n == 7:
        start_time = time.time()
        deadline = start_time + 115
        
        S6 = build_standard(6)
        
        seen = set()
        perms_order = []
        for i in range(len(S6) - 5):
            window = tuple(S6[i:i + 6])
            if len(set(window)) == 6 and window not in seen:
                seen.add(window)
                perms_order.append(window)
        
        num_perms = len(perms_order)
        
        costs = [[0]*num_perms for _ in range(num_perms)]
        for i in range(num_perms):
            pi = perms_order[i]
            for j in range(num_perms):
                pj = perms_order[j]
                ov = 0
                for k in range(6, -1, -1):
                    if k == 0:
                        ov = 0
                        break
                    if pi[-k:] == pj[:k]:
                        ov = k
                        break
                costs[i][j] = 13 - ov

        # Simulated Annealing
        path = list(range(num_perms))
        
        def get_total_cost(path):
            total = 0
            for i in range(len(path) - 1):
                total += costs[path[i]][path[i+1]]
            return total
        
        current_cost = get_total_cost(path)
        best_cost = current_cost
        best_path = path[:]
        
        T_start = 10.0
        T_end = 0.1
        alpha = 0.999
        T = T_start
        
        while time.time() < deadline - 20:
            i = random.randint(0, num_perms - 2)
            j = random.randint(i + 1, num_perms - 1)
            
            old_c1 = costs[path[i-1]][path[i]] if i > 0 else 0
            old_c2 = costs[path[j]][path[j+1]] if j < num_perms - 1 else 0
            old_cost = old_c1 + old_c2
            
            new_c1 = costs[path[i-1]][path[j]] if i > 0 else 0
            new_c2 = costs[path[i]][path[j+1]] if j < num_perms - 1 else 0
            new_cost = new_c1 + new_c2
            
            delta = new_cost - old_cost
            
            if delta < 0 or (random.random() < math.exp(-delta / T)):
                path[i:j+1] = reversed(path[i:j+1])
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_path = path[:]
            
            T *= alpha
            if T < T_end:
                T = T_end
        
        # Final 2-Opt polish
        path = best_path[:]
        current_cost = get_total_cost(path)
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            for i in range(num_perms - 1):
                if time.time() > deadline - 15:
                    break
                for j in range(i + 1, num_perms):
                    c1 = costs[path[i-1]][path[i]] if i > 0 else 0
                    c2 = costs[path[j]][path[j+1]] if j < num_perms - 1 else 0
                    new_c1 = costs[path[i-1]][path[j]] if i > 0 else 0
                    new_c2 = costs[path[i]][path[j+1]] if j < num_perms - 1 else 0
                    delta = (new_c1 + new_c2) - (c1 + c2)
                    if delta < 0:
                        path[i:j+1] = reversed(path[i:j+1])
                        current_cost += delta
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_path = path[:]
        
        # Reconstruct S7 using best_path
        S7 = []
        for idx in best_path:
            P = perms_order[idx]
            block = list(P) + [7] + list(P)
            max_possible_overlap = min(len(S7), len(block) - 1)
            overlap = 0
            for k in range(max_possible_overlap, -1, -1):
                if k == 0:
                    overlap = 0
                    break
                if S7[-k:] == block[:k]:
                    overlap = k
                    break
            S7.extend(block[overlap:])
        
        # Verify
        all_perms = set(permutations(range(1, 8)))
        covered = set()
        for i in range(len(S7) - 6):
            w = tuple(S7[i:i+7])
            if w in all_perms:
                covered.add(w)
        
        if len(covered) == len(all_perms):
            # Now apply local removal search on S7
            S = S7[:]
            counts = {}
            for i in range(len(S) - 6):
                p = tuple(S[i:i+7])
                counts[p] = counts.get(p, 0) + 1
            
            best_S = S[:]
            best_len = len(S)
            
            while time.time() < deadline - 2:
                # Try removing characters from right to left
                for i in range(len(S) - 1, -1, -1):
                    if time.time() > deadline - 2:
                        break
                    
                    # Check if removal is valid
                    start_old = max(0, i - 6)
                    end_old = min(i, len(S) - 7)
                    
                    valid = True
                    old_perms_to_dec = []
                    
                    for j in range(start_old, end_old + 1):
                        p = tuple(S[j:j+7])
                        if counts.get(p, 0) == 1:
                            valid = False
                            break
                        old_perms_to_dec.append(p)
                    
                    if not valid:
                        continue
                    
                    # Perform removal
                    for p in old_perms_to_dec:
                        counts[p] -= 1
                        if counts[p] == 0:
                            del counts[p]
                    
                    # Add new straddling windows
                    start_new = max(0, i - 6)
                    end_new = min(i - 1, len(S) - 1 - 7)
                    
                    for k in range(start_new, end_new + 1):
                        window_chars = []
                        for idx in range(k, k+7):
                            s_idx = idx if idx < i else idx + 1
                            window_chars.append(S[s_idx])
                        p_new = tuple(window_chars)
                        counts[p_new] = counts.get(p_new, 0) + 1
                    
                    S.pop(i)
                    best_S = S[:]
                    best_len = len(S)
                    
                    if time.time() > deadline - 2:
                        break
                
                # If we made progress, continue; otherwise break
                if len(S) == best_len:
                    # No progress in this pass, break
                    break
        
        return "".join(map(str, best_S))
    
    S = build_standard(n)
    return "".join(map(str, S))
