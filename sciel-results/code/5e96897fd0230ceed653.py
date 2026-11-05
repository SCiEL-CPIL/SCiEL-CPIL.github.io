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
        deadline = start_time + 145
        
        # Build S6 using standard construction
        S6 = build_standard(6)
        
        # Extract all 6-permutations in order of appearance in S6
        seen = set()
        perms_order = []
        perm_index = {}
        for i in range(len(S6) - 5):
            window = tuple(S6[i:i + 6])
            if len(set(window)) == 6:
                if window not in seen:
                    seen.add(window)
                    perm_index[window] = len(perms_order)
                    perms_order.append(window)
        
        num_perms = len(perms_order)
        
        # Build cost matrix for transitions between permutation blocks
        # Each block is P + [7] + P, length 13
        # Overlap between block_i and block_j:
        # block_i ends with P_i, block_j starts with P_j
        # Overlap = max k such that P_i[-k:] == P_j[:k]
        # Cost = 13 - overlap
        
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
        
        # Initial solution: natural order
        path = list(range(num_perms))
        
        def get_total_cost(p):
            total = 0
            for i in range(len(p) - 1):
                total += costs[p[i]][p[i+1]]
            return total
        
        current_cost = get_total_cost(path)
        best_cost = current_cost
        best_path = path[:]
        
        # Simulated Annealing with multiple move types
        T_start = 5.0
        T_end = 0.05
        alpha = 0.9995
        T = T_start
        
        step = 0
        no_improve_count = 0
        patience = 300000
        
        while time.time() < deadline - 10:
            move_type = random.random()
            
            if move_type < 0.6:
                # 2-opt reversal
                i = random.randint(0, num_perms - 2)
                j = random.randint(i + 1, num_perms - 1)
                
                old_c1 = costs[path[i-1]][path[i]] if i > 0 else 0
                old_c2 = costs[path[j]][path[j+1]] if j < num_perms - 1 else 0
                new_c1 = costs[path[i-1]][path[j]] if i > 0 else 0
                new_c2 = costs[path[i]][path[j+1]] if j < num_perms - 1 else 0
                
                delta = (new_c1 + new_c2) - (old_c1 + old_c2)
                
                if delta < 0 or (T > T_end and random.random() < math.exp(-delta / T)):
                    path[i:j+1] = path[i:j+1][::-1]
                    current_cost += delta
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_path = path[:]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        
            elif move_type < 0.8:
                # Swap two elements
                i = random.randint(0, num_perms - 1)
                j = random.randint(0, num_perms - 1)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                
                a = path[i]
                b = path[j]
                
                neighbors = set()
                for pos in [i-1, i+1, j-1, j+1]:
                    if 0 <= pos < num_perms:
                        neighbors.add(pos)
                
                old_cost = 0
                new_cost = 0
                
                for pos in neighbors:
                    if pos == i or pos == j:
                        continue
                    if pos + 1 == i or pos - 1 == i:
                        if pos < i:
                            old_cost += costs[path[pos]][a]
                            new_cost += costs[path[pos]][b]
                        else:
                            old_cost += costs[a][path[pos]]
                            new_cost += costs[b][path[pos]]
                    if pos + 1 == j or pos - 1 == j:
                        if pos < j:
                            old_cost += costs[path[pos]][b]
                            new_cost += costs[path[pos]][a]
                        else:
                            old_cost += costs[b][path[pos]]
                            new_cost += costs[a][path[pos]]
                
                if abs(i - j) == 1:
                    if i < j:
                        old_cost += costs[a][b]
                        new_cost += costs[b][a]
                
                delta = new_cost - old_cost
                
                if delta < 0 or (T > T_end and random.random() < math.exp(-delta / T)):
                    path[i], path[j] = path[j], path[i]
                    current_cost += delta
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_path = path[:]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
            else:
                # Or-opt: move a segment of length 1, 2, or 3 to another position
                seg_len = random.choice([1, 2, 3])
                i = random.randint(0, num_perms - seg_len)
                j = random.randint(0, num_perms - seg_len)
                if abs(i - j) <= seg_len:
                    continue
                
                segment = path[i:i+seg_len]
                remaining = path[:i] + path[i+seg_len:]
                insert_pos = j if j < i else j - seg_len
                new_path = remaining[:insert_pos] + segment + remaining[insert_pos:]
                
                # Compute delta by comparing costs around the affected regions
                old_delta = 0
                new_delta = 0
                
                # Old edges broken
                if i > 0:
                    old_delta += costs[path[i-1]][path[i]]
                if i + seg_len < num_perms:
                    old_delta += costs[path[i+seg_len-1]][path[i+seg_len]]
                if j > 0 and j < num_perms:
                    old_delta += costs[path[j-1]][path[j]]
                if j + seg_len < num_perms:
                    old_delta += costs[path[j+seg_len-1]][path[j+seg_len]]
                
                # New edges created
                if insert_pos > 0:
                    new_delta += costs[new_path[insert_pos-1]][new_path[insert_pos]]
                if insert_pos + seg_len < len(new_path):
                    new_delta += costs[new_path[insert_pos+seg_len-1]][new_path[insert_pos+seg_len]]
                
                delta = new_delta - old_delta
                
                if delta < 0 or (T > T_end and random.random() < math.exp(-delta / T)):
                    path = new_path
                    current_cost += delta
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_path = path[:]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
            
            step += 1
            T *= alpha
            if T < T_end:
                T = T_end
            
            if no_improve_count > patience:
                path = best_path[:]
                current_cost = best_cost
                for _ in range(10):
                    i = random.randint(0, num_perms - 2)
                    j = random.randint(i + 1, num_perms - 1)
                    path[i:j+1] = path[i:j+1][::-1]
                current_cost = get_total_cost(path)
                T = T_start * 0.5
                no_improve_count = 0
        
        # Final local search polish
        path = best_path[:]
        current_cost = get_total_cost(path)
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for i in range(num_perms - 1):
                if time.time() > deadline - 3:
                    break
                for j in range(i + 1, num_perms):
                    c1 = costs[path[i-1]][path[i]] if i > 0 else 0
                    c2 = costs[path[j]][path[j+1]] if j < num_perms - 1 else 0
                    new_c1 = costs[path[i-1]][path[j]] if i > 0 else 0
                    new_c2 = costs[path[i]][path[j+1]] if j < num_perms - 1 else 0
                    delta = (new_c1 + new_c2) - (c1 + c2)
                    if delta < 0:
                        path[i:j+1] = path[i:j+1][::-1]
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
            return "".join(map(str, S7))
        else:
            S_std = build_standard(7)
            return "".join(map(str, S_std))
    
    S = build_standard(n)
    return "".join(map(str, S))
