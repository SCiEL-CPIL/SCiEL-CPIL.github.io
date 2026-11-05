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
        deadline = start_time + 140
        
        S6 = build_standard(6)
        
        seen = set()
        perms_order = []
        for i in range(len(S6) - 5):
            window = tuple(S6[i:i + 6])
            if len(set(window)) == 6 and window not in seen:
                seen.add(window)
                perms_order.append(window)
        
        num_perms = len(perms_order)
        
        # Build cost matrix: cost[i][j] = cost of going from perm i to perm j
        # Block for perm P is: P + [7] + P (length 13)
        # Overlap between end of block_i and start of block_j
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
        
        # Multi-start greedy nearest-neighbor + 2-opt
        best_cost = float('inf')
        best_path = list(range(num_perms))
        
        def get_total_cost(path):
            total = 0
            for i in range(len(path) - 1):
                total += costs[path[i]][path[i+1]]
            return total
        
        # Multiple random starts
        num_starts = 20
        for s in range(num_starts):
            if time.time() > deadline - 5:
                break
            
            if s == 0:
                # Standard order
                path = list(range(num_perms))
            else:
                # Random start
                path = list(range(num_perms))
                random.shuffle(path)
            
            current_cost = get_total_cost(path)
            
            # 2-opt improvement
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
                            path[i:j+1] = list(reversed(path[i:j+1]))
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
