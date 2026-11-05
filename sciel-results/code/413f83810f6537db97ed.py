import time
import sys
import random
from itertools import permutations

def construct(n):
    if n <= 1:
        return "1" if n == 1 else ""
    
    # Standard construction for n <= 5 is optimal or close enough, 
    # but for n=7 we need optimization.
    # We will build up to n=6 using standard method, 
    # then optimize the ordering of blocks for n=7.
    
    # Step 1: Build S up to n-1 (or n=6 if n=7)
    # For n=7, we build S_6.
    # For general n, we build S_{n-1}.
    
    def build_standard(m):
        if m == 1:
            return [1]
        S = [1]
        for k in range(2, m + 1):
            seen = set()
            perms_order = []
            for i in range(len(S) - (k - 1) + 1):
                window = tuple(S[i:i + k - 1])
                # Only valid permutations (distinct elements)
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
        # Standard is 33. Optimal is 33.
        return "".join(map(str, build_standard(4)))
    if n == 5:
        # Standard is 153. Optimal is 153? 
        # Actually, for n=5, shortest is 153.
        return "".join(map(str, build_standard(5)))
    if n == 6:
        # Standard is 873. Shortest is 873.
        return "".join(map(str, build_standard(6)))
    
    if n == 7:
        start_time = time.time()
        deadline = start_time + 110
        
        # 1. Build S_6
        S6 = build_standard(6)
        
        # 2. Extract permutations of 1..6 from S6
        seen = set()
        perms_order = []
        for i in range(len(S6) - 5):
            window = tuple(S6[i:i + 6])
            if len(set(window)) == 6 and window not in seen:
                seen.add(window)
                perms_order.append(window)
        
        # perms_order has 720 elements
        num_perms = len(perms_order)
        
        # 3. Precompute overlap costs
        # Cost[i][j] = 13 - overlap(perms_order[i], perms_order[j])
        # We want to minimize sum of costs.
        # overlap(A, B) is max k such that A[-k:] == B[:k]
        
        # To save memory and time, we might not store the full matrix if 720x720 is too big?
        # 720*720 = 518400. Small enough.
        costs = [[0]*num_perms for _ in range(num_perms)]
        
        for i in range(num_perms):
            pi = perms_order[i]
            for j in range(num_perms):
                pj = perms_order[j]
                # Calculate overlap
                max_k = 6
                ov = 0
                for k in range(max_k, -1, -1):
                    if k == 0:
                        ov = 0
                        break
                    if pi[-k:] == pj[:k]:
                        ov = k
                        break
                costs[i][j] = 13 - ov
        
        # 4. Optimize the order using Local Search
        # Start with the standard order (index 0 to 719)
        path = list(range(num_perms))
        
        def get_total_cost(path):
            total = 0
            for i in range(len(path) - 1):
                total += costs[path[i]][path[i+1]]
            return total
        
        current_cost = get_total_cost(path)
        best_cost = current_cost
        best_path = path[:]
        
        # 2-Opt Search
        # Try reversing segments
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            random.shuffle(path) # Random restarts if stuck? 
            # Actually, let's do deterministic 2-opt first on the current best.
            # If we want to be safe, we do a few random restarts.
            
            # Let's do a simple 2-opt on the current path
            for i in range(num_perms - 1):
                if time.time() > deadline - 10:
                    break
                for j in range(i + 1, num_perms):
                    # Reverse path[i..j]
                    # Cost change:
                    # Old edges: (i-1, i) and (j, j+1)
                    # New edges: (i-1, j) and (i, j+1)
                    # Note: path is 0-indexed.
                    
                    c1 = costs[path[i-1]][path[i]] if i > 0 else 0
                    c2 = costs[path[j]][path[j+1]] if j < num_perms - 1 else 0
                    
                    new_c1 = costs[path[i-1]][path[j]] if i > 0 else 0
                    new_c2 = costs[path[i]][path[j+1]] if j < num_perms - 1 else 0
                    
                    delta = (new_c1 + new_c2) - (c1 + c2)
                    
                    if delta < 0:
                        # Apply reversal
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
            return "".join(map(str, S7))
        else:
            # Fallback to standard if optimization failed or produced invalid result
            S_std = build_standard(7)
            return "".join(map(str, S_std))
    
    # Fallback for n > 7 (not expected, but safe)
    S = build_standard(n)
    return "".join(map(str, S))
