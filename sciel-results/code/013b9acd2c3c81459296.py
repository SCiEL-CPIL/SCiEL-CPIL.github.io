import time
import random
import numpy as np
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    if n < 7:
        return _classic_construct(n)
    
    if n == 7:
        # 1. Generate permutations
        perms = list(permutations(range(1, 7)))
        N = len(perms) # 5040
        
        # 2. Precompute Overlap Matrix
        O = np.zeros((N, N), dtype=np.uint8)
        
        maps = [None] * 7
        for k in range(1, 7):
            m = {}
            for j in range(N):
                p = perms[j]
                pref = p[:k]
                if pref not in m:
                    m[pref] = []
                m[pref].append(j)
            maps[k] = m
            
        for i in range(N):
            p_i = perms[i]
            for k in range(6, 0, -1):
                suff = p_i[-k:]
                if suff in maps[k]:
                    candidates = maps[k][suff]
                    for j in candidates:
                        if O[i][j] == 0:
                            O[i][j] = k
                            
        # 3. Get Initial Order (Classic)
        order = _get_classic_order(7)
        
        # 4. Calculate Initial Energy
        def get_energy(order, O):
            e = 0
            for k in range(len(order) - 1):
                e += 7 - O[order[k]][order[k+1]]
            return e
            
        current_energy = get_energy(order, O)
        best_energy = current_energy
        best_order = order[:]
        
        # 5. Simulated Annealing
        T = 5.0
        T_min = 0.001
        cooling_rate = 0.9995
        
        random.seed(42)
        
        while time.time() < deadline:
            move_type = random.randint(0, 1)
            
            if move_type == 0:
                # Swap adjacent
                i = random.randint(0, N - 2)
                j = i + 1
                
                u = order[i-1] if i > 0 else None
                v = order[i]
                w = order[i+1]
                x = order[i+2] if i+2 < N else None
                
                old_e = 0
                new_e = 0
                
                if u is not None:
                    old_e += 7 - O[u][v]
                    new_e += 7 - O[u][w]
                
                old_e += 7 - O[v][w]
                new_e += 7 - O[w][v]
                
                if x is not None:
                    old_e += 7 - O[w][x]
                    new_e += 7 - O[v][x]
                    
                delta = new_e - old_e
                
                if delta < 0 or random.random() < np.exp(-delta / T):
                    order[i], order[i+1] = order[i+1], order[i]
                    current_energy += delta
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_order = order[:]
                        
            else:
                # Short 2-opt: Reverse segment [i, j]
                # Pick i
                i = random.randint(0, N - 2)
                # Pick length L
                L = random.randint(2, 10)
                j = i + L - 1
                if j >= N:
                    j = N - 1
                    L = j - i + 1
                    if L < 2: continue
                
                # Calculate delta E
                # Old edges: (i-1, i), (i, i+1), ..., (j-1, j), (j, j+1)
                # New edges: (i-1, j), (j, j-1), ..., (i+1, i), (i, j+1)
                
                old_e = 0
                new_e = 0
                
                # Left boundary
                if i > 0:
                    old_e += 7 - O[order[i-1]][order[i]]
                    new_e += 7 - O[order[i-1]][order[j]]
                
                # Internal edges (reversed)
                for k in range(i, j):
                    old_e += 7 - O[order[k]][order[k+1]]
                    new_e += 7 - O[order[k+1]][order[k]]
                    
                # Right boundary
                if j < N - 1:
                    old_e += 7 - O[order[j]][order[j+1]]
                    new_e += 7 - O[order[i]][order[j+1]]
                    
                delta = new_e - old_e
                
                if delta < 0 or random.random() < np.exp(-delta / T):
                    # Reverse segment
                    order[i:j+1] = order[i:j+1][::-1]
                    current_energy += delta
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_order = order[:]

            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min

        # Construct word from best_order
        word = ""
        for idx in range(len(best_order)):
            p = perms[best_order[idx]]
            block = "".join(map(str, p))
            if idx == 0:
                word = block
            else:
                ov = O[best_order[idx-1]][best_order[idx]]
                word += block[ov:]
        
        return word

    else:
        return _classic_construct(n)

def _classic_construct(n):
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))

def _get_classic_order(n):
    word = _classic_construct(n)
    word_list = list(map(int, word))
    perms, seen = [], set()
    for i in range(len(word_list) - n + 1):
        window = tuple(word_list[i:i + n])
        if len(set(window)) == n and window not in seen:
            seen.add(window)
            perms.append(window)
    all_perms = list(permutations(range(1, n+1)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    order = [perm_to_idx[p] for p in perms]
    return order
