import numpy as np
import time
import random

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Fallback rank 39 construction
    def construct_39():
        rank = 39
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        def add_term(u_indices, v_indices, w_indices):
            nonlocal col
            for idx in u_indices:
                U[idx][col] += 1
            for idx in v_indices:
                V[idx][col] += 1
            for idx in w_indices:
                W[idx][col] += 1
            col += 1

        def add_strassen_block(k_start):
            b00 = 0 * 7 + k_start
            b01 = 0 * 7 + k_start + 1
            b10 = 1 * 7 + k_start
            b11 = 1 * 7 + k_start + 1
            
            c00 = k_start * 2 + 0
            c01 = (k_start + 1) * 2 + 0
            c10 = k_start * 2 + 1
            c11 = (k_start + 1) * 2 + 1
            
            add_term([0, 4], [b00, b11], [c00, c11])
            add_term([3, 4], [b00], [c10, c11])
            W[c11][col-1] = -1
            
            add_term([0], [b01, b11], [c01, c11])
            V[b11][col-1] = -1
            
            add_term([4], [b10, b00], [c00, c10])
            V[b00][col-1] = -1
            
            add_term([0, 1], [b11], [c00, c01])
            W[c00][col-1] = -1
            
            add_term([3, 0], [b00, b01], [c11])
            U[0][col-1] = -1
            
            add_term([1, 4], [b10, b11], [c00])
            U[4][col-1] = -1

        add_strassen_block(0)
        add_strassen_block(2)
        add_strassen_block(4)
        
        b06 = 0 * 7 + 6
        b16 = 1 * 7 + 6
        c06 = 6 * 2 + 0
        c16 = 6 * 2 + 1
        
        add_term([0], [b06], [c06])
        add_term([1], [b16], [c06])
        add_term([3], [b06], [c16])
        add_term([4], [b16], [c16])
        
        for k in range(7):
            b2k = 2 * 7 + k
            c0k = k * 2 + 0
            c1k = k * 2 + 1
            add_term([2], [b2k], [c0k])
            add_term([5], [b2k], [c1k])
            
        return {"U": U, "V": V, "W": W}

    start_time = time.time()
    deadline = start_time + 140 # 10s margin for safety

    # Grid of allowed values (half-integers)
    grid_vals = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

    # Prepare triples and maps
    triples = []
    triple_map_U = [[] for _ in range(n * m)]
    triple_map_V = [[] for _ in range(m * p)]
    triple_map_W = [[] for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            a = i * m + j
            for k in range(p):
                b = j * p + k
                g = k * n + i
                triples.append((a, b, g))
                triple_map_U[a].append(idx)
                triple_map_V[b].append(idx)
                triple_map_W[g].append(idx)
                idx += 1
    num_eq = len(triples)

    def try_rank(R):
        if time.time() > deadline:
            return None
            
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # Sparse initialization
        for _ in range(R * 3):
             row = np.random.randint(0, n * m)
             col = np.random.randint(0, R)
             U[row, col] = random.choice(grid_vals)
             
             row = np.random.randint(0, m * p)
             col = np.random.randint(0, R)
             V[row, col] = random.choice(grid_vals)
             
             row = np.random.randint(0, p * n)
             col = np.random.randint(0, R)
             W[row, col] = random.choice(grid_vals)
             
        # Initialize S (sums for each triple)
        S = np.zeros(num_eq)
        for idx, (a, b, g) in enumerate(triples):
            S[idx] = np.sum(U[a] * V[b] * W[g])
            
        current_energy = np.sum((S - 1.0)**2)
        best_energy = current_energy
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        T = 10.0
        alpha = 0.999
        max_iters = 50000
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            factor_idx = np.random.randint(0, 3)
            if factor_idx == 0:
                mat = U
                rows = n * m
                map_list = triple_map_U
            elif factor_idx == 1:
                mat = V
                rows = m * p
                map_list = triple_map_V
            else:
                mat = W
                rows = p * n
                map_list = triple_map_W
                
            r = np.random.randint(0, rows)
            c = np.random.randint(0, R)
            
            old_val = mat[r, c]
            new_val = random.choice(grid_vals)
            
            if old_val == new_val:
                continue
                
            delta_val = new_val - old_val
            
            # Calculate delta E and update S
            delta_E = 0.0
            affected_indices = map_list[r]
            for idx in affected_indices:
                a, b, g = triples[idx]
                if factor_idx == 0:
                    factor_prod = V[b, c] * W[g, c]
                elif factor_idx == 1:
                    factor_prod = U[a, c] * W[g, c]
                else:
                    factor_prod = U[a, c] * V[b, c]
                    
                delta_S = delta_val * factor_prod
                old_S = S[idx]
                new_S = old_S + delta_S
                delta_E += (new_S - 1.0)**2 - (old_S - 1.0)**2
                S[idx] = new_S
                
            mat[r, c] = new_val
            
            if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
                current_energy += delta_E
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_energy < 1e-12:
                         valid = True
                         for idx, (a, b, g) in enumerate(triples):
                             if abs(S[idx] - 1.0) > 1e-9:
                                 valid = False
                                 break
                         if valid:
                             return {
                                 "U": best_U.tolist(),
                                 "V": best_V.tolist(),
                                 "W": best_W.tolist()
                             }
            else:
                # Revert
                mat[r, c] = old_val
                for idx in affected_indices:
                    a, b, g = triples[idx]
                    if factor_idx == 0:
                        factor_prod = V[b, c] * W[g, c]
                    elif factor_idx == 1:
                        factor_prod = U[a, c] * W[g, c]
                    else:
                        factor_prod = U[a, c] * V[b, c]
                    delta_S = delta_val * factor_prod
                    S[idx] -= delta_S
                current_energy -= delta_E
                
            T *= alpha
            
        return None

    # Try ranks 38 down to 35
    for R in range(38, 34, -1):
        if time.time() > deadline:
            break
        result = try_rank(R)
        if result:
            return result

    # Fallback
    return construct_39()
# EVOLVE-BLOCK-END
