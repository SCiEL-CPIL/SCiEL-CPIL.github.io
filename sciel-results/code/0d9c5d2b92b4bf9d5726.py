import numpy as np
import time
import itertools
import random

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # Tensor shape: (n*m, m*p, p*n) = (20, 25, 20)
    # T[i,j,k] = 1 if i//m == j%m and j//m == k%n, else 0
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Generate candidate rank-1 tensors
    # Each candidate is (u, v, w) where u in Z^{n*m}, v in Z^{m*p}, w in Z^{p*n}
    # and u ⊗ v ⊗ w approximates some part of T
    
    # Strategy: use structured candidates based on Strassen-like patterns
    # and random sparse vectors
    
    candidates = []
    
    # Add the known 85-term decomposition as a baseline
    # We'll generate it the same way as before but as candidates
    def get_baseline_candidates():
        cands = []
        
        # 2x2x2 Strassen factors
        s_u = [
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, -1, 1, 0, 0, -1]
        ]
        s_v = [
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, -1, 0, 1, 0, 1]
        ]
        s_w = [
            [1, 0, 0, 1, -1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ]
        
        # 4x4x4 part: 7 outer terms x 7 inner terms = 49 candidates
        terms = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
        ]
        
        for t_idx, (a_coeff, b_coeff, c_coeff) in enumerate(terms):
            for inner_k in range(7):
                u_vec = np.array(s_u)[:, inner_k]
                v_vec = np.array(s_v)[:, inner_k]
                w_vec = np.array(s_w)[:, inner_k]
                
                u_global = np.zeros(n*m)
                v_global = np.zeros(m*p)
                w_global = np.zeros(p*n)
                
                for (a_br, a_bc), a_c in a_coeff.items():
                    for lr in range(2):
                        for lc in range(2):
                            local_idx = lr*2 + lc
                            if u_vec[local_idx] != 0:
                                gi = 2*a_br + lr
                                gj = 2*a_bc + lc
                                u_global[gi*5 + gj] += a_c * u_vec[local_idx]
                
                for (b_br, b_bc), b_c in b_coeff.items():
                    for lr in range(2):
                        for lc in range(2):
                            local_idx = lr*2 + lc
                            if v_vec[local_idx] != 0:
                                gj = 2*b_br + lr
                                gk = 2*b_bc + lc
                                v_global[gj*5 + gk] += b_c * v_vec[local_idx]
                
                for (c_br, c_bc), c_c in c_coeff.items():
                    for lr in range(2):
                        for lc in range(2):
                            local_idx = lr*2 + lc
                            if w_vec[local_idx] != 0:
                                gk = 2*c_bc + lc
                                gi = 2*c_br + lr
                                w_global[gk*4 + gi] += c_c * w_vec[local_idx]
                
                cands.append((u_global, v_global, w_global))
        
        # 4x4x1 part: 16 candidates
        for i in range(4):
            for j in range(4):
                u = np.zeros(n*m)
                v = np.zeros(m*p)
                w = np.zeros(p*n)
                u[i*5+j] = 1
                v[j*5+4] = 1
                w[4*4+i] = 1
                cands.append((u, v, w))
        
        # 4x1x4 part: 16 candidates
        for i in range(4):
            for k in range(4):
                u = np.zeros(n*m)
                v = np.zeros(m*p)
                w = np.zeros(p*n)
                u[i*5+4] = 1
                v[4*5+k] = 1
                w[k*4+i] = 1
                cands.append((u, v, w))
        
        # 4x1x1 part: 4 candidates
        for i in range(4):
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            u[i*5+4] = 1
            v[4*5+4] = 1
            w[4*4+i] = 1
            cands.append((u, v, w))
        
        return cands
    
    baseline_cands = get_baseline_candidates()
    candidates.extend(baseline_cands)
    
    # Verify baseline is correct
    def verify_decomposition(cands, selected):
        T_recon = np.zeros_like(T, dtype=int)
        for idx in selected:
            u, v, w = cands[idx]
            T_recon += np.einsum('i,j,k->ijk', u, v, w)
        return np.array_equal(T_recon, T)
    
    # Check if all 85 baseline candidates work
    all_indices = list(range(len(baseline_cands)))
    if not verify_decomposition(candidates, all_indices):
        # Fallback to original construction
        return _original_construct()
    
    # Greedy set cover: try to find a subset of candidates that exactly equals T
    # Start with the full baseline, try to remove candidates
    
    # First, try greedy addition from scratch with a larger candidate pool
    # Generate additional random candidates
    random.seed(42)
    for _ in range(200):
        if time.time() > deadline - 30:
            break
        # Generate sparse random candidates
        u = np.zeros(n*m)
        v = np.zeros(m*p)
        w = np.zeros(p*n)
        
        # Sparse: 1-3 non-zero entries each
        for _ in range(random.randint(1, 3)):
            idx = random.randint(0, n*m-1)
            u[idx] = random.choice([-2, -1, 1, 2])
        for _ in range(random.randint(1, 3)):
            idx = random.randint(0, m*p-1)
            v[idx] = random.choice([-2, -1, 1, 2])
        for _ in range(random.randint(1, 3)):
            idx = random.randint(0, p*n-1)
            w[idx] = random.choice([-2, -1, 1, 2])
        
        # Only keep if it makes progress (reduces some error)
        T_recon = np.einsum('i,j,k->ijk', u, v, w)
        if np.any(T_recon != 0):
            candidates.append((u, v, w))
    
    # Greedy algorithm: start empty, add candidates that maximize progress
    # "Progress" = number of tensor entries correctly matched that weren't before
    
    T_target = T.astype(int)
    T_current = np.zeros_like(T_target)
    remaining = T_target.copy()
    
    selected = []
    available = list(range(len(candidates)))
    
    # Greedy: repeatedly pick the candidate that maximizes the reduction in |remaining|
    # while not introducing errors
    
    max_iter = 100
    for _ in range(max_iter):
        if time.time() > deadline - 20:
            break
        if np.all(remaining == 0):
            break
        if not available:
            break
        
        best_idx = None
        best_score = -1
        best_new_remaining = None
        
        for idx in available:
            u, v, w = candidates[idx]
            T_add = np.einsum('i,j,k->ijk', u, v, w)
            new_remaining = remaining - T_add
            
            # Score: number of entries that become exactly 0, minus entries that become non-zero
            correct = np.sum((remaining != 0) & (new_remaining == 0))
            wrong = np.sum((remaining == 0) & (new_remaining != 0))
            score = correct - wrong
            
            if score > best_score:
                best_score = score
                best_idx = idx
                best_new_remaining = new_remaining
        
        if best_idx is None or best_score <= 0:
            break
        
        selected.append(best_idx)
        remaining = best_new_remaining
        available.remove(best_idx)
        
        if np.all(remaining == 0):
            break
    
    # If greedy found a solution, use it
    if np.all(remaining == 0) and len(selected) < len(baseline_cands):
        # Build the result
        U = np.zeros((n*m, len(selected)), dtype=float)
        V = np.zeros((m*p, len(selected)), dtype=float)
        W = np.zeros((p*n, len(selected)), dtype=float)
        
        for col, idx in enumerate(selected):
            u, v, w = candidates[idx]
            U[:, col] = u
            V[:, col] = v
            W[:, col] = w
        
        # Convert to list of lists with integer/half-integer values
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: use the baseline 85-term decomposition
    return _original_construct()

def _original_construct():
    n, m, p = DIMENSIONS
    rank = 85
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def get_strassen_2x2_factors():
        u = [[0]*7 for _ in range(4)]
        v = [[0]*7 for _ in range(4)]
        w = [[0]*7 for _ in range(4)]
        
        u[0][0]=1; u[3][0]=1
        v[0][0]=1; v[3][0]=1
        w[0][0]=1; w[3][0]=1
        
        u[2][1]=1; u[3][1]=1
        v[0][1]=1
        w[2][1]=1; w[3][1]=-1
        
        u[0][2]=1
        v[1][2]=1; v[3][2]=-1
        w[1][2]=1; w[3][2]=1
        
        u[3][3]=1
        v[2][3]=1; v[0][3]=-1
        w[0][3]=1; w[2][3]=1
        
        u[0][4]=1; u[1][4]=1
        v[3][4]=1
        w[0][4]=-1; w[1][4]=1
        
        u[0][5]=-1; u[2][5]=1
        v[0][5]=1; v[1][5]=1
        w[3][5]=1
        
        u[1][6]=1; u[3][6]=-1
        v[2][6]=1; v[3][6]=1
        w[0][6]=1
        
        return u, v, w

    s_u, s_v, s_w = get_strassen_2x2_factors()
    
    def add_4x4_term(a_block_coeffs, b_block_coeffs, c_block_coeffs, start_col):
        for inner_k in range(7):
            u_vec = [s_u[r][inner_k] for r in range(4)]
            v_vec = [s_v[r][inner_k] for r in range(4)]
            w_vec = [s_w[r][inner_k] for r in range(4)]
            
            col_idx = start_col + inner_k
            
            for (a_br, a_bc), a_coeff in a_block_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        u_val = u_vec[local_idx]
                        if u_val != 0:
                            global_i = 2 * a_br + lr
                            global_j = 2 * a_bc + lc
                            global_u_row = global_i * 5 + global_j
                            U[global_u_row][col_idx] += a_coeff * u_val
            
            for (b_br, b_bc), b_coeff in b_block_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        v_val = v_vec[local_idx]
                        if v_val != 0:
                            global_j = 2 * b_br + lr
                            global_k = 2 * b_bc + lc
                            global_v_row = global_j * 5 + global_k
                            V[global_v_row][col_idx] += b_coeff * v_val
            
            for (c_br, c_bc), c_coeff in c_block_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        w_val = w_vec[local_idx]
                        if w_val != 0:
                            global_k = 2 * c_bc + lc
                            global_i = 2 * c_br + lr
                            global_w_row = global_k * 4 + global_i
                            W[global_w_row][col_idx] += c_coeff * w_val

    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    for t_idx, (a_coeff, b_coeff, c_coeff) in enumerate(terms):
        start_col = t_idx * 7
        add_4x4_term(a_coeff, b_coeff, c_coeff, start_col)
        
    col = 49
    
    for i in range(4):
        for j in range(4):
            U[i*5 + j][col] = 1
            V[j*5 + 4][col] = 1
            W[4*4 + i][col] = 1
            col += 1
            
    for i in range(4):
        for k in range(4):
            U[i*5 + 4][col] = 1
            V[4*5 + k][col] = 1
            W[k*4 + i][col] = 1
            col += 1
            
    for i in range(4):
        U[i*5 + 4][col] = 1
        V[4*5 + 4][col] = 1
        W[4*4 + i][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
