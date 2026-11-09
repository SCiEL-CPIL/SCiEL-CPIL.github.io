"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _build_rank28():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    S = [
        ({(0,1),(3,1)}, {(0,1),(3,1)}, {(0,1),(3,1)}),
        ({(2,1),(3,1)}, {(0,1)}, {(1,1),(3,-1)}),
        ({(0,1)}, {(1,1),(3,-1)}, {(2,1),(3,1)}),
        ({(3,1)}, {(2,1),(0,-1)}, {(0,1),(1,1)}),
        ({(0,1),(1,1)}, {(3,1)}, {(0,-1),(2,1)}),
        ({(2,1),(0,-1)}, {(0,1),(1,1)}, {(3,1)}),
        ({(1,1),(3,-1)}, {(2,1),(3,1)}, {(0,1)}),
    ]
    
    j_offsets = [0, 2]
    k_offsets = [0, 2]
    
    col = 0
    for j_off in j_offsets:
        for k_off in k_offsets:
            for u_terms, v_terms, w_terms in S:
                for (li, coeff) in u_terms:
                    i, j = li // 2, li % 2
                    gi, gj = i, j + j_off
                    U[gi * m + gj][col] += coeff
                
                for (li, coeff) in v_terms:
                    j, k = li // 2, li % 2
                    gj, gk = j + j_off, k + k_off
                    V[gj * p + gk][col] += coeff
                
                for (li, coeff) in w_terms:
                    k, i = li // 2, li % 2
                    gk, gi = k + k_off, i
                    W[gk * n + gi][col] += coeff
                
                col += 1
    
    actual_rank = col
    U = [row[:actual_rank] for row in U]
    V = [row[:actual_rank] for row in V]
    W = [row[:actual_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n=2, m=4, p=4, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_find_rank12_sub():
    """
    Try to find a rank 12 decomposition for M_{2,4,2} with half-integer entries.
    This corresponds to integer entries U', V', W' satisfying sum U'V'W' = 8.
    """
    n_sub, m_sub, p_sub = 2, 4, 2
    target_rank = 12
    
    # Search for integer U', V' with small values, solve for integer W'
    # Equation: sum_r U'[i,j,r] V'[j,k,r] W'[k,i,r] = 8
    # Unknowns W'[k,i,r] for k in 0..1, i in 0..1, r in 0..11 (48 unknowns)
    # Equations: 4 (for i=0,1, k=0,1)
    
    # We can set most W' to 0 and solve for the rest?
    # Or use least squares and check integrality.
    
    for _ in range(2000):
        # Random sparse integer U', V'
        # Values in {-2, -1, 1, 2}
        U_prime = np.zeros((n_sub * m_sub, target_rank), dtype=int)
        V_prime = np.zeros((m_sub * p_sub, target_rank), dtype=int)
        
        for r in range(target_rank):
            # Sparsity: ~2 non-zeros
            nz_u = random.sample(range(n_sub * m_sub), 2)
            for idx in nz_u:
                U_prime[idx, r] = random.choice([-2, -1, 1, 2])
            
            nz_v = random.sample(range(m_sub * p_sub), 2)
            for idx in nz_v:
                V_prime[idx, r] = random.choice([-2, -1, 1, 2])
        
        # Build linear system for W_prime
        # A W_flat = b
        # A has 4 rows (for each i,k pair)
        # W_flat has 48 elements (4 * 12)
        A = np.zeros((4, n_sub * p_sub * target_rank))
        b = np.full(4, 8.0)
        
        for i in range(n_sub):
            for k in range(p_sub):
                eq_idx = i * p_sub + k
                for r in range(target_rank):
                    # coeff for W'[k,i,r] is sum_j U'[i,j,r] V'[j,k,r]
                    coeff = 0
                    for j in range(m_sub):
                        coeff += U_prime[i * m_sub + j, r] * V_prime[j * p_sub + k, r]
                    if coeff != 0:
                        w_idx = (k * n_sub + i) * target_rank + r
                        A[eq_idx, w_idx] = coeff
        
        # Solve
        W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
        
        # Check if W_flat is close to integer
        # And check reconstruction
        W_prime_sol = W_flat.reshape((n_sub * p_sub, target_rank))
        
        # Check integrality
        if not np.all(np.abs(W_prime_sol - np.round(W_prime_sol)) < 1e-5):
            continue
        
        W_prime_int = np.round(W_prime_sol).astype(int)
        
        # Verify exactness with integers
        valid = True
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    s_val = 0
                    for r in range(target_rank):
                        s_val += U_prime[i*m_sub+j, r] * V_prime[j*p_sub+k, r] * W_prime_int[k*n_sub+i, r]
                    if s_val != 8:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        
        if valid:
            # Convert back to half-integers
            U_sol = (U_prime / 2.0).tolist()
            V_sol = (V_prime / 2.0).tolist()
            W_sol = (W_prime_int / 2.0).tolist()
            return U_sol, V_sol, W_sol
            
    return None

def construct():
    start = time.time()
    deadline = start + 140.0
    
    result = _build_rank28()
    best_rank = len(result["U"][0])
    
    # Try to find a better solution
    if time.time() < deadline - 20:
        sub_sol = _try_find_rank12_sub()
        if sub_sol:
            U_sub, V_sub, W_sub = sub_sol
            # Construct full 2x4x4 solution
            # Rank 24
            U_full = [[0] * 24 for _ in range(8)]
            V_full = [[0] * 24 for _ in range(16)]
            W_full = [[0] * 24 for _ in range(8)]
            
            # k=0,1 block
            for r in range(12):
                for i in range(2):
                    for j in range(4):
                        U_full[i*4+j][r] = U_sub[i*4+j][r]
                for j in range(4):
                    for k in range(2):
                        V_full[j*4+k][r] = V_sub[j*2+k][r]
                for k in range(2):
                    for i in range(2):
                        W_full[k*2+i][r] = W_sub[k*2+i][r]
                        
            # k=2,3 block
            for r in range(12):
                for i in range(2):
                    for j in range(4):
                        U_full[i*4+j][r+12] = U_sub[i*4+j][r]
                for j in range(4):
                    for k in range(2):
                        V_full[j*4+2+k][r+12] = V_sub[j*2+k][r]
                for k in range(2):
                    for i in range(2):
                        W_full[(2+k)*2+i][r+12] = W_sub[k*2+i][r]
            
            # Verify
            if _verify_decomposition(U_full, V_full, W_full):
                result = {"U": U_full, "V": V_full, "W": W_full}
                best_rank = 24
    
    return result
# EVOLVE-BLOCK-END
