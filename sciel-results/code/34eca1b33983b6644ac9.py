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
    Try to find a rank 12 decomposition for M_{2,4,2}.
    If found, return U, V, W for the sub-problem.
    U: 8 x 12, V: 8 x 12, W: 4 x 12.
    """
    n_sub, m_sub, p_sub = 2, 4, 2
    target_rank = 12
    
    # We will search for integer solutions in {-1, 0, 1} for U and V,
    # and solve for W using least squares, then check if W is integer/half-integer.
    
    for _ in range(500): # 500 restarts
        # Randomly initialize U and V with sparse integers
        U_rand = np.zeros((n_sub * m_sub, target_rank))
        V_rand = np.zeros((m_sub * p_sub, target_rank))
        
        # Sparsity pattern: each column has ~2 non-zeros
        for r in range(target_rank):
            # U column
            nz_u = random.sample(range(n_sub * m_sub), 2)
            for idx in nz_u:
                U_rand[idx, r] = random.choice([-1, 1])
            
            # V column
            nz_v = random.sample(range(m_sub * p_sub), 2)
            for idx in nz_v:
                V_rand[idx, r] = random.choice([-1, 1])
        
        # Solve for W
        # Equation: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
        # For fixed (i,k), we have: sum_r (U[i,j,r] V[j,k,r]) W[k,i,r] = 1
        # This is linear in W.
        # We have 4 equations (for i=0,1 and k=0,1) and 12 unknowns per (i,k)?
        # No, W has shape 4 x 12.
        # The equations are coupled.
        # Let's build the system A W_flat = b
        # A has 4 rows? No, 4 equations total?
        # Wait, the tensor is 2x4x2.
        # Equations: for each i in 0..1, k in 0..1:
        # sum_j sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
        # This is 4 equations.
        # Unknowns: 12 (W[0,0..11], W[1,0..11], W[2,0..11], W[3,0..11] -> 4*12=48 unknowns).
        # 4 equations, 48 unknowns. Highly underdetermined.
        # We can set 44 unknowns to 0 and solve for 4?
        # Or use minimum norm solution.
        
        A = np.zeros((4, n_sub * p_sub * target_rank))
        b = np.ones(4)
        
        for i in range(n_sub):
            for k in range(p_sub):
                eq_idx = i * p_sub + k
                for r in range(target_rank):
                    for j in range(m_sub):
                        coeff = U_rand[i * m_sub + j, r] * V_rand[j * p_sub + k, r]
                        if coeff != 0:
                            # W index: k * n_sub + i, column r
                            w_idx = (k * n_sub + i) * target_rank + r
                            A[eq_idx, w_idx] += coeff
        
        # Solve A W_flat = b
        # Use least squares
        W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
        
        # Check if solution is close to integer/half-integer
        # And check if the reconstruction is exact
        W_sol = W_flat.reshape((n_sub * p_sub, target_rank))
        
        # Verify
        valid = True
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    s_val = 0.0
                    for r in range(target_rank):
                        s_val += U_rand[i*m_sub+j, r] * V_rand[j*p_sub+k, r] * W_sol[k*n_sub+i, r]
                    if abs(s_val - 1.0) > 1e-5:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        
        if valid:
            # Check if entries are integer or half-integer
            # Multiply by 2 and check if close to integer
            W_check = W_sol * 2
            if np.all(np.abs(W_check - np.round(W_check)) < 1e-5):
                W_int = np.round(W_check) / 2.0
                # Return as lists
                return U_rand.tolist(), V_rand.tolist(), W_int.tolist()
        
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
            # U_full: 8 x 24
            # V_full: 16 x 24
            # W_full: 8 x 24
            
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
                        
            # k=2,3 block (use same U, V but shifted W? No, V and W indices change)
            # For k=2,3, we need V[j, 2] and V[j, 3].
            # Our sub-solution V is 4x2. We need to map it to 4x4.
            # The sub-solution computes C = A B_sub where B_sub is 4x2.
            # For the full problem, B is 4x4.
            # We can use the same U and V structure for the second half of B?
            # V_full[j, 2] corresponds to V_sub[j, 0]
            # V_full[j, 3] corresponds to V_sub[j, 1]
            # W_full[2, i] and W_full[3, i] correspond to W_sub[0, i] and W_sub[1, i]
            
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
