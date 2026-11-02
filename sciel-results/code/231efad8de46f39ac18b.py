import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def _build_strassen_56():
    """Build the 8-block Strassen decomposition (rank 56)."""
    n, m, p = 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    R = num_blocks * 7   # 56
    
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    # Standard Strassen for 2x2 * 2x2
    U_coeffs = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    V_coeffs = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    W_contribs = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1}
    ]
    
    for q in range(num_blocks):
        b_00 = 0 * p + 2 * q
        b_01 = 0 * p + (2 * q + 1)
        b_10 = 1 * p + 2 * q
        b_11 = 1 * p + (2 * q + 1)
        
        c_00 = (2 * q) * n + 0
        c_01 = (2 * q + 1) * n + 0
        c_10 = (2 * q) * n + 1
        c_11 = (2 * q + 1) * n + 1
        
        b_map = {0: b_00, 1: b_01, 2: b_10, 3: b_11}
        c_map = {0: c_00, 1: c_01, 2: c_10, 3: c_11}
        
        for s in range(7):
            r = q * 7 + s
            for a_idx, coeff in U_coeffs[s].items():
                U[a_idx, r] = coeff
            for local_idx, coeff in V_coeffs[s].items():
                V[b_map[local_idx], r] = coeff
            for local_out_idx, coeff in W_contribs[s].items():
                W[c_map[local_out_idx], r] = coeff
    
    return U, V, W

def _verify(U, V, W, n, m, p, num_tests=3):
    """Verify the decomposition is correct."""
    for seed in range(num_tests):
        np.random.seed(seed)
        A = np.random.randint(-3, 4, size=(n, m))
        B = np.random.randint(-3, 4, size=(m, p))
        C = A @ B
        
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        UA = U.T @ A_flat
        VB = V.T @ B_flat
        C_decomp = W @ (UA * VB)
        C_flat = C.T.flatten()
        
        if not np.allclose(C_decomp, C_flat, rtol=1e-9, atol=1e-9):
            return False
    return True

def _try_eliminate_term(U, V, W, term_to_remove, n, m, p):
    """
    Try to eliminate one term by solving for W with fixed U and V.
    Returns (U_new, V_new, W_new) if successful, else None.
    """
    R0 = U.shape[1]
    R = R0 - 1
    
    # Keep indices
    keep = [i for i in range(R0) if i != term_to_remove]
    
    U_new = U[:, keep].copy()
    V_new = V[:, keep].copy()
    
    # For each output index c, solve for W_new[c, :]
    # Equation: sum_r W_new[c, r] * U_new[a, r] * V_new[b, r] = T[a, b, c]
    
    W_new = np.zeros((p * n, R), dtype=float)
    
    num_A = n * m  # 4
    num_B = m * p  # 32
    num_C = p * n  # 32
    
    # Build target tensor values
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for valid (i,j,k)
    
    for c in range(num_C):
        k = c // n
        i = c % n
        
        # For this c, the valid (a, b) pairs:
        # j=0: a = i*m + 0, b = 0*p + k = k
        # j=1: a = i*m + 1, b = 1*p + k = p + k
        
        equations = []
        rhs = []
        
        for j in range(m):
            a = i * m + j
            b = j * p + k
            # T[a, b, c] = 1 (since this is a valid product index)
            # The equation: sum_r W_new[c, r] * U_new[a, r] * V_new[b, r] = 1
            
            row = U_new[a, :] * V_new[b, :]
            equations.append(row)
            rhs.append(1)
        
        # Also, for other (a, b) pairs that map to the same c but aren't valid:
        # Actually, for a given c = k*n + i, the only valid (a,b) are those above.
        # For invalid (a,b) pairs that would give the same c, T[a,b,c] = 0.
        # But those (a,b) pairs don't correspond to valid product indices, so they
        # don't contribute to the tensor. The tensor is zero for those.
        # 
        # Wait, I need to be more careful. The tensor T has entries for ALL (a,b,c) triples.
        # T[a,b,c] = 1 iff there exist (i,j,k) such that a = i*m+j, b = j*p+k, c = k*n+i.
        # For a fixed c = k*n+i, the valid (a,b) are exactly those with j in [0,m).
        # For other (a,b) with the same c, T[a,b,c] = 0.
        #
        # But in the bilinear form, the equation is:
        # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for ALL (a,b,c).
        #
        # So for fixed c, I need:
        # sum_r W[c,r] * U[a,r] * V[b,r] = T[a,b,c] for all (a,b).
        #
        # That's num_A * num_B = 4 * 32 = 128 equations per c!
        # But most of them have T[a,b,c] = 0.
        
        # Let me redo this properly.
        # For fixed c, equations for all (a, b):
        # sum_r W[c, r] * U_new[a, r] * V_new[b, r] = T[a, b, c]
        
        # Build the system
        A_sys = np.zeros((num_A * num_B, R))
        b_sys = np.zeros(num_A * num_B)
        
        for a in range(num_A):
            for b in range(num_B):
                idx = a * num_B + b
                A_sys[idx, :] = U_new[a, :] * V_new[b, :]
                # T[a, b, c]:
                # a = i'*m + j', b = j'*p + k', c = k*n + i
                # T = 1 iff i' = i, k' = k, and j' = j (consistent)
                i_prime = a // m
                j_prime = a % m
                k_prime = b // p
                j_check = b % p  # this should equal k_prime... wait
                # b = j'*p + k', so j' = b // p, k' = b % p
                # c = k*n + i, so k = c // n, i = c % n
                # T = 1 iff i' == i AND k' == k AND j' == j_prime (which is automatic)
                # Wait: a = i'*m + j', b = j'*p + k', c = k*n + i
                # T[a,b,c] = 1 iff i' == i AND k' == k
                # (j' is determined by both a and b, and must be consistent: j' from a must equal j' from b)
                # j' from a: a % m
                # j' from b: b // p
                # These must be equal for consistency, but since we're iterating over all (a,b),
                # some combinations are inconsistent. For those, T = 0.
                
                if i_prime == i and k_prime == k and (a % m) == (b // p):
                    b_sys[idx] = 1
                else:
                    b_sys[idx] = 0
        
        # Solve the underdetermined system: A_sys @ W_new[c, :] = b_sys
        # Use least squares (minimum norm solution)
        try:
            sol, residuals, rank, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
            W_new[c, :] = sol
        except:
            return None
        
        # Check if the solution satisfies the equations
        if not np.allclose(A_sys @ sol, b_sys, rtol=1e-9, atol=1e-9):
            return None
    
    # Verify the full decomposition
    U_check = U_new
    V_check = V_new
    W_check = W_new
    
    for seed in range(5):
        np.random.seed(seed)
        A_mat = np.random.randint(-3, 4, size=(n, m))
        B_mat = np.random.randint(-3, 4, size=(m, p))
        C_mat = A_mat @ B_mat
        
        A_flat = A_mat.flatten()
        B_flat = B_mat.flatten()
        
        UA = U_check.T @ A_flat
        VB = V_check.T @ B_flat
        C_decomp = W_check @ (UA * VB)
        C_flat = C_mat.T.flatten()
        
        if not np.allclose(C_decomp, C_flat, rtol=1e-9, atol=1e-9):
            return None
    
    # Try to round to half-integers
    W_rounded = np.round(W_new * 2) / 2
    
    # Check if rounded version works
    for seed in range(5):
        np.random.seed(seed)
        A_mat = np.random.randint(-3, 4, size=(n, m))
        B_mat = np.random.randint(-3, 4, size=(m, p))
        C_mat = A_mat @ B_mat
        
        A_flat = A_mat.flatten()
        B_flat = B_mat.flatten()
        
        UA = U_check.T @ A_flat
        VB = V_check.T @ B_flat
        C_decomp = W_rounded @ (UA * VB)
        C_flat = C_mat.T.flatten()
        
        if not np.allclose(C_decomp, C_flat, rtol=1e-9, atol=1e-9):
            return None
    
    return U_new, V_new, W_rounded

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with the known rank-56 decomposition
    U, V, W = _build_strassen_56()
    
    # Verify it works
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by eliminating terms
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]
    
    for term_to_remove in range(U.shape[1]):
        if time.time() > deadline:
            break
        
        result = _try_eliminate_term(U, V, W, term_to_remove, n, m, p)
        if result is not None:
            U_new, V_new, W_new = result
            new_rank = U_new.shape[1]
            if new_rank < best_rank:
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = new_rank
                # Update U, V, W for further elimination attempts
                U, V, W = U_new, V_new, W_new
                # Reset term_to_remove since indices changed
                term_to_remove = 0
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
