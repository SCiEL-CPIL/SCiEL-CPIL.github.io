"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

import numpy as np
import time

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify if the tensor decomposition is exact."""
    rank = len(U[0]) if U else 0
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1

    T_approx = np.zeros_like(T)
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for r in range(rank):
        T_approx += U_np[:, r][:, np.newaxis, np.newaxis] * \
                    V_np[:, r][np.newaxis, :, np.newaxis] * \
                    W_np[:, r][np.newaxis, np.newaxis, :]
                    
    return np.allclose(T_approx, T, atol=1e-10)

def _construct_strassen_238():
    """
    Constructs rank 32 decomposition for 2x3x8 matrix multiplication.
    Uses 4 blocks of 2x3x2.
    Each 2x3x2 block uses Strassen (rank 7) for 2x2x2 part and Rank-1 (rank 1) for 2x1x2 part.
    Total rank: 4 * (7 + 1) = 32.
    """
    n, m, p = 2, 3, 8
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(col, u_idx, u_coeff, v_idx, v_coeff, w_idx, w_coeff):
        for idx, c in zip(u_idx, u_coeff):
            U[idx][col] += c
        for idx, c in zip(v_idx, v_coeff):
            V[idx][col] += c
        for idx, c in zip(w_idx, w_coeff):
            W[idx][col] += c

    current_col = 0
    
    for k_block in range(4):
        k0 = k_block * 2
        k1 = k_block * 2 + 1
        
        e_idx = 0 * p + k0
        f_idx = 0 * p + k1
        g_idx = 1 * p + k0
        h_idx = 1 * p + k1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        a_idx = 0
        b_idx = 1
        c_idx = 3
        d_idx = 4
        
        # M1 = (a+d)(e+h)
        add_term(current_col, [a_idx, d_idx], [1, 1], [e_idx, h_idx], [1, 1], [w_00, w_11], [1, 1])
        current_col += 1
        
        # M2 = (c+d)e
        add_term(current_col, [c_idx, d_idx], [1, 1], [e_idx], [1], [w_10, w_11], [1, -1])
        current_col += 1
        
        # M3 = a(f-h)
        add_term(current_col, [a_idx], [1], [f_idx, h_idx], [1, -1], [w_01, w_11], [1, 1])
        current_col += 1
        
        # M4 = d(g-e)
        add_term(current_col, [d_idx], [1], [g_idx, e_idx], [1, -1], [w_00, w_10], [1, 1])
        current_col += 1
        
        # M5 = (a+b)h
        add_term(current_col, [a_idx, b_idx], [1, 1], [h_idx], [1], [w_00, w_01], [-1, 1])
        current_col += 1
        
        # M6 = (c-a)(e+f)
        add_term(current_col, [c_idx, a_idx], [1, -1], [e_idx, f_idx], [1, 1], [w_11], [1])
        current_col += 1
        
        # M7 = (b-d)(g+h)
        add_term(current_col, [b_idx, d_idx], [1, -1], [g_idx, h_idx], [1, 1], [w_00], [1])
        current_col += 1
        
        # Part 2: j = 2 (2x1x2)
        # This part corresponds to A[:,2] * B[2,:].
        # The tensor slice T[i, 2, k] = 1 for i in {0,1}, k in {k0, k1}.
        # This can be represented by a single rank-1 term.
        a2_idx = 0 * m + 2
        c2_idx = 1 * m + 2
        
        e2_idx = 2 * p + k0
        f2_idx = 2 * p + k1
        
        w_indices = [w_00, w_01, w_10, w_11]
        w_coeffs = [1, 1, 1, 1]
        
        add_term(current_col, [a2_idx, c2_idx], [1, 1], [e2_idx, f2_idx], [1, 1], w_indices, w_coeffs)
        current_col += 1

    return {"U": U, "V": V, "W": W}

def _construct_schoolbook():
    """
    Constructs rank 48 schoolbook decomposition.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(base_U, base_V, base_W, target_rank, max_time=60.0):
    """
    Try to reduce rank by finding linear dependencies among the rank-44 terms.
    Uses randomized approach: try to express some terms as linear combinations of others.
    """
    start_time = time.time()
    n, m, p = 2, 3, 8
    nm, mp, pn = n * m, m * p, p * n
    current_rank = len(base_U[0])
    
    # Convert to numpy
    U0 = np.array(base_U, dtype=float).T  # (rank, nm)
    V0 = np.array(base_V, dtype=float).T  # (rank, mp)
    W0 = np.array(base_W, dtype=float).T  # (rank, pn)
    
    # The tensor is T = sum_r U0[r] ⊗ V0[r] ⊗ W0[r]
    # We want to find if some columns are linearly dependent in a tensor sense.
    
    # Strategy: try removing one term at a time and see if the remaining can still span the tensor.
    # For each term r, check if U0[r] ⊗ V0[r] ⊗ W0[r] is in the span of the others.
    
    # This is expensive, so we'll do a limited search.
    best_U, best_V, best_W = base_U, base_V, base_W
    best_rank = current_rank
    
    # Try removing each term and verify
    for r_remove in range(current_rank):
        if time.time() - start_time > max_time - 5:
            break
        
        # Create reduced matrices without term r_remove
        keep = [i for i in range(current_rank) if i != r_remove]
        U_red = [base_U[i] for i in keep]
        V_red = [base_V[i] for i in keep]
        W_red = [base_W[i] for i in keep]
        
        if _verify_decomposition(U_red, V_red, W_red, n, m, p):
            best_U, best_V, best_W = U_red, V_red, W_red
            best_rank = current_rank - 1
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}

def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s kill
    
    # Candidate 1: Rank 44 Strassen-based
    cand1 = _construct_strassen_238()
    if _verify_decomposition(cand1["U"], cand1["V"], cand1["W"], 2, 3, 8):
        # Try to reduce rank further
        if time.time() - start_time < deadline - 30:
            cand_reduced = _try_reduce_rank(cand1["U"], cand1["V"], cand1["W"], 43, max_time=25.0)
            if len(cand_reduced["U"][0]) < len(cand1["U"][0]):
                if _verify_decomposition(cand_reduced["U"], cand_reduced["V"], cand_reduced["W"], 2, 3, 8):
                    return cand_reduced
        return cand1
        
    # Candidate 2: Rank 48 Schoolbook (Fallback)
    cand2 = _construct_schoolbook()
    if _verify_decomposition(cand2["U"], cand2["V"], cand2["W"], 2, 3, 8):
        return cand2
        
    return cand2
# EVOLVE-BLOCK-END
