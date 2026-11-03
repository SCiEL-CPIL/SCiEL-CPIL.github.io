"""Tensor decomposition for 4x4 x 4x9 matrix multiplication using numerical HOOI optimization."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _build_target_tensor(n, m, p):
    """
    Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i, j, k] = 1 if i = i0*m + j0, j = j0*p + k0, k = k0*n + i0
    where i0, j0, k0 are the indices of A, B, C respectively.
    """
    T = np.zeros((n * m, m * p, p * n))
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                i = i0 * m + j0
                j = j0 * p + k0
                k = k0 * n + i0
                T[i, j, k] = 1.0
    return T

def _hooi_decompose(T, rank, max_iter=300, tol=1e-10, seed=42):
    """
    Higher-Order Orthogonal Iteration (HOOI) for tensor decomposition.
    T approx sum_r U[:,r] V[:,r] W[:,r]
    """
    n, m, p = T.shape
    rng = np.random.default_rng(seed)
    
    # Initialize
    U = rng.standard_normal((n, rank))
    V = rng.standard_normal((m, rank))
    W = rng.standard_normal((p, rank))
    
    # Normalize initial factors
    for r in range(rank):
        norm = np.linalg.norm(U[:, r])
        if norm > 1e-10:
            U[:, r] /= norm
        norm = np.linalg.norm(V[:, r])
        if norm > 1e-10:
            V[:, r] /= norm
        norm = np.linalg.norm(W[:, r])
        if norm > 1e-10:
            W[:, r] /= norm

    prev_err = np.inf
    for it in range(max_iter):
        # Update U
        # T_W = T matmul W -> shape (n, m, rank)
        # Reshape to (n, m*rank)
        # SVD
        T_W = np.tensordot(T, W, axes=([2], [0]))  # (n, m, rank)
        T_W_flat = T_W.reshape(n, m * rank)
        _, _, Vt = np.linalg.svd(T_W_flat, full_matrices=False)
        U_new = Vt.T  # (n, rank)
        
        # Update V
        T_U = np.tensordot(T, U, axes=([0], [0]))  # (m, p, rank)
        T_U_flat = T_U.reshape(m, p * rank)
        _, _, Vt = np.linalg.svd(T_U_flat, full_matrices=False)
        V_new = Vt.T  # (m, rank)
        
        # Update W
        T_V = np.tensordot(T, V, axes=([1], [0]))  # (p, n, rank)
        T_V_flat = T_V.reshape(p, n * rank)
        _, _, Wt = np.linalg.svd(T_V_flat, full_matrices=False)
        W_new = Wt.T  # (p, rank)
        
        # Check convergence
        U = U_new
        V = V_new
        W = W_new
        
        # Calculate error
        T_recon = np.tensordot(U, V, axes=([1], [0]))  # (n, m, rank)
        T_recon = np.tensordot(T_recon, W, axes=([2], [0]))  # (n, m, p)
        err = np.linalg.norm(T - T_recon)
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
        
    return U, V, W

def _reconstruct_rational(U, V, W, T, max_denom=20):
    """
    Attempt to reconstruct exact integer/half-integer coefficients.
    We scale the factors to make them integers.
    """
    n, m, p = T.shape
    rank = U.shape[1]
    
    # We need to find scalars a_r, b_r, c_r such that a_r U[:,r], b_r V[:,r], c_r W[:,r]
    # are rational with small denominators, and product a_r b_r c_r = 1 (or adjusted).
    # Actually, the decomposition is T = sum U_r V_r W_r.
    # We can absorb scalings: U_r -> alpha_r U_r, V_r -> beta_r V_r, W_r -> gamma_r W_r
    # with alpha_r beta_r gamma_r = 1.
    
    # Simple heuristic: for each r, find the best rational approximation for the vector U[:,r]
    # assuming it is of the form (integers / D) for some small D.
    
    # Let's try to find a global scale first.
    # The entries of T are 0 or 1.
    # The entries of U, V, W from HOOI are floating point.
    
    # For each r, let's try to snap U[:,r] to a vector of rationals with denominator D.
    # Then snap V[:,r] and W[:,r].
    # This is a combinatorial problem.
    
    # Alternative: Use the fact that the tensor is simple.
    # If HOOI converges to the exact solution (up to scaling), the vectors should be linearly dependent over Q.
    
    # Let's try to snap each column of U to integers by finding the best D.
    # This is expensive.
    
    # Simpler approach for this specific task:
    # If the rank is low enough, we might be able to solve for the factors directly from the tensor slices.
    # But HOOI gives us the directions.
    
    # Let's assume the output of HOOI is close to the exact solution.
    # We can try to "clean" the values.
    
    U_int = np.zeros_like(U)
    V_int = np.zeros_like(V)
    W_int = np.zeros_like(W)
    
    success = True
    for r in range(rank):
        # Try to find D such that U[:,r] * D is close to integers
        best_D = 1
        best_err = np.inf
        for D in range(1, max_denom + 1):
            U_scaled = U[:, r] * D
            err = np.linalg.norm(U_scaled - np.round(U_scaled))
            if err < best_err:
                best_err = err
                best_D = D
        
        if best_err > 1e-6:
            success = False
            break
            
        U_int[:, r] = np.round(U[:, r] * best_D) / best_D
        
        # Do same for V and W
        best_D_v = 1
        best_err_v = np.inf
        for D in range(1, max_denom + 1):
            V_scaled = V[:, r] * D
            err = np.linalg.norm(V_scaled - np.round(V_scaled))
            if err < best_err_v:
                best_err_v = err
                best_D_v = D
        
        if best_err_v > 1e-6:
            success = False
            break
            
        V_int[:, r] = np.round(V[:, r] * best_D_v) / best_D_v
        
        best_D_w = 1
        best_err_w = np.inf
        for D in range(1, max_denom + 1):
            W_scaled = W[:, r] * D
            err = np.linalg.norm(W_scaled - np.round(W_scaled))
            if err < best_err_w:
                best_err_w = err
                best_D_w = D
        
        if best_err_w > 1e-6:
            success = False
            break
            
        W_int[:, r] = np.round(W[:, r] * best_D_w) / best_D_w
        
    if not success:
        return None, None, None
        
    return U_int, V_int, W_int

def _verify_decomp(U, V, W, T, atol=1e-6):
    """Verify if U, V, W decompose T."""
    T_recon = np.tensordot(U, V, axes=([1], [0]))
    T_recon = np.tensordot(T_recon, W, axes=([2], [0]))
    return np.allclose(T_recon, T, atol=atol)

def _try_rank(n, m, p, rank, max_iter=200, seed=0):
    """Try to find a decomposition of given rank."""
    T = _build_target_tensor(n, m, p)
    
    # Try multiple seeds
    for s in range(5):
        U, V, W = _hooi_decompose(T, rank, max_iter=max_iter, seed=seed + s)
        
        # Try to rationalize
        U_r, V_r, W_r = _reconstruct_rational(U, V, W, T)
        
        if U_r is not None:
            if _verify_decomp(U_r, V_r, W_r, T):
                return U_r, V_r, W_r
        else:
            # Maybe the floating point result is good enough for the evaluator?
            # The evaluator requires integer/half-integer entries.
            # If we can't snap, we can't use it.
            pass
            
    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Theoretical lower bound for M_{4,4,9} is n*m*p - (n-1)(m-1)(p-1)? No.
    # Lower bound is max(n*m, m*p, p*n) = max(16, 36, 36) = 36.
    # But tensor rank is higher.
    # For M_{n,m,p}, rank >= n*m*p / max(n,m,p).
    # 144 / 9 = 16.
    # Known: M_{4,4,4} rank 49.
    # M_{4,4,9} likely around 96-100?
    # Current best is 114.
    
    # Let's try ranks from 96 up to 113.
    # If we find one, it's an improvement.
    
    best_rank = 114
    best_U, best_V, best_W = None, None, None
    
    # Fallback: Constructive Strassen
    # Re-implementing the seed's constructive method to have a guaranteed baseline
    # But we want to beat it.
    
    # Start with a high rank to ensure convergence, then lower?
    # HOOI is sensitive to initialization.
    
    # Let's try a few promising ranks.
    # 96, 100, 105, 110
    
    ranks_to_try = [96, 100, 105, 110]
    
    for r in ranks_to_try:
        if time.time() > deadline - 20:
            break
            
        U, V, W = _try_rank(n, m, p, r, max_iter=100)
        
        if U is not None:
            best_U, best_V, best_W = U, V, W
            best_rank = r
            break # Found a valid decomposition, return it
    
    if best_U is not None:
        # Convert to lists
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # If numerical search fails, fallback to constructive 114
    # Re-using the logic from the seed program for fallback
    def _strassen_2x2():
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = -1; W[2, 3] = 1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1; W[1, 4] = 1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        return U, V, W

    def _decomp_4x4x4():
        n, m, p = 4, 4, 4
        rank = 49
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        inner_U, inner_V, inner_W = _strassen_2x2()
        outer_A = [[1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, -1], [0, 1, 0, -1]]
        outer_B = [[1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, -1], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]]
        outer_C = [[1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]]
        for t in range(7):
            col_offset = t * 7
            for r in range(7):
                col = col_offset + r
                for li in range(2):
                    for lj in range(2):
                        b = li * 2 + lj
                        coeff_U = outer_A[t][b] * inner_U[li*2 + lj, r]
                        coeff_V = outer_B[t][b] * inner_V[li*2 + lj, r]
                        coeff_W = outer_C[t][b] * inner_W[li*2 + lj, r]
                        for ei in range(2):
                            for ej in range(2):
                                g_i = 2 * (b // 2) + ei
                                g_j = 2 * (b % 2) + ej
                                U[g_i * 4 + g_j, col] = coeff_U
                                V[g_j * 4 + g_i, col] = coeff_V
                                W[g_i * 4 + g_j, col] = coeff_W
        return U, V, W

    def _schoolbook_nm_p(n, m, p):
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def _decomp_4x4x9_combined():
        n, m, p = 4, 4, 9
        total_rank = 49 + 49 + 16
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        U444, V444, W444 = _decomp_4x4x4()
        U441, V441, W441 = _schoolbook_nm_p(4, 4, 1)
        U[:, :49] = U444
        for j in range(m):
            for k in range(4):
                V[j * 9 + k, :49] = V444[j * 4 + k, :]
        for k in range(4):
            for i in range(n):
                W[k * 4 + i, :49] = W444[k * 4 + i, :]
        U[:, 49:98] = U444
        for j in range(m):
            for k in range(4):
                V[j * 9 + (4 + k), 49:98] = V444[j * 4 + k, :]
        for k in range(4):
            for i in range(n):
                W[(4 + k) * 4 + i, 49:98] = W444[k * 4 + i, :]
        U[:, 98:114] = U441
        for j in range(m):
            V[j * 9 + 8, 98:114] = V441[j * 1 + 0, :]
        for i in range(n):
            W[8 * 4 + i, 98:114] = W441[0 * 4 + i, :]
        return U, V, W

    U, V, W = _decomp_4x4x9_combined()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
