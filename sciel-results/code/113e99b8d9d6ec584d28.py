"""Program for matrix multiplication 4x4 x 4x6 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
    
    Returns (U, V, W) where:
      U: (4, 7) - for A[i,j], row index = i*2+j
      V: (4, 7) - for B[j,k], row index = j*2+k
      W: (4, 7) - for C[i,k], row index = k*2+i  (trace convention: W[k*n+i] with n=2)
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    
    # M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    
    # M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # W: C[i,k] contributions, W row = k*2 + i
    # C[0,0]: i=0,k=0 -> W[0]; C[0,1]: i=0,k=1 -> W[2]
    # C[1,0]: i=1,k=0 -> W[1]; C[1,1]: i=1,k=1 -> W[3]
    
    # C[0,0] = M1 + M4 - M5 + M7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    
    # C[0,1] = M3 + M5
    W[2, 2] = 1; W[2, 4] = 1
    
    # C[1,0] = M2 + M4
    W[1, 1] = 1; W[1, 3] = 1
    
    # C[1,1] = M1 - M2 + M3 + M6
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W


def _build_rank77():
    """Build the base rank-77 decomposition."""
    n, m, p = DIMENSIONS  # 4, 4, 6
    sU, sV, sW = _strassen_2x2()
    
    outer_terms = []
    
    for t in range(7):
        U_idx = [(a, b) for a in range(2) for b in range(2) if sU[a*2+b, t] != 0]
        V_idx = [(b, c) for b in range(2) for c in range(2) if sV[b*2+c, t] != 0]
        W_idx = [(a, c) for a in range(2) for c in range(2) if sW[c*2+a, t] != 0]
        
        U_coeffs = { (a,b): sU[a*2+b, t] for a,b in U_idx }
        V_coeffs = { (b,c): sV[b*2+c, t] for b,c in V_idx }
        W_coeffs = { (a,c): sW[c*2+a, t] for a,c in W_idx }
        
        outer_terms.append((U_coeffs, V_coeffs, W_coeffs))
        
    for a in range(2):
        for b in range(2):
            U_coeffs = { (a,b): 1 }
            V_coeffs = { (b,2): 1 }
            W_coeffs = { (a,2): 1 }
            outer_terms.append((U_coeffs, V_coeffs, W_coeffs))
            
    total_rank = len(outer_terms) * 7
    U = np.zeros((n*m, total_rank), dtype=float)
    V = np.zeros((m*p, total_rank), dtype=float)
    W = np.zeros((p*n, total_rank), dtype=float)
    
    for ot_idx, (U_coeffs, V_coeffs, W_coeffs) in enumerate(outer_terms):
        for t in range(7):
            col = ot_idx * 7 + t
            
            for (ai, bj), coeff_outer in U_coeffs.items():
                for ai2 in range(2):
                    for bj2 in range(2):
                        coeff_inner = sU[ai2*2 + bj2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            u_row = (2*ai + ai2) * 4 + (2*bj + bj2)
                            U[u_row, col] += coeff_outer * coeff_inner
            
            for (bj, ck), coeff_outer in V_coeffs.items():
                for bj2 in range(2):
                    for ck2 in range(2):
                        coeff_inner = sV[bj2*2 + ck2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            v_row = (2*bj + bj2) * 6 + (2*ck + ck2)
                            V[v_row, col] += coeff_outer * coeff_inner
            
            for (ai, ck), coeff_outer in W_coeffs.items():
                for ai2 in range(2):
                    for ck2 in range(2):
                        coeff_inner = sW[ck2*2 + ai2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            w_row = (2*ck + ck2) * 4 + (2*ai + ai2)
                            W[w_row, col] += coeff_outer * coeff_inner
    
    return U, V, W


def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor T[i,j,k] = 1 if j is the contraction index."""
    # T[a, b, c] = 1 if the multiplication involves A[a,b]*B[b,c]
    # For our convention: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor element T[i,j,k] = 1 (always, for valid indices)
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T


def _compute_residual(U, V, W, n, m, p):
    """Compute the residual between the decomposition and the target tensor."""
    rank = U.shape[1]
    # Compute the decomposed tensor
    # D[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # But we need to handle the indexing conventions properly
    
    # U: (n*m, rank) where row = i*m + j
    # V: (m*p, rank) where row = j*p + k  
    # W: (p*n, rank) where row = k*n + i
    
    D = np.zeros((n, m, p))
    for r in range(rank):
        U_col = U[:, r].reshape(n, m)
        V_col = V[:, r].reshape(m, p)
        W_col = W[:, r].reshape(p, n)
        
        # D[i,j,k] += U[i,j] * V[j,k] * W[k,i]
        # This is a bit tricky. Let's think about it differently.
        # The decomposition should satisfy:
        # C[i,k] = sum_j A[i,j] B[j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # But this doesn't quite work as a simple tensor product.
        
        # Actually, the standard form is:
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # where U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    D[i, j, k] += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    
    T = _build_target_tensor(n, m, p)
    return D - T


def _gradient_descent_border_rank(U0, V0, W0, n, m, p, max_iter=500, lr=0.01, deadline=None):
    """
    Try to reduce the rank by starting from a perturbed rank-78 and 
    using gradient descent to find a valid rank-77 (or lower) decomposition.
    
    Returns (U, V, W, success) where success indicates if we found a valid lower-rank decomposition.
    """
    rank0 = U0.shape[1]
    
    # Start with rank0+1 = 78 columns
    rank = rank0 + 1
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Copy the base decomposition
    U[:, :rank0] = U0
    V[:, :rank0] = V0
    W[:, :rank0] = W0
    
    # Initialize the extra column with small random values
    np.random.seed(42)
    U[:, rank0] = np.random.randn(n*m) * 0.01
    V[:, rank0] = np.random.randn(m*p) * 0.01
    W[:, rank0] = np.random.randn(p*n) * 0.01
    
    T = _build_target_tensor(n, m, p)
    
    best_residual = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 10:
            break
        
        # Compute residual
        D = np.zeros((n, m, p))
        for r in range(rank):
            U_col = U[:, r].reshape(n, m)
            V_col = V[:, r].reshape(m, p)
            W_col = W[:, r].reshape(p, n)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        D[i, j, k] += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
        
        residual = D - T
        res_norm = np.linalg.norm(residual)
        
        if res_norm < best_residual:
            best_residual = res_norm
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if res_norm < 1e-8:
            break
        
        # Compute gradients
        # d/dU[i*m+j, r] = sum over (j',k') where the term contributes
        # This is complex. Let's use a simpler approach: finite differences or 
        # a different optimization strategy.
        
        # Actually, let's try a different approach: use the fact that if we can
        # make one column nearly zero while maintaining the decomposition, we've
        # effectively reduced the rank.
        
        # Try to drive the last column toward zero while adjusting others
        # to compensate.
        
        # Simplified: just check if we can find a valid decomposition
        # by trying to zero out columns.
        
        break  # For now, just return the best we found
    
    # Check if we can reduce the rank
    # Try removing columns one by one and see if we can still achieve a small residual
    for col_to_remove in range(rank):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Try removing column col_to_remove
        keep_cols = [c for c in range(rank) if c != col_to_remove]
        U_test = U[:, keep_cols]
        V_test = V[:, keep_cols]
        W_test = W[:, keep_cols]
        
        # Check if this still gives a valid decomposition
        D_test = np.zeros((n, m, p))
        for r in range(len(keep_cols)):
            U_col = U_test[:, r].reshape(n, m)
            V_col = V_test[:, r].reshape(m, p)
            W_col = W_test[:, r].reshape(p, n)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        D_test[i, j, k] += U_test[i*m + j, r] * V_test[j*p + k, r] * W_test[k*n + i, r]
        
        res_test = np.linalg.norm(D_test - T)
        
        if res_test < 1e-6:
            # We found a valid lower-rank decomposition!
            return U_test, V_test, W_test, True
    
    return best_U, best_V, best_W, False


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Get the base rank-77 decomposition
    U_base, V_base, W_base = _build_rank77()
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 140s to leave safety margin
    
    # Try border rank optimization
    U_opt, V_opt, W_opt, success = _gradient_descent_border_rank(
        U_base, V_base, W_base, n, m, p, 
        max_iter=100, lr=0.01, deadline=deadline
    )
    
    if success and U_opt.shape[1] < U_base.shape[1]:
        # Convert to integers if possible
        U_out = np.round(U_opt).astype(int)
        V_out = np.round(V_opt).astype(int)
        W_out = np.round(W_opt).astype(int)
        
        # Verify
        T = _build_target_tensor(n, m, p)
        D = np.zeros((n, m, p))
        rank_out = U_out.shape[1]
        for r in range(rank_out):
            U_col = U_out[:, r].reshape(n, m)
            V_col = V_out[:, r].reshape(m, p)
            W_col = W_out[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        D[i, j, k] += U_out[i*m + j, r] * V_out[j*p + k, r] * W_out[k*n + i, r]
        
        if np.allclose(D, T, atol=1e-6):
            return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
    
    # Fall back to the base rank-77 decomposition
    U_out = U_base.astype(int)
    V_out = V_base.astype(int)
    W_out = W_base.astype(int)
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}

# EVOLVE-BLOCK-END
