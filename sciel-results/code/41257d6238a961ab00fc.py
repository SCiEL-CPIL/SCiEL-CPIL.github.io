"""Matrix multiplication 2x2 * 2x13 rank minimization via tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # 140s limit (10s safety margin before 150s)
    n, m, p = DIMENSIONS  # A: n x m, B: m x p, C: n x p (output indexed W[k*n+i, r])
    
    # Schoolbook rank
    base_rank = n * m * p
    U_base = np.zeros((n * m, base_rank), dtype=np.int64)
    V_base = np.zeros((m * p, base_rank), dtype=np.int64)
    W_base = np.zeros((p * n, base_rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_base[i * m + j, col] = 1
                V_base[j * p + k, col] = 1
                W_base[k * n + i, col] = 1
                col += 1
    
    best_rank = base_rank
    best_decomp = (U_base, V_base, W_base)
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that U, V, W decompose the matmul tensor."""
        U_arr = np.array(U, dtype=np.int64)
        V_arr = np.array(V, dtype=np.int64)
        W_arr = np.array(W, dtype=np.int64)
        r = U_arr.shape[1]
        
        # Reconstruct: C[i,k] = sum over r of (sum over j of A[i,j]*B[j,k])
        # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # The matmul tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for valid j
        # Decomposition: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        
        # Check via random test matrices
        rng = np.random.RandomState(42)
        A_test = rng.randint(-5, 6, size=(n, m)).astype(np.int64)
        B_test = rng.randint(-5, 6, size=(m, p)).astype(np.int64)
        
        # Compute expected C
        C_expected = A_test @ B_test
        
        # Compute from decomposition:
        # For each r: U_r gives coefficients for A entries, V_r for B entries, W_r for C entries
        # Product term: (U_r · A_flat) * (V_r · B_flat) contributes W_r * that product to C
        U_flat = U_arr  # (n*m, r)
        V_flat = V_arr  # (m*p, r)
        W_flat = W_arr  # (p*n, r)
        
        A_flat = A_test.reshape(-1)  # (n*m,)
        B_flat = B_test.reshape(-1)  # (m*p,)
        
        # sum_r (U[:,r] @ A_flat) * (V[:,r] @ B_flat) -> vector in W space
        terms = (U_flat @ A_flat) * (V_flat @ B_flat)  # (r,)
        C_flat_computed = W_flat.T @ terms  # (p*n,)
        
        C_computed = C_flat_computed.reshape((p, n)).T  # (n, p)
        
        return np.allclose(C_computed, C_expected, atol=tol)
    
    def try_rank_reduction(base_U, base_V, base_W, target_rank, max_iters=500):
        """Try to find a rank reduction using perturbation / alternating minimization."""
        r0 = base_U.shape[1]
        U = base_U.astype(np.float64).copy()
        V = base_V.astype(np.float64).copy()
        W = base_W.astype(np.float64).copy()
        
        for _ in range(max_iters):
            if time.time() > deadline - 20:
                break
            
            # Randomly perturb a random column
            c = np.random.randint(0, target_rank)
            U[:, c] *= (1.0 + np.random.randn() * 0.1)
            V[:, c] *= (1.0 + np.random.randn() * 0.1)
            W[:, c] *= (1.0 + np.random.randn() * 0.1)
            
            # Quick verification with small test
            if not verify_decomposition(U, V, W, tol=1e-5):
                # Revert
                U[:, c] = base_U[:, c].astype(np.float64)
                V[:, c] = base_V[:, c].astype(np.float64)
                W[:, c] = base_W[:, c].astype(np.float64)
        
        if verify_decomposition(U, V, W, tol=1e-10):
            # Convert to int/half-int if possible
            U_int = np.round(U * 2) / 2
            V_int = np.round(V * 2) / 2
            W_int = np.round(W * 2) / 2
            if verify_decomposition(U_int, V_int, W_int, tol=1e-10):
                return True, (U_int, V_int, W_int)
        
        return False, (U, V, W)
    
    # Strategy: use Strassen-like approach for the 2x2 block, then handle the 13 columns
    # The 2x2 part can potentially be done in rank 7 (Strassen)
    # But we need to check if the 13-dimensional output allows savings
    
    # Alternative: exploit cyclic structure of 13 columns
    # Represent B as polynomials and use FFT-like decomposition
    
    # For now, try direct rank reduction from 52
    # Key insight: the 2x2 matmul has structure - we can use Strassen for the 2x2 part
    # Strassen for 2x2: rank 7 instead of 8
    # But B has 13 columns, so each Strassen product needs 13 multiplications
    # Unless we can share across columns...
    
    # Let's try: group columns of B and use the 2x2 Strassen decomposition
    # Actually, for 2x2 * 2x13, the natural approach is:
    # Compute 2x2 products for each column: 13 independent 2x2 multiplications
    # Each 2x2 can be done in rank 7 (Strassen), total rank = 13 * 7 = 91 (worse!)
    # Or 13 * 8 = 104 (worse than schoolbook 52)
    
    # So schoolbook is better here. The key question is whether we can do better than 52.
    
    # Lower bound considerations: 
    # - The tensor rank of 2x2 * 2xp matmul is at least 2p (I believe)
    # - Actually, it's known that for A(n x m) * B(m x p), rank >= max(n*p, m*p, n*m)
    # - Here: max(26, 26, 4) = 26. So rank >= 26.
    # - Schoolbook gives 52. Can we get closer to 26?
    
    # For the specific case of 2x2 * 2x13, let me think about structure.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is a bilinear form with 4*13 = 52 input variables and 2*13 = 26 output variables.
    # Each output C[i,k] depends on 4 products (A[i,0]*B[0,k], A[i,1]*B[1,k]) - wait no,
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] - that's 2 terms per output, 26 outputs = 52 terms total.
    
    # Actually each C[i,k] has exactly 2 terms: C[i,k] = sum_j A[i,j]*B[j,k]
    # So the bilinear map has 26 outputs, each requiring 2 products.
    # Total "scalar multiplications" = 52, matching schoolbook rank.
    
    # The question: can we combine these 52 products into fewer rank-1 tensors?
    # A rank-1 tensor corresponds to: one product (a*F(A))(b*G(B)) mapped to output via H(C).
    # To reduce rank, we'd need some products to serve multiple outputs or inputs to be shared.
    
    # For example, if we define M1 = A[0,0]*B[0,0] + A[0,1]*B[1,0], this isn't rank-1.
    
    # Let me try a different approach: use the fact that 13 is prime and see if 
    # any algebraic structure helps.
    
    # Actually, let me just try random search for rank reduction.
    rng = np.random.RandomState(12345)
    
    for trial_rank in range(51, 25, -1):
        if time.time() > deadline - 30:
            break
        
        success, (U_try, V_try, W_try) = try_rank_reduction(
            U_base, V_base, W_base, trial_rank, max_iters=200
        )
        if success:
            best_rank = trial_rank
            best_decomp = (U_try, V_try, W_try)
            break  # First success at this rank level
    
    # If no reduction found, return schoolbook
    if best_rank == base_rank:
        return {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}
    else:
        U_final, V_final, W_final = best_decomp
        # Convert to lists
        return {
            "U": U_final.tolist() if hasattr(U_final, 'tolist') else [[float(x) for x in row] for row in U_final],
            "V": V_final.tolist() if hasattr(V_final, 'tolist') else [[float(x) for x in row] for row in V_final],
            "W": W_final.tolist() if hasattr(W_final, 'tolist') else [[float(x) for x in row] for row in W_final]
        }
# EVOLVE-BLOCK-END
