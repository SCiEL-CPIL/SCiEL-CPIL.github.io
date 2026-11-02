import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that sum_r U[:,r] V[:,r] W[:,r] equals the matmul tensor."""
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    # Build the decomposition tensor
    T_decomp = np.zeros((nm, mp, pn), dtype=np.float64)
    for r in range(rank):
        T_decomp += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    # Build the target tensor
    T_target = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0
    return np.allclose(T_decomp, T_target, atol=1e-8)

def _rank_17_decomposition(n, m, p):
    """
    Construct the known rank-17 decomposition.
    Combines Strassen (rank 7) for 2x2x2 and schoolbook for the rest.
    """
    U = np.zeros((n * m, 17))
    V = np.zeros((m * p, 17))
    W = np.zeros((p * n, 17))
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: 0,1,3,4 (i*m+j)
    # B indices: 0,1,3,4 (j*p+k)
    # C indices: 0,1,2,3 (k*n+i)
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A10+A11)B00 -> C10, -C11
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3: A00(B01-B11) -> C01, C11
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4: A11(B10-B00) -> C00, C10
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5: (A00+A01)B11 -> -C00, C01
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6: (A10-A00)(B00+B01) -> C11
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C00
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    # A indices: 2, 5
    # B indices: 6, 7
    # C indices: 0,1,2,3
    
    # r=7: A02 B20 -> C00
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    # r=8: A02 B21 -> C01
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1
    
    # r=9: A12 B20 -> C10
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1
    
    # r=10: A12 B21 -> C11
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1
    
    # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    # A indices: 0,1,2,3,4,5
    # B indices: 2,5,8
    # C indices: 4,5
    
    # r=11: A00 B02 -> C02
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1
    
    # r=12: A01 B12 -> C02
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1
    
    # r=13: A02 B22 -> C02
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1
    
    # r=14: A10 B02 -> C12
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1
    
    # r=15: A11 B12 -> C12
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1
    
    # r=16: A12 B22 -> C12
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    return U, V, W

def _find_rank_16(n, m, p, time_budget):
    """
    Attempt to find a rank-16 decomposition using numerical optimization.
    """
    nm, mp, pn = n * m, m * p, p * n
    rank = 16
    
    # Pre-compute target tensor
    T_target = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Flatten target for faster residual calculation
    T_flat = T_target.flatten()
    
    # Objective function: Sum of squared residuals
    def objective(params):
        # Unpack parameters into U, V, W
        size_u = nm * rank
        size_v = mp * rank
        size_w = pn * rank
        
        U_flat = params[:size_u]
        V_flat = params[size_u:size_u + size_v]
        W_flat = params[size_u + size_v:size_u + size_v + size_w]
        
        U = U_flat.reshape(nm, rank)
        V = V_flat.reshape(mp, rank)
        W = W_flat.reshape(pn, rank)
        
        # Compute decomposition tensor
        T_decomp = np.zeros((nm, mp, pn), dtype=np.float64)
        for r in range(rank):
            T_decomp += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        
        # Residual
        residual = T_decomp.flatten() - T_flat
        return np.sum(residual * residual)
    
    # Initial guess: Start with rank-17 and merge two terms
    U17, V17, W17 = _rank_17_decomposition(n, m, p)
    
    # Try merging different pairs of terms from rank 17 to initialize rank 16
    # We pick two random terms and try to combine them into one, then optimize
    rng = np.random.default_rng(42)
    
    deadline = time.time() + time_budget
    
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
            
        # Pick two indices to merge
        idx1, idx2 = rng.choice(17, size=2, replace=False)
        
        # Initialize U, V, W with rank 16
        # Copy all terms except idx2
        mask = np.ones(17, dtype=bool)
        mask[idx2] = False
        mask_indices = np.where(mask)[0]
        
        U_init = np.zeros((nm, rank))
        V_init = np.zeros((mp, rank))
        W_init = np.zeros((pn, rank))
        
        # Fill first 15 terms
        for new_idx, old_idx in enumerate(mask_indices[:15]):
            U_init[:, new_idx] = U17[:, old_idx]
            V_init[:, new_idx] = V17[:, old_idx]
            W_init[:, new_idx] = W17[:, old_idx]
            
        # Merge idx1 and idx2 into the 16th term (index 15)
        # Simple average or sum? Let's try sum first, then scale down? 
        # Or just random perturbation around the sum?
        # The term is U1 V1 W1 + U2 V2 W2. 
        # We want U16 V16 W16 to approximate this.
        # This is a rank-1 approximation of a rank-2 tensor.
        # Let's just use the sum of the vectors as a starting point, which is bad but maybe optimizable.
        # Better: Use SVD of the merged tensor to get best rank-1 approx.
        
        T_merged = np.zeros((nm, mp, pn))
        T_merged += np.einsum('i,j,k->ijk', U17[:, idx1], V17[:, idx1], W17[:, idx1])
        T_merged += np.einsum('i,j,k->ijk', U17[:, idx2], V17[:, idx2], W17[:, idx2])
        
        # Reshape to (nm, mp*pn) for SVD
        T_mat = T_merged.reshape(nm, mp * pn)
        U_svd, s, Vt_svd = np.linalg.svd(T_mat, full_matrices=False)
        
        # Best rank 1 approx: u * v^T
        u_vec = U_svd[:, 0]
        v_vec = Vt_svd[0, :]
        
        # Scale to match magnitude roughly
        # The term contributes to the sum.
        # Let's set U16 = u_vec, V16 W16 combined = v_vec?
        # No, V and W are separate.
        # We need to split v_vec (which is in B x C space) into V and W.
        # This is hard. 
        # Alternative: Just random initialization for the 16th term, or reuse one of them.
        
        # Let's try a different init: Random perturbation of the whole matrix.
        # Or, simpler: Just start from scratch with random values and optimize.
        # But random start in high dim is hard.
        
        # Let's stick to the merge idea but initialize V and W randomly for the merged term,
        # and optimize everything.
        
        U_init[:, 15] = u_vec
        # Split v_vec into V and W? 
        # v_vec is size mp*pn.
        # Let's pick a random split.
        V_init[:, 15] = rng.standard_normal(mp)
        W_init[:, 15] = rng.standard_normal(pn)
        # Scale to match v_vec magnitude?
        # Actually, let's just let the optimizer figure it out.
        
        # Flatten
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Optimize
        try:
            res = minimize(objective, x0, method='BFGS', options={'maxiter': 500, 'gtol': 1e-8})
            if res.success and res.fun < 1e-6:
                # Extract solution
                size_u = nm * rank
                size_v = mp * rank
                U = res.x[:size_u].reshape(nm, rank)
                V = res.x[size_u:size_u + size_v].reshape(mp, rank)
                W = res.x[size_u + size_v:].reshape(pn, rank)
                
                # Verify
                if _verify_decomposition(U, V, W, n, m, p):
                    return U, V, W
                    
        except Exception:
            pass
            
    return None

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    time_budget = 120 # Total limit is 120s, leave margin
    
    # Try to find rank 16
    result = _find_rank_16(n, m, p, time_budget - 10)
    
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 17
    U, V, W = _rank_17_decomposition(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
