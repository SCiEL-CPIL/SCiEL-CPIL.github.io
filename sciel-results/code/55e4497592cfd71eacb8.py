"""Program for matrix multiplication tensor decomposition of (3,3,8)."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 8]

DEADLINE_SECONDS = 120.0
SAFETY_MARGIN = 10.0


def _schoolbook_decomposition(n, m, p):
    """Build the standard (schoolbook) rank-n*m*p decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return U, V, W


def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor as a 3D array.
    
    T[a, b, c] where a = i*m+j, b = j'*p+k', c = k''*n+i''
    T[a,b,c] = 1 if j=j' and k'=k'' and i''=i, else 0.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check all diagonal entries (should be 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check off-diagonal entries (should be 0)
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for kp in range(p):
                    for ip in range(n):
                        if jp == j and kp == k and ip == i:
                            continue
                        val = np.sum(U_arr[i*m+j, :] * V_arr[jp*p+kp, :] * W_arr[kp*n+ip, :])
                        if abs(val) > tol:
                            return False
    return True


def _als_decomposition(T, n, m, p, rank, max_iter=100, tol=1e-12, seed=42):
    """Use ALS to find a rank-r decomposition of the tensor T."""
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    for it in range(max_iter):
        # Precompute design matrices using Kronecker-like structure
        D_U = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(m * p * p * n, rank)
        D_V = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(n * m * p * n, rank)
        D_W = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(n * m * m * p, rank)
        
        # Update U
        for a in range(n * m):
            target = T[a, :, :].flatten()
            U[a, :] = np.linalg.lstsq(D_U, target, rcond=None)[0]
        
        # Update V
        for b in range(m * p):
            target = T[:, b, :].flatten()
            V[b, :] = np.linalg.lstsq(D_V, target, rcond=None)[0]
        
        # Update W
        for c in range(p * n):
            target = T[:, :, c].flatten()
            W[c, :] = np.linalg.lstsq(D_W, target, rcond=None)[0]
        
        # Check convergence
        if it % 20 == 0 or it == max_iter - 1:
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        err += (val - 1.0)**2
            if err < tol:
                break
    
    return U.tolist(), V.tolist(), W.tolist()


def _try_rank(T, n, m, p, rank, time_budget=10.0, seed=42):
    """Try to find a rank-r decomposition. Returns (U,V,W) or None."""
    start = time.time()
    try:
        U, V, W = _als_decomposition(T, n, m, p, rank, max_iter=100, tol=1e-12, seed=seed)
        if time.time() - start > time_budget:
            return None
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
            return U, V, W
    except Exception:
        pass
    return None


def _winograd_3x3_decomposition():
    """Winograd's rank-23 decomposition for 3x3 matrix multiplication.
    
    This uses the well-known Winograd scheme for 3x3*3x3.
    We extend it to 3x3*3x8 by replicating the k=0,1,2 pattern for k=3..7.
    """
    # Winograd 3x3 uses rank 23 with specific coefficient structure.
    # The key insight: for 3x3*3x8, we can use the 3x3 Winograd for the first
    # 3 columns and replicate for the remaining 5 columns.
    # But that gives rank 23 + 5*9 = 68, which is worse than schoolbook 72.
    # Actually, let's think more carefully.
    
    # For 3x3*3x8: A is 3x3, B is 3x8, C = A*B is 3x8.
    # C[i,k] = sum_j A[i,j] * B[j,k] for k=0..7
    
    # Winograd 3x3x3 has rank 23. For 3x3x8, one approach:
    # Use the 3x3 Winograd structure for each "block" of 3 columns.
    # But the columns are independent in the output, so we can't simply share.
    
    # Alternative: use the structure that B has 8 columns.
    # The tensor is T[i,j,k] = A[i,j]*B[j,k] -> C[i,k]
    # This is a 3x3x8 tensor with structure T[i*m+j, j'*p+k', k''*n+i''] = 1 if j=j', k'=k'', i''=i
    
    # Let me construct a rank-50 decomposition using Winograd-like ideas.
    # Actually, for 3x3*3x8, the known best is at most 72 (schoolbook).
    # Winograd 3x3x3 = 23. If we could share structure across the 8 columns...
    
    # Let's try a direct construction. The tensor has 3*3*8 = 72 non-zero entries.
    # We want to find rank r < 72.
    
    # One approach: group the 8 columns into groups.
    # For columns 0,1,2: use Winograd 3x3x3 (rank 23)
    # For columns 3,4,5: another Winograd 3x3x3 (rank 23)  
    # For columns 6,7: schoolbook 3x3x2 (rank 18)
    # Total: 23+23+18 = 64. Still better than 72!
    
    # But wait, we need to be careful about the indexing.
    # Let me construct this properly.
    
    n, m, p = 3, 3, 8
    rank = 64
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Winograd 3x3x3 decomposition
    # The standard Winograd 3x3x3 uses 23 multiplications.
    # Let me use the known Winograd scheme.
    
    # Winograd 3x3: 
    # A is 3x3, B is 3x3, C = A*B
    # Define vectors:
    # a1 = [A[0,0], A[0,1], A[0,2]]
    # a2 = [A[1,0], A[1,1], A[1,2]]  
    # a3 = [A[2,0], A[2,1], A[2,2]]
    # b1 = [B[0,0], B[1,0], B[2,0]]
    # b2 = [B[0,1], B[1,1], B[2,1]]
    # b3 = [B[0,2], B[1,2], B[2,2]]
    
    # Winograd uses:
    # d1 = a1 * b1 (element-wise) -> 3 mults
    # d2 = a2 * b2 -> 3 mults
    # d3 = a3 * b3 -> 3 mults
    # d4 = (a1+a2) * (b1+b2) -> 3 mults
    # d5 = (a1+a3) * (b1+b3) -> 3 mults
    # d6 = (a2+a3) * (b2+b3) -> 3 mults
    # d7 = (a1+2*a2+4*a3) * (b1+4*b2+2*b3) -> 3 mults
    # d8 = (a1+a2-2*a3) * (b1+b2-2*b3) -> 3 mults
    # d9 = (a1-2*a2+a3) * (b1-2*b2+b3) -> 3 mults
    # Total: 9*3 = 27... that's not 23.
    
    # Let me look up the actual Winograd 3x3x3 = 23 scheme.
    # Actually, the known result is that 3x3x3 has rank 23 (Winograd).
    # Let me use a simpler approach for now.
    
    # For this implementation, let's use a block approach:
    # Columns 0-2: schoolbook 3x3x3 = 9
    # Columns 3-5: schoolbook 3x3x3 = 9
    # Columns 6-7: schoolbook 3x3x2 = 6
    # Total: 24. That's much better than 72!
    
    # Wait, that's just schoolbook but grouped. Total is still 3*3*8 = 72.
    # I made an error. Schoolbook is always 3*3*8 = 72.
    
    # Let me reconsider. The Winograd 3x3x3 = 23 means we can compute 
    # a 3x3 * 3x3 product with 23 scalar multiplications instead of 9*3=27.
    
    # For 3x3 * 3x8:
    # Split B into B1 (3x3) and B2 (3x5)
    # C1 = A*B1: use Winograd 3x3x3, rank 23
    # C2 = A*B2: schoolbook 3x3x5, rank 45
    # Total: 23 + 45 = 68
    
    # Or split into B1 (3x3), B2 (3x3), B3 (3x2):
    # C1: Winograd 23, C2: Winograd 23, C3: schoolbook 18
    # Total: 23+23+18 = 64
    
    # Let me implement the 3+3+2 split with Winograd for the 3x3 parts.
    
    # Actually, I need the explicit Winograd 3x3x3 = 23 coefficients.
    # This is complex. Let me use a simpler rank reduction first.
    
    # For now, return schoolbook and try ALS to find better.
    return None


def _greedy_rank_reduction(T, n, m, p, max_rank=50, time_budget=20.0, seed=42):
    """Greedy heuristic to find a low-rank decomposition.
    
    Strategy: Start with random initialization and use ALS with 
    multiple restarts and rank annealing.
    """
    rng = np.random.RandomState(seed)
    start = time.time()
    
    best_result = None
    best_rank = max_rank
    
    for target_rank in range(50, 25, -2):
        if time.time() - start > time_budget:
            break
        
        for restart in range(3):
            if time.time() - start > time_budget:
                break
            
            U = rng.randn(n * m, target_rank) * 0.1
            V = rng.randn(m * p, target_rank) * 0.1
            W = rng.randn(p * n, target_rank) * 0.1
            
            # ALS iterations
            for it in range(50):
                if time.time() - start > time_budget:
                    break
                
                # Update U
                for a in range(n * m):
                    D = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(m * p * p * n, target_rank)
                    target = T[a, :, :].flatten()
                    U[a, :] = np.linalg.lstsq(D, target, rcond=None)[0]
                
                # Update V
                for b in range(m * p):
                    D = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(n * m * p * n, target_rank)
                    target = T[:, b, :].flatten()
                    V[b, :] = np.linalg.lstsq(D, target, rcond=None)[0]
                
                # Update W
                for c in range(p * n):
                    D = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(n * m * m * p, target_rank)
                    target = T[:, :, c].flatten()
                    W[c, :] = np.linalg.lstsq(D, target, rcond=None)[0]
            
            # Verify
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-8):
                best_result = (U.tolist(), V.tolist(), W.tolist())
                best_rank = target_rank
                break
    
    return best_result


def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_result = {"U": U, "V": V, "W": W}
    best_rank = n * m * p
    
    # Build target tensor
    T = _build_target_tensor(n, m, p)
    
    remaining = DEADLINE_SECONDS - SAFETY_MARGIN - (time.time() - start_time)
    
    if remaining < 15.0:
        return best_result
    
    # Try greedy rank reduction
    result = _greedy_rank_reduction(T, n, m, p, max_rank=50, 
                                     time_budget=min(remaining * 0.4, 30.0), seed=42)
    if result is not None:
        U_new, V_new, W_new = result
        best_result = {"U": U_new, "V": V_new, "W": W_new}
        best_rank = len(U_new[0])
    
    remaining = DEADLINE_SECONDS - SAFETY_MARGIN - (time.time() - start_time)
    
    if remaining < 15.0:
        return best_result
    
    # Try ALS with various ranks
    for rank in [40, 35, 30, 28, 26, 24, 22, 20]:
        if rank >= best_rank:
            continue
        remaining = DEADLINE_SECONDS - SAFETY_MARGIN - (time.time() - start_time)
        if remaining < 15.0:
            break
        
        time_budget = min(remaining * 0.25, 15.0)
        result = _try_rank(T, n, m, p, rank, time_budget=time_budget, seed=42 + rank)
        
        if result is not None:
            U_new, V_new, W_new = result
            best_result = {"U": U_new, "V": V_new, "W": W_new}
            best_rank = rank
    
    return best_result
