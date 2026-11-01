"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build the tensor T[i*m+j, j*p+k, k*n+i] = 1
    # Reconstruct: for each (i,j,k), check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    # and for all other index combinations, the sum should be 0.
    # We check all 6*24*16 = 2304 entries (but many are zero by structure)
    # Actually we need to check: for all (a, b, c) where a in [0, nm), b in [0, mp), c in [0, pn)
    # the value sum_c U[a,c]*V[b,c]*W[c_idx,c] should equal T[a,b,c_idx]
    # where T[a,b,c_idx] = 1 iff a = i*m+j, b = j'*p+k, c_idx = k'*n+i' with j=j' and k=k' and i=i'
    # More precisely: T[i*m+j, j*p+k, k*n+i] = 1, all other entries 0.
    # Let's build the full tensor and check.
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Reconstruct from decomposition
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    if not np.allclose(recon, T, atol=1e-9):
        return False
    return True

def _round_and_verify(U, V, W, n, m, p):
    """Round to nearest half-integer and verify."""
    U_r = np.round(2.0 * np.array(U, dtype=float)) / 2.0
    V_r = np.round(2.0 * np.array(V, dtype=float)) / 2.0
    W_r = np.round(2.0 * np.array(W, dtype=float)) / 2.0
    if _verify(U_r, V_r, W_r, n, m, p):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    # Try rounding to integers
    U_i = np.round(np.array(U, dtype=float))
    V_i = np.round(np.array(V, dtype=float))
    W_i = np.round(np.array(W, dtype=float))
    if _verify(U_i, V_i, W_i, n, m, p):
        return U_i.tolist(), V_i.tolist(), W_i.tolist()
    return None

def _build_rank44_decomposition(n, m, p):
    """
    Build a rank-44 decomposition for 2x3 x 3x8 matrix multiplication.
    
    Strategy: Decompose the 3x8 part into blocks and use Strassen for 2x2x2.
    
    The 2x3 * 3x8 multiplication can be viewed as:
    - The input A is 2x3, B is 3x8
    - We split B into 4 blocks of 3x2: B = [B1, B2, B3, B4] where Bj is 3x2
    - The output C = A*B is 2x8, split into 4 blocks of 2x2: C = [C1, C2, C3, C4]
    - Ci = A * Bi
    
    For each Ci = A * Bi where A is 2x3 and Bi is 3x2:
    We can write A = [a1, a2, a3] (2x1 columns) and Bi = [b1, b2] (3x1 columns)
    Ci = a1*b1^T + a2*b2^T + a3*b3^T... wait, that's not right.
    
    Actually A is 2x3 and Bi is 3x2, so Ci = A * Bi is 2x2.
    Ci = sum_j A[:,j] * Bi[j,:] where A[:,j] is 2x1 and Bi[j,:] is 1x2.
    So Ci = A[:,0]*Bi[0,:] + A[:,1]*Bi[1,:] + A[:,2]*Bi[2,:]
    
    This is a 2x2x2 matrix multiplication if we think of it differently.
    Actually, let's think of it as: Ci[i,k] = sum_j A[i,j] * Bi[j,k]
    This is exactly a 2x2x2 tensor contraction! The indices are i in {0,1}, j in {0,1,2}, k in {0,1}.
    But j has 3 values, not 2. So it's not directly 2x2x2.
    
    Alternative approach: Use the fact that 2x3*3x2 can be done with rank... 
    Actually, let's use a different decomposition.
    
    Let me reconsider. The standard approach for 2x3*3x8:
    - Split the 3x8 matrix into 4 blocks of 3x2
    - Each 2x3 * 3x2 multiplication has a known minimal rank
    - But we need to share computations across blocks
    
    Actually, a better approach: Use Strassen on 2x2x2 subproblems.
    Split the 3x8 into: first 2 columns handled with one 2x2x2 Strassen, 
    and the remaining... this gets complicated.
    
    Let me use a simpler known construction. The rank of 2x3*3x8 is at most 44.
    One construction:
    - Write the 3x8 as [3x2 | 3x2 | 3x2 | 3x2]
    - For each 3x2 block, the 2x3*3x2 product can be computed with rank 4 (schoolbook for 2x2x2 is 4, but we have 3 in the middle)
    - Actually 2x3*3x2: T[i,j,k] where i in 2, j in 3, k in 2. Rank is at most 6 (schoolbook).
    - 4 blocks * 6 = 24, plus we need to handle the coupling... 
    
    Hmm, let me think differently. Let's use the Kronecker product structure.
    
    Actually, the known result is that the rank of 2x3*3x8 is 44.
    Here's a construction using Strassen:
    
    Think of 2x3*3x8 as (2x3)*(3x8). 
    We can write 3x8 = 3x(2+2+2+2) and 2x3 = (2x3).
    
    Alternative: Use the fact that [2,3,8] can be embedded in [2,4,8] which has rank 32 (Strassen-like),
    but that's not directly helpful.
    
    Let me try a direct construction. We'll use 44 rank-1 terms.
    
    Actually, let me just implement a known rank-44 decomposition.
    The key insight: split the 8 columns into 4 groups of 2.
    For each group of 2 columns, we have a 2x3 * 3x2 subproblem.
    
    A 2x3 * 3x2 multiplication: C[i,k] = sum_j A[i,j]*B[j,k], i in {0,1}, j in {0,1,2}, k in {0,1}
    This tensor has 2*3*2 = 12 entries, all 1 in the right positions.
    Its rank is at most 6 (schoolbook: one term per (i,j,k) with j=0,1,2 and i,k fixed... no)
    Schoolbook: for each i in {0,1}, j in {0,1,2}, k in {0,1}: term with U[i,j]=1, V[j,k]=1, W[k,i]=1. That's 6 terms.
    
    But we can do better. The 2x3*3x2 tensor can be decomposed with rank 5 or less?
    Actually, I recall that the rank of 2x3*3x2 is 6 (same as schoolbook) for this small case.
    
    So 4 blocks * 6 = 24 terms for the block-diagonal part. But we need 44, so there's more structure.
    
    Let me try a different approach: use Strassen's algorithm adapted.
    
    Actually, I think the correct approach is:
    - The 2x3*3x8 can be seen as computing 4 outputs of 2x2 each
    - Use a shared Strassen-like decomposition
    
    Let me just implement the ALS approach to find a rank-44 decomposition, then round it.
    """
    n, m, p = 2, 3, 8
    target_rank = 44
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Use ALS to find a rank-44 decomposition
    np.random.seed(42)
    
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    start_time = time.time()
    deadline = 100.0  # seconds
    
    for attempt in range(5):
        if time.time() > deadline - 20:
            break
        
        # Initialize
        U = np.random.randn(n*m, target_rank) * 0.1
        V = np.random.randn(m*p, target_rank) * 0.1
        W = np.random.randn(p*n, target_rank) * 0.1
        
        # ALS iterations
        for it in range(200):
            if time.time() > deadline - 10:
                break
            
            # Update U: for each row a, U[a,:] = (V.T * W.T * T[a,:,:].reshape(-1)) / (V.T * W.T * ones)
            # More efficiently:
            # T[a,:,:] reshaped is a vector of length m*p*p*n
            # U[a,c] = sum_{b,d} T[a,b,d] * V[b,c] * W[d,c] / sum_{b,d} V[b,c] * W[d,c]
            VW = V.T * W.T  # (target_rank, m*p) * ... no
            # Let's do it properly
            # For each c: U[:,c] should satisfy sum_c U[a,c]*V[b,c]*W[d,c] = T[a,b,d]
            # Fix V,W: for each a, U[a,:] = argmin sum_{b,d} (T[a,b,d] - sum_c U[a,c]*V[b,c]*W[d,c])^2
            # This is: for each a, U[a,:] = (M_a * (V*W)) / ((V*W)*(V*W)) where M_a is T[a,:,:] flattened
            # Actually: let X = V * W element-wise, shape (m*p, p*n, rank) -> no
            # X[b,d,c] = V[b,c]*W[d,c]
            # Then for each a: U[a,c] = sum_{b,d} T[a,b,d]*X[b,d,c] / sum_{b,d} X[b,d,c]^2
            
            # Compute X = V[:,None,:] * W[None,:,:] shape (m*p, p*n, rank)
            X = V[:, None, :] * W[None, :, :]  # (m*p, p*n, rank)
            # Numerator: for each a, c: sum_{b,d} T[a,b,d] * X[b,d,c]
            # = einsum('abd,bdc->ac', T, X)
            num_U = np.einsum('abd,bdc->ac', T, X)
            # Denominator: for each c: sum_{b,d} X[b,d,c]^2
            den_U = np.sum(X**2, axis=(0,1))  # (rank,)
            den_U = np.where(den_U == 0, 1e-10, den_U)
            U = num_U / den_U[None, :]
            
            # Update V
            X2 = U[:, None, :] * W[None, :, :]  # (n*m, p*n, rank)
            num_V = np.einsum('abd,adc->bc', T, X2)
            den_V = np.sum(X2**2, axis=(0,2))
            den_V = np.where(den_V == 0, 1e-10, den_V)
            V = num_V / den_V[None, :]
            
            # Update W
            X3 = U[:, None, :] * V[:, None, :]  # (n*m, m*p, rank)
            num_W = np.einsum('abd,abc->dc', T, X3)
            den_W = np.sum(X3**2, axis=(0,1))
            den_W = np.where(den_W == 0, 1e-10, den_W)
            W = num_W / den_W[None, :]
        
        # Check error
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(recon - T)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-8:
            break
    
    if best_U is None:
        # Fallback to schoolbook
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
    
    # Try to round
    result = _round_and_verify(best_U, best_V, best_W, n, m, p)
    if result is not None:
        U_r, V_r, W_r = result
        return {"U": U_r, "V": V_r, "W": W_r}
    
    # If rounding fails, use the unrounded (should still be valid if error is small enough)
    # But the problem requires integer/half-integer entries
    # Let's try harder to round
    # Scale and round
    for scale in [1, 2, 4, 8]:
        U_s = np.round(scale * best_U) / scale
        V_s = np.round(scale * best_V) / scale
        W_s = np.round(scale * best_W) / scale
        if _verify(U_s, V_s, W_s, n, m, p):
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    # Fallback: return schoolbook
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

def construct():
    """
    Rank-44 decomposition of 2x3*3x8 matrix multiplication using ALS search
    with deterministic fallback to schoolbook (rank 48).
    """
    n, m, p = DIMENSIONS
    
    # First try the rank-44 construction
    try:
        result = _build_rank44_decomposition(n, m, p)
        # Verify it's valid
        U = np.array(result["U"], dtype=float)
        V = np.array(result["V"], dtype=float)
        W = np.array(result["W"], dtype=float)
        if U.shape[1] < n*m*p:  # Better than schoolbook
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
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
# EVOLVE-BLOCK-END
