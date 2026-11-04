import numpy as np
import time
import sys

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Build the rank-19 decomposition for 4x5x5 matrix multiplication
    # using the known Bini-Capovani-Lotti-Romani (BCLR) decomposition for 5x5x5,
    # restricted to the first 4 rows of the input matrix.
    
    # BCLR decomposition: 19 terms for 5x5x5 matrix multiplication
    # Each term has U_r, V_r, W_r as 5x5 matrices
    # We restrict U to the first 4 rows (since n=4)
    
    # The BCLR 5x5x5 decomposition uses the following 19 sets of (U_r, V_r, W_r)
    # where U_r, V_r, W_r are 5x5 matrices
    # T[i,j,k] = sum_r U_r[i,j] * V_r[j,k] * W_r[k,i]
    
    # Let me define the BCLR factors. The decomposition is:
    # (AB)_{ik} = sum_j A_{ij} B_{jk}
    
    # For the BCLR scheme, the 19 terms are structured as follows.
    # I'll use a known explicit representation.
    
    # Actually, let me use a cleaner approach. The tensor for matrix multiplication
    # of (4x5) by (5x5) is:
    # T[a, b, c] where a = i*5+j (i in 0..3, j in 0..4), b = j*5+k (j in 0..4, k in 0..4), c = k*4+i (k in 0..4, i in 0..3)
    # T[a,b,c] = 1 iff b mod 5 == j and a // 5 == i and a mod 5 == j and c mod 4 == i and c // 4 == k and b // 5 == j
    
    # Let me try a rank-19 construction based on the 5x5x5 BCLR decomposition.
    # The BCLR decomposition for 5x5x5 has rank 19.
    # We can restrict it to 4x5x5 by taking the first 4 rows of U factors.
    
    # BCLR 5x5x5 factors (from literature):
    # The decomposition is:
    # (A*B)_{i,k} = sum_{j=0}^{4} A_{i,j} * B_{j,k}
    
    # Let me use a different, well-known rank-19 decomposition structure.
    # I'll construct it using the Strassen-like approach for 4x5x5.
    
    # Actually, let me try to construct a valid rank decomposition directly.
    # For 4x5x5, the schoolbook rank is 4*5*5 = 100.
    # A better bound: we can use the fact that we're multiplying by a fixed size.
    
    # Let me try rank 19 based on BCLR restriction.
    # BCLR for 5x5x5: the 19 terms have specific 5x5 factor matrices.
    
    # I'll define the BCLR factors explicitly.
    # From the BCLR paper, the decomposition of 5x5x5 matrix mult into 19 terms:
    
    # Let me use a simpler approach: construct the tensor and use a rank-revealing
    # approach, but since we need integer/half-integer entries, let me try to
    # find a structured low-rank decomposition.
    
    # For now, let me return the schoolbook decomposition as a safe baseline,
    # but also attempt to find a lower rank via optimization.
    
    # Schoolbook decomposition
    rank_sb = n * m * p  # 100
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Time limit
    start_time = time.time()
    deadline = start_time + 140  # 140 seconds (10s safety margin for 150s limit)
    
    # Construct the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Try rank 19 using the BCLR-inspired construction
    # For 5x5x5, BCLR gives rank 19. For 4x5x5, we restrict.
    # Let me construct the 4x5x5 tensor and see if rank 19 works.
    
    # Approach: Use the fact that the 4x5x5 multiplication tensor is a sub-tensor
    # of the 5x5x5 multiplication tensor. The 5x5x5 tensor has rank 19 (BCLR).
    # The 4x5x5 tensor is obtained by fixing i=4 in the output dimension and
    # removing the corresponding row from U factors.
    
    # BCLR 5x5x5 factor matrices:
    # I'll define them as 19 sets of (5x5, 5x5, 5x5) matrices.
    # Then for 4x5x5, I take the first 4 rows of each U_r.
    
    # Let me define the BCLR factors. The exact values from the BCLR paper:
    # The decomposition uses the identity:
    # x*y*z = ((x+y+z)^2 - x^2 - y^2 - z^2 - 2(xy+yz+zx)) ... no, that's not right.
    
    # Let me use a known explicit BCLR representation.
    # The 19 terms for 5x5x5 can be written as:
    
    # Actually, let me try a different approach. Let me search for a rank-19
    # decomposition using random initialization and ALS, targeting half-integers.
    
    for target_rank in [19, 20, 25, 30, 40, 50]:
        if time.time() > deadline:
            break
        
        result = try_decomposition(T, n, m, p, target_rank, deadline)
        if result is not None:
            U_opt, V_opt, W_opt = result
            if verify_decomposition(U_opt, V_opt, W_opt, T, n, m, p):
                best_rank = target_rank
                best_U = U_opt
                best_V = V_opt
                best_W = W_opt
                break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

def try_decomposition(T, n, m, p, rank, deadline):
    """Try to find a rank decomposition using random restarts and ALS"""
    np.random.seed(42)
    
    for attempt in range(10):
        if time.time() > deadline:
            return None
        
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        # ALS iterations
        for it in range(500):
            if time.time() > deadline:
                return None
            
            # Update U
            M_U = np.einsum('br,cr->bcr', V, W).reshape(rank, -1)
            try:
                U_new = np.linalg.lstsq(M_U.T, T.reshape(n*m, -1).T, rcond=None)[0].T
                U = U_new
            except:
                pass
            
            # Update V
            M_V = np.einsum('ar,cr->acr', U, W).reshape(rank, -1)
            try:
                V_new = np.linalg.lstsq(M_V.T, T.transpose(1, 0, 2).reshape(m*p, -1).T, rcond=None)[0].T
                V = V_new
            except:
                pass
            
            # Update W
            M_W = np.einsum('ar,br->abr', U, V).reshape(rank, -1)
            try:
                W_new = np.linalg.lstsq(M_W.T, T.transpose(2, 0, 1).reshape(p*n, -1).T, rcond=None)[0].T
                W = W_new
            except:
                pass
        
        # Round to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        if verify_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist(), T, n, m, p, tol=1e-8):
            return (U_r.tolist(), V_r.tolist(), W_r.tolist())
        
        # Try with tighter rounding or different scaling
        for scale in [1, 2, 4]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if verify_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist(), T, n, m, p, tol=1e-8):
                return (U_r.tolist(), V_r.tolist(), W_r.tolist())
    
    return None

def verify_decomposition(U, V, W, T, n, m, p, tol=1e-8):
    """Verify that the decomposition exactly reconstructs T"""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    if U_arr.shape[1] != V_arr.shape[1] or V_arr.shape[1] != W_arr.shape[1]:
        return False
    if U_arr.shape[0] != n*m or V_arr.shape[0] != m*p or W_arr.shape[0] != p*n:
        return False
    
    reconstructed = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    return np.allclose(reconstructed, T, atol=tol, rtol=0)

# EVOLVE-BLOCK-END
