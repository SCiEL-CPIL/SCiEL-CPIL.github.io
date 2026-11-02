import time
import numpy as np

DIMENSIONS = [2, 2, 7]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # Safety margin for 150s limit

def _time_remaining():
    return _DEADLINE - time.time()

def _build_target_tensor():
    """Build the (2,2,7) matrix multiplication tensor T[i,j,k] where C[i,k] = sum_j A[i,j]B[j,k]."""
    n, m, p = 2, 2, 7
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T):
    """Verify U, V, W decompose the target tensor."""
    T_rec = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_rec, T, atol=1e-9)

def _search_rank_subtensor(T_sub, rank, time_limit=10.0):
    """
    Search for a rank-`rank` decomposition of a sub-tensor T_sub.
    T_sub has shape (dim_a, dim_b, dim_c).
    """
    dim_a, dim_b, dim_c = T_sub.shape
    start = time.time()
    
    num_trials = 20
    max_iters = 200
    
    for trial in range(num_trials):
        if time.time() - start > time_limit:
            break
        
        # Random initialization
        U = np.random.randn(dim_a, rank) * 0.5
        V = np.random.randn(dim_b, rank) * 0.5
        W = np.random.randn(dim_c, rank) * 0.5
        
        for it in range(max_iters):
            if time.time() - start > time_limit:
                break
            
            # Update U
            for r in range(rank):
                VW = np.outer(V[:, r], W[:, r])
                denom = np.sum(VW ** 2)
                if denom < 1e-12:
                    continue
                U[:, r] = np.einsum('abc,bc->a', T_sub, VW) / denom
            
            # Update V
            for r in range(rank):
                UW = np.outer(U[:, r], W[:, r])
                denom = np.sum(UW ** 2)
                if denom < 1e-12:
                    continue
                V[:, r] = np.einsum('abc,ac->b', T_sub, UW) / denom
            
            # Update W
            for r in range(rank):
                UV = np.outer(U[:, r], V[:, r])
                denom = np.sum(UV ** 2)
                if denom < 1e-12:
                    continue
                W[:, r] = np.einsum('abc,ab->c', T_sub, UV) / denom
            
            # Check convergence
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.linalg.norm(T_rec - T_sub)
            if residual < 1e-9:
                if _verify_decomposition(U, V, W, T_sub):
                    return U, V, W
    return None

def _construct_rank25():
    """Fallback rank-25 Strassen block decomposition."""
    n, m, p = 2, 2, 7
    rank = 25
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_strassen(col_start, k_start):
        # M1 = (a+d)(e+h)
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[k_start, col_start+0] += 1
        V[p + k_start + 1, col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1

        # M2 = (c+d)e
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1
        W[2*(k_start+1) + 1, col_start+1] -= 1

        # M3 = a(f-h)
        U[0, col_start+2] += 1
        V[k_start+1, col_start+2] += 1
        V[p + k_start + 1, col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1
        W[2*(k_start+1) + 1, col_start+2] += 1

        # M4 = d(g-e)
        U[3, col_start+3] += 1
        V[p + k_start, col_start+3] += 1
        V[k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1
        W[2*k_start + 1, col_start+3] += 1

        # M5 = (a+b)h
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[p + k_start + 1, col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 0, col_start+4] += 1

        # M6 = (c-a)(e+f)
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[k_start, col_start+5] += 1
        V[k_start+1, col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1

        # M7 = (b-d)(g+h)
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[p + k_start, col_start+6] += 1
        V[p + k_start + 1, col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1

    add_strassen(0, 0)
    add_strassen(7, 2)
    add_strassen(14, 4)
    
    col_start = 21
    # Schoolbook for k=6
    U[0, col_start+0] += 1
    V[0*p + 6, col_start+0] += 1
    W[6*n + 0, col_start+0] += 1
    
    U[1, col_start+1] += 1
    V[1*p + 6, col_start+1] += 1
    W[6*n + 0, col_start+1] += 1
    
    U[2, col_start+2] += 1
    V[0*p + 6, col_start+2] += 1
    W[6*n + 1, col_start+2] += 1
    
    U[3, col_start+3] += 1
    V[1*p + 6, col_start+3] += 1
    W[6*n + 1, col_start+3] += 1
    
    return U, V, W

def construct():
    T = _build_target_tensor()
    n, m, p = 2, 2, 7
    
    # Strategy: Try to find a lower rank decomposition.
    # We prioritize ranks 24, 23, 22, 21, 20, 19.
    # We use the sub-block strategy:
    # Block 1: cols 0-3 (2x2x4). Target Rank 11.
    # Block 2: cols 4-6 (2x2x3). Target Rank 8.
    # Total Rank 19.
    
    # Define sub-tensors
    # T is (4, 14, 14)
    # Indices: a (0-3), b (0-13), c (0-13)
    # b index: j*p + k. j=0 (0-6), j=1 (7-13).
    # c index: k*n + i. k=0 (0-1), k=1 (2-3), ..., k=6 (12-13).
    
    # Sub-tensor for cols 0-3 (k=0..3)
    # b indices: 0..6 (j=0, k=0..3) and 7..10 (j=1, k=0..3)
    # c indices: 0..7 (k=0..3, i=0..1)
    T_sub1 = T[:, 0:11, 0:8] # b: 0..10, c: 0..7
    # Wait, b indices for k=0..3:
    # j=0: 0,1,2,3
    # j=1: 7,8,9,10
    # So b slice is not contiguous.
    # We need to extract the correct slices.
    
    # b_indices for k in [0,3]:
    # j=0: [0,1,2,3]
    # j=1: [7,8,9,10]
    b_idx1 = [0,1,2,3, 7,8,9,10]
    # c_indices for k in [0,3]:
    # k=0: [0,1]
    # k=1: [2,3]
    # k=2: [4,5]
    # k=3: [6,7]
    c_idx1 = [0,1, 2,3, 4,5, 6,7]
    
    T_sub1 = T[:, b_idx1, c_idx1] # Shape (4, 8, 8)
    
    # Sub-tensor for cols 4-6 (k=4..6)
    # b_indices for k in [4,5,6]:
    # j=0: [4,5,6]
    # j=1: [11,12,13]
    b_idx2 = [4,5,6, 11,12,13]
    # c_indices for k in [4,5,6]:
    # k=4: [8,9]
    # k=5: [10,11]
    # k=6: [12,13]
    c_idx2 = [8,9, 10,11, 12,13]
    
    T_sub2 = T[:, b_idx2, c_idx2] # Shape (4, 6, 6)
    
    # Search for Rank 11 in T_sub1
    result1 = None
    if _time_remaining() > 20:
        result1 = _search_rank_subtensor(T_sub1, 11, time_limit=15.0)
    
    # Search for Rank 8 in T_sub2
    result2 = None
    if _time_remaining() > 10:
        result2 = _search_rank_subtensor(T_sub2, 8, time_limit=10.0)
    
    if result1 is not None and result2 is not None:
        U1, V1, W1 = result1
        U2, V2, W2 = result2
        
        # Combine
        U_full = np.zeros((4, 19))
        V_full = np.zeros((14, 19))
        W_full = np.zeros((14, 19))
        
        U_full[:, 0:11] = U1
        U_full[:, 11:19] = U2
        
        # Map V indices back
        V_full[b_idx1, 0:11] = V1
        V_full[b_idx2, 11:19] = V2
        
        # Map W indices back
        W_full[c_idx1, 0:11] = W1
        W_full[c_idx2, 11:19] = W2
        
        if _verify_decomposition(U_full, V_full, W_full, T):
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback: Search for Rank 24, 23, 22 globally
    target_ranks = [24, 23, 22]
    for r in target_ranks:
        if _time_remaining() < 5.0:
            break
        result = _search_rank_subtensor(T, r, time_limit=10.0)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, T):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback to Rank 25
    U, V, W = _construct_rank25()
    if _verify_decomposition(U, V, W, T):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": [[0]*25 for _ in range(4)], "V": [[0]*25 for _ in range(14)], "W": [[0]*25 for _ in range(14)]}

