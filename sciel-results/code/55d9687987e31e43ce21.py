import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 4, 8
    
    # Try rank 40 construction
    U, V, W = _build_rank_40(n, m, p)
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to rank 56
    U56, V56, W56 = _build_rank_56(n, m, p)
    return {"U": U56, "V": V56, "W": W56}

def _build_rank_40(n, m, p):
    """
    Rank 40 construction: Split j into two groups of 2.
    For group {j1,j2}, we compute A_i1*B_j1 + A_i2*B_j2 for i in {0,1}, k in {0..7}.
    This is a (2x2)*(2x8) = 2x8 multiplication.
    
    For each (2x2)*(2x8) subproblem, we use a rank-20 scheme:
    Split k into 4 pairs. For each (2x2)*(2x2) block, use rank 5 (not 7) 
    by exploiting that we only need 4 output values, not the full 2x2.
    
    Actually, let me use a different approach: 
    For (2x2)*(2x8), write it as 4 independent (2x2)*(2x2) blocks.
    Each (2x2)*(2x2) has rank 7 via Strassen, giving 4*7=28 per group, total 56.
    
    To get rank 40, we need 20 per group. Let's try:
    For (2x2)*(2x8), use 5 rank-4 schemes on overlapping (2x2)*(2x4) subblocks.
    A (2x2)*(2x4) can be done in rank 10 (two (2x2)*(2x2) in rank 5 each? No, rank 7 each = 14).
    
    Alternative: Use the fact that for fixed j-pair, the problem is:
    C[i,k] = sum_{j in pair} A[i,j] * B[j,k]
    This is equivalent to computing 2 linear combinations of the 8 columns of B,
    where the coefficients are the 2x2 A block.
    
    Let me try a direct rank-20 scheme for (2x2)*(2x8):
    We need to compute for i=0,1 and k=0..7:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    This is: C[0,:] = A[0,0]*B[0,:] + A[0,1]*B[1,:]
    C[1,:] = A[1,0]*B[0,:] + A[1,1]*B[1,:]
    
    So we're computing 2 linear combinations of 8-length vectors.
    Total: 4 scalar multiplications per k, but tensor-wise:
    Each C[i,k] depends on 2 products: A[i,j]*B[j,k] for j=0,1.
    
    The tensor for this subproblem has 2*2*8 = 32 entries, each being a product of 2 variables.
    Rank is at most 16 (each product is rank 1). Can we do better?
    
    Actually, the minimal rank of (2x2)*(2x8) is known. Let me just use rank 16 per group = 32 total.
    """
    total_rank = 32
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # For each j-pair (j1, j2):
    for jp in range(2):
        j1 = 2 * jp
        j2 = 2 * jp + 1
        base_col = 16 * jp
        
        # For each k, we have:
        # C[0,k] += A[0,j1]*B[j1,k] + A[0,j2]*B[j2,k]
        # C[1,k] += A[1,j1]*B[j1,k] + A[1,j2]*B[j2,k]
        
        # Each term A[i,j]*B[j,k] is rank 1.
        # 4 terms per k, 8 k values = 32 terms, but we can combine.
        
        # Actually, for fixed i,j,k: A[i,j]*B[j,k] contributes to C[i,k].
        # The tensor entry is U[i*m+j] * V[j*p+k] * W[k*n+i].
        # We need sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] = 1 if (i,j,k) is a valid triple.
        
        # For the j-pair, the valid triples are (i, j1, k) and (i, j2, k) for i in {0,1}, k in {0..7}.
        # That's 2*2*8 = 32 triples. Each is rank 1, so rank <= 32 for the subproblem.
        # Total rank = 2 * 32 = 64. That's worse!
        
        # I made an error. The full problem has 2*4*8 = 64 triples, each rank 1, so rank <= 64.
        # Strassen gives 56. Let me just stick with 56.
        
        pass
    
    # This approach doesn't work. Return dummy.
    return U, V, W

def _build_rank_56(n, m, p):
    """Rank 56 decomposition via Strassen on 2x2 sub-blocks."""
    total_rank = 56
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for u in range(2):
        for v in range(4):
            base_col = 7 * (4 * u + v)
            
            a11_idx = 0 * 4 + 2 * u
            a12_idx = 0 * 4 + 2 * u + 1
            a21_idx = 1 * 4 + 2 * u
            a22_idx = 1 * 4 + 2 * u + 1
            
            b11_idx = 2 * u * 8 + 2 * v
            b12_idx = 2 * u * 8 + 2 * v + 1
            b21_idx = (2 * u + 1) * 8 + 2 * v
            b22_idx = (2 * u + 1) * 8 + 2 * v + 1
            
            c11_idx = 2 * v * 2 + 0
            c12_idx = (2 * v + 1) * 2 + 0
            c21_idx = 2 * v * 2 + 1
            c22_idx = (2 * v + 1) * 2 + 1
            
            U[a11_idx][base_col] += 1
            U[a22_idx][base_col] += 1
            V[b11_idx][base_col] += 1
            V[b22_idx][base_col] += 1
            W[c11_idx][base_col] += 1
            W[c22_idx][base_col] += 1
            
            U[a21_idx][base_col + 1] += 1
            U[a22_idx][base_col + 1] += 1
            V[b11_idx][base_col + 1] += 1
            W[c21_idx][base_col + 1] += 1
            W[c22_idx][base_col + 1] -= 1
            
            U[a11_idx][base_col + 2] += 1
            V[b12_idx][base_col + 2] += 1
            V[b22_idx][base_col + 2] -= 1
            W[c12_idx][base_col + 2] += 1
            W[c22_idx][base_col + 2] += 1
            
            U[a22_idx][base_col + 3] += 1
            V[b21_idx][base_col + 3] += 1
            V[b11_idx][base_col + 3] -= 1
            W[c11_idx][base_col + 3] += 1
            W[c21_idx][base_col + 3] += 1
            
            U[a11_idx][base_col + 4] += 1
            U[a12_idx][base_col + 4] += 1
            V[b22_idx][base_col + 4] += 1
            W[c11_idx][base_col + 4] -= 1
            W[c12_idx][base_col + 4] += 1
            
            U[a21_idx][base_col + 5] += 1
            U[a11_idx][base_col + 5] -= 1
            V[b11_idx][base_col + 5] += 1
            V[b12_idx][base_col + 5] += 1
            W[c22_idx][base_col + 5] += 1
            
            U[a12_idx][base_col + 6] += 1
            U[a22_idx][base_col + 6] -= 1
            V[b21_idx][base_col + 6] += 1
            V[b22_idx][base_col + 6] += 1
            W[c11_idx][base_col + 6] += 1
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for s in range(rank):
                    val += U_arr[i*m+j, s] * V_arr[j*p+k, s] * W_arr[k*n+i, s]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True
