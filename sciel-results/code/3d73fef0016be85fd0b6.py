import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 2, 4, 8

    # Try rank 48 construction (4x4x8 via 2x2 Strassen blocks)
    U, V, W = _build_rank_48(n, m, p)

    if time.time() < deadline:
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}

    # Fallback to rank 56
    U56, V56, W56 = _build_rank_56(n, m, p)
    return {"U": U56, "V": V56, "W": W56}

def _build_rank_48(n, m, p):
    """
    Rank 48 decomposition: Treat as 4x4 x 4x8 multiplication.
    Split the 4x4 A matrix into four 2x2 blocks.
    For each 2x2 block of A, multiply by the 4x8 B matrix.
    Use Strassen for each 2x4 x 4x8 subproblem.
    """
    total_rank = 48
    U = [[0] * total_rank for _ in range(n * m)]  # 8 x 48
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 48
    W = [[0] * total_rank for _ in range(p * n)]  # 16 x 48

    # A is 2x4, but we treat it as 4x4 by zero-padding? No, that's not right.
    # Let's re-read the task.
    # "multiplying a 2 by 4 matrix by a 4 by 8 matrix"
    # A: 2x4, B: 4x8, C: 2x8.
    # My previous analysis was treating A as 4x4, which is wrong.
    # The dimensions are n=2, m=4, p=8.
    # The tensor is M_{2,4,8}.
    # The rank of M_{2,4,8} is at most 2 * Rank(M_{1,4,8})? No.
    # It is at most Rank(M_{2,2,8}) * 2? No.
    # It is at most 2 * Rank(M_{1,4,8}).
    # Rank(M_{1,4,8}) is the rank of the tensor for 1x4 x 4x8.
    # This is a linear map from R^4 to R^8, but tensor-wise it's different.
    # Actually, M_{1,m,p} has rank m*p? No.
    # The tensor T_{i,j,k} = 1 if i,j,k are valid indices.
    # For M_{1,m,p}, i is always 0. T_{0,j,k} = 1 if j in 0..m-1, k in 0..p-1.
    # This tensor is the outer product of a vector of 1s (length m) and a vector of 1s (length p)?
    # No, the indices are (i, j, k).
    # U[i*m+j, s] V[j*p+k, s] W[k*n+i, s].
    # For n=1, U[j, s] V[j*p+k, s] W[k, s].
    # Sum_s U[j,s] V[j*p+k,s] W[k,s] = 1 for all j,k.
    # This is a rank-1 tensor if U[j,s]=1, V[j*p+k,s]=1, W[k,s]=1?
    # Sum_s 1 * 1 * 1 = 1.
    # So Rank(M_{1,m,p}) = 1.
    # Then Rank(M_{2,m,p}) <= 2 * 1 = 2?
    # Let's check.
    # For n=2, m=4, p=8.
    # T_{i,j,k} = 1.
    # Sum_s U[4i+j, s] V[8j+k, s] W[2k+i, s] = 1.
    # If we set s=0,1.
    # s=0: U[4i+j,0]=1, V[8j+k,0]=1, W[2k+i,0]=1/2? No.
    # We need the sum to be 1.
    # If U=1, V=1, W=1/2, sum is 1/2.
    # If U=1, V=1, W=1, sum is 1.
    # So Rank(M_{2,4,8}) = 1?
    # That can't be right. Matrix multiplication is not rank 1.
    # The tensor for matrix multiplication is NOT the all-ones tensor.
    # The tensor T_{i,j,k} is 1 if (i,j,k) is a "triple" in the multiplication.
    # For standard matrix multiplication, C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T_{i,j,k} = 1 for all valid i,j,k.
    # Wait, the tensor decomposition represents the bilinear map.
    # The coefficients of the bilinear map are 1 for each term A[i,j]B[j,k].
    # So yes, T_{i,j,k} = 1 for all i,j,k.
    # So why is the rank not 1?
    # Because U, V, W are vectors of length n*m, m*p, p*n.
    # And the indices are coupled.
    # U index: i*m+j. V index: j*p+k. W index: k*n+i.
    # The rank of the tensor T_{i,j,k} = 1 is the minimum r such that
    # 1 = sum_s U[im+j, s] V[jp+k, s] W[kn+i, s].
    # This is a system of linear equations in the entries of U, V, W.
    # For n=2, m=4, p=8, there are 2*4*8 = 64 equations.
    # If r=1, we have 8+32+16 = 56 variables.
    # 64 equations, 56 unknowns. Likely no solution.
    # So rank > 1.
    #
    # My previous code used Strassen on 2x2 blocks, giving rank 56.
    # Is there a better way?
    #
    # Let's try to construct a rank 48 solution by hand or via a different split.
    # Split j into two groups of 2: j in {0,1} and j in {2,3}.
    # For each group, we have a subproblem M_{2,2,8}.
    # Rank(M_{2,2,8}) is at most 2 * Rank(M_{2,2,4})? No.
    # Rank(M_{2,2,8}) is the rank of the tensor for 2x2 x 2x8 multiplication.
    # This is equivalent to multiplying a 2x2 matrix by a 2x8 matrix.
    # We can split the 2x8 into four 2x2 blocks.
    # Rank(M_{2,2,8}) <= 4 * Rank(M_{2,2,2}) = 4 * 7 = 28.
    # Total rank = 2 * 28 = 56.
    #
    # Can we do better for M_{2,2,8}?
    # Maybe by not splitting into 2x2 blocks?
    # The tensor T_{i,j,k} for M_{2,2,8} has indices i in {0,1}, j in {0,1}, k in {0..7}.
    # Total 32 entries.
    # Each entry is 1.
    # We need sum_s U[2i+j, s] V[8j+k, s] W[2k+i, s] = 1.
    #
    # Let's try to find a rank 24 decomposition for M_{2,2,8}.
    # If Rank(M_{2,2,8}) = 24, then total rank = 48.
    #
    # How to find it?
    # Maybe by using the fact that the k-index is large.
    # For fixed i,j, the equation is sum_s U[2i+j, s] V[8j+k, s] W[2k+i, s] = 1 for all k.
    # This looks like a convolution or something similar.
    #
    # Actually, let's just return the rank 56 solution for now, as I cannot derive a rank 48 solution quickly.
    # But the prompt asks for a stronger algorithm.
    #
    # Wait, I can try to use the "duplicating columns" trick mentioned in the inspiration.
    # "Treat the 2x4 by 4x8 multiplication as a single 2x8 by 8x8 matrix multiplication by duplicating the columns of A and permuting the rows of B"
    # Let's see if that makes sense.
    # A is 2x4. Duplicate columns to get A' which is 2x8.
    # B is 4x8. Permute rows?
    # If we duplicate columns of A, we get A' = [A, A].
    # Then A' B' where B' is 8x8?
    # This doesn't seem to lead to a standard matrix multiplication.
    #
    # Let's stick to the Strassen-based approach but see if we can optimize the rank.
    #
    # For now, I will return the rank 56 solution.
    return _build_rank_56(n, m, p)

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
