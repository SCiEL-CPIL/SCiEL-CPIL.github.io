"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x8 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns: {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
             with integer/half-integer entries that exactly decompose the matrix multiplication tensor.
             Score is the rank (lower is better).
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 32
    pn = p * n  # 32
    
    # The matrix multiplication tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    # This is equivalent to: for each (i,j,k), the vector u_{ij} dot v_{jk} dot w_{ki} = 1
    
    # Schoolbook has rank n*m*p = 128. We want to do better.
    
    # Strategy: Use structured low-rank decompositions.
    # One approach: exploit the structure that T[i,j,k] = 1 (all ones).
    # 
    # Consider: if we can write the all-ones tensor as a sum of rank-1 tensors
    # where each rank-1 tensor has a simple structure.
    #
    # Key insight: T[i,j,k] = 1 = 1*1*1. 
    # If we use factors that are outer products of simpler structures, we can reduce rank.
    #
    # Approach 1: Block decomposition
    # Partition the indices into blocks and use block-level rank-1 tensors.
    #
    # Approach 2: Use the fact that for matrix multiplication, there are known
    # sub-multiplicative bounds. For 4x4 * 4x8, the known best upper bound is around 55-60.
    #
    # Let's try a structured approach:
    # The tensor T[i,j,k] = 1 can be decomposed as:
    # sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
    #
    # If we set u_r[i*m+j] = a_r[i] * b_r[j], v_r[j*p+k] = c_r[j] * d_r[k], w_r[k*n+i] = e_r[k] * f_r[i]
    # Then the product is a_r[i]*b_r[j]*c_r[j]*d_r[k]*e_r[k]*f_r[i]
    # = (a_r[i]*f_r[i]) * (b_r[j]*c_r[j]) * (d_r[k]*e_r[k])
    #
    # For this to equal 1 for all (i,j,k), we need:
    # sum_r (a_r[i]*f_r[i]) * (b_r[j]*c_r[j]) * (d_r[k]*e_r[k]) = 1
    #
    # This is equivalent to: let A_r[i] = a_r[i]*f_r[i], B_r[j] = b_r[j]*c_r[j], C_r[k] = d_r[k]*e_r[k]
    # Then sum_r A_r[i] * B_r[j] * C_r[k] = 1
    #
    # This is a rank-r decomposition of the all-ones tensor of size n x m x p.
    # The minimum rank for an all-ones tensor of size n x m x p is max(n,m,p) if we allow
    # arbitrary real entries, but we need integer/half-integer entries.
    #
    # Actually, the all-ones tensor 1_{n x m x p} has tensor rank max(n, m, p) over the reals.
    # Proof: The tensor can be written as sum_{s=1}^{max(n,m,p)} u_s v_s w_s where
    # for s <= n: u_s = e_s (standard basis), v_s = 1_m, w_s = 1_p
    # This gives sum_s e_s[i] * 1_m[j] * 1_p[k] = 1 if i is in [0,n)
    # Wait, that's not quite right. Let me reconsider.
    #
    # The all-ones tensor of size n x m x p: T[i,j,k] = 1
    # Rank-1 decomposition: T = sum_s u_s ⊗ v_s ⊗ w_s
    # T[i,j,k] = sum_s u_s[i] v_s[j] w_s[k]
    #
    # If we take s from 1 to n:
    # u_s[i] = 1 if i == s-1 else 0 (i.e., u_s = e_{s-1})
    # v_s[j] = 1 for all j
    # w_s[k] = 1 for all k
    # Then sum_s u_s[i] v_s[j] w_s[k] = 1 * 1 * 1 = 1 for all (i,j,k)
    # So the rank is at most n.
    #
    # Similarly, rank is at most m or at most p.
    # So the tensor rank of the all-ones tensor is min(n, m, p).
    #
    # For n=4, m=4, p=8: rank = min(4,4,8) = 4.
    #
    # BUT WAIT: This is for the all-ones tensor. The matrix multiplication tensor is NOT
    # the all-ones tensor in general. Let me re-read the problem.
    #
    # "Find a rank-minimal exact tensor decomposition for multiplying a 4 by 4 matrix by a 4 by 8 matrix."
    #
    # The matrix multiplication tensor for (n,m,p) has entries T[i,j,k] where the multiplication
    # C = AB has C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor is T[i,j,k] which represents the coefficient of A[i,j]*B[j,k] in C[i,k].
    # So T[i,j,k] = 1 for all valid (i,j,k) because each product A[i,j]*B[j,k] contributes
    # exactly once to C[i,k].
    #
    # So yes, the matrix multiplication tensor IS the all-ones tensor of size n x m x p!
    #
    # Therefore, the tensor rank is min(n, m, p) = min(4, 4, 8) = 4.
    #
    # Let's construct a rank-4 decomposition:
    # For s in range(n) = range(4):
    #   u_s[i*m+j] = 1 if i == s else 0
    #   v_s[j*p+k] = 1 for all (j,k)
    #   w_s[k*n+i] = 1 for all (k,i)
    #
    # Then sum_s u_s[i*m+j] * v_s[j*p+k] * w_s[k*n+i]
    # = sum_s [1 if i==s else 0] * 1 * 1
    # = 1 (since exactly one s equals i)
    #
    # This works! And all entries are 0 or 1, which are integers.
    
    rank = min(n, m, p)
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    for s in range(rank):
        # u_s[i*m+j] = 1 if i == s else 0
        for i in range(n):
            for j in range(m):
                if i == s:
                    U[i * m + j, s] = 1
        # v_s[j*p+k] = 1 for all (j,k)
        for j in range(m):
            for k in range(p):
                V[j * p + k, s] = 1
        # w_s[k*n+i] = 1 for all (k,i)
        for k in range(p):
            for i in range(n):
                W[k * n + i, s] = 1
    
    # Verify the decomposition
    # Compute the reconstructed tensor and check it equals the all-ones tensor
    # T[i,j,k] = sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
    
    # For verification, check a few entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                if val != 1:
                    # Fallback to schoolbook if verification fails
                    rank_fb = n * m * p
                    U_fb = np.zeros((nm, rank_fb))
                    V_fb = np.zeros((mp, rank_fb))
                    W_fb = np.zeros((pn, rank_fb))
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_fb[ii * m + jj, col] = 1
                                V_fb[jj * p + kk, col] = 1
                                W_fb[kk * n + ii, col] = 1
                                col += 1
                    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
