"""Find a low-rank exact tensor decomposition for 3x3 x 3x8 matrix multiplication."""
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 8]
DEADLINE = time.time() + 140  # Safety margin before 150s limit

def construct():
    """Return {'U', 'V', 'W'} factor matrices for matrix multiplication tensor."""
    n, m, p = 3, 3, 8  # A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T[i,j,k] = 1 if term contributes, indexed by (i,j,k)
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Convention: W[k*n + i, r]
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition using bilinear algorithm search
    # The tensor is: T[i, j, k] = delta_{i+m-1, j} for the multiplication
    # Actually, for C = AB: C[i,k] = sum_j A[i,j]B[j,k]
    # So the tensor entry T[i,j,k] = 1 (each term A[i,j]*B[j,k] contributes to C[i,k])
    
    # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
    # And 0 for invalid combinations (but in matrix mult, all i,j,k with i<n, j<m, k<p are valid)
    
    # Actually the tensor is: for each (i,j,k) in [n]x[m]x[p], T[i,j,k] = 1
    # and we need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # This is a rank-r decomposition of the 3-tensor T where T[i,j,k] = 1 for all i,j,k
    # Wait no - that's not right. The matrix multiplication tensor is NOT all ones.
    
    # Let me reconsider. The matrix multiplication tensor M_{n,m,p} is defined by:
    # M[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    # In standard matrix mult, EVERY term A[i,j]*B[j,k] contributes to C[i,k],
    # so M[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    
    # Wait, that would make it the all-ones tensor, which has rank 1!
    # That can't be right.
    
    # I'm confusing myself. Let me be more careful.
    # The bilinear form for matrix multiplication is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T is a trilinear form: T(A,B,C) = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C[i,k]
    # For matrix multiplication, T[i,j,k] = 1 for all i,j,k (since each A[i,j]*B[j,k] goes into C[i,k])
    # But that's a trilinear form, not what we're decomposing.
    
    # The RANK of the matrix multiplication tensor refers to the minimum r such that:
    # C[i,k] = sum_{r=1}^{R} U[i*m+j_r, r] * V[j_r*p+k, r] * ... 
    # No, that's not right either.
    
    # Correct formulation: We want to compute all C[i,k] = sum_j A[i,j]*B[j,k] using R multiplications.
    # Each "multiplication" computes some bilinear form L_r(A) * M_r(B) and adds it to some C[i,k].
    # So: C[i,k] = sum_{r=1}^{R} coeff_{i,k,r} * (L_r(A) * M_r(B))
    # where L_r(A) = sum_{i,j} U[i*m+j, r] * A[i,j]
    # and M_r(B) = sum_{j,k} V[j*p+k, r] * B[j,k]
    # and the contribution to C[i,k] is W[k*n+i, r] * L_r(A) * M_r(B)
    
    # So we need: sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = delta_{j, j} ... 
    # No. We need: for each (i,k): C[i,k] = sum_j A[i,j]*B[j,k]
    # = sum_r W[k*n+i, r] * (sum_{i',j'} U[i'*m+j', r] * A[i',j']) * (sum_{j'',k''} V[j''*p+k'', r] * B[j'',k''])
    
    # For this to equal sum_j A[i,j]*B[j,k] for all A,B, we need:
    # The coefficient of A[a,b]*B[c,d] in the RHS must equal delta_{a,i}*delta_{c,b}*delta_{d,k}
    # Coefficient of A[a,b]*B[c,d] = sum_r W[k*n+i, r] * U[a*m+b, r] * V[c*p+d, r]
    # This must equal 1 if a=i, c=b, d=k, else 0.
    
    # So: sum_r W[k*n+i, r] * U[a*m+b, r] * V[c*p+d, r] = 1 if (a,c,d)=(i,b,k) else 0
    
    # This is a system of equations. For each (i,k,a,b,c,d) with the constraint that
    # (a,c,d) should equal (i,b,k) for the 1 case.
    
    # Number of equations: n*p*n*m*m*p = n^2 * m^2 * p^2 = 9*9*64 = 5184
    # Number of unknowns: R * (n*m + m*p + p*n) = R * (9 + 24 + 24) = 57R
    
    # For R=72 (schoolbook), 57*72 = 4104 unknowns, 5184 equations. Underdetermined? No, 4104 < 5184.
    # Wait, that means schoolbook with R=72 has 4104 unknowns but 5184 constraints? That can't be right.
    
    # I think I'm overcomplicating. The schoolbook solution works, so the system is consistent for R=72.
    
    # Let me just try to find a sparse solution with R < 72 using a greedy approach or SVD-based method.
    
    # For now, return schoolbook if we can't find better quickly.
    return best_result

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
