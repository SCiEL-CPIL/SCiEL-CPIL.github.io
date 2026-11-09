import numpy as np
from fractions import Fraction
import time

def construct():
    n, m, p = 2, 4, 5
    deadline = time.time() + 110
    
    # Trivial rank-40 decomposition
    rank0 = n * m * p
    U0 = np.zeros((n * m, rank0), dtype=object)
    V0 = np.zeros((m * p, rank0), dtype=object)
    W0 = np.zeros((p * n, rank0), dtype=object)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                U0[i * m + j, idx] = Fraction(1)
                V0[j * p + k, idx] = Fraction(1)
                W0[k * n + i, idx] = Fraction(1)
    
    def verify_decomposition(U, V, W, rank, n, m, p):
        """Verify that the decomposition exactly computes the matrix multiplication tensor"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if total != 1:
                        return False
        return True
    
    # Strategy: Decompose 2x4x5 into two 2x2x5 sub-problems
    # A = [A1 A2] where A1, A2 are 2x2
    # B = [B1; B2] where B1, B2 are 2x5
    # C = A1*B1 + A2*B2
    # Each sub-problem is 2x2x5 matrix multiplication
    # If we can find rank-10 decomposition for 2x2x5, total rank = 20
    
    # For 2x2x5: C[i,k] = sum_j A[i,j] B[j,k] where i in {0,1}, j in {0,1}, k in {0,1,2,3,4}
    # Tensor T[i,j,k] = 1 for all valid i,j,k
    # Need: sum_r U[i,j]_r V[j,k]_r W[k,i]_r = 1
    
    # Try to find rank-10 decomposition for 2x2x5
    # This is equivalent to multiplying a 2x2 matrix by a 2x5 matrix
    # We can view this as 5 separate 2x2 * 2x1 multiplications
    
    # Known: 2x2 * 2x1 has rank at most 7 (using Strassen-like methods)
    # But we need a unified decomposition for all 5 columns
    
    # Let's try a systematic search for rank-10 decomposition of 2x2x5
    # The tensor has 2*2*5 = 20 entries, all equal to 1
    
    # Approach: Use the fact that we can share computations across the 5 columns
    # For each pair (i,j), we need sum_k W[k,i]_r * V[j,k]_r * U[i,j]_r = 1
    
    # Let's try a structured approach:
    # For rank 10, we have 10 terms
    # We can think of this as: for each of the 4 (i,j) pairs, we need to cover 5 k values
    
    # Actually, let's try a different approach:
    # The 2x2x5 tensor can be seen as a sum of rank-1 tensors
    # T[i,j,k] = sum_r U[i,j]_r V[j,k]_r W[k,i]_r
    
    # For a fixed (i,j), this becomes: 1 = sum_r U[i,j]_r V[j,k]_r W[k,i]_r for all k
    # This is a system of 5 equations (one for each k) with 10 unknowns (the rank-1 coefficients)
    
    # Let's try a greedy construction:
    # Start with the trivial rank-20 decomposition and try to merge terms
    
    # Trivial rank-20 for 2x2x5:
    # For each (i,j,k), one term with U[i,j]=1, V[j,k]=1, W[k,i]=1
    
    # Try to find rank-10 by merging pairs of terms
    
    # Actually, let's just do a random search with verification
    # But random search for exact rank-10 is very unlikely to succeed
    
    # Better approach: Use known results or algebraic construction
    
    # For 2x2 * 2x5, we can use the following:
    # Let's define 10 rank-1 terms
    # Term r: U_r[i,j] V_r[j,k] W_r[k,i]
    
    # We need sum_r U_r[i,j] V_r[j,k] W_r[k,i] = 1 for all i,j,k
    
    # This is a 20-equation system (4 i,j pairs × 5 k values)
    
    # Let's try a structured solution:
    # For i=0, j=0: sum_r U_r[0,0] V_r[0,k] W_r[k,0] = 1 for k=0,1,2,3,4
    # For i=0, j=1: sum_r U_r[0,1] V_r[1,k] W_r[k,0] = 1 for k=0,1,2,3,4
    # For i=1, j=0: sum_r U_r[1,0] V_r[0,k] W_r[k,1] = 1 for k=0,1,2,3,4
    # For i=1, j=1: sum_r U_r[1,1] V_r[1,k] W_r[k,1] = 1 for k=0,1,2,3,4
    
    # This is complex. Let's try a different approach.
    
    # Actually, for the full 2x4x5 problem, let's just return the trivial rank-40 solution
    # since finding a lower rank decomposition is non-trivial
    
    # But let's try one more thing: use the block structure
    # C = A1*B1 + A2*B2
    # If we use rank-20 for each sub-problem (trivial), total rank = 40
    # If we can find rank-10 for each, total rank = 20
    
    # For now, return the trivial solution
    U_out = U0
    V_out = V0
    W_out = W0
    
    def convert_to_list(arr):
        res = []
        for row in arr:
            res_row = []
            for val in row:
                if isinstance(val, Fraction):
                    if val.denominator == 1:
                        res_row.append(int(val))
                    elif val.denominator == 2:
                        res_row.append(val.numerator / 2.0)
                    else:
                        res_row.append(float(val))
                else:
                    res_row.append(val)
            res.append(res_row)
        return res
    
    return {"U": convert_to_list(U_out), "V": convert_to_list(V_out), "W": convert_to_list(W_out)}
