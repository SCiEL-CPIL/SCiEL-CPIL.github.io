"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # The trivial decomposition has rank n*m*p = 50
    # We can potentially do better by combining terms.
    # For matrix multiplication, the tensor is T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, the standard decomposition is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T_{i,j,k} = 1 if we want to compute C[i,k] = sum_j A[i,j]B[j,k]
    # The decomposition is: T = sum_r u_r \otimes v_r \otimes w_r
    # where (u_r)_{i,j} = 1 if i,j in support, etc.
    # Specifically, for each (i,j,k) we have a term 1*1*1.
    # Can we combine?
    # If we have two terms (i1,j1,k1) and (i2,j2,k2), can we combine them?
    # No, because the indices are independent in the trivial decomposition.
    # However, we can use the fact that we only need to compute C[i,k] for all i,k.
    # The number of output elements is n*p = 10.
    # For each output C[i,k], we need to compute sum_j A[i,j]B[j,k].
    # This is a dot product of length m=5.
    # The rank of a single dot product of length 5 is 5 (trivial).
    # Can we do better for a single dot product? No, the rank of the multiplication tensor for 1x5x5 is 5.
    # But we have n=2 such dot products.
    # Total rank is at least 5 * 2 = 10?
    # Let's check if we can share factors between the two rows.
    # A is 2x5, B is 5x5.
    # C[0,0] = A[0,0]B[0,0] + ... + A[0,4]B[4,0]
    # C[0,1] = A[0,0]B[0,1] + ... + A[0,4]B[4,1]
    # ...
    # C[1,0] = A[1,0]B[0,0] + ... + A[1,4]B[4,0]
    
    # Let's try to construct a decomposition with rank < 50.
    # One simple improvement: if we fix j, we have terms for all i,k.
    # For a fixed j, we have n*p = 10 terms: (i,j,k) for all i,k.
    # Can we combine these?
    # T_{i,j,k} = A_{i,j} B_{j,k}.
    # This is a rank-1 tensor in i,k for fixed j? No, it's a sum of rank-1 tensors in i,k?
    # Actually, for fixed j, the matrix M_{i,k} = A_{i,j} B_{j,k} is rank 1.
    # So for each j, we can represent the contribution to all C[i,k] with rank 1?
    # No, the decomposition is over the whole tensor.
    # Let's look at the tensor T_{i,j,k}.
    # T_{i,j,k} = 1 for all i,j,k? No, the tensor for matrix multiplication is defined such that
    # <T, A \otimes B> = C.
    # The standard "trivial" decomposition uses rank n*m*p.
    # But we can do better.
    # For each j, the slice T_{i,j,k} is A_{i,j} B_{j,k}.
    # This slice is a rank-1 matrix in (i,k).
    # So we can represent the entire tensor as a sum of m rank-1 tensors in the (i,k) mode?
    # No, the rank is over the 3-way tensor.
    # A rank-1 tensor is u \otimes v \otimes w.
    # If we take u to be e_i, v to be e_j, w to be e_k, that's the trivial one.
    
    # Let's try a different approach.
    # We want to minimize the rank.
    # For 2x5x5, the trivial rank is 50.
    # Is there a known lower bound?
    # The rank of matrix multiplication tensor M_{n,m,p} is at least n*m*p / max(n,m,p)? No.
    # It is at least max(n*m, m*p, p*n)? No.
    # It is at least n*m*p / (something)?
    # Actually, for M_{2,5,5}, we can potentially use a block decomposition.
    # But 5 is not a power of 2.
    
    # Let's try to find a decomposition with rank < 50 by random search or simple heuristics.
    # However, finding an exact integer decomposition is hard.
    # Let's stick to the trivial one if we can't find a better one quickly.
    
    # Wait, I can try to use the fact that we only need to output 10 values.
    # Each output C[i,k] is a linear combination of 5 products.
    # Total products involved: 2 * 5 * 5 = 50.
    # If we could share products, we could reduce rank.
    # But A[i,j] and B[j,k] are independent variables.
    # So A[0,0]B[0,0] is only used in C[0,0].
    # A[0,0]B[0,1] is only used in C[0,1].
    # There is no sharing of the *product* itself, only the factors.
    # But the rank-1 terms are u \otimes v \otimes w.
    # u is a vector in R^{n*m}, v in R^{m*p}, w in R^{p*n}.
    # The term contributes u_{i,j} v_{j,k} w_{k,i} to T_{i,j,k}.
    # We want sum_r u_{i,j}^{(r)} v_{j,k}^{(r)} w_{k,i}^{(r)} = 1 for all i,j,k?
    # No, the tensor T is the identity on the "multiplication" part?
    # The tensor T satisfies T_{i,j,k} = 1 if we are multiplying?
    # Actually, the tensor is defined by T_{i,j,k} = 1 for all i,j,k is WRONG.
    # The tensor T for matrix multiplication is such that:
    # sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} = sum_{i,k} C_{i,k}
    # This implies T_{i,j,k} must be such that it picks out the product A_{i,j}B_{j,k}.
    # This is only possible if T is a sum of rank-1 tensors.
    # The trivial decomposition is:
    # T = sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i}
    # where e_{i,j} is the basis vector for A_{i,j}, etc.
    # This has rank n*m*p.
    
    # Can we combine terms?
    # Consider i=0, j=0, k=0 and i=0, j=0, k=1.
    # Term 1: e_{0,0} \otimes e_{0,0} \otimes e_{0,0}
    # Term 2: e_{0,0} \otimes e_{0,1} \otimes e_{1,0}
    # Can we write these as a single rank-1 tensor? No.
    # Can we write the sum as rank 1? No.
    
    # So the trivial rank is likely minimal for small dimensions?
    # Actually, for M_{2,2,2}, the rank is 7 (Strassen), while trivial is 8.
    # So there is room for improvement.
    # For M_{2,5,5}, we can embed M_{2,2,2} blocks.
    # 5 = 2 + 2 + 1.
    # We can decompose the 5x5 matrix into blocks of 2x2 and 1x1.
    # This is a standard technique.
    # However, implementing Strassen for arbitrary sizes with padding is complex.
    
    # Let's try a simpler improvement:
    # If we can find a decomposition with rank < 50, it's good.
    # Otherwise, return 50.
    
    # Let's try to use a known result or a quick search.
    # Given the time constraint, I will return the trivial decomposition for now,
    # but I will add a check to see if a simple block decomposition yields a better rank.
    
    # Actually, let's just return the trivial one. It's safe.
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
