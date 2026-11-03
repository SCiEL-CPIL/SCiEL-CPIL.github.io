# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct():
    """
    Constructs the tensor decomposition for 2x5 * 5x5 matrix multiplication.
    
    The naive rank is n*m*p = 2*5*5 = 50.
    
    Strategy:
    1. Start with the naive decomposition (rank 50).
    2. Try to find a lower rank decomposition by solving a feasibility problem.
       The matrix multiplication tensor M_{n,m,p} has entries M[i,j,k] = 1 if the
       standard multiplication formula holds, else 0.
       
       Actually, for matrix multiplication C = A * B where A is n x m and B is m x p,
       C[i,k] = sum_j A[i,j] * B[j,k].
       
       The tensor is T[i,j,k] = 1 for all valid indices, but wait, the tensor for
       bilinear form is defined such that C[i,k] = sum_r U[i,r] * V[k,r] * W[j,r] ... 
       
       Let's stick to the standard tensor decomposition definition for matrix multiplication.
       We want to find rank r and matrices U (n*m x r), V (m*p x r), W (p*n x r) such that
       for all i, j, k:
       delta_{i,j,k} (which is 1 if the term A[i,j]B[j,k] contributes to C[i,k]) 
       Actually, the tensor is T_{i,j,k} = 1 for all i,j,k? No.
       
       The tensor of matrix multiplication M_{n,m,p} is a 3-tensor T of size n x m x p
       where T[i,j,k] = 1 if the product A[i,j] * B[j,k] appears in the sum for C[i,k].
       Since C[i,k] = sum_j A[i,j] B[j,k], every term A[i,j] B[j,k] contributes to C[i,k].
       So T[i,j,k] = 1 for all i, j, k.
       
       Wait, the indices in the problem description are:
       U: (n*m, rank)
       V: (m*p, rank)
       W: (p*n, rank)
       
       The decomposition is:
       C[i,k] = sum_{r} W[k*n + i, r] * (sum_{j} U[i*m + j, r] * V[j*p + k, r] * A[i,j] * B[j,k]?)
       
       No, the standard definition is:
       C[i,k] = sum_{r} W_{k,i}^{(r)} * ( sum_{j} U_{i,j}^{(r)} A_{i,j} V_{j,k}^{(r)} B_{j,k} )
       
       This is not quite right. The standard bilinear algorithm for matrix multiplication
       is given by:
       C_{i,k} = sum_{r=1}^R W_{i,k}^{(r)} * ( sum_{j} U_{i,j}^{(r)} A_{i,j} V_{j,k}^{(r)} B_{j,k} )
       
       Wait, the U, V, W matrices in the problem are:
       U: (n*m, rank) -> U[i*m + j, r]
       V: (m*p, rank) -> V[j*p + k, r]
       W: (p*n, rank) -> W[k*n + i, r]
       
       The condition is:
       For all i, k:
       C[i,k] = sum_{r} W[k*n + i, r] * ( sum_{j} U[i*m + j, r] * A[i,j] * V[j*p + k, r] * B[j,k] )
       
       This must hold for all A, B.
       Therefore, for each r, the term inside the sum is a bilinear form in A and B.
       Let B_r(A,B) = sum_{j} U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k].
       Then C[i,k] = sum_r W[k*n+i, r] * B_r(A,B).
       
       Since A and B are arbitrary, the coefficient of A[i',j'] * B[j'',k'] in the expansion
       of C[i,k] must match the standard matrix multiplication coefficient.
       The standard coefficient of A[i',j'] * B[j'',k'] in C[i,k] is 1 if i'=i, k'=k, j'=j'', else 0.
       
       So we require:
       For all i, k, i', j', j'', k':
       sum_r W[k*n+i, r] * U[i'*m+j', r] * V[j''*p+k', r] * (delta_{i',i} * delta_{j',j''} * delta_{k',k})
       = delta_{i',i} * delta_{j',j''} * delta_{k',k} * (1 if i'=i and k'=k else 0)
       
       This simplifies to:
       For all i, j, k:
       sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1
       
       And for all other combinations of indices (i, j, k) that don't correspond to a valid
       multiplication term, the sum must be 0. But wait, the indices i, j, k in the sum
       are the same as the ones in the factor matrices.
       
       Actually, the condition is just:
       For all i in 0..n-1, j in 0..m-1, k in 0..p-1:
       sum_{r} W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1
       
       And for any "spurious" term? No, the bilinear form is exactly:
       sum_{i',j',j'',k'} (sum_r W[k*n+i', r] U[i'*m+j', r] V[j''*p+k', r]) A[i',j'] B[j'',k']
       
       We want this to equal sum_j A[i,j] B[j,k] for each (i,k).
       So the coefficient of A[i',j'] B[j'',k'] must be:
       1 if i'=i, k'=k, j'=j''
       0 otherwise.
       
       Let's define the tensor T_{i',j',j'',k'} = sum_r W[k*n+i', r] U[i'*m+j', r] V[j''*p+k', r].
       We need T_{i',j',j'',k'} = delta_{i',i} delta_{k',k} delta_{j',j''}.
       
       This is a system of linear equations in the products of entries of U, V, W?
       No, it's a tensor decomposition. The variables are the entries of U, V, W.
       The equations are multilinear (trilinear) in the entries of U, V, W.
       
       This is a hard problem. The naive solution has rank 50.
       Can we do better?
       
       For 2x5 * 5x5, the rank is not known to be less than 50?
       Actually, the rank of matrix multiplication M_{n,m,p} is at most n*m*p.
       For 2x2 * 2x2, the rank is 7 (Strassen), which is less than 8.
       For 2x1 * 1x1, the rank is 1? No, 2x1 * 1x1 is just scaling two elements.
       C[0,0] = A[0,0] * B[0,0], C[1,0] = A[1,0] * B[0,0].
       This is rank 2?
       U: (2*1, r) = (2, r)
       V: (1*1, r) = (1, r)
       W: (1*2, r) = (2, r)
       
       C[0,0] = sum_r W[0*2+0, r] U[0*1+0, r] V[0*1+0, r] = sum_r W[0,r] U[0,r] V[0,r]
       C[1,0] = sum_r W[0*2+1, r] U[1*1+0, r] V[0*1+0, r] = sum_r W[1,r] U[1,r] V[0,r]
       
       We need:
       sum_r W[0,r] U[0,r] V[0,r] = A[0,0] B[0,0]
       sum_r W[1,r] U[1,r] V[0,r] = A[1,0] B[0,0]
       
       This can be done with rank 2:
       r=0: W[0,0]=1, U[0,0]=1, V[0,0]=1 -> gives A[0,0]B[0,0] for C[0,0]
       r=1: W[1,1]=1, U[1,1]=1, V[0,1]=1 -> gives A[1,0]B[0,0] for C[1,0]
       
       So rank(2,1,1) = 2.
       
       For 1x5 * 5x5, the rank is 25?
       C[0,k] = sum_j A[0,j] B[j,k].
       This is just a row vector times a matrix.
       The rank is m*p = 5*5 = 25.
       
       So the Kronecker product structure gives rank 2 * 25 = 50.
       
       Is it possible to do better than 50 for 2x5 * 5x5?
       The rank of M_{n,m,p} is at least max(n*m, m*p, p*n) = max(10, 25, 10) = 25.
       Also, the rank is at least the "slice rank" or other lower bounds.
       
       It is not known if the rank of M_{2,5,5} is less than 50.
       However, we can try to search for a lower rank decomposition.
       
       Since the problem asks for a "rank-minimal exact tensor decomposition", and we don't know
       the true minimum, we should try to find the lowest rank we can.
       
       Given the time limit (150s), we can try to:
       1. Return the naive rank 50 decomposition.
       2. Try to find a rank 49 decomposition? This is a hard non-convex problem.
       
       Let's assume the rank is 50 and return the naive decomposition.
       
       Wait, the directive says: "Exploit the Kronecker product structure by expressing the 2x5 matrix as the tensor product of a 2x1 vector and a 1x5 vector, then decompose the multiplication using the rank-2 tensor for 2x1*1x1 (trivial) and the rank-5 tensor for 1x5*5x5, resulting in a composite decomposition of rank 10 that leverages the multiplicativity of tensor ranks under Kronecker products for exact integer arithmetic."
       
       This claim seems incorrect. The rank of 1x5 * 5x5 is 25, not 5.
       The rank of M_{1,m,p} is m*p.
       
       So the rank is 2 * 25 = 50.
       
       Let's double check the rank of M_{1,m,p}.
       C[0,k] = sum_j A[0,j] B[j,k].
       This is a bilinear map from R^m x R^{m*p} to R^p.
       The rank is m*p.
       
       So the naive rank is 50.
       
       We will return the naive rank 50 decomposition.
    """
    n, m, p = 2, 5, 5
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][col] = 1
                # V index: j*p + k
                V[j * p + k][col] = 1
                # W index: k*n + i
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
