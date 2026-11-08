"""Matrix multiplication 2x3x9 rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 9
    # We use the schoolbook decomposition which has rank n*m*p = 2*3*9 = 54.
    # This is the baseline and is always valid.
    # 
    # For a structural change, we try to find a lower rank decomposition.
    # The known rank for M_{2,3,9} is not trivially less than 54 in small cases.
    # However, we can try a direct construction.
    #
    # Let's try the Kronecker product structure:
    # M_{2,3,9} = M_{2,3,1} x M_{1,1,3} x M_{1,1,3}  (this is the Kronecker product of tensors)
    # Rank of M_{2,3,1}: The tensor T[i,j,k] for i in {0,1}, j in {0,1,2}, k=0.
    # T[i,j,0] = 1 for all i,j. This is an outer product of vector of 2 ones and vector of 3 ones.
    # So rank is 1? No. T_{ijk} = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    # For M_{2,3,1}: U is 6xR, V is 3xR, W is 2xR.
    # We need sum_r U[i*3+j, r] V[j*1+0, r] W[0*2+i, r] = 1.
    # = sum_r U[i*3+j, r] V[j, r] W[i, r] = 1.
    # This can be done with R=1: U[i*3+j, 0] = 1, V[j, 0] = 1, W[i, 0] = 1.
    # Then sum = 1*1*1 = 1. So rank(M_{2,3,1}) = 1.
    #
    # M_{1,1,3}: U is 1xR, V is 3xR, W is 3xR.
    # T[i=0,j=0,k] = 1 for k in {0,1,2}.
    # sum_r U[0,r] V[0*3+k, r] W[k*1+0, r] = U[0,r] V[k,r] W[k,r] = 1.
    # This requires rank 3 (since V and W are 3xR and we need 3 orthogonal "channels").
    # Actually, U[0,r] can be 1 for all r. Then we need sum_r V[k,r] W[k,r] = 1 for each k.
    # If we set V[k,k]=1, W[k,k]=1 for k=0,1,2, then sum_r V[k,r]W[k,r] = V[k,k]W[k,k] = 1.
    # So rank(M_{1,1,3}) = 3.
    #
    # By the Kronecker product property of tensor rank (rank of A x B <= rank(A)*rank(B)),
    # rank(M_{2,3,9}) <= rank(M_{2,3,1}) * rank(M_{1,1,3}) * rank(M_{1,1,3}) = 1 * 3 * 3 = 9.
    #
    # Let's construct this rank-9 decomposition explicitly.
    #
    # M_{2,3,9} = M_{2,3,1} x M_{1,1,3} x M_{1,1,3}
    # Index structure:
    # A is 2x3, B is 3x9. We view B as 3x(3x3) = 3x3x3.
    # So B[j, k1*3 + k2] where j in {0,1,2}, k1 in {0,1,2}, k2 in {0,1,2}.
    # C is 2x9. C[i, k1*3 + k2].
    # C[i, k1*3+k2] = sum_j A[i,j] B[j, k1*3+k2].
    #
    # The tensor T[i, j, k1*3+k2] = 1 if we multiply A[i,j] by B[j, k1*3+k2].
    # Actually, the multiplication tensor M_{n,m,p} has entries M_{i,j,k} = delta_{j, j'} for A[i,j'] * B[j,k].
    # So M_{i,j,k} = 1 if the j index matches, 0 otherwise.
    #
    # For M_{2,3,9} with B reshaped as B[j, a*3+b]:
    # M_{i, j, a*3+b} = 1 for all i in {0,1}, j in {0,1,2}, a in {0,1,2}, b in {0,1,2}.
    # Wait, that's not right. The tensor entry M_{i,j,k} is 1 if and only if the multiplication
    # C_{i,k} = sum_j A_{i,j} B_{j,k} has the term A_{i,j} B_{j,k} contributing.
    # So M_{i,j,k} = 1 for all valid i, j, k. No, that's the all-ones tensor.
    #
    # Let me re-read the convention.
    # The matrix multiplication tensor T of size n x m x p is defined by
    # T_{i,j,k} = 1 if the (i,k)-th entry of the product depends on A_{i,j} * B_{j,k}.
    # Yes, so T_{i,j,k} = 1 for all i in [n], j in [m], k in [p].
    # So the tensor is the all-ones tensor of size 2x3x9.
    #
    # Wait, that can't be right. If T_{i,j,k} = 1 for all i,j,k, then
    # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
    # But that's not the standard matrix multiplication. 
    #
    # Let me re-read the task: "Matrix multiplication 2,3,9"
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 9 matrix."
    #
    # The standard matrix multiplication tensor has T_{i,j,k} = 1 if the product C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So T_{i,j,k} is 1 for all i, j, k in the valid ranges. 
    #
    # Actually, I think I was confused. Let me check the schoolbook decomposition.
    # In the schoolbook, for each (i,j,k), we have a rank-1 term.
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1.
    # Then sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 if (i,j,k) matches the column, 0 otherwise.
    # So the reconstructed tensor has 1 at position (i, j, k) and 0 elsewhere.
    # This means the tensor T_{i,j,k} = delta_{i,i'} delta_{j,j'} delta_{k,k'} for the specific (i',j',k') of the column.
    #
    # So the full tensor is the sum of all these delta tensors, which is the all-ones tensor.
    #
    # So we need to decompose the all-ones tensor of size 2x3x9 into rank-1 tensors.
    # The all-ones tensor of size n x m x p has rank = 1 (it's the outer product of three vectors of ones).
    # Wait, that's not right either. The all-ones tensor T_{i,j,k} = 1 for all i,j,k.
    # This is the outer product of vec(1_{n*m}) and ... no.
    #
    # T_{i,j,k} = 1 = (1 for i) * (1 for j) * (1 for k).
    # So T = u v^T w^T where u is a vector of ones of length n*m, v is a vector of ones of length m*p, 
    # w is a vector of ones of length p*n? No.
    #
    # The tensor is 3-dimensional. T_{i,j,k} where i in [n], j in [m], k in [p].
    # But in the problem, the indices are:
    # U has rows indexed by (i,j) -> i*m+j, so n*m rows.
    # V has rows indexed by (j,k) -> j*p+k, so m*p rows.
    # W has rows indexed by (k,i) -> k*n+i, so p*n rows.
    #
    # So the tensor is not T_{i,j,k} but rather a tensor with indices (i,j), (j,k), (k,i).
    # This is the standard matrix multiplication tensor.
    #
    # The schoolbook decomposition gives rank = n*m*p.
    # For n=2, m=3, p=9, rank = 54.
    #
    # Can we do better? The rank of the matrix multiplication tensor M_{n,m,p} is known to be 
    # less than n*m*p for some values. For M_{2,3,9}, the known rank is not obviously less than 54.
    #
    # Let's just return the schoolbook decomposition, which is guaranteed to be correct.
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
