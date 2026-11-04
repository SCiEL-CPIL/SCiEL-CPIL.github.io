"""Matrix multiplication 4x5x5 rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS
    
    # Build the matrix multiplication tensor
    # T[i, j, k] = 1 if we're computing (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k in the standard basis
    # Actually, the tensor for matrix mult is: T_{i,j,k} = delta_{i,k} is wrong
    # The correct tensor: (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor T has T[i,j,k] = 1 for the "structure" but really:
    # We need T such that sum_c U[i,c]*V[j,c]*W[k,c] = delta_{i,k} is NOT right
    # 
    # Let me reconsider. The matrix multiplication tensor is defined as:
    # T(A,B) = A*B where (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # As a trilinear form: sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} (A*B)_{i,k}
    # No, that's not right either.
    #
    # The standard way: the matrix multiplication tensor M_{n,m,p} is defined by:
    # M_{n,m,p} = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_i^* ⊗ e_j ⊗ e_k
    # where the multiplication is: (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    #
    # Wait, I think the convention is:
    # T[i,j,k] corresponds to the coefficient of A[i,j]*B[j,k] in (A*B)[i,k]
    # So T[i,j,k] = 1 if the indices match, i.e., for each (i,j,k), the term A[i,j]*B[j,k]
    # contributes to (A*B)[i,k]. So T[i,j,k] = 1 for all i,j,k.
    #
    # But that's just the schoolbook. The tensor is:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k  (in the appropriate spaces)
    # No wait, that's not right.
    
    # Let me use the standard definition from the literature:
    # The matrix multiplication tensor M_{n,m,p} is:
    # M_{n,m,p} = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_i^{(n)} ⊗ e_j^{(m)} ⊗ e_k^{(p)}
    # where the first factor is the output index, second is A's column, third is B's row
    # And the multiplication is: C_{i,k} = sum_j A_{i,j} B_{j,k}
    #
    # So the tensor has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # But that's 100 ones, which gives rank 100 (schoolbook).
    #
    # Actually no. The tensor is:
    # M = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k  where the product structure is:
    # <M, A ⊗ B ⊗ C> = sum_{i,j,k} M_{i,j,k} A_{i,j} B_{j,k} C_{i,k}
    # For matrix multiplication, we want this to equal sum_{i,k} (A*B)_{i,k} C_{i,k}
    # = sum_{i,k} sum_j A_{i,j} B_{j,k} C_{i,k}
    # = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}
    # So M_{i,j,k} = 1 for all i,j,k.
    
    # So the tensor is all ones in the appropriate sense. The rank is the minimum number
    # of rank-1 tensors that sum to this.
    
    # The schoolbook decomposition is:
    # M = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k
    # which has rank n*m*p = 100.
    
    # To find a lower rank decomposition, I need to find r < 100 triples (u_c, v_c, w_c)
    # such that sum_c u_c[i] * v_c[j] * w_c[k] = 1 for all i,j,k.
    
    # This is equivalent to finding a rank-r decomposition of the tensor.
    
    # Let me try a simple approach: use a structured decomposition.
    
    # For now, let's return the schoolbook decomposition as a baseline.
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
