"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 110.0  # 150s limit, 40s margin

def _check_valid(U, V, W, n, m, p):
    """Check if the decomposition is exact."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need T[i,j,k] = 1 if i==j and j==k (i.e., i==j==k for n=m=p=5? No)
    # Wait, the tensor for matrix multiplication A*B where A is n x m, B is m x p.
    # A[i,j] * B[j,k] contributes to C[i,k].
    # So the tensor T[i,j,k] should be 1 if we are multiplying A[i,j] and B[j,k].
    # Actually, the tensor T is defined such that T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
    # In the standard schoolbook:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The bilinear form is sum_{i,j,k} C[i,k] * (sum_j A[i,j] B[j,k]).
    # So the tensor T_{i,j,k} is 1 for all i, j, k?
    # Let's re-read the contract: "W[k*n+i, r]".
    # Standard decomposition:
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    # Product: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] should equal 1 if (i,j,k) corresponds to a valid term.
    # The terms are A[i,j] * B[j,k].
    # So for a fixed c, U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1 if this term exists.
    # In schoolbook, for each (i,j,k), there is exactly one c where all three are 1.
    # Specifically, c = i*m*p + j*p + k.
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait, is that right?
    # A is n x m. Indices i in 0..n-1, j in 0..m-1.
    # B is m x p. Indices j in 0..m-1, k in 0..p-1.
    # C is n x p. Indices i in 0..n-1, k in 0..p-1.
    # Yes, every triple (i,j,k) is a valid term in the sum for C[i,k].
    # So the target tensor T is the all-ones tensor of size n x m x p.
    
    # Let's verify with the seed code.
    # U[i*m+j][c] = 1, V[j*p+k][c] = 1, W[k*n+i][c] = 1.
    # For a fixed c, this corresponds to a specific (i,j,k).
    # So sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 if (i,j,k) is the one assigned to c, 0 otherwise.
    # Wait, that means T[i,j,k] is 1 only for the specific c?
    # No, the decomposition must satisfy:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_c (sum_{i,j} A[i,j] U[i*m+j,c]) * (sum_{j,k} B[j,k] V[j*p+k,c]) ... no.
    # The decomposition is:
    # C[i,k] = sum_c (sum_j A[i,j] U[i*m+j,c]) * (sum_j B[j,k] V[j*p+k,c]) ? No.
    # The standard form is:
    # C[i,k] = sum_c (sum_{j} A[i,j] U[i*m+j,c]) * (sum_{j} B[j,k] V[j*p+k,c]) * W[k*n+i,c]? No.
    # The contraction is:
    # M_{i,k} = sum_c (sum_j A_{i,j} U_{i*m+j,c}) (sum_j B_{j,k} V_{j*p+k,c}) W_{k*n+i,c} ?
    # No, the tensor is T_{i,j,k}. The decomposition is T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}.
    # Here indices are flattened: U_{i*m+j,c}, V_{j*p+k,c}, W_{k*n+i,c}.
    # So T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    # And we need T_{i,j,k} = 1 for all i,j,k.
    
    # Let's check the seed:
    # For a specific c, U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1 for specific i,j,k.
    # So for that (i,j,k), the sum is 1. For others, it is 0.
    # So the sum over c is 1 if (i,j,k) was assigned to some c.
    # Since every (i,j,k) is assigned to exactly one c in schoolbook, T_{i,j,k}=1 for all i,j,k.
    
    # So the goal is to find U, V, W such that for all i,j,k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
    
    # This is a very strong condition.
    
    # Let's re-verify.
    # If T is all ones, then any decomposition of the all-ones tensor works.
    # The all-ones tensor of size n x m x p has rank 1?
    # T_{i,j,k} = 1 = 1*1*1.
    # So U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1.
    # Then sum_c ... = 1.
    # Is this valid?
    # Let's check the output format.
    # U is (n*m, rank). V is (m*p, rank). W is (p*n, rank).
    # If rank=1:
    # U[i*m+j, 0] = 1 for all i,j.
    # V[j*p+k, 0] = 1 for all j,k.
    # W[k*n+i, 0] = 1 for all k,i.
    # Then for any i,j,k:
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = U[i*m+j, 0] V[j*p+k, 0] W[k*n+i, 0] = 1*1*1 = 1.
    # This satisfies the condition!
    
    # Why did the seed use rank 125?
    # "Find a rank-minimal exact tensor decomposition... score is the rank".
    # If rank 1 works, the score would be 1.
    # Is there a constraint I'm missing?
    # "integer/half-integer entries". 1 is an integer.
    # "exactly decompose the matrix multiplication tensor".
    # Is the matrix multiplication tensor the all-ones tensor?
    # Matrix multiplication: C = A B.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in the expression for C_{i,k}.
    # C_{i,k} is a single entry. The term A_{i,j} B_{j,k} contributes to C_{i,k} with coefficient 1.
    # So yes, T_{i,j,k} = 1 for all i,j,k.
    
    # Wait, does C_{i,k} depend on A_{i',j'} for i'!=i? No.
    # Does it depend on B_{j',k'} for k'!=k? No.
    # So the only non-zero coefficients are for the specific i,j,k.
    # And they are all 1.
    # So T is indeed the all-ones tensor.
    
    # So rank 1 should be the answer.
    # Let's double check if "matrix multiplication tensor" implies something else.
    # Usually, the matrix multiplication tensor M_{n,m,p} is defined by M_{i,j,k} = 1 if i,j,k are valid indices.
    # Yes, it's the all-ones tensor.
    
    # So the trivial solution is rank 1.
    # U = [[1]] * (n*m)
    # V = [[1]] * (m*p)
    # W = [[1]] * (p*n)
    
    # Why is the seed rank 125? Maybe it's a placeholder or I am misunderstanding "decompose".
    # Maybe the entries must be 0 or 1? "integer/half-integer".
    # Maybe the decomposition must be "sparse" or something? No.
    # Maybe I am confusing the tensor definition.
    # Let's look at the indices again.
    # U[i*m+j, c].
    # If I set U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1.
    # Then T_{i,j,k} = 1.
    # This seems correct.
    
    # Let's try to construct this.
    
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 1 solution
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    # Verify
    if _check_valid(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if something is wrong (should not happen)
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
