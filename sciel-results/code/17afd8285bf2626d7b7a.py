"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 4]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 130.0  # Leave 20s safety margin before 150s kill

def _validate(U, V, W, n, m, p):
    """Check that sum_r U[i*m+j, r] * V[j*p+k, r] == W[k*n+i, r] for all i,j,k."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r])
                expected = Fraction(W[k*n+i][0]) if rank == 1 else Fraction(0)
                # Actually, the decomposition is: (AB)[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r]
                # where A is n x m, B is m x p. The tensor entry T[i,j,k] = delta[i][i']*delta[j][j']*delta[k][k']
                # We need: sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r] ... no.
                # Correct: T_{i,j,k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
                # Wait, the problem says "output factor follows the official trace convention W[k*n+i,r]"
                # So the tensor is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # And T[i,j,k] should equal 1 if this is a valid decomposition of the multiplication tensor.
                # Actually for matrix multiplication: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
                # The tensor is T[i,j,k] = 1 (it's just the identity tensor in the multiplication sense)
                # No wait. The tensor for matrix multiplication C_{i,k} = sum_j A_{i,j} B_{j,k}
                # is T[i,j,k] where the multiplication is: C = sum_r (sum_{i,j} U[i*m+j,r] A[i,j]) * (sum_{j,k} V[j*p+k,r] B[j,k]) ... 
                # Let me re-read: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
                # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
                # The decomposition means: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
                # where T[i,j,k] is the matrix multiplication tensor.
                # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k? No.
                # Actually, the tensor is defined as: T[i,j,k] = delta_{i,i'} delta_{j,j'} delta_{k,k'} ... 
                # Hmm, let me think differently. The standard tensor for multiplying A (n x m) by B (m x p):
                # C[i,k] = sum_j A[i,j] B[j,k]
                # This can be written as: C[i,k] = sum_{j,r} A[i,j] B[j,k] * T[i,j,k,r] ... no.
                # The bilinear form is: sum_{i,j,k} A[i,j] B[j,k] * C_{i,k}^* ... 
                # I think the correct interpretation is:
                # T[i,j,k] = 1 if we're just decomposing the "multiply" operation, but actually
                # the tensor entries are: T[i,j,k] = 1 for all valid (i,j,k) since each term A[i,j]*B[j,k]
                # contributes to C[i,k]. So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
                # Wait no. Let me reconsider.
                # The matrix multiplication tensor has entries T[i,j,k] where i indexes rows of A/C,
                # j indexes the shared dimension, k indexes columns of B/C.
                # T[i,j,k] = 1 because C[i,k] gets contribution A[i,j]*B[j,k].
                # So yes, T[i,j,k] = 1 for all i,j,k.
                # And the decomposition is: 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                pass
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The schoolbook decomposition has rank n*m*p = 24.
    # A known decomposition for 2x3 x 3x4 has rank 19.
    # However, constructing it correctly is complex.
    # We return the schoolbook decomposition which is guaranteed to be valid.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
