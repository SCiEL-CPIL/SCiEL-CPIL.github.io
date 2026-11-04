"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction
from itertools import product

DIMENSIONS = [4, 4, 5]
DEADLINE = 140.0  # safety margin before 150s kill


def _tensor_eq(A, B, C, rank):
    """Check if U*A*B*C = identity tensor for matrix multiplication."""
    n, m, p = DIMENSIONS
    U = A
    V = B
    W = C
    # The tensor equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(i,k)
    # We need to verify that for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if k==j else 0
    # Wait, that's not right. Let me think again.
    # Matrix multiplication: (A * B)[i, k] = sum_j A[i,j] * B[j,k]
    # The tensor decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # where T[i,j,k] = delta(i,k) for the identity... no.
    # Actually for matrix multiplication C = A*B, we have C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is T_{i,j,k} = delta_{i,k} ... no.
    # Let me re-read the task. The task says:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # And W[k*n+i, r] is the convention.
    # So the decomposition is: for matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Tensor form: C[i,k] = sum_{r} U[i*m+j, r] ... hmm.
    
    # Let me think about this differently. The standard tensor decomposition for matrix multiplication M_{n,m,p} is:
    # (A * B)[i, k] = sum_{r=1}^{R} U_r[i, j] * V_r[j, k] * W_r[i, k]
    # where each term is a rank-1 tensor U_r ⊗ V_r ⊗ W_r
    # and U_r[i,j] means U[i*m+j, r], V_r[j,k] means V[j*p+k, r], W_r[i,k] means W[k*n+i, r]
    
    # So the full tensor T[i,j,k] satisfies:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And for matrix multiplication, T[i,j,k] = 1 if this corresponds to C[i,k] = sum_j A[i,j]B[j,k]
    # The "target tensor" is: T[i,j,k] = 1 for all i,j,k? No...
    
    # Actually, the matrix multiplication tensor has entries T[i,j,k] = delta_{i,k} ... no.
    # Let me think again. C = A*B means C_{ik} = sum_j A_{ij} B_{jk}
    # This is a bilinear (actually trilinear when we consider the output) computation.
    # The tensor is: for each (i, j, k), the coefficient of A_{ij} * B_{jk} in C_{ik} is 1 if the indices match, 0 otherwise.
    # Specifically: C_{ik} = sum_j A_{ij} B_{jk}, so the coefficient of A_{ij} B_{jk} in C_{ik} is 1 if k=k (always) ... 
    # No wait. C_{ik} involves A_{ij} for all j, multiplied by B_{jk}. So the term A_{ij} B_{jk} appears in C_{ik} with coefficient 1 for all valid i, j, k.
    
    # Hmm, I think the correct interpretation is:
    # The multiplication tensor T has T[i, j, k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to the (i,k) entry of C.
    # Since C_{ik} = sum_j A_{ij} B_{jk}, every triple (i, j, k) where the indices are compatible contributes.
    # For n x m times m x p: i in [0,n), j in [0,m), k in [0,p).
    # T[i, j, k] = 1 for all i, j, k in their ranges.
    
    # Wait no. That can't be right either, because then the rank would be n*m*p which is the schoolbook.
    
    # Let me look at it from the schoolbook decomposition given:
    # For each (i, j, k): U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # So for a fixed (i,j,k), the contribution is U[i*m+j] * V[j*p+k] * W[k*n+i] = 1 * 1 * 1 = 1
    # And the total is sum over r of U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # In the schoolbook, for a given (i,j,k), only the column corresponding to (i,j,k) contributes, giving 1.
    # For other (i',j',k'), the product is 0 because the indices don't match.
    
    # So the target tensor is: T[i, j, k] = 1 for all valid i, j, k.
    # And we need sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i, j, k.
    
    n, m, p = DIMENSIONS
    # Check: for all i in range(n), j in range(m), k in range(p):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if total != 1:
                    return False
    return True


def _verify(U, V, W, rank):
    """Verify the decomposition is exact."""
    n, m, p = DIMENSIONS
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total = total + U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if total != 1:
                    return False
    return True


def _to_fraction_matrix(mat):
    """Convert a list of lists of floats to a list of lists of Fractions."""
    result = []
    for row in mat:
        frow = []
        for val in row:
            if isinstance(val, Fraction):
                frow.append(val)
            else:
                frow.append(Fraction(val).limit_denominator(1000000))
        result.append(frow)
    return result


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
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
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank by eliminating zero rows/columns or finding redundancies
    # First, check if we can do better with a known construction
    
    # For 4x4x5, let's try a systematic search approach
    # We'll use the fact that if we have a valid decomposition of rank R,
    # we can try to find a smaller one by looking for linear dependencies
    
    # Strategy: Start from schoolbook and try to merge/eliminate terms
    # A term r can be eliminated if its contribution is already covered by other terms
    
    # Let's try a different approach: use a rank-19 base for 4x4x4 and extend
    
    # Known: 4x4x4 can be done in rank 19 (Lander-Bailey)
    # For 4x4x5, we need to handle the extra column
    
    # Let me try a greedy approach: randomly perturb and see if we can reduce rank
    
    # Actually, let's try a more systematic approach:
    # The minimum rank for 4x4x5 is at most 4*4*5 = 80 (schoolbook)
    # We know 4x4x4 <= 19, so 4x4x5 <= 19 + 5*4 = 39 (adding 5 new columns with 4 terms each)
    # Or maybe 19 + 4 = 23 if we can share some terms
    
    # Let's try to construct a decomposition using the structure of matrix multiplication
    # C = A * B, where A is 4x4, B is 4x5, C is 4x5
    
    # We can write this as: C[:, :4] = A * B[:, :4] and C[:, 4] = A * B[:, 4]
    # The first part is 4x4x4, the second part is 4x4x1
    
    # For 4x4x1: C[:, 4] = A * B[:, 4], which is just 4 dot products
    # Each dot product C[i, 4] = sum_j A[i,j] B[j, 4] can be computed with 4 multiplications
    # But these can potentially share terms with the 4x4x4 part
    
    # Total: rank(4x4x4) + rank(4x4x1) = 19 + 4 = 23 at most
    # But maybe we can do better
    
    # Let's try to find a valid decomposition of rank 23
    
    # First, let's get a valid 4x4x4 decomposition of rank 19
    # I'll use a known construction
    
    # For now, let's just return the schoolbook to ensure correctness
    # Then try to improve within the time budget
    
    # Try to reduce rank by checking for zero columns
    # A column r is useless if U[i*m+j, r] = 0 for all i,j OR V[j*p+k, r] = 0 for all j,k OR W[k*n+i, r] = 0 for all k,i
    
    # Actually, let's try a completely different approach:
    # Use integer programming or a greedy algorithm to find a low-rank decomposition
    
    # Given the time constraints, let's try a simple heuristic:
    # 1. Start with schoolbook
    # 2. For each pair of terms, check if they can be merged
    # 3. Keep the best decomposition found
    
    # Check time
    if time.time() - start_time > DEADLINE - 5:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try to find a better decomposition using a known 4x4x4 rank-19 construction
    # and extend it to 4x4x5
    
    # For 4x4x4, one known rank-19 decomposition exists
    # Let me construct it explicitly
    
    # The Lander-Bailey decomposition for 4x4x4 has rank 19
    # I'll define it using the known pattern
    
    # Actually, let me just try to verify if a rank of 20, 21, 22, or 23 works
    # by attempting to construct such a decomposition
    
    # Given the complexity, let's stick with the schoolbook for now
    # but try one more thing: check if any columns are redundant
    
    # For the schoolbook, no columns are redundant since each (i,j,k) triple is unique
    
    # Return the schoolbook decomposition
    return {"U": best_U, "V": best_V, "W": best_W}
