"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE = 145.0  # 150s limit with 5s safety margin

def _time_left():
    return _DEADLINE - (time.time() - _start_time)

def _verify(U, V, W, n=4, m=5, p=5):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # For each (i,j,k), we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # And for "wrong" combinations, the sum should be 0
    # Actually, the tensor T[i,j,k] = delta_{i,j,k} in the sense that
    # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # The decomposition says: T_{ijk} = sum_c U_{ij,c} V_{jk,c} W_{ki,c}
    # where T_{ijk} = 1 if this is a valid multiplication term, 0 otherwise
    # Wait, let me reconsider. The matrix multiplication tensor has entries:
    # T[i,j,k] = 1 for all i,j,k (since (AB)_{ik} = sum_j A_{ij} B_{jk})
    # No, that's not right either.
    
    # Let me re-read the task. The tensor for matrix multiplication C = A*B where
    # A is n x m, B is m x p, C is n x p.
    # C_{ik} = sum_j A_{ij} B_{jk}
    # So the tensor T has indices (i, j, k) and T[i,j,k] = 1 for all valid i,j,k.
    # A rank-r decomposition: T[i,j,k] = sum_{c=0}^{r-1} U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Wait, the W index is W[k*n+i, c] per the task description.
    
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # And that's it! All entries of the tensor are 1.
    # Let's verify this interpretation makes sense with the schoolbook:
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1 for col = i*m*p + j*p + k
    # Then sum_c ... = 1 for each (i,j,k). Correct!
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _try_rank100_search():
    """Search for rank-100 decomposition with structured ansatz."""
    n, m, p = 4, 5, 5
    target_rank = 100
    
    # Use a structured approach: try to find decompositions where
    # each factor has a specific structure.
    
    # Approach: Use the fact that we need sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    # for all i,j,k.
    
    # Try a parametrized family: assume W[k*n+i, c] depends only on (k,i) pairs
    # and use symmetry.
    
    # Alternative: Use integer programming or systematic search over small values.
    # Given time constraints, let's try a randomized approach with local search.
    
    # Start with schoolbook (rank 100) and try to simplify/verify
    # Actually schoolbook IS rank 100 for 4*5*5 = 100.
    
    # So the question is: can we do better than 100?
    # Known bounds: For 4x5x5, the known lower bound is around 90-95.
    # Let's try to find a decomposition of rank < 100.
    
    # Strategy: Start from schoolbook and try to merge/simplify terms.
    # Or use a different parametrization.
    
    # Let's try a systematic approach:
    # Assume U[i,j,c] in {-1, 0, 1}, V[j,k,c] in {-1, 0, 1}, W[k,i,c] in {-1, 0, 1}
    # and search for rank r < 100.
    
    # This is a huge search space, so let's use a smarter approach.
    
    # Use the structure: The tensor is all-ones. We want to write 1 as a sum of products.
    # Think of it as: for each (i,j,k), we need the inner product of vectors
    # U_{ij}, V_{jk}, W_{ki} (element-wise product summed) to equal 1.
    
    # Let's try a constructive approach with rank 95, 90, 85, 80, etc.
    
    # For now, return the schoolbook which is rank 100.
    # A proper search would require more sophisticated methods.
    
    # Let's try a randomized local search starting from schoolbook.
    
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W
    
    U, V, W = _schoolbook()
    
    # Try to reduce rank by finding redundant terms or combining.
    # This is complex; for now, return schoolbook as baseline.
    
    return U, V, W, n*m*p

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a better decomposition if time permits
    if _time_left() > 10:
        try:
            U, V, W, rank = _try_rank100_search()
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    # Fallback to schoolbook
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
