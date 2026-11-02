"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 4, 7

    # The matrix multiplication tensor T[i,j,k] for C[i,k] = sum_j A[i,j]*B[j,k]
    # We need to find rank-r decomposition:
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over c
    # such that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta(i,j,k) for all valid (i,j,k)
    # where delta is 1 if the indices correspond to a valid multiplication term, else 0.
    # Actually, the multiplication tensor is: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # The tensor has shape (n*m, m*p, p*n) but only entries where the indices are "compatible" matter.
    # Wait, let me reconsider. The standard formulation:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has indices (i, j, k) and T[i,j,k] = 1 for all valid i,j,k.
    # In the factorization: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    # where T[i,j,k] = 1 for all i,j,k (since every (i,j,k) triple contributes to the product).
    # Actually no - the tensor is just the identity in a sense. Let me think again.
    
    # The matrix multiplication tensor M has entries M[i,j,k] = 1 for all i in {0..n-1}, j in {0..m-1}, k in {0..p-1}.
    # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = M[i,j,k] = 1 for all i,j,k.
    # And for invalid index combinations, the product should be 0, but since we only evaluate on valid (i,j,k),
    # we just need the equation to hold for all valid triples.
    
    # Schoolbook: rank = n*m*p = 84. Each term picks one (i,j,k) triple.
    # U[i*m+j, c] = 1 if c corresponds to (i,j,k_c), else 0
    # V[j*p+k, c] = 1 if c corresponds to (i_c,j,k), else 0  
    # W[k*n+i, c] = 1 if c corresponds to (i_c,j_c,k), else 0
    
    # Let's try to find a lower rank decomposition.
    # Known results: for 3x4 by 4x7, the minimum rank is not well-known in closed form.
    # Let's try a systematic search for small rank reductions.
    
    # Start with schoolbook
    rank_schoolbook = n * m * p  # 84
    
    # Try to find a decomposition of rank r < 84
    # We'll use a greedy/iterative approach:
    # 1. Start with schoolbook
    # 2. Try to merge or eliminate terms
    
    # Alternative approach: use the known structure of matrix multiplication.
    # For small matrices, we can try to find explicit low-rank decompositions.
    
    # Let's try a direct construction approach.
    # For 3x4 * 4x7, let's see if we can find a rank < 84 decomposition.
    
    # One approach: use the fact that some entries can be shared.
    # For example, if we have terms for (i1,j,k1) and (i2,j,k2), we might be able to combine them.
    
    # Let's try a randomized search with verification.
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(len(U[0])):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def schoolbook_decomposition():
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
        return U, V, W, rank
    
    U, V, W, rank = schoolbook_decomposition()
    
    # Try to reduce rank by finding linear dependencies or using known techniques.
    # For now, return schoolbook if we can't find better quickly.
    
    # Let's try a simple approach: see if we can find a rank 83 decomposition
    # by checking if any column can be removed or modified.
    
    # Actually, let's try a more systematic approach using the structure.
    # The key insight: for matrix multiplication, there are known lower bounds.
    # For 3x4 * 4x7, the Strassen-like algorithms might help, but they're complex.
    
    # Given the time constraint, let's return the schoolbook decomposition.
    # This guarantees correctness with rank 84.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
