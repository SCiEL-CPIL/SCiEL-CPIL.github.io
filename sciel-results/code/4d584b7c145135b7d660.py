"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The decomposition should satisfy:
    # For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # (since the multiplication tensor has all-ones entries)
    
    # Check a few random (i,j,k) triples
    import random
    random.seed(42)
    for _ in range(100):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
        if abs(val - 1) > 1e-9:
            return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 150s kill
    
    def schoolbook():
        """Return the trivial rank-105 decomposition."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomp(U, V, W, rank):
        """Fully verify all (i,j,k) triples."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = float(np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i]))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Laderman's algorithm for 3x3 matrix multiplication (rank 23)
    # Applied to the 3x3x3 sub-block: rows 0-2 of A, cols 0-2 of B, outputs (i,k) for i,k in {0,1,2}
    # This covers 27 triples with rank 23 instead of 27
    
    # Laderman's 23-term decomposition for 3x3 matrix multiplication
    # Indices: A[i,j], B[j,k], C[i,k] for i,j,k in {0,1,2}
    laderman_terms = [
        # (U_indices, V_indices, W_indices)
        # Each term is a list of (matrix, row, col, coeff) tuples
        # U: A[i*m+j], V: B[j*p+k], W: C[k*n+i]
        
        # Term 0: A[0,0] * B[0,0] -> C[0,0]
        [((0, 0), 1), ((0, 0), 1), ((0, 0), 1)],
        # Term 1: A[0,1] * B[1,0] -> C[0,0]
        [((0, 1), 1), ((1, 0), 1), ((0, 0), 1)],
        # Term 2: A[1,0] * B[0,1] -> C[1,1]
        [((1, 0), 1), ((0, 1), 1), ((1, 1), 1)],
        # Term 3: A[1,1] * B[1,1] -> C[1,1]
        [((1, 1), 1), ((1, 1), 1), ((1, 1), 1)],
        # Term 4: A[2,0] * B[0,2] -> C[2,2]
        [((2, 0), 1), ((0, 2), 1), ((2, 2), 1)],
        # Term 5: A[2,1] * B[1,2] -> C[2,2]
        [((2, 1), 1), ((1, 2), 1), ((2, 2), 1)],
        # Term 6: A[2,2] * B[2,2] -> C[2,2]
        [((2, 2), 1), ((2, 2), 1), ((2, 2), 1)],
        # Term 7: (A[0,0]+A[0,1]+A[1,0]+A[1,1]) * (B[0,0]+B[0,1]+B[1,0]+B[1,1]) -> C[0,0], C[0,1], C[1,0], C[1,1]
        # We need to split this into individual contributions
        # Actually, let me use the standard Laderman formulation
    ]
    
    # Let me use a cleaner approach: directly implement Laderman's 23 multiplications
    # Standard Laderman algorithm for 3x3:
    
    laderman = []
    
    # M1 = A[0,0] * B[0,0]
    laderman.append([(0,0,1), (0,0,1), (0,0,1)])
    # M2 = A[0,1] * B[1,0]
    laderman.append([(0,1,1), (1,0,1), (0,0,1)])
    # M3 = A[1,0] * B[0,1]
    laderman.append([(1,0,1), (0,1,1), (1,1,1)])
    # M4 = A[1,1] * B[1,1]
    laderman.append([(1,1,1), (1,1,1), (1,1,1)])
    # M5 = A[2,0] * B[0,2]
    laderman.append([(2,0,1), (0,2,1), (2,2,1)])
    # M6 = A[2,1] * B[1,2]
    laderman.append([(2,1,1), (1,2,1), (2,2,1)])
    # M7 = A[2,2] * B[2,2]
    laderman.append([(2,2,1), (2,2,1), (2,2,1)])
    # M8 = (A[0,0]+A[0,1]) * (B[0,1]-B[1,1]) -> C[0,1]
    laderman.append([(0,0,1), (0,1,1), (0,1,1)])
    laderman.append([(0,1,1), (0,1,1), (0,1,1)])
    laderman.append([(0,1,1), (1,1,-1), (0,1,1)])
    laderman.append([(0,1,1), (1,1,-1), (0,1,1)])
    
    # This is getting complex. Let me use a known correct Laderman implementation.
    
    # Actually, let me try a different approach: use Strassen for 2x2x2 blocks
    # and combine with schoolbook for the rest.
    
    # Strassen 2x2x2 blocks:
    # n=3: Strassen on {0,1}, schoolbook on {2}
    # m=5: Strassen on {0,1} and {2,3}, schoolbook on {4}
    # p=7: Strassen on {0,1}, {2,3}, {4,5}, schoolbook on {6}
    
    n_blocks = [(0, 1)]
    m_blocks = [(0, 1), (2, 3)]
    p_blocks = [(0, 1), (2, 3), (4, 5)]
    
    # Count: 6 blocks * 7 terms = 42 Strassen terms
    # Remaining triples: 105 - 6*8 = 105 - 48 = 57
    # Total rank: 42 + 57 = 99
    rank = 99
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for (a, b) in n_blocks:
        for (c, d) in m_blocks:
            for (e, f) in p_blocks:
                # M1 = (A[a,c]+A[b,d]) * (B[c,e]+B[d,f])
                # Contributes: C[e,a] += 1, C[f,b] += 1
                U[a*m+c][col] += 1
                U[b*m+d][col] += 1
                V[c*p+e][col] += 1
                V[d*p+f][col] += 1
                W[e*n+a][col] += 1
                W[f*n+b][col] += 1
                col += 1
                
                # M2 = (A[b,c]+A[b,d]) * B[c,e]
                # Contributes: C[f,a] += 1
                U[b*m+c][col] += 1
                U[b*m+d][col] += 1
                V[c*p+e][col] += 1
                W[f*n+a][col] += 1
                col += 1
                
                # M3 = A[a,c] * (B[d,f]-B[d,e])
                # Contributes: C[e,f] += 1
                U[a*m+c][col] += 1
                V[d*p+f][col] += 1
                V[d*p+e][col] -= 1
                W[e*n+f][col] += 1
                col += 1
                
                # M4 = A[b,d] * (B[c,f]-B[c,e])
                # Contributes: C[e,a] += 1
                U[b*m+d][col] += 1
                V[c*p+f][col] += 1
                V[c*p+e][col] -= 1
                W[e*n+a][col] += 1
                col += 1
                
                # M5 = (A[a,c]+A[a,d]) * B[d,f]
                # Contributes: C[e,f] += 1, C[e,a] -= 1
                U[a*m+c][col] += 1
                U[a*m+d][col] += 1
                V[d*p+f][col] += 1
                W[e*n+f][col] += 1
                W[e*n+a][col] -= 1
                col += 1
                
                # M6 = (A[b,c]-A[a,c]) * (B[c,e]+B[c,f])
                # Contributes: C[f,a] += 1
                U[b*m+c][col] += 1
                U[a*m+c][col] -= 1
                V[c*p+e][col] += 1
                V[c*p+f][col] += 1
                W[f*n+a][col] += 1
                col += 1
                
                # M7 = (A[b,d]-A[a,d]) * (B[d,e]+B[d,f])
                # Contributes: C[e,a] -= 1, C[f,b] += 1
                U[b*m+d][col] += 1
                U[a*m+d][col] -= 1
                V[d*p+e][col] += 1
                V[d*p+f][col] += 1
                W[e*n+a][col] -= 1
                W[f*n+b][col] += 1
                col += 1
    
    # Schoolbook for remaining (uncovered) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                covered = False
                for (a, b) in n_blocks:
                    if i in (a, b):
                        for (c, d) in m_blocks:
                            if j in (c, d):
                                for (e, f) in p_blocks:
                                    if k in (e, f):
                                        covered = True
                if not covered:
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Full verification
    if verify_decomp(U, V, W, rank):
        return {"U": U, "V": V, "W": W}
    
    # If verification fails, fall back to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
