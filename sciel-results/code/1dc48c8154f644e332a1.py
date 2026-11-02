"""Bootstrap SEARCH for exact tensor decomposition of 3x4x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """Return the best exact tensor decomposition found within a time budget.
    The interface is construct() -> {'U', 'V', 'W'} with integer/half-integer entries.
    """
    start = time.time()
    deadline = start + 140.0

    n, m, p = DIMENSIONS

    def _schoolbook():
        """Return the classical rank-nmp decomposition."""
        R = n * m * p
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _strassen_block_coeffs():
        """
        Return the 7 Strassen terms as lists of (u_idx, v_idx, w_idx) tuples
        for a 2x2x2 block with local indices (0,1) in each dimension.
        
        For C = A*B:
        C[0,0] = M1 - M4 + M5 + M7
        C[0,1] = M3 + M4
        C[1,0] = M2 + M5
        C[1,1] = M1 - M2 - M6 + M7
        
        M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        M2 = (A[1,0]+A[1,1])*B[0,0]
        M3 = A[0,0]*(B[1,0]-B[1,1])
        M4 = A[1,1]*(B[0,1]-B[0,0])
        M5 = (A[0,0]+A[1,0])*B[1,1]
        M6 = (A[0,1]-A[0,0])*(B[0,0]+B[1,0])
        M7 = (A[1,0]-A[1,1])*(B[0,1]+B[1,1])
        
        In our tensor: U[i*m+j], V[j*p+k], W[k*n+i]
        """
        terms = []
        # M1: (A00+A11)(B00+B11)
        terms.append([(0,0,0,1), (1,1,0,1), (0,0,1,1), (1,1,1,1)])
        # M2: (A10+A11)B00
        terms.append([(1,0,0,1), (1,1,0,1), (0,0,1,1)])
        # M3: A00(B10-B11)
        terms.append([(0,0,0,1), (1,0,1,1), (1,1,-1,1)])
        # M4: A11(B01-B00)
        terms.append([(1,1,0,1), (0,1,1,1), (0,0,-1,1)])
        # M5: (A00+A10)B11
        terms.append([(0,0,0,1), (1,0,0,1), (1,1,1,1)])
        # M6: (A01-A00)(B00+B10)
        terms.append([(0,1,0,1), (0,0,-1,1), (0,0,1,1), (1,0,1,1)])
        # M7: (A10-A11)(B01+B11)
        terms.append([(1,0,0,1), (1,1,-1,1), (0,1,1,1), (1,1,1,1)])
        return terms

    def _strassen_embedded():
        """
        Construct decomposition using non-overlapping 2x2x2 Strassen blocks.
        n=3, m=4, p=7.
        i-blocks: {0,1} (only one non-overlapping pair)
        j-blocks: {0,1}, {2,3}
        k-blocks: {0,1}, {2,3}, {4,5}
        Total: 1*2*3 = 6 blocks, each saves 1 rank.
        Rank: 84 - 6 = 78
        """
        strassen_terms = _strassen_block_coeffs()
        
        blocks = []
        for i_off in [0]:
            for j_off in [0, 2]:
                for k_off in [0, 2, 4]:
                    blocks.append((i_off, j_off, k_off))
        
        num_blocks = len(blocks)
        R = num_blocks * 7 + (n * m * p - num_blocks * 8)
        
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        
        col = 0
        covered = set()
        
        for i_off, j_off, k_off in blocks:
            i0, i1 = i_off, i_off + 1
            j0, j1 = j_off, j_off + 1
            k0, k1 = k_off, k_off + 1
            
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered.add((i_off + di, j_off + dj, k_off + dk))
            
            for term in strassen_terms:
                for (li, lj, lk, coeff) in term:
                    ii = i_off + li
                    jj = j_off + lj
                    kk = k_off + lk
                    U[ii * m + jj][col] += coeff
                    V[jj * p + kk][col] += coeff
                    W[kk * n + ii][col] += coeff
                col += 1
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in covered:
                        continue
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        if col != R:
            return _schoolbook()
        
        return {"U": U, "V": V, "W": W}

    def _verify(U, V, W):
        """Verify exact reconstruction of matrix multiplication."""
        R = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        rng = np.random.RandomState(42)
        for trial in range(3):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_true = A @ B
            C_approx = np.zeros((n, p))
            A_flat = A.reshape(-1)
            B_flat = B.reshape(-1)
            for r in range(R):
                ua = Ua[:, r]
                va = Va[:, r]
                wa = Wa[:, r]
                u_dot_a = ua @ A_flat
                v_dot_b = va @ B_flat
                C_approx += (wa * (u_dot_a * v_dot_b)).reshape(p, n).T
            if not np.allclose(C_true, C_approx, atol=1e-10):
                return False
        return True

    # Try Strassen-embedded decomposition
    best = _strassen_embedded()
    if _verify(best["U"], best["V"], best["W"]):
        return best
    
    # Fallback: schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
