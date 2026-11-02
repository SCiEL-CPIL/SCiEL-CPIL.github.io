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

    def _strassen_2x2x2(U, V, W, i0, j0, k0, col):
        """
        Apply Strassen's algorithm to a 2x2x2 sub-block at offset (i0, j0, k0).
        Returns the next column index.
        """
        strassen_terms = _strassen_block_coeffs()
        for term in strassen_terms:
            for (li, lj, lk, coeff) in term:
                ii = i0 + li
                jj = j0 + lj
                kk = k0 + lk
                U[ii * m + jj][col] += coeff
                V[jj * p + kk][col] += coeff
                W[kk * n + ii][col] += coeff
            col += 1
        return col

    def _strassen_embedded_v2():
        """
        Construct a rank-57 decomposition by applying Strassen to the 2x2x2 blocks
        within the 2x4x6 sub-problem, and using schoolbook for the remaining parts.
        
        Partition:
        - i: {0,1} (Strassen-capable), {2} (schoolbook)
        - j: {0,1}, {2,3} (both Strassen-capable)
        - k: {0,1}, {2,3}, {4,5} (Strassen-capable), {6} (schoolbook)
        
        For the 2x4x6 part (i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}):
        - Split j into {0,1} and {2,3}
        - Split k into {0,1}, {2,3}, {4,5}
        - For each (j_block, k_block) pair, we have a 2x2x2 sub-problem
        - Apply Strassen to each: 7 terms
        - Total Strassen terms: 2 * 3 * 7 = 42
        
        For the remaining parts:
        - i in {0,1}, j in {0,1,2,3}, k = 6: 2*4*1 = 8 terms
        - i = 2, j in {0,1,2,3}, k in {0,1,2,3,4,5}: 1*4*6 = 24 terms
        - i = 2, j in {0,1,2,3}, k = 6: 1*4*1 = 4 terms
        Total schoolbook: 8 + 24 + 4 = 36? No wait, let me recount.
        
        Actually:
        - i in {0,1}, k = 6: 2*4*1 = 8 terms
        - i = 2, k in {0,1,2,3,4,5}: 1*4*6 = 24 terms
        - i = 2, k = 6: 1*4*1 = 4 terms
        Total: 8 + 24 + 4 = 36
        
        But wait, that gives 42 + 36 = 78, same as before. Let me reconsider.
        
        Actually, I think I made an error. Let me carefully count:
        Total elements: 3*4*7 = 84
        
        Strassen-covered: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}
        = 2*4*6 = 48 elements
        Using 42 Strassen terms (2 j-blocks * 3 k-blocks * 7 terms)
        
        Remaining: 84 - 48 = 36 elements
        - i in {0,1}, k = 6: 2*4*1 = 8
        - i = 2, k in {0,1,2,3,4,5}: 1*4*6 = 24
        - i = 2, k = 6: 1*4*1 = 4
        Total: 8 + 24 + 4 = 36
        
        So total rank = 42 + 36 = 78. Same as before.
        
        Hmm, so the current program is already at 78. Let me verify this is correct.
        """
        strassen_terms = _strassen_block_coeffs()
        
        U = [[0] * (n * m * p) for _ in range(n * m)]
        V = [[0] * (n * m * p) for _ in range(m * p)]
        W = [[0] * (n * m * p) for _ in range(p * n)]
        
        col = 0
        covered = set()
        
        # Apply Strassen to 2x2x2 blocks within the 2x4x6 part
        for j_off in [0, 2]:
            for k_off in [0, 2, 4]:
                i0, j0, k0 = 0, j_off, k_off
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            covered.add((i0 + di, j0 + dj, k0 + dk))
                col = _strassen_2x2x2(U, V, W, i0, j0, k0, col)
        
        # Schoolbook for remaining elements
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in covered:
                        continue
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        R = col
        U = [row[:R] for row in U]
        V = [row[:R] for row in V]
        W = [row[:R] for row in W]
        
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
    best = _strassen_embedded_v2()
    if _verify(best["U"], best["V"], best["W"]):
        return best
    
    # Fallback: schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
