"""Combined program: Strassen-embedded exact tensor decomposition for 3x4x7."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the tensor decomposition.
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)
    The decomposition satisfies:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    """
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
    
    def _strassen_embedded():
        """
        Construct a decomposition by replacing multiple 2x2x2 schoolbook blocks
        with Strassen's rank-7 decomposition. Each 2x2x2 block saves 1 rank.
        
        We can fit Strassen blocks in dimensions where we have at least 2 consecutive
        indices. For 3x4x7:
        - i dimension: 3 -> can have 1 block of size 2 (indices 0,1 or 1,2)
        - j dimension: 4 -> can have 2 blocks of size 2 (indices 0,1 and 2,3)
        - k dimension: 7 -> can have 3 blocks of size 2 (indices 0,1, 2,3, 4,5)
        
        Each Strassen block covers 2x2x2 = 8 (i,j,k) triples and uses 7 terms.
        Schoolbook for those 8 triples would use 8 terms. So each block saves 1 rank.
        
        Maximum number of non-overlapping Strassen blocks:
        We need to partition the 3x4x7 grid into 2x2x2 blocks plus leftover 1D slabs.
        
        i: 3 = 2 + 1 -> 1 block
        j: 4 = 2 + 2 -> 2 blocks
        k: 7 = 2 + 2 + 2 + 1 -> 3 blocks
        
        Max blocks = 1 * 2 * 3 = 6
        Total saved = 6
        Rank = 84 - 6 = 78
        """
        # Define Strassen block helper
        def strassen_block(i_off, j_off, k_off, U, V, W, col):
            """Add Strassen decomposition for block at given offsets. Returns next col."""
            i0, i1 = i_off, i_off + 1
            j0, j1 = j_off, j_off + 1
            k0, k1 = k_off, k_off + 1
            
            # Term 1: M1 = (A00+A11)(B00+B11)
            # A00 is at (i0, j0), A11 is at (i1, j1)
            # B00 is at (j0, k0), B11 is at (j1, k1)
            # C00 is at (k0, i0), C11 is at (k1, i1)
            U[i0*m + j0][col] = 1
            U[i1*m + j1][col] = 1
            V[j0*p + k0][col] = 1
            V[j1*p + k1][col] = 1
            W[k0*n + i0][col] = 1
            W[k1*n + i1][col] = 1
            col += 1
            
            # Term 2: M2 = (A10+A11)B00
            # A10 is at (i1, j0), A11 is at (i1, j1)
            # B00 is at (j0, k0)
            # C10 is at (k0, i1), C11 is at (k1, i1)
            U[i1*m + j0][col] = 1
            U[i1*m + j1][col] = 1
            V[j0*p + k0][col] = 1
            W[k0*n + i1][col] = 1
            W[k1*n + i1][col] = -1
            col += 1
            
            # Term 3: M3 = A00(B10-B11)
            # A00 is at (i0, j0)
            # B10 is at (j1, k0), B11 is at (j1, k1)
            # C01 is at (k1, i0), C11 is at (k1, i1) -> wait, C indices are (k, i)
            # C01 means k=0, i=1? No, in the block local coords:
            # C00 -> (k0, i0), C01 -> (k1, i0), C10 -> (k0, i1), C11 -> (k1, i1)
            # So C01 is (k1, i0)
            U[i0*m + j0][col] = 1
            V[j1*p + k0][col] = 1
            V[j1*p + k1][col] = -1
            W[k1*n + i0][col] = 1
            col += 1
            
            # Term 4: M4 = A11(B01-B00)
            # A11 is at (i1, j1)
            # B01 is at (j0, k1), B00 is at (j0, k0)
            # C00 is (k0, i0), C01 is (k1, i0)
            U[i1*m + j1][col] = 1
            V[j0*p + k1][col] = 1
            V[j0*p + k0][col] = -1
            W[k0*n + i0][col] = -1
            W[k1*n + i0][col] = 1
            col += 1
            
            # Term 5: M5 = (A00+A10)B11
            # A00 is (i0, j0), A10 is (i1, j0)
            # B11 is (j1, k1)
            # C00 is (k0, i0), C10 is (k0, i1)
            U[i0*m + j0][col] = 1
            U[i1*m + j0][col] = 1
            V[j1*p + k1][col] = 1
            W[k0*n + i0][col] = 1
            W[k0*n + i1][col] = 1
            col += 1
            
            # Term 6: M6 = (A01-A00)(B00+B10)
            # A01 is (i0, j1), A00 is (i0, j0)
            # B00 is (j0, k0), B10 is (j1, k0)
            # C11 is (k1, i1)
            U[i0*m + j1][col] = 1
            U[i0*m + j0][col] = -1
            V[j0*p + k0][col] = 1
            V[j1*p + k0][col] = 1
            W[k1*n + i1][col] = -1
            col += 1
            
            # Term 7: M7 = (A10-A11)(B01+B11)
            # A10 is (i1, j0), A11 is (i1, j1)
            # B01 is (j0, k1), B11 is (j1, k1)
            # C00 is (k0, i0), C11 is (k1, i1)
            U[i1*m + j0][col] = 1
            U[i1*m + j1][col] = -1
            V[j0*p + k1][col] = 1
            V[j1*p + k1][col] = 1
            W[k0*n + i0][col] = 1
            W[k1*n + i1][col] = 1
            col += 1
            
            return col
        
        # Determine all possible Strassen block positions
        # i blocks: start at 0 (covers 0,1)
        i_blocks = [0]
        # j blocks: start at 0 (covers 0,1), start at 2 (covers 2,3)
        j_blocks = [0, 2]
        # k blocks: start at 0 (covers 0,1), start at 2 (covers 2,3), start at 4 (covers 4,5)
        k_blocks = [0, 2, 4]
        
        # All possible blocks
        all_blocks = []
        for i_off in i_blocks:
            for j_off in j_blocks:
                for k_off in k_blocks:
                    all_blocks.append((i_off, j_off, k_off))
        
        num_blocks = len(all_blocks)  # 6
        R = num_blocks * 7 + (n * m * p - num_blocks * 8)
        
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        
        col = 0
        covered = set()
        
        for block in all_blocks:
            i_off, j_off, k_off = block
            col = strassen_block(i_off, j_off, k_off, U, V, W, col)
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered.add((i_off + di, j_off + dj, k_off + dk))
        
        # Fill remaining with schoolbook terms
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
        """Verify that the decomposition exactly reconstructs the matrix multiplication."""
        N = len(U)
        M = len(V)
        P = len(W)
        R = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        # Check: for all i, j, k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # And for all (i,j,k) that are NOT the diagonal, the sum should be 0?
        # Actually, the tensor T[i,j,k] = 1 if this is a valid multiplication term.
        # The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k.
        # So we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for ALL i,j,k.
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(np.dot(Ua[u_idx], Va[v_idx]), Wa[w_idx])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        return max_err < 1e-10
    
    # Try Strassen-embedded first
    start = time.time()
    result = _strassen_embedded()
    
    # Verify
    if _verify(result["U"], result["V"], result["W"]):
        return result
    else:
        # Fall back to schoolbook
        return _schoolbook()
# EVOLVE-BLOCK-END
