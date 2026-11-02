"""Bootstrap SEARCH for exact tensor decomposition of 3x4x7 matrix multiplication."""
import time
import itertools
import numpy as np
import scipy.sparse.linalg as spla
import scipy.linalg as sla

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """Return the best exact tensor decomposition found within a time budget.
    The interface is construct() -> {'U', 'V', 'W'} with integer/half-integer entries.
    """
    start = time.time()
    deadline = start + 140.0  # safety margin below 150s kill

    def _elapsed():
        return time.time() - start

    def _time_left(frac=0.7):
        return deadline - _elapsed() - 10.0  # 10s safety margin

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 84

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
        
        Blocks:
          Block 1: i in {0,1}, j in {0,1}, k in {0,1}
          Block 2: i in {0,1}, j in {2,3}, k in {0,1}
          Block 3: i in {0,1}, j in {0,1}, k in {2,3}
        Total: 3 blocks * 1 rank saved = rank 81
        """
        # Define the Strassen 2x2x2 decomposition pattern
        # For a block with i_off, j_off, k_off offsets:
        # i indices: i_off, i_off+1
        # j indices: j_off, j_off+1
        # k indices: k_off, k_off+1
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # Strassen terms (relative to block-local indices 0,1):
        # Term 1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        # Term 2: (A10+A11)B00       -> C10 += M2, C11 += M2
        # Term 3: A00(B10-B11)       -> C01 += M3, C11 -= M3
        # Term 4: A11(B01-B00)       -> C00 -= M4, C01 += M4
        # Term 5: (A00+A10)B11       -> C00 += M5, C10 += M5
        # Term 6: (A01-A00)(B00+B10) -> C11 -= M6
        # Term 7: (A10-A11)(B01+B11) -> C00 += M7
        
        def strassen_block(i_off, j_off, k_off, U, V, W, col):
            """Add Strassen decomposition for block at given offsets. Returns next col."""
            i0, i1 = i_off, i_off + 1
            j0, j1 = j_off, j_off + 1
            k0, k1 = k_off, k_off + 1
            
            # Term 1: M1 = (A00+A11)(B00+B11)
            U[i0*m + j0][col] = 1
            U[i1*m + j1][col] = 1
            V[j0*p + k0][col] = 1
            V[j1*p + k1][col] = 1
            W[k0*n + i0][col] = 1
            W[k1*n + i1][col] = 1
            col += 1
            
            # Term 2: M2 = (A10+A11)B00
            U[i1*m + j0][col] = 1
            U[i1*m + j1][col] = 1
            V[j0*p + k0][col] = 1
            W[k0*n + i1][col] = 1
            W[k1*n + i1][col] = 1
            col += 1
            
            # Term 3: M3 = A00(B10-B11)
            U[i0*m + j0][col] = 1
            V[j1*p + k0][col] = 1
            V[j1*p + k1][col] = -1
            W[k1*n + i0][col] = 1
            W[k1*n + i1][col] = -1
            col += 1
            
            # Term 4: M4 = A11(B01-B00)
            U[i1*m + j1][col] = 1
            V[j0*p + k1][col] = 1
            V[j0*p + k0][col] = -1
            W[k0*n + i0][col] = -1
            W[k1*n + i0][col] = 1
            col += 1
            
            # Term 5: M5 = (A00+A10)B11
            U[i0*m + j0][col] = 1
            U[i1*m + j0][col] = 1
            V[j1*p + k1][col] = 1
            W[k0*n + i0][col] = 1
            W[k0*n + i1][col] = 1
            col += 1
            
            # Term 6: M6 = (A01-A00)(B00+B10)
            U[i0*m + j1][col] = 1
            U[i0*m + j0][col] = -1
            V[j0*p + k0][col] = 1
            V[j1*p + k0][col] = 1
            W[k1*n + i1][col] = -1
            col += 1
            
            # Term 7: M7 = (A10-A11)(B01+B11)
            U[i1*m + j0][col] = 1
            U[i1*m + j1][col] = -1
            V[j0*p + k1][col] = 1
            V[j1*p + k1][col] = 1
            W[k0*n + i0][col] = 1
            col += 1
            
            return col
        
        # Number of Strassen blocks
        num_blocks = 3
        # Each block uses 7 terms, rest uses schoolbook
        # Total (i,j,k) triples: n*m*p = 84
        # Each Strassen block covers 8 triples
        # Remaining triples: 84 - 3*8 = 60
        # Total rank: 3*7 + 60 = 81
        R = num_blocks * 7 + (n * m * p - num_blocks * 8)
        
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        
        col = 0
        
        # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
        col = strassen_block(0, 0, 0, U, V, W, col)
        
        # Block 2: i in {0,1}, j in {2,3}, k in {0,1}
        col = strassen_block(0, 2, 0, U, V, W, col)
        
        # Block 3: i in {0,1}, j in {0,1}, k in {2,3}
        col = strassen_block(0, 0, 2, U, V, W, col)
        
        # Mark which (i,j,k) triples are covered by Strassen blocks
        covered = set()
        for i_off in [0]:
            for j_off in [0, 2]:
                for k_off in [0]:
                    for di in range(2):
                        for dj in range(2):
                            for dk in range(2):
                                covered.add((i_off + di, j_off + dj, k_off + dk))
        for i_off in [0]:
            for j_off in [0]:
                for k_off in [2]:
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
        
        # Verify col matches R
        if col != R:
            return _schoolbook()
            
        return {"U": U, "V": V, "W": W}

    def _winograd_embedded():
        """
        Attempt a Winograd-like convolutional embedding.
        This maps the inner product dimension onto a cyclic convolution.
        For 3x4x7, we can try to exploit structure in the k-dimension (size 7).
        
        Winograd's algorithm for 2x2 matrix multiplication uses rank 7 (same as Strassen).
        For larger matrices, Winograd can sometimes be better.
        
        However, for 3x4x7, the dimensions are small enough that Strassen blocks are
        the primary known improvement. Let's try a different block structure.
        
        Alternative: Use 3x3x3 Winograd blocks if possible.
        Winograd 3x3x3 has rank 19 (vs 27 schoolbook), saving 8 ranks.
        We have one 3x3x3 block possible: i in {0,1,2}, j in {0,1,2}, k in {0,1,2}.
        But we have m=4, p=7. So we can fit one 3x3x3 block.
        Remaining: 84 - 27 = 57 terms.
        Total rank: 19 + 57 = 76.
        
        Let's implement Winograd 3x3x3.
        The Winograd 3x3x3 decomposition is complex. Let's use a known set of coefficients.
        Actually, implementing Winograd 3x3x3 correctly is error-prone.
        Let's stick to Strassen for now, but try more blocks or different combinations.
        
        Can we add another Strassen block?
        Current covered:
        Block 1: i in {0,1}, j in {0,1}, k in {0,1}
        Block 2: i in {0,1}, j in {2,3}, k in {0,1}
        Block 3: i in {0,1}, j in {0,1}, k in {2,3}
        
        Uncovered i=2. So no more 2x2 blocks in i-dimension.
        What about j-dimension?
        Block 2 covers j in {2,3}. Block 1,3 cover j in {0,1}.
        So all j are covered for i in {0,1}.
        
        What about k-dimension?
        Block 1,2 cover k in {0,1}. Block 3 covers k in {2,3}.
        Remaining k: 4,5,6.
        
        So we can't add more 2x2x2 Strassen blocks easily.
        
        Let's try a different approach: Random search for rank 80.
        """
        return None

    def _verify(U, V, W):
        """Verify that the decomposition exactly reconstructs the matrix multiplication."""
        N = len(U)
        M = len(V)
        P = len(W)
        R = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        rng = np.random.RandomState(42)
        for trial in range(5):
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

    def _random_search(rank, max_iters=1000):
        """
        Random search for a decomposition of given rank.
        This is unlikely to succeed for rank < 84, but we try.
        """
        if time.time() > deadline - 20:
            return None
        
        rng = np.random.RandomState(12345)
        vals = [-1.0, -0.5, 0.5, 1.0]
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                return None
            
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Sparse random initialization
            for r in range(rank):
                # Each term has sparse support
                nz_u = rng.randint(1, 3)
                nz_v = rng.randint(1, 3)
                nz_w = rng.randint(1, 3)
                
                idx_u = rng.choice(n * m, nz_u, replace=False)
                idx_v = rng.choice(m * p, nz_v, replace=False)
                idx_w = rng.choice(p * n, nz_w, replace=False)
                
                U[idx_u, r] = rng.choice(vals, nz_u)
                V[idx_v, r] = rng.choice(vals, nz_v)
                W[idx_w, r] = rng.choice(vals, nz_w)
            
            # Quick verification
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_true = A @ B
            C_approx = np.zeros((n, p))
            for r in range(rank):
                u_scalar = np.dot(U[:, r], A.flatten())
                v_scalar = np.dot(V[:, r], B.flatten())
                C_approx += u_scalar * v_scalar * W[:, r].reshape(p, n).T
            
            if np.allclose(C_approx, C_true, atol=1e-6):
                # Full verification
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                if _verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try the Strassen-embedded decomposition
    best = _strassen_embedded()
    if _verify(best["U"], best["V"], best["W"]):
        best_rank = len(best["U"][0])
    else:
        best = _schoolbook()
        best_rank = rank_schoolbook

    # Try random search for slightly lower ranks
    # This is a long shot, but we have time budget
    for R in range(best_rank - 1, max(27, best_rank - 5), -1):
        if time.time() > deadline - 15:
            break
        result = _random_search(R, max_iters=200)
        if result is not None:
            best = result
            best_rank = R
            break

    return best

# EVOLVE-BLOCK-END
