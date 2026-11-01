"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Schoolbook fallback (rank 32)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Strassen decomposition for 2x2 x 2x2
    # Terms are (u, v, w) where u, v, w are length-4 vectors
    # u indexes A (flattened row-major: [A00, A01, A10, A11])
    # v indexes B (flattened row-major: [B00, B01, B10, B11])
    # w indexes C (flattened row-major: [C00, C01, C10, C11])
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),   # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])    # M7
    ]

    def build_strassen_228():
        """Build rank-28 decomposition using 4 Strassen blocks."""
        rank = 28
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]

        num_blocks = p // 2  # 4 blocks

        for block_idx in range(num_blocks):
            for term_idx in range(7):
                col = block_idx * 7 + term_idx
                u, v, w = strassen_terms[term_idx]

                # Fill U: index is i (0..3) for A[i//2, i%2]
                for i in range(4):
                    U[i][col] = u[i]

                # Fill V: B is 2x8, block covers columns 2*block_idx and 2*block_idx+1
                # Global B index: row j (0..1), col k_global
                # Local v index: j*2 + k_local where k_local is 0 or 1
                for j in range(2):
                    for k_local in range(2):
                        k_global = 2 * block_idx + k_local
                        b_idx = j * p + k_global
                        v_idx = j * 2 + k_local
                        V[b_idx][col] = v[v_idx]

                # Fill W: C is 8x2 (p x n), index is k_global*n + i
                # Global C index: row k_global, col i
                # Local w index: i*2 + k_local where k_local is 0 or 1
                # The convention is W[k*n+i, r]
                # C is p x n. C[k, i] corresponds to index k*n + i.
                # The local w vector is for a 2x2 block of C.
                # The block of C is at rows 2*block_idx and 2*block_idx+1, cols 0 and 1.
                # So local index (i, k_local) maps to global index (2*block_idx + k_local)*n + i.
                for i in range(2):
                    for k_local in range(2):
                        k_global = 2 * block_idx + k_local
                        c_idx = k_global * n + i
                        w_idx = i * 2 + k_local
                        W[c_idx][col] = w[w_idx]

        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Verify the decomposition exactly."""
        rank = len(U[0])
        # Build target tensor T[a,b,c]
        T = np.zeros((nm, mp, pn), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1

        U_np = np.array(U, dtype=int)
        V_np = np.array(V, dtype=int)
        W_np = np.array(W, dtype=int)

        # Compute T_decomp[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        T_decomp = np.zeros((nm, mp, pn), dtype=int)
        for r in range(rank):
            T_decomp += np.einsum('a,b,c->abc', U_np[:, r], V_np[:, r], W_np[:, r])

        return np.array_equal(T_decomp, T)

    # Try Strassen first
    result_28 = build_strassen_228()
    if verify(result_28["U"], result_28["V"], result_28["W"]):
        return result_28

    # If Strassen fails (shouldn't happen if logic is correct), try to optimize or fallback
    # Given the directive to explore improvements, let's try a quick optimization search for rank < 32
    # But Strassen is known to be rank 7 for 2x2x2, so 4*7=28 should work.
    # If it fails, there is a bug. Let's double check the Strassen terms.
    
    # Re-verifying Strassen terms manually:
    # M1 = (A00+A11)(B00+B11) -> u=[1,0,0,1], v=[1,0,0,1]
    # C00 += M1, C11 += M1
    # w for M1: C00=1, C11=1 -> w=[1,0,0,1] (C00, C01, C10, C11). Correct.
    
    # M2 = (A10+A11)B00 -> u=[0,0,1,1], v=[1,0,0,0]
    # C10 += M2, C11 -= M2
    # w for M2: C10=1, C11=-1 -> w=[0,0,1,-1]. Correct.
    
    # M3 = A00(B01-B11) -> u=[1,0,0,0], v=[0,1,0,-1]
    # C01 += M3, C11 += M3
    # w for M3: C01=1, C11=1 -> w=[0,1,0,1]. Correct.
    
    # M4 = A11(B10-B00) -> u=[0,0,0,1], v=[-1,0,1,0]
    # C00 += M4, C10 += M4
    # w for M4: C00=1, C10=1 -> w=[1,0,1,0]. Correct.
    
    # M5 = (A00+A01)B11 -> u=[1,1,0,0], v=[0,0,0,1]
    # C00 -= M5, C10 += M5
    # w for M5: C00=-1, C10=1 -> w=[-1,1,0,0]. Correct.
    
    # M6 = (A10-A00)(B00+B01) -> u=[-1,0,1,0], v=[1,1,0,0]
    # C11 += M6
    # w for M6: C11=1 -> w=[0,0,0,1]. Correct.
    
    # M7 = (A01-A11)(B10+B11) -> u=[0,1,0,-1], v=[0,0,1,1]
    # C00 += M7
    # w for M7: C00=1 -> w=[1,0,0,0]. Correct.

    # The terms look correct. The indexing in build_strassen_228 looks correct.
    # If verify fails, it's a subtle bug. Let's try a direct construction with explicit indices.

    def build_strassen_228_direct():
        """Build rank-28 decomposition using 4 Strassen blocks with explicit index mapping."""
        rank = 28
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]

        # Strassen terms for 2x2x2
        # u, v, w are length-4 vectors: [00, 01, 10, 11]
        terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # M2
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # M3
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),   # M4
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # M5
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # M6
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])    # M7
        ]

        num_blocks = p // 2  # 4 blocks

        for block_idx in range(num_blocks):
            for term_idx in range(7):
                col = block_idx * 7 + term_idx
                u, v, w = terms[term_idx]

                # Fill U: A is 2x2, flattened row-major: [A00, A01, A10, A11]
                # U index: i (0..3)
                for i in range(4):
                    U[i][col] = u[i]

                # Fill V: B is 2x8
                # Local v indices 0,1,2,3 correspond to B[0,0], B[0,1], B[1,0], B[1,1] within the block
                # Block covers columns 2*block_idx and 2*block_idx+1
                # Global B index: row j (0..1), col k_global
                # Local index 0: j=0, k_local=0 -> k_global=2*block_idx -> b_idx = 0*8 + 2*block_idx
                # Local index 1: j=0, k_local=1 -> k_global=2*block_idx+1 -> b_idx = 0*8 + 2*block_idx+1
                # Local index 2: j=1, k_local=0 -> k_global=2*block_idx -> b_idx = 1*8 + 2*block_idx
                # Local index 3: j=1, k_local=1 -> k_global=2*block_idx+1 -> b_idx = 1*8 + 2*block_idx+1
                
                v_indices = [2*block_idx, 2*block_idx+1, 8+2*block_idx, 8+2*block_idx+1]
                for l in range(4):
                    V[v_indices[l]][col] = v[l]

                # Fill W: C is 8x2 (p x n)
                # Convention: W[k*n + i, r] where k is row (0..7), i is col (0..1)
                # Local w indices 0,1,2,3 correspond to C[0,0], C[0,1], C[1,0], C[1,1] within the block
                # Block covers rows 2*block_idx and 2*block_idx+1
                # Local index 0: k_local=0, i=0 -> k_global=2*block_idx -> c_idx = 2*block_idx*2 + 0
                # Local index 1: k_local=0, i=1 -> k_global=2*block_idx -> c_idx = 2*block_idx*2 + 1
                # Local index 2: k_local=1, i=0 -> k_global=2*block_idx+1 -> c_idx = (2*block_idx+1)*2 + 0
                # Local index 3: k_local=1, i=1 -> k_global=2*block_idx+1 -> c_idx = (2*block_idx+1)*2 + 1
                
                w_indices = [2*block_idx*2, 2*block_idx*2+1, (2*block_idx+1)*2, (2*block_idx+1)*2+1]
                for l in range(4):
                    W[w_indices[l]][col] = w[l]

        return {"U": U, "V": V, "W": W}

    # Try the direct construction
    result_28_direct = build_strassen_228_direct()
    if verify(result_28_direct["U"], result_28_direct["V"], result_28_direct["W"]):
        return result_28_direct

    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
