import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    def build_strassen_2x2_blocks():
        """Standard Strassen on 2x2 blocks. Rank 189."""
        block_size = 2
        blocks_per_dim = 3
        rank = (blocks_per_dim ** 3) * 7
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        strassen_patterns = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]

        col_idx = 0
        for I in range(blocks_per_dim):
            for J in range(blocks_per_dim):
                for K in range(blocks_per_dim):
                    for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                        for (pi, pj), val in u_coeffs.items():
                            r = I * block_size + pi
                            c = J * block_size + pj
                            U[r * m + c, col_idx] += val
                        for (pi, pj), val in v_coeffs.items():
                            r = J * block_size + pi
                            c = K * block_size + pj
                            V[r * p + c, col_idx] += val
                        for (pi, pj), val in w_coeffs.items():
                            k = K * block_size + pj
                            i = I * block_size + pi
                            W[k * n + i, col_idx] += val
                        col_idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    def verify_decomposition(U, V, W):
        """Verify if the decomposition is correct."""
        # Check a few random cases
        for _ in range(5):
            A = np.random.randint(-10, 10, (n, m))
            B = np.random.randint(-10, 10, (m, p))
            C_true = A @ B
            
            # Compute C using decomposition
            # C_ik = sum_j sum_r U_ij,r V_jk,r W_ki,r
            # Note: W index is k*n + i
            C_comp = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    s = 0
                    for j in range(m):
                        for r in range(len(U[0])):
                            s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    C_comp[i, k] = s
            if not np.allclose(C_true, C_comp):
                return False
        return True

    def build_laderman_6x6():
        """Attempt Strassen on 2x2 blocks with Laderman 3x3 base. Rank 161."""
        # This requires valid Laderman patterns for 3x3.
        # Since I am not 100% confident in the hardcoded patterns for full 3x3,
        # I will skip this to avoid returning an invalid solution.
        # If I had verified patterns, I would construct U, V, W here.
        return None

    # Try Laderman first
    if time.time() < deadline:
        sol_lad = build_laderman_6x6()
        if sol_lad is not None:
            if verify_decomposition(np.array(sol_lad['U']), np.array(sol_lad['V']), np.array(sol_lad['W'])):
                return sol_lad

    # Fallback to Strassen 189
    sol, rank = build_strassen_2x2_blocks()
    return sol
# EVOLVE-BLOCK-END
