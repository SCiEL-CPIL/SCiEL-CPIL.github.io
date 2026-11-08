"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard: the evaluator kills at 150s wall-clock.
    # We return a valid result long before that.
    _t0 = time.time()
    _deadline = _t0 + 145.0  # 5s safety margin

    n, m, p = DIMENSIONS
    # Using Strassen decomposition for 2x2 blocks.
    # 6x6 matrix is 3x3 blocks of 2x2.
    # 27 block multiplications, each of rank 7.
    # Total rank = 27 * 7 = 189.
    
    block_size = 2
    blocks_per_dim = 3 # 6 / 2

    rank = (blocks_per_dim ** 3) * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Precompute block offsets for speed
    offsets = [i * block_size for i in range(blocks_per_dim)]
    
    # Strassen patterns for 2x2 matrices
    # Each pattern is a tuple: (u_coeffs, v_coeffs, w_coeffs)
    # coeffs are dicts mapping (row, col) to value
    strassen_patterns = [
        # M0 = (A00+A11)(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M1 = (A10+A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M2 = A00(B01-B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M3 = A11(B10-B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M4 = (A00+A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M5 = (A10-A00)(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M6 = (A01-A11)(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        off_I = offsets[I]
        for J in range(blocks_per_dim):
            off_J = offsets[J]
            for K in range(blocks_per_dim):
                off_K = offsets[K]
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    # Update U
                    for (pi, pj), val in u_coeffs.items():
                        r = off_I + pi
                        c = off_J + pj
                        U[r * m + c][col_idx] += val

                    # Update V
                    for (pi, pj), val in v_coeffs.items():
                        r = off_J + pi
                        c = off_K + pj
                        V[r * p + c][col_idx] += val

                    # Update W
                    for (pi, pj), val in w_coeffs.items():
                        # W index is k*n + i
                        # k is col of C (which is K block, pj local)
                        # i is row of C (which is I block, pi local)
                        k = off_K + pj
                        i = off_I + pi
                        W[k * n + i][col_idx] += val

                    col_idx += 1

                # Periodic deadline check (every block-triple iteration)
                if time.time() > _deadline:
                    # Return what we have; rank is fixed so partial fill
                    # would be invalid. In practice this never triggers
                    # because the loop is ~27*7 = 189 iterations of
                    # small dict loops, finishing in < 1s.
                    break
            if time.time() > _deadline:
                break
        if time.time() > _deadline:
            break

    # Final safety: if we somehow didn't finish, return the full
    # (already-constructed) result. The loop is deterministic and fast.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
