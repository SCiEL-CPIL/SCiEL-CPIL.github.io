"""
Program for 5x5 matrix multiplication tensor decomposition.
Uses a block decomposition of 5 = 2 + 2 + 1.
For 2x2x2 blocks, it uses Strassen's algorithm (rank 7).
For other blocks, it uses schoolbook multiplication.
Total rank: 8 * 7 (Strassen) + (125 - 8 * 8) (Schoolbook remainder) = 56 + 61 = 117.
"""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Block sizes for partitioning 5 = 2 + 2 + 1
    block_sizes = [2, 2, 1]
    
    # Block offsets
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)
    
    # Strassen decomposition for 2x2x2
    # Returns list of (U_vec, V_vec, W_vec)
    # Vectors have length 4, indices [0,1,2,3] correspond to [r0c0, r0c1, r1c0, r1c1]
    def get_strassen_factors():
        factors = []
        # M1 = (a11+a22)(b11+b22) -> c11(+1), c22(+1)
        factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        # M2 = (a21+a22)b11 -> c21(+1), c22(-1)
        factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
        # M3 = a11(b12-b22) -> c12(+1), c22(+1)
        factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
        # M4 = a22(b21-b11) -> c11(+1), c21(+1)
        factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
        # M5 = (a11+a12)b22 -> c11(-1), c12(+1)
        factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
        # M6 = (a21-a11)(b11+b12) -> c22(+1)
        factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        # M7 = (a12-a22)(b21+b22) -> c11(+1)
        factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
        return factors

    strassen_factors = get_strassen_factors()
    
    # Calculate total rank
    total_rank = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk
    
    # Allocate matrices
    # U: (n*m) x rank
    # V: (m*p) x rank
    # W: (p*n) x rank
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(3):
        if time.time() > deadline:
            break
        for j in range(3):
            if time.time() > deadline:
                break
            for k in range(3):
                if time.time() > deadline:
                    break
                    
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                
                # Global offsets
                off_i = offsets[i]
                off_j = offsets[j]
                off_k = offsets[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    # Use Strassen
                    for (uf, vf, wf) in strassen_factors:
                        # uf, vf, wf are lists of 4 floats
                        # Indices in local 2x2 block:
                        # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                        
                        # U indices
                        # local (lr, lc) -> global (off_i + lr, off_j + lc)
                        # U_idx = row * m + col
                        u_indices = [
                            (off_i + 0) * m + (off_j + 0),
                            (off_i + 0) * m + (off_j + 1),
                            (off_i + 1) * m + (off_j + 0),
                            (off_i + 1) * m + (off_j + 1)
                        ]
                        
                        # V indices
                        # local (lr, lc) -> global (off_j + lr, off_k + lc)
                        # V_idx = row * p + col
                        v_indices = [
                            (off_j + 0) * p + (off_k + 0),
                            (off_j + 0) * p + (off_k + 1),
                            (off_j + 1) * p + (off_k + 0),
                            (off_j + 1) * p + (off_k + 1)
                        ]
                        
                        # W indices
                        # local (lr, lc) -> global (off_k + lc, off_i + lr)  <-- Note: W is col-major in index k*n+i
                        # W_idx = col_C * n + row_C
                        # col_C = off_k + lc
                        # row_C = off_i + lr
                        w_indices = [
                            (off_k + 0) * n + (off_i + 0),
                            (off_k + 1) * n + (off_i + 0),
                            (off_k + 0) * n + (off_i + 1),
                            (off_k + 1) * n + (off_i + 1)
                        ]
                        
                        for idx in range(4):
                            if uf[idx] != 0:
                                U[u_indices[idx]][col] += uf[idx]
                            if vf[idx] != 0:
                                V[v_indices[idx]][col] += vf[idx]
                            if wf[idx] != 0:
                                W[w_indices[idx]][col] += wf[idx]
                        col += 1
                else:
                    # Schoolbook for this block
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                # U_idx
                                U_idx = (off_i + li) * m + (off_j + lj)
                                # V_idx
                                V_idx = (off_j + lj) * p + (off_k + lk)
                                # W_idx
                                W_idx = (off_k + lk) * n + (off_i + li)
                                
                                U[U_idx][col] = 1.0
                                V[V_idx][col] = 1.0
                                W[W_idx][col] = 1.0
                                col += 1
    
    # Verify col == total_rank
    # In case of timeout, we might have partial matrices, but the problem requires exact decomposition.
    # Given the speed, timeout is unlikely.
    if col != total_rank:
        # Fallback to schoolbook if interrupted? 
        # But we must return valid matrices. 
        # If interrupted, the matrices are incomplete/invalid.
        # However, 117 iterations is very fast.
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
