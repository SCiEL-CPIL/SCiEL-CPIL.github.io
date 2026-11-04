"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Optimization: Use vectorized check for speed
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We check if T is all 1s.
    # Since rank might be small, we can just loop.
    # But let's use numpy einsum for verification if rank is not huge.
    # Actually, simple loop is fine for 96 elements.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_decomposition(n, m, p, i_blocks, j_blocks, k_blocks):
    """
    Builds the U, V, W matrices for a given block structure.
    i_blocks: list of block sizes for dim n (e.g., [2, 1])
    j_blocks: list of block sizes for dim m (e.g., [2, 2])
    k_blocks: list of block sizes for dim p (e.g., [2, 2, 2, 2])
    
    Returns (U, V, W, rank) or None if invalid.
    """
    # Calculate total rank
    rank = 0
    for bi in i_blocks:
        for bj in j_blocks:
            for bk in k_blocks:
                if bi == 2 and bj == 2 and bk == 2:
                    rank += 7
                else:
                    rank += bi * bj * bk
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    i_offset = 0
    for bi in i_blocks:
        j_offset = 0
        for bj in j_blocks:
            k_offset = 0
            for bk in k_blocks:
                # Determine block type
                if bi == 2 and bj == 2 and bk == 2:
                    # Strassen Block
                    # Local indices:
                    # A: i_offset, i_offset+1 ; j_offset, j_offset+1
                    # B: j_offset, j_offset+1 ; k_offset, k_offset+1
                    # C: i_offset, i_offset+1 ; k_offset, k_offset+1
                    
                    # Terms: (A_coeffs, B_coeffs, C_coeffs)
                    # A index order: [0,0], [0,1], [1,0], [1,1] -> idx = di*2+dj
                    # B index order: [0,0], [0,1], [1,0], [1,1] -> idx = dj*2+dk
                    # C index order: [0,0], [0,1], [1,0], [1,1] -> idx = di*2+dk (Wait, W row is k*n+i, so local C index is dk*2+di? No.)
                    
                    # Let's re-verify the W index mapping in the original code.
                    # Original: W[(k0+dk)*n + (i0+di)]
                    # So W row index is determined by global k and i.
                    # In the block, local k is dk, local i is di.
                    # So W row is (k_offset + dk)*n + (i_offset + di).
                    
                    # The 7 terms from the original code:
                    terms = [
                        # p1 = (a+b)(e+f)
                        ([1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]), # C coeffs: [C00, C01, C10, C11] ? 
                        # Original code: W[(k0+dk)*n + (i0+di)][col] = c_coeffs[di*2+dk]
                        # So c_coeffs index is di*2+dk.
                        # C00: di=0, dk=0 -> idx 0
                        # C01: di=0, dk=1 -> idx 1
                        # C10: di=1, dk=0 -> idx 2
                        # C11: di=1, dk=1 -> idx 3
                        # So c_coeffs list is [C00, C01, C10, C11]
                        
                        ([1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]), # p1
                        ([0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 1, 1]), # p2
                        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 1, 0]), # p3
                        ([1, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1]), # p4
                        ([1, 0, 0, 0], [1, 0, -1, 0], [1, 1, 0, 0]), # p5
                        ([0, 0, 0, 1], [0, -1, 0, 1], [1, 0, 1, 0]), # p6
                        ([0, 1, 0, -1], [1, 0, 1, 0], [1, 0, 0, 1]), # p7
                    ]
                    
                    for t in range(7):
                        a_c = terms[t][0]
                        b_c = terms[t][1]
                        c_c = terms[t][2]
                        
                        for di in range(2):
                            for dj in range(2):
                                U[(i_offset+di)*m + (j_offset+dj)][col] = a_c[di*2+dj]
                        for dj in range(2):
                            for dk in range(2):
                                V[(j_offset+dj)*p + (k_offset+dk)][col] = b_c[dj*2+dk]
                        for di in range(2):
                            for dk in range(2):
                                W[(k_offset+dk)*n + (i_offset+di)][col] = c_c[di*2+dk]
                        col += 1
                else:
                    # Schoolbook Block
                    for di in range(bi):
                        for dj in range(bj):
                            for dk in range(bk):
                                U[(i_offset+di)*m + (j_offset+dj)][col] = 1
                                V[(j_offset+dj)*p + (k_offset+dk)][col] = 1
                                W[(k_offset+dk)*n + (i_offset+di)][col] = 1
                                col += 1
                k_offset += bk
            j_offset += bj
        i_offset += bi
        
    if col != rank:
        return None
        
    return U, V, W, rank

def _get_block_partitions(dim):
    """
    Generate all valid partitions of 'dim' into blocks of size 1 or 2.
    Returns a list of lists, e.g., for 3: [[2,1], [1,2], [1,1,1]]
    """
    def gen(rem, current):
        if rem == 0:
            yield list(current)
            return
        if rem >= 2:
            current.append(2)
            yield from gen(rem-2, current)
            current.pop()
        if rem >= 1:
            current.append(1)
            yield from gen(rem-1, current)
            current.pop()
            
    return list(gen(dim, []))

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 100
    
    # Generate all possible block structures
    i_parts = _get_block_partitions(n)
    j_parts = _get_block_partitions(m)
    k_parts = _get_block_partitions(p)
    
    best_rank = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Iterate through all combinations
    # n=3 -> 3 options
    # m=4 -> 5 options (2+2, 2+1+1, 1+2+1, 1+1+2, 1+1+1+1)
    # p=8 -> 21 options (Fibonacci-like)
    # Total ~ 3 * 5 * 21 = 315 combinations. Very fast.
    
    for i_b in i_parts:
        for j_b in j_parts:
            for k_b in k_parts:
                if time.time() > deadline:
                    break
                
                res = _build_decomposition(n, m, p, i_b, j_b, k_b)
                if res is None:
                    continue
                U, V, W, rank = res
                
                # Verify
                if _verify(U, V, W, n, m, p):
                    if rank < best_rank:
                        best_rank = rank
                        best_U = U
                        best_V = V
                        best_W = W
                        # print(f"Found rank {rank} with blocks {i_b}, {j_b}, {k_b}")

    if best_U is not None:
        return {"U": best_U, "V": best_V, "W": best_W}
        
    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
