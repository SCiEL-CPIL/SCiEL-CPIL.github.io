import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]


def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Verify using numpy einsum for speed."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64).reshape(n, m, rank)
    V_arr = np.array(V, dtype=np.float64).reshape(m, p, rank)
    W_arr = np.array(W, dtype=np.float64).reshape(p, n, rank)
    reconstructed = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    expected = np.ones((n, m, p), dtype=np.float64)
    return np.allclose(expected, reconstructed, atol=tol)


def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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
    return U, V, W


def _build_4x4x5_strassen_boundary():
    """Build rank-44 decomposition for 4x4 x 4x5 using Strassen(4x4x4) + schoolbook(4x4x4x1)."""
    n, m, p = 4, 4, 5
    
    # Strassen 2x2 terms
    U44 = [[0] * 28 for _ in range(16)]
    V44 = [[0] * 28 for _ in range(16)]
    W44 = [[0] * 28 for _ in range(16)]
    
    # Strassen 2x2 block structure
    # Term definitions: (U_blocks, V_blocks, W_blocks) where each is list of (block_row, block_col, coeff)
    strassen_terms = [
        # m0: (a00+a11)(b00+b11) -> c00+c11
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # m1: (a10+a11)b00 -> c10-c00... wait let me use standard Strassen
        # Standard Strassen for 2x2:
        # m0 = (a00+a11)(b00+b11)
        # m1 = (a10+a11)b00
        # m2 = a00(b01-b11)
        # m3 = a11(b10-b00)
        # m4 = (a00+a01)b11
        # m5 = (a10-a00)(b00+b01)
        # m6 = (a01-a11)(b10+b11)
        # c00 = m0+m3-m4-m6
        # c01 = m2+m4
        # c10 = m1+m5
        # c11 = m0-m1+m2+m5
    ]
    
    # Let me just directly construct the 4x4x4 Strassen decomposition
    # Split into 2x2 blocks. Block (br, bc) covers rows br*2..br*2+1, cols bc*2..bc*2+1
    
    U_blocks = [
        [(0,0,1),(1,1,1)],   # m0: a00+a11
        [(1,0,1),(1,1,1)],   # m1: a10+a11
        [(0,0,1)],           # m2: a00
        [(1,1,1)],           # m3: a11
        [(0,0,1),(0,1,1)],   # m4: a00+a01
        [(1,0,1),(0,0,-1)],  # m5: a10-a00
        [(0,1,1),(1,1,-1)],  # m6: a01-a11
    ]
    V_blocks = [
        [(0,0,1),(1,1,1)],   # m0: b00+b11
        [(0,0,1)],           # m1: b00
        [(0,1,1),(1,1,-1)],  # m2: b01-b11
        [(1,0,1),(0,0,-1)],  # m3: b10-b00
        [(1,1,1)],           # m4: b11
        [(0,0,1),(0,1,1)],   # m5: b00+b01
        [(1,0,1),(1,1,1)],   # m6: b10+b11
    ]
    W_blocks = [
        [(0,0,1),(1,1,1)],   # m0: c00+c11
        [(1,0,1),(1,1,-1)],  # m1: c10-c11... 
        # Actually W gives output C. Let me be more careful.
        # c00 = m0+m3-m4-m6
        # c01 = m2+m4
        # c10 = m1+m5
        # c11 = m0-m1+m2+m5
    ]
    
    # W block contributions for each output block (br_out, bc_out):
    # c00: m0(+1), m3(+1), m4(-1), m6(-1)
    # c01: m2(+1), m4(+1)
    # c10: m1(+1), m5(+1)
    # c11: m0(+1), m1(-1), m2(+1), m5(+1)
    
    W_contribs = {
        (0,0): [(0,1),(3,1),(4,-1),(6,-1)],
        (0,1): [(2,1),(4,1)],
        (1,0): [(1,1),(5,1)],
        (1,1): [(0,1),(1,-1),(2,1),(5,1)],
    }
    
    for s in range(7):
        for lr in range(2):
            for lc in range(2):
                c = s * 4 + lr * 2 + lc
                # U
                for (br, bc, coeff) in U_blocks[s]:
                    for lr2 in range(2):
                        for lc2 in range(2):
                            i = br * 2 + lr2
                            j = bc * 2 + lc2
                            U44[i * 4 + j][c] += coeff
                # V
                for (br, bc, coeff) in V_blocks[s]:
                    for lr2 in range(2):
                        for lc2 in range(2):
                            j = br * 2 + lr2
                            k = bc * 2 + lc2
                            V44[j * 4 + k][c] += coeff
                # W
                for (br_out, bc_out, coeff) in W_contribs.get((s % 2, s // 2 % 2), []):
                    pass  # This is getting complicated, let me just verify the standard construction
    
    # Let me use a simpler verified approach: just do schoolbook for 4x4x4 and add boundary
    # Actually, let me just build the rank-44 directly and verify
    rank = 44
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(20)]
    W = [[0] * rank for _ in range(20)]
    
    # For now, use schoolbook for 4x4x4 (rank 64) - no, that's too high
    # Let me just use the known rank-44 construction
    
    # Actually, the simplest correct approach: schoolbook for all 80 terms
    # But we want rank 44. Let me construct it properly.
    
    # 4x4x4 Strassen (rank 28) for columns 0-3
    # 4x4x4x1 schoolbook (rank 16) for column 4
    
    # I'll construct the 4x4x4 Strassen by composing 2x2 Strassen
    # Each 2x2 Strassen has rank 7. For 4x4, we have 4 blocks of 2x2 in A and B.
    # The block-level multiplication is 2x2 x 2x2 -> 2x2, done via Strassen (rank 7).
    # But each "element" of the block matrices is itself a 2x2 matrix, so we need
    # to multiply 2x2 matrices, which takes rank 4 (schoolbook) or rank 7 (Strassen).
    # Using schoolbook for inner: rank = 7 * 4 = 28
    
    # Block (br, bc) of A covers rows [br*2, br*2+1], cols [bc*2, bc*2+1]
    # Block (br, bc) of B covers rows [br*2, br*2+1], cols [bc*2, bc*2+1]
    # Block (br, bc) of C covers rows [br*2, br*2+1], cols [bc*2, bc*2+1]
    
    # Strassen at block level:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    # C00 = M0+M3-M4-M6
    # C01 = M2+M4
    # C10 = M1+M5
    # C11 = M0-M1+M2+M5
    
    # For each Strassen term s, and each inner schoolbook term (lr,lc) for the 2x2 multiply:
    # U[i,j,c] where i = br_A*2+lr_A, j = bc_A*2+lc_A
    # V[j,k,c] where j = br_B*2+lr_B, k = bc_B*2+lc_B  
    # W[k,i,c] where k = br_C*2+lr_C, i = bc_C*2+lc_C
    
    # For term s, the A-blocks involved, B-blocks involved, C-blocks involved:
    A_block_terms = [
        [(0,0,1),(1,1,1)],   # M0
        [(1,0,1),(1,1,1)],   # M1
        [(0,0,1)],           # M2
        [(1,1,1)],           # M3
        [(0,0,1),(0,1,1)],   # M4
        [(1,0,1),(0,0,-1)],  # M5
        [(0,1,1),(1,1,-1)],  # M6
    ]
    B_block_terms = [
        [(0,0,1),(1,1,1)],   # M0
        [(0,0,1)],           # M1
        [(0,1,1),(1,1,-1)],  # M2
        [(1,0,1),(0,0,-1)],  # M3
        [(1,1,1)],           # M4
        [(0,0,1),(0,1,1)],   # M5
        [(1,0,1),(1,1,1)],   # M6
    ]
    C_block_terms = [
        [(0,0,1),(1,1,1)],   # M0
        [(1,0,1),(1,1,-1)],  # M1: C10 += M1, C11 -= M1
        [(0,1,1),(1,1,1)],   # M2: C01 += M2, C11 += M2
        [(0,0,1),(1,0,1)],   # M3: C00 += M3, C10... wait
        # Let me redo C contributions:
        # C00: M0(+1), M3(+1), M4(-1), M6(-1)
        # C01: M2(+1), M4(+1)
        # C10: M1(+1), M5(+1)
        # C11: M0(+1), M1(-1), M2(+1), M5(+1)
    ]
    
    C_block_terms = [
        [(0,0,1),(1,1,1)],   # M0: C00+=M0, C11+=M0
        [(1,0,1),(1,1,-1)],  # M1: C10+=M1, C11-=M1
        [(0,1,1),(1,1,1)],   # M2: C01+=M2, C11+=M2
        [(0,0,1)],           # M3: C00+=M3
        [(0,0,-1),(0,1,1)],  # M4: C00-=M4, C01+=M4
        [(1,0,1),(1,1,1)],   # M5: C10+=M5, C11+=M5
        [(0,0,-1)],          # M6: C00-=M6
    ]
    
    for s in range(7):
        for lr in range(2):
            for lc in range(2):
                c = s * 4 + lr * 2 + lc
                for (br_A, bc_A, coeff_A) in A_block_terms[s]:
                    for lr_A in range(2):
                        for lc_A in range(2):
                            i = br_A * 2 + lr_A
                            j = bc_A * 2 + lc_A
                            U[i * 4 + j][c] += coeff_A
                for (br_B, bc_B, coeff_B) in B_block_terms[s]:
                    for lr_B in range(2):
                        for lc_B in range(2):
                            j = br_B * 2 + lr_B
                            k = bc_B * 2 + lc_B
                            V[j * 5 + k][c] += coeff_B  # k is 0-3 for 4x4 part
                for (br_C, bc_C, coeff_C) in C_block_terms[s]:
                    for lr_C in range(2):
                        for lc_C in range(2):
                            k = br_C * 2 + lr_C
                            i = bc_C * 2 + lc_C
                            W[k * 4 + i][c] += coeff_C
    
    # Boundary: column 4 of B (k=4)
    # C[i,4] = sum_j A[i,j] * B[j,4]
    for i in range(4):
        for j in range(4):
            c = 28 + i * 4 + j
            U[i * 4 + j][c] = 1
            V[j * 5 + 4][c] = 1
            W[4 * 4 + i][c] = 1
    
    return U, V, W


def _try_reduce_rank(U, V, W, n, m, p, max_iter=500, tol=1e-9):
    """Attempt to reduce rank by removing provably-redundant rank-1 terms."""
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    original_rank = rank
    start_time = time.time()
    
    for _ in range(max_iter):
        if time.time() - start_time > 100:
            break
        removed = False
        for c in range(rank - 1, -1, -1):
            U_test = [row[:c] + row[c + 1:] for row in U]
            V_test = [row[:c] + row[c + 1:] for row in V]
            W_test = [row[:c] + row[c + 1:] for row in W]
            if _verify_decomposition_fast(U_test, V_test, W_test, n, m, p, tol):
                U = U_test
                V = V_test
                W = W_test
                rank -= 1
                removed = True
                break
        if not removed:
            break
    
    if rank < original_rank:
        return U, V, W, rank
    return None


def _random_search(U_init, V_init, W_init, n, m, p, target_rank, time_limit=30, seed=42):
    """Random search for lower rank decomposition."""
    rng = random.Random(seed)
    start_time = time.time()
    best = (len(U_init[0]), U_init, V_init, W_init)
    
    while time.time() - start_time < time_limit:
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        for c in range(target_rank):
            for i in range(n * m):
                U[i][c] = rng.choice([-1, -0.5, 0, 0.5, 1])
            for i in range(m * p):
                V[i][c] = rng.choice([-1, -0.5, 0, 0.5, 1])
            for i in range(p * n):
                W[i][c] = rng.choice([-1, -0.5, 0, 0.5, 1])
        
        if _verify_decomposition_fast(U, V, W, n, m, p):
            return target_rank, U, V, W
    
    return best


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Build structured 4x4x5 Strassen + boundary
    if time.time() < deadline:
        U_struct, V_struct, W_struct = _build_4x4x5_strassen_boundary()
        if _verify_decomposition_fast(U_struct, V_struct, W_struct, n, m, p):
            best_rank = len(U_struct[0])
            best_result = {"U": U_struct, "V": V_struct, "W": W_struct}
            
            # Attempt rank reduction
            if time.time() < deadline:
                reduced = _try_reduce_rank(U_struct, V_struct, W_struct, n, m, p, max_iter=500)
                if reduced is not None:
                    U_r, V_r, W_r, rank_r = reduced
                    if rank_r < best_rank and _verify_decomposition_fast(U_r, V_r, W_r, n, m, p):
                        best_rank = rank_r
                        best_result = {"U": U_r, "V": V_r, "W": W_r}
    
    # Try random search for lower ranks
    if time.time() < deadline and best_rank > 30:
        target_ranks = [42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16]
        for target_rank in target_ranks:
            if time.time() > deadline - 10:
                break
            if target_rank >= best_rank:
                continue
            
            found_rank, U_found, V_found, W_found = _random_search(
                best_result["U"], best_result["V"], best_result["W"],
                n, m, p, target_rank, time_limit=5, seed=42 + target_rank
            )
            if found_rank < best_rank:
                best_rank = found_rank
                best_result = {"U": U_found, "V": V_found, "W": W_found}
                if best_rank <= 20:
                    break
    
    return best_result
