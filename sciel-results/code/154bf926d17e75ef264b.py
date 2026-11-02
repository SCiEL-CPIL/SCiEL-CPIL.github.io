"""Rewritten: find a small-rank exact integer decomposition for 3x4 @ 4x8."""
import itertools
import random
import time
import sys
import numpy as np
from fractions import Fraction
from functools import lru_cache

# Deterministic global random state for reproducibility
random.seed(20240801)

# Known small-rank bilinear multiplications
# Strassen-7 for 2x2 @ 2x2 (rank 7)
def strassen7(n, m, p):
    # Returns (U, V, W) of rank 7 for 2x2 @ 2x2.
    U = [
        [1, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]
    V = [
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
    ]
    W = [
        [1, 1, 0, 0],
        [1, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 1],
    ]
    Ua = np.array(U)
    Va = np.array(V)
    Wa = np.array(W)
    # Verify
    C_tensor = np.zeros((4, 4))
    for idx in range(7):
        Uv = Ua[:, idx]
        Vv = Va[:, idx]
        Wv = Wa[:, idx]
        # Reconstruct
        mat = np.zeros((4, 4))
        for a in range(4):
            for b in range(4):
                mat[a, b] += Uv[a] * Vv[b] * Wv[a + 2 * b] # wait Wv is 1-index in 2x2?
        # Check indices
    # Manual check:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C11 = M1 - M2 + M3 + M4
    # C12 = M5 + M2
    # C21 = M3 + M6
    # C22 = M1 - M4 + M6 + M7
    # M1 = (A21+A22)A22
    # M2 = (A11+A12)B11
    # M3 = (A11+A21)B11
    # M4 = (A12-A22)B22
    # M5 = (A11-A21)B12
    # M6 = (A12-A22)B21
    # M7 = (A11+A21)B22
    # Indices: U: 1,2,3,4. V: 5,6,7,8? No U (4x1), V (4x1), W (4x1).
    # Let's just use the standard Strassen matrices from literature.
    
    # Correct matrices (standard):
    U = [
        [0, 0, 1, 1, 1, 0, 1],
        [1, 1, 1, 0, 1, 0, 1],
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0]
    ]
    V = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 1]
    ]
    W = [
        [1, -1, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, -1, 0, 1, 1]
    ]
    return U, V, W

def schoolbook_block(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def combine_blocks(n, m, p, blocks):
    # blocks: list of (bn, bm, bp, U, V, W)
    rank_sum = 0
    U_list = []
    V_list = []
    W_list = []
    offsets_u = []
    for bn, bm, bp, U, V, W in blocks:
        offsets_u.append(rank_sum)
        rank_sum += U.shape[1]
    U = np.zeros((n * m, rank_sum))
    V = np.zeros((m * p, rank_sum))
    W = np.zeros((p * n, rank_sum))
    for idx, (bn, bm, bp, Ub, Vb, Wb) in enumerate(blocks):
        offset = offsets_u[idx]
        r = Ub.shape[1]
        for i in range(bn):
            for j in range(bm):
                U[i * m + j, offset:offset + r] = Ub[i * bm + j, :]
        for j in range(bm):
            for k in range(bp):
                V[j * p + k, offset:offset + r] = Vb[j * bp + k, :]
        for k in range(bp):
            for i in range(bn):
                W[k * n + i, offset:offset + r] = Wb[k * bn + i, :]
    return U, V, W

def is_decomp_correct(n, m, p, U, V, W):
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    
    # Construct tensor
    T = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        T[:, :, r] = np.outer(np.outer(U[:, r], V[:, r]), W[:, r]).reshape(n*m, m*p, p*n)
        # Check: T[i, j, k] = U[i, r] * V[j, r] * W[k, r]
    # Target tensor
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1
    if not np.allclose(T, T_target):
        return False
    return True

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100
    
    # Strategy:
    # 1. Try 2x4 @ 4x4 with rank 28 via (2x2, 2x2) and (2x2, 2x2) split.
    # Wait, m=4, p=8.
    # Blocks:
    # 3 x 4 x 8
    # 1. (3, 4, 4) -> rank?
    # 2. (3, 4, 4) -> rank?
    
    # Let's use Strassen (rank 7) for 2x2x2.
    # For 3x4x4:
    # Can we do it in rank 21?
    # Maybe 1x4x4 (rank 4) + 2x4x4?
    
    # Best shot:
    # Rank 21 for 2x4 @ 4x4. (Strassen for 2x4 @ 4x4 is 7*2 = 14? No.)
    # Split 2x4 @ 4x4 into (2x2, 2x2) and (2x2, 2x2)? No, m=4, p=4.
    # A(2x4), B(4x4).
    # C(2x4).
    # C_top_left (2x2) = A_top_left (2x2) B_top_left (2x2) + A_top_right (2x2) B_bottom_left (2x2)
    # C_top_right (2x2) = A_top_left (2x2) B_top_right (2x2) + A_top_right (2x2) B_bottom_right (2x2)
    # So we have 4 terms of 2x2@2x2. Each takes rank 7. Total 28.
    
    # 3x4x8:
    # Split p=8 into 4,4.
    # (3x4 @ 4x4) + (3x4 @ 4x4).
    # Each 3x4 @ 4x4:
    # Split n=3 into 1,2.
    # (1x4 @ 4x4) + (2x4 @ 4x4).
    # 1x4 @ 4x4:
    # A(1x4), B(4x4) -> C(1x4).
    # This is just dot product. Rank 4.
    # 2x4 @ 4x4:
    # Rank 28.
    # Total for 3x4 @ 4x4: 32.
    # Total for 3x4x8: 64.
    
    # Schoolbook is 96.
    # We can do better with Strassen.
    
    # Let's try to implement the block decomposition.
    
    U_sb, V_sb, W_sb = schoolbook_block(n, m, p)
    if is_decomp_correct(n, m, p, U_sb, V_sb, W_sb):
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = n * m * p
    else:
        # This should never fail
        raise Exception("Schoolbook failed?!")

    # Construct custom block decomposition
    # Rank 64 decomposition
    blocks = []
    
    # 1x4 @ 4x4 block (top row of A)
    # A is 1x4. B is 4x4. C is 1x4.
    # C = A @ B.
    # This is 4 dot products.
    # U: 1x4 x rank 4.
    U_1x4x4 = np.eye(4).reshape(1, 4, 4)[:, :, :] # wait
    # A is 1x4.
    # For j in 4: U[0, j, j] = 1
    # V[j, k, j] = 1
    # W[k, 0, j] = 1
    U1 = np.zeros((1 * 4, 4))
    V1 = np.zeros((4 * 4, 4))
    W1 = np.zeros((4 * 1, 4))
    for j in range(4):
        for k in range(4):
            # term j
            U1[0 * 4 + j, j] = 1
            V1[j * 4 + k, j] = 1
            W1[k * 1 + 0, j] = 1
            
    # 2x4 @ 4x4 block (bottom 2 rows of A)
    # This is 4 terms of 2x2 @ 2x2.
    # Strassen 7 for 2x2@2x2.
    U_s7, V_s7, W_s7 = strassen7(2, 2, 2)
    # Verify Strassen
    # U_s7 (4x7), V_s7 (4x7), W_s7 (4x7)
    if not is_decomp_correct(2, 2, 2, np.array(U_s7), np.array(V_s7), np.array(W_s7)):
        # Fallback schoolbook
        U_s7, V_s7, W_s7 = schoolbook_block(2, 2, 2)
    
    # We need 2x4 @ 4x4.
    # Split into (2,2,2) blocks.
    # A (2x4) = [A11 A12] (2x2 blocks)
    # B (4x4) = [B11 B12; B21 B22]
    # C (2x4) = [C11 C12] = [A11 B11 + A12 B21, A11 B12 + A12 B22]
    # So 4 multiplications: M11=A11B11, M12=A11B12, M21=A12B21, M22=A12B22.
    # Rank = 4 * 7 = 28.
    # U (8 x 28). V (16 x 28). W (8 x 28).
    U_2x4x4 = np.zeros((8, 28))
    V_2x4x4 = np.zeros((16, 28))
    W_2x4x4 = np.zeros((8, 28))
    
    for term_idx in range(4):
        offset = term_idx * 7
        # term 0: A11 (rows 0,1) B11 (cols 0,1) -> C11 (cols 0,1)
        # A rows 0,1. B rows 0,1. C cols 0,1.
        # In 2x4@4x4:
        # A is 2x4. A11 is cols 0,1.
        # B is 4x4. B11 is rows 0,1, cols 0,1.
        # C is 2x4. C11 is cols 0,1.
        
        # U index in 2x4x4: i (0,1) in 2x4? No, i in 2x2.
        # Let's map carefully.
        # A(2x4), B(4x4), C(2x4).
        # Term: A_sub (2x2) @ B_sub (2x2) -> C_sub (2x2).
        # A_sub uses A's cols ca..ca+1.
        # B_sub uses B's rows cb..cb+1 and cols cc..cc+1.
        # C_sub uses C's cols cc..cc+1.
        
        terms = [
            (0, 1, 0, 1), # A cols 0,1. B rows 0,1, cols 0,1. C cols 0,1.
            (0, 1, 2, 3), # A cols 0,1. B rows 0,1, cols 2,3. C cols 2,3.
            (2, 3, 0, 1), # A cols 2,3. B rows 2,3, cols 0,1. C cols 0,1.
            (2, 3, 2, 3), # A cols 2,3. B rows 2,3, cols 2,3. C cols 2,3.
        ]
        for idx, (ca, cb, cc) in enumerate([(0,0,0), (0,1,1), (1,0,0), (1,1,1)]):
            offset = idx * 7
            # A_sub: cols ca*2..ca*2+1
            # B_sub: rows cb*2..cb*2+1, cols cc*2..cc*2+1
            # C_sub: cols cc*2..cc*2+1
            # U: A is 2x4. U indices 0..7.
            for u_i in range(4):
                row_a = u_i // 2 # 0 or 1
                col_a = u_i % 2  # 0 or 1
                col_a_2x4 = ca * 2 + col_a
                U_2x4x4[row_a * 4 + col_a_2x4, offset:offset+7] = U_s7[u_i, :]
            
            for v_i in range(4):
                row_b = v_i // 2
                col_b = v_i % 2
                row_b_4x4 = cb * 2 + row_b
                col_b_4x4 = cc * 2 + col_b
                V_2x4x4[row_b_4x4 * 4 + col_b_4x4, offset:offset+7] = V_s7[v_i, :]
            
            for w_i in range(4):
                row_c = w_i // 2
                col_c = w_i % 2
                col_c_2x4 = cc * 2 + col_c
                W_2x4x4[row_c * 4 + col_c_2x4, offset:offset+7] = W_s7[w_i, :]
                
    # Combine 1x4x4 and 2x4x4 to form 3x4x4
    # Rank 4 + 28 = 32.
    U_3x4x4 = np.zeros((12, 32))
    V_3x4x4 = np.zeros((16, 32))
    W_3x4x4 = np.zeros((12, 32))
    
    # 1x4x4 part (rows 0 of A, rows 0 of C)
    U_3x4x4[0:4, 0:4] = U1
    V_3x4x4[:, 0:4] = V1
    W_3x4x4[0:4, 0:4] = W1
    
    # 2x4x4 part (rows 1,2 of A, rows 1,2 of C)
    U_3x4x4[4:12, 4:32] = U_2x4x4
    V_3x4x4[:, 4:32] = V_2x4x4
    W_3x4x4[4:12, 4:32] = W_2x4x4 # Wait W is (p*n, rank) = (4*3, 32) = (12, 32).
    # W index is k*n+i. k in 0..3, i in 0..2.
    # W_2x4x4 index is k*2+i? No W is 2x4. (p=4, n=2). W shape (8, 28).
    # W_3x4x4 shape (12, 32).
    # We need to map W_2x4x4 (k,i where i in 0,1) to W_3x4x4 (k, i' where i' in 1,2).
    # W_2x4x4 index = k*2 + i.
    # W_3x4x4 index = k*3 + (i+1).
    
    for k in range(4):
        for i in range(2):
            W_3x4x4[k * 3 + (i+1), 4:32] = W_2x4x4[k * 2 + i, :]
            
    # 3x4x4 done. Rank 32.
    
    # 3x4x8: Split p=8 into 4,4.
    # Two blocks of 3x4x4.
    # Block 1: p=4 (cols 0..3 of B, cols 0..3 of C)
    # Block 2: p=4 (cols 4..7 of B, cols 4..7 of C)
    U_3x4x8 = np.zeros((12, 64))
    V_3x4x8 = np.zeros((32, 64))
    W_3x4x8 = np.zeros((24, 64))
    
    # Block 1
    U_3x4x8[:, 0:32] = U_3x4x4
    # V is 4x8. Rows 0..3.
    V_3x4x8[0:16, 0:32] = V_3x4x4
    # W is 4x3. k in 0..3, i in 0..2.
    # W_3x4x8 index k*3+i.
    W_3x4x8[0:12, 0:32] = W_3x4x4
    
    # Block 2
    U_3x4x8[:, 32:64] = U_3x4x4
    # V is 4x8. Rows 4..7? No V is (m*p, rank) = (4*8, 64) = (32, 64).
    # V index j*p+k. j in 0..3, k in 0..7.
    # Block 2 uses k=4..7.
    # V_3x4x4 index j*4+k.
    # V_3x4x8 index j*8+k.
    for j in range(4):
        for k in range(4):
            V_3x4x8[j * 8 + (k+4), 32:64] = V_3x4x4[j * 4 + k, :]
            
    # W is 8x3. k in 0..7.
    # W_3x4x8 index k*3+i.
    # Block 2 uses k=4..7.
    # W_3x4x4 index k*3+i (k in 0..3).
    for k in range(4):
        for i in range(3):
            W_3x4x8[(k+4) * 3 + i, 32:64] = W_3x4x4[k * 3 + i, :]
            
    # Verify 3x4x8
    if is_decomp_correct(3, 4, 8, U_3x4x8, V_3x4x8, W_3x4x8):
        if 64 < best_rank:
            best_U, best_V, best_W = U_3x4x8, V_3x4x8, W_3x4x8
            best_rank = 64

    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
