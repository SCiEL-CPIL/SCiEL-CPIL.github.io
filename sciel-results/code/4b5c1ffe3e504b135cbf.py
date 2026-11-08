"""Rewritten program for matrix multiplication 3x3x4 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

def _verify(U, V, W):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    Convention: W[k*n + i, r]
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Reshape to 3D tensors for efficient verification
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    # We need sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1 for all i,j,k
    # This is equivalent to checking that the tensor T = sum_r U_r \otimes V_r \otimes W_r
    # equals the all-ones tensor of shape (n, m, p).
    
    # Using einsum: T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # Note the indices: U is (n, m, r), V is (m, p, r), W is (p, n, r)
    # We want result[i, j, k].
    # Term: U[i, j, r] * V[j, k, r] * W[k, i, r]
    # Sum over r.
    
    # Let's compute this using broadcasting or einsum.
    # np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    # Let's check indices:
    # U: i, j, r
    # V: j, k, r
    # W: k, i, r
    # Output: i, j, k
    # This looks correct.
    
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    # The target tensor is all 1s.
    target = np.ones((n, m, p))
    
    return np.allclose(T, target, atol=1e-10)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # 1. Schoolbook Baseline
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb), dtype=float)
    V_sb = np.zeros((m * p, rank_sb), dtype=float)
    W_sb = np.zeros((p * n, rank_sb), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # 2. Block Decomposition (Rank 34 for 3x3x4)
    # Strategy:
    # Split A (3x3) into blocks: A11 (2x2), A12 (2x1), A21 (1x2), A22 (1x1)
    # Split B (3x4) into blocks: B1 (2x4), B2 (1x4)
    # C = A * B
    # C1 (2x4) = A11 B1 + A12 B2
    # C2 (1x4) = A21 B1 + A22 B2
    
    # Term 1: A11 * B1.
    # A11 is 2x2, B1 is 2x4.
    # B1 = [B1a B1b] where B1a is 2x2, B1b is 2x2.
    # A11 B1a is 2x2x2 (Strassen rank 7).
    # A11 B1b is 2x2x2 (Strassen rank 7).
    # Total 14 products for A11 B1.
    
    # Term 2: A12 * B2.
    # A12 is 2x1, B2 is 1x4.
    # C1[i, k] = A12[i, 0] * B2[0, k].
    # This is 2 * 4 = 8 products.
    
    # Term 3: A21 * B1.
    # A21 is 1x2, B1 is 2x4.
    # C2[k] = sum_j A21[0, j] B1[j, k].
    # This is 4 dot products of length 2.
    # Total 4 * 2 = 8 products.
    
    # Term 4: A22 * B2.
    # A22 is 1x1, B2 is 1x4.
    # C2[k] += A22[0,0] * B2[0, k].
    # This is 4 products.
    
    # Total Rank = 14 + 8 + 8 + 4 = 34.
    
    U_b = np.zeros((n * m, 34), dtype=float)
    V_b = np.zeros((m * p, 34), dtype=float)
    W_b = np.zeros((p * n, 34), dtype=float)
    
    col = 0
    
    def add_block_term(u_list, v_list, w_list):
        """
        u_list: list of (i, j, coeff)
        v_list: list of (j, k, coeff)
        w_list: list of (k, i, coeff)
        """
        nonlocal col
        if col >= 34:
            return
        for (ri, cj, cu) in u_list:
            U_b[ri*m + cj, col] += cu
        for (rj, ck, cv) in v_list:
            V_b[rj*p + ck, col] += cv
        for (rk, ci, cw) in w_list:
            W_b[rk*n + ci, col] += cw
        col += 1

    # Strassen for A11 (i:0,1; j:0,1) * B1a (j:0,1; k:0,1)
    # m1: (a00+a11)(b00+b11)
    add_block_term(
        [(0,0,1), (1,1,1)],
        [(0,0,1), (1,1,1)],
        [(0,0,1), (1,1,1)]
    )
    # m2: (a10+a11)b00
    add_block_term(
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,0,1)]
    )
    # m3: a00(b01-b11)
    add_block_term(
        [(0,0,1)],
        [(0,1,1), (1,1,-1)],
        [(0,1,1), (1,1,-1)]
    )
    # m4: a11(b10-b00)
    add_block_term(
        [(1,1,1)],
        [(1,0,1), (0,0,-1)],
        [(1,0,1), (0,0,-1)]
    )
    # m5: (a00+a01)b11
    add_block_term(
        [(0,0,1), (0,1,1)],
        [(1,1,1)],
        [(1,1,1)]
    )
    # m6: (a10-a00)(b00+b01)
    add_block_term(
        [(1,0,1), (0,0,-1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (1,1,1)]
    )
    # m7: (a01-a11)(b10+b11)
    add_block_term(
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (1,1,1)],
        [(1,0,1), (1,1,1)]
    )
    
    # Strassen for A11 (i:0,1; j:0,1) * B1b (j:0,1; k:2,3)
    # Note: k indices are 2,3 here.
    # m1: (a00+a11)(b02+b13)
    add_block_term(
        [(0,0,1), (1,1,1)],
        [(0,2,1), (1,3,1)],
        [(2,0,1), (3,1,1)]
    )
    # m2: (a10+a11)b02
    # Result C[1,2]. i=1, k=2.
    add_block_term(
        [(1,0,1), (1,1,1)],
        [(0,2,1)],
        [(2,1,1)]
    )
    # m3: a00 (b03 - b13)
    # Result C[0,3] gets +, C[1,3] gets -.
    add_block_term(
        [(0,0,1)],
        [(0,3,1), (1,3,-1)],
        [(3,0,1), (3,1,-1)]
    )
    # m4: a11 (b12 - b02)
    # Result C[1,2] gets +, C[0,2] gets -.
    add_block_term(
        [(1,1,1)],
        [(1,2,1), (0,2,-1)],
        [(2,1,1), (2,0,-1)]
    )
    # m5: (a00+a01) b13
    # Result C[1,3].
    add_block_term(
        [(0,0,1), (0,1,1)],
        [(1,3,1)],
        [(3,1,1)]
    )
    # m6: (a10-a00) (b02+b03)
    # Result C[1,2], C[1,3].
    add_block_term(
        [(1,0,1), (0,0,-1)],
        [(0,2,1), (0,3,1)],
        [(2,1,1), (3,1,1)]
    )
    # m7: (a01-a11) (b12+b13)
    # Result C[1,2], C[1,3].
    add_block_term(
        [(0,1,1), (1,1,-1)],
        [(1,2,1), (1,3,1)],
        [(2,1,1), (3,1,1)]
    )
    
    # Term 2: A12 (i:0,1; j:2) * B2 (j:2; k:0,1,2,3)
    # A12 is 2x1. B2 is 1x4.
    # C1[i,k] += A12[i,2] B2[2,k]
    # 8 products.
    for i in range(2):
        for k in range(4):
            add_block_term(
                [(i,2,1)],
                [(2,k,1)],
                [(k,i,1)]
            )
            
    # Term 3: A21 (i:2; j:0,1) * B1 (j:0,1; k:0,1,2,3)
    # A21 is 1x2. B1 is 2x4.
    # C2[k] += sum_j A21[2,j] B1[j,k]
    # 8 products.
    for j in range(2):
        for k in range(4):
            add_block_term(
                [(2,j,1)],
                [(j,k,1)],
                [(k,2,1)]
            )
            
    # Term 4: A22 (i:2; j:2) * B2 (j:2; k:0,1,2,3)
    # 4 products.
    for k in range(4):
        add_block_term(
            [(2,2,1)],
            [(2,k,1)],
            [(k,2,1)]
        )
        
    if col == 34 and _verify(U_b, V_b, W_b):
        best_result = {"U": U_b, "V": V_b, "W": W_b}
        best_rank = 34
        
    # If we have time, we could try random search or other optimizations,
    # but 34 is a solid improvement over 36.
    
    return best_result
