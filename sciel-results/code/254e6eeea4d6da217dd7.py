import time
import random
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook_3x3x2():
    """Return U, V, W for 3x3x2 multiplication with rank 18."""
    n, m, p = 3, 3, 2
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
    return U, V, W, rank

def _schoolbook_3x2x3():
    """Return U, V, W for 3x2x3 multiplication with rank 18."""
    n, m, p = 3, 2, 3
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
    return U, V, W, rank

def _schoolbook_3x2x2():
    """Return U, V, W for 3x2x2 multiplication with rank 12."""
    n, m, p = 3, 2, 2
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
    return U, V, W, rank

def _schoolbook_3x3x3():
    """Return U, V, W for 3x3x3 multiplication with rank 27."""
    n, m, p = 3, 3, 3
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
    return U, V, W, rank

def _verify_3x3x3(U, V, W):
    """Verify 3x3x3 decomposition."""
    n, m, p = 3, 3, 3
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _search_3x3x3(deadline):
    """Search for a lower rank 3x3x3 decomposition."""
    best_rank = 27
    best_U, best_V, best_W = _schoolbook_3x3x3()
    
    # Try ranks 23 to 26
    for target_rank in range(23, 27):
        if time.time() > deadline - 5:
            break
        attempts = 0
        max_attempts = 500
        while attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            # Random initialization with small integers/half-integers
            candidates = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
            U = [[random.choice(candidates) for _ in range(target_rank)] for _ in range(9)]
            V = [[random.choice(candidates) for _ in range(target_rank)] for _ in range(9)]
            W = [[random.choice(candidates) for _ in range(target_rank)] for _ in range(9)]
            
            if _verify_3x3x3(U, V, W):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
        
        if best_rank == target_rank:
            continue
        else:
            # If not found at this rank, try higher? No, we want lowest.
            # But if we fail at 23, we should try 24, etc.
            pass
            
    return best_U, best_V, best_W, best_rank

def _assemble(U1, V1, W1, r1, U2, V2, W2, r2, U3, V3, W3, r3, U4, V4, W4, r4):
    """
    Assemble the 3x5x5 decomposition from 4 blocks.
    Block 1: 3x3x3 (i<3, j<3, k<3)
    Block 2: 3x3x2 (i<3, j<3, k>=3) -> V cols 3-4, W rows 3-4?
    Wait, indices:
    A (3x5): i in 0..2, j in 0..4
    B (5x5): j in 0..4, k in 0..4
    C (3x5): i in 0..2, k in 0..4
    
    Split j into j1 (0..2) and j2 (3..4).
    Split k into k1 (0..2) and k2 (3..4).
    
    C = A B
    C_{i,k1} = sum_{j1} A_{i,j1} B_{j1,k1} + sum_{j2} A_{i,j2} B_{j2,k1}
    C_{i,k2} = sum_{j1} A_{i,j1} B_{j1,k2} + sum_{j2} A_{i,j2} B_{j2,k2}
    
    Block 1 (j1,k1): 3x3x3. U1, V1, W1.
    Block 2 (j1,k2): 3x3x2. U2, V2, W2.
    Block 3 (j2,k1): 3x2x3. U3, V3, W3.
    Block 4 (j2,k2): 3x2x2. U4, V4, W4.
    
    Total Rank = r1 + r2 + r3 + r4.
    U_total (15 x Rank)
    V_total (25 x Rank)
    W_total (15 x Rank)
    """
    total_rank = r1 + r2 + r3 + r4
    U_total = [[0] * total_rank for _ in range(15)]
    V_total = [[0] * total_rank for _ in range(25)]
    W_total = [[0] * total_rank for _ in range(15)]
    
    offset = 0
    
    # Block 1: 3x3x3
    # U1 rows: i*3+j (0..8). Maps to U_total rows i*5+j (0..8)
    # V1 rows: j*3+k (0..8). Maps to V_total rows j*5+k (0..8)
    # W1 rows: k*3+i (0..8). Maps to W_total rows k*5+i (0..8)
    for u_i in range(9):
        i = u_i // 3
        j = u_i % 3
        U_total[i*5 + j] = U_total[i*5 + j] + [U1[u_i][c] for c in range(r1)] # Note: need to add to existing zeros
    # Actually, simpler to just copy
    for c in range(r1):
        for u_i in range(9):
            i = u_i // 3
            j = u_i % 3
            U_total[i*5 + j][offset + c] = U1[u_i][c]
        for v_i in range(9):
            j = v_i // 3
            k = v_i % 3
            V_total[j*5 + k][offset + c] = V1[v_i][c]
        for w_i in range(9):
            k = w_i // 3
            i = w_i % 3
            W_total[k*5 + i][offset + c] = W1[w_i][c]
    offset += r1
    
    # Block 2: 3x3x2 (j1, k2)
    # U2 rows: i*3+j (0..8). Maps to U_total rows i*5+j (0..8)
    # V2 rows: j*2+k (0..5). j in 0..2, k in 0..1 (local). Global k = 3+k_local.
    # V2 row index v_i: j = v_i // 2, k_local = v_i % 2. Global V row: j*5 + (3+k_local).
    # W2 rows: k*3+i (0..5). k_local in 0..1, i in 0..2. Global W row: (3+k_local)*5 + i.
    for c in range(r2):
        for u_i in range(9):
            i = u_i // 3
            j = u_i % 3
            U_total[i*5 + j][offset + c] += U2[u_i][c]
        for v_i in range(6):
            j = v_i // 2
            k_loc = v_i % 2
            V_total[j*5 + 3 + k_loc][offset + c] += V2[v_i][c]
        for w_i in range(6):
            k_loc = w_i // 3
            i = w_i % 3
            W_total[(3+k_loc)*5 + i][offset + c] += W2[w_i][c]
    offset += r2
    
    # Block 3: 3x2x3 (j2, k1)
    # U3 rows: i*2+j (0..5). i in 0..2, j_local in 0..1. Global U row: i*5 + (3+j_local).
    # V3 rows: j*3+k (0..5). j_local in 0..1, k in 0..2. Global V row: (3+j_local)*5 + k.
    # W3 rows: k*3+i (0..8). k in 0..2, i in 0..2. Global W row: k*5 + i.
    for c in range(r3):
        for u_i in range(6):
            i = u_i // 2
            j_loc = u_i % 2
            U_total[i*5 + 3 + j_loc][offset + c] += U3[u_i][c]
        for v_i in range(6):
            j_loc = v_i // 3
            k = v_i % 3
            V_total[(3+j_loc)*5 + k][offset + c] += V3[v_i][c]
        for w_i in range(9):
            k = w_i // 3
            i = w_i % 3
            W_total[k*5 + i][offset + c] += W3[w_i][c]
    offset += r3
    
    # Block 4: 3x2x2 (j2, k2)
    # U4 rows: i*2+j (0..5). i in 0..2, j_local in 0..1. Global U row: i*5 + (3+j_local).
    # V4 rows: j*2+k (0..3). j_local in 0..1, k_local in 0..1. Global V row: (3+j_local)*5 + (3+k_local).
    # W4 rows: k*3+i (0..5). k_local in 0..1, i in 0..2. Global W row: (3+k_local)*5 + i.
    for c in range(r4):
        for u_i in range(6):
            i = u_i // 2
            j_loc = u_i % 2
            U_total[i*5 + 3 + j_loc][offset + c] += U4[u_i][c]
        for v_i in range(4):
            j_loc = v_i // 2
            k_loc = v_i % 2
            V_total[(3+j_loc)*5 + 3 + k_loc][offset + c] += V4[v_i][c]
        for w_i in range(6):
            k_loc = w_i // 3
            i = w_i % 3
            W_total[(3+k_loc)*5 + i][offset + c] += W4[w_i][c]
            
    return U_total, V_total, W_total, total_rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Get sub-decompositions
    U2, V2, W2, r2 = _schoolbook_3x3x2()
    U3, V3, W3, r3 = _schoolbook_3x2x3()
    U4, V4, W4, r4 = _schoolbook_3x2x2()
    
    # Search for better 3x3x3
    U1, V1, W1, r1 = _search_3x3x3(deadline)
    
    U, V, W, total_rank = _assemble(U1, V1, W1, r1, U2, V2, W2, r2, U3, V3, W3, r3, U4, V4, W4, r4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
