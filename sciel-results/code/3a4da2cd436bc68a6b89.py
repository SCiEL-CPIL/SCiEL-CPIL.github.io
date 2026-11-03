import time
import random
import numpy as np

DIMENSIONS = [3, 4, 8]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if val != 1:
                    return False
    return True

def _find_2x2x4_rank10(deadline):
    """Search for rank 10 decomposition of 2x2x4."""
    # U is 2*2 = 4 rows, V is 2*4 = 8 rows, W is 4*2 = 8 rows
    # Rank 10
    for _ in range(5000):
        if time.time() > deadline:
            return None
        U = [[random.randint(-2, 2) for _ in range(10)] for _ in range(4)]
        V = [[random.randint(-2, 2) for _ in range(10)] for _ in range(8)]
        W = [[random.randint(-2, 2) for _ in range(10)] for _ in range(8)]
        
        # Check
        # C[i,k] = sum_j A[i,j] B[j,k]
        # U[i,j], V[j,k], W[k,i]
        # Indices:
        # i in 0..1, j in 0..1, k in 0..3
        # U rows: i*2+j
        # V rows: j*4+k
        # W rows: k*2+i
        ok = True
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    val = 0
                    for c in range(10):
                        val += U[i*2+j][c] * V[j*4+k][c] * W[k*2+i][c]
                    if val != 1:
                        ok = False
                        break
                if not ok: break
            if not ok: break
        if ok:
            return U, V, W
    return None

def _construct_3x4x8():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Try to find 2x2x4 rank 10
    sol_224 = _find_2x2x4_rank10(deadline)
    
    if sol_224:
        U224, V224, W224 = sol_224
        rank_224 = 10
    else:
        # Fallback to 2x2x2 Strassen blocks (rank 14 for 2x2x4)
        # 2x2x4 is two 2x2x2 blocks
        # We can just use the 2x2x2 Strassen and duplicate for the two k-pairs
        U224 = []
        V224 = []
        W224 = []
        # Strassen 2x2x2
        U2 = [
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ]
        V2 = [
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, -1, 0, 1, 0, 1]
        ]
        W2 = [
            [1, 0, 0, 1, -1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ]
        # Construct 2x2x4 by placing two 2x2x2 blocks
        # k=0,1 uses first 7 cols, k=2,3 uses next 7 cols
        for r in range(4):
            U224.append(U2[r] + [0]*7)
        for r in range(4):
            V224.append(V2[r] + [0]*7)
        for r in range(4):
            W224.append(W2[r] + [0]*7)
        # Second block
        for r in range(4):
            U224.append([0]*7 + U2[r])
        for r in range(4):
            V224.append([0]*7 + V2[r])
        for r in range(4):
            W224.append([0]*7 + W2[r])
        rank_224 = 14

    # Now build 3x4x8
    # Split into i=0,1 (2x4x8) and i=2 (1x4x8)
    # 2x4x8: 4 blocks of 2x2x4
    # 1x4x8: 8 blocks of 1x2x2 (rank 4 each, total 32)
    
    total_rank = 4 * rank_224 + 32
    U = [[0] * total_rank for _ in range(3*4)]
    V = [[0] * total_rank for _ in range(4*8)]
    W = [[0] * total_rank for _ in range(8*3)]
    
    col_offset = 0
    # 2x4x8 part (i=0,1)
    # j in 0..3, k in 0..7
    # Split j into 0,1 and 2,3. Split k into 0,1; 2,3; 4,5; 6,7.
    # 4 blocks of 2x2x4.
    for jp in range(2): # j pair
        for kp in range(4): # k pair
            j0, j1 = 2*jp, 2*jp+1
            k0, k1 = 4*kp, 4*kp+1 # Wait, k is 0..7. Pairs: (0,1), (2,3), (4,5), (6,7)
            # My 2x2x4 block handles k=0,1,2,3? No, 2x2x4 handles k=0..3.
            # So I need to map the 4 k-pairs to the 4 k's of 2x2x4?
            # No, 2x2x4 has k=0..3.
            # So I need 4 blocks of 2x2x2?
            # Let's re-evaluate.
            # 2x4x8:
            # i=0,1. j=0..3. k=0..7.
            # Split j into (0,1) and (2,3).
            # Split k into (0,1), (2,3), (4,5), (6,7).
            # This gives 2 * 4 = 8 blocks of 2x2x2.
            # Rank 8 * 7 = 56.
            # My previous logic: 2x4x4 is two 2x2x4.
            # 2x2x4 is two 2x2x2.
            # So 2x4x8 is 4 * 2x2x4 = 8 * 2x2x2.
            # If I have a 2x2x4 decomposition, I use it 4 times.
            
            # U rows for 2x2x4 block:
            # i=0,1. j=j0,j1.
            # U indices: i*4 + j? No, U is for 2x2x4, so U is 2x2=4 rows.
            # In the global 3x4x8, U is 3x4=12 rows.
            # For i=0,1, j=j0,j1.
            # U row index: i*4 + j.
            
            u_rows = [0*4 + j0, 0*4 + j1, 1*4 + j0, 1*4 + j1]
            v_rows = [j0*8 + k0, j0*8 + k1, j1*8 + k0, j1*8 + k1]
            # W rows: k=0..3 in 2x2x4.
            # Here k0, k1 are the two k's.
            # Wait, 2x2x4 has k=0..3.
            # So I need 4 k's.
            # My loop kp in range(4) gives 4 pairs.
            # So I need 4 blocks of 2x2x2?
            # If I use 2x2x4, it covers 4 k's.
            # So I only need 2 blocks of 2x2x4 for the k dimension?
            # k=0..3 and k=4..7.
            # So 2 blocks of 2x2x4 for each j-pair?
            # j-pair 0: k=0..3, k=4..7.
            # j-pair 1: k=0..3, k=4..7.
            # Total 4 blocks of 2x2x4.
            
            # Let's restart the loop structure.
            pass

    # Correct Loop:
    col_offset = 0
    for jp in range(2): # j pair
        for kp in range(2): # k half (0..3, 4..7)
            j0, j1 = 2*jp, 2*jp+1
            k_base = 4*kp
            k0, k1, k2, k3 = k_base, k_base+1, k_base+2, k_base+3
            
            u_rows = [0*4 + j0, 0*4 + j1, 1*4 + j0, 1*4 + j1]
            v_rows = [j0*8 + k0, j0*8 + k1, j0*8 + k2, j0*8 + k3,
                      j1*8 + k0, j1*8 + k1, j1*8 + k2, j1*8 + k3]
            w_rows = [k0*3 + 0, k1*3 + 0, k2*3 + 0, k3*3 + 0,
                      k0*3 + 1, k1*3 + 1, k2*3 + 1, k3*3 + 1]
            
            for i in range(4):
                for c in range(rank_224):
                    U[u_rows[i]][col_offset + c] += U224[i][c]
            for i in range(8):
                for c in range(rank_224):
                    V[v_rows[i]][col_offset + c] += V224[i][c]
            for i in range(8):
                for c in range(rank_224):
                    W[w_rows[i]][col_offset + c] += W224[i][c]
            col_offset += rank_224

    # 1x4x8 part (i=2)
    # 8 blocks of 1x2x2
    for jp in range(2):
        for kp in range(4):
            j0, j1 = 2*jp, 2*jp+1
            k0, k1 = 4*kp, 4*kp+1 # Wait, k pairs are (0,1), (2,3)...
            # kp in 0..3. k0=4*kp, k1=4*kp+1.
            # This misses k=1,3,5,7?
            # No, kp=0: k0=0, k1=1.
            # kp=1: k0=4, k1=5.
            # This is wrong. k pairs should be (0,1), (2,3), (4,5), (6,7).
            # So k0 = 2*kp, k1 = 2*kp+1.
            
            k0, k1 = 2*kp, 2*kp+1
            
            u_rows = [2*4 + j0, 2*4 + j1]
            v_rows = [j0*8 + k0, j0*8 + k1, j1*8 + k0, j1*8 + k1]
            w_rows = [k0*3 + 2, k1*3 + 2]
            
            # 1x2x2 schoolbook
            U1 = [[1, 0, 1, 0], [0, 1, 0, 1]]
            V1 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            W1 = [[1, 1, 0, 0], [0, 0, 1, 1]]
            
            for i in range(2):
                for c in range(4):
                    U[u_rows[i]][col_offset + c] += U1[i][c]
                    W[w_rows[i]][col_offset + c] += W1[i][c]
            for i in range(4):
                for c in range(4):
                    V[v_rows[i]][col_offset + c] += V1[i][c]
            col_offset += 4

    return {"U": U, "V": V, "W": W}

def construct():
    start = time.time()
    result = _construct_3x4x8()
    n, m, p = DIMENSIONS
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # Fallback schoolbook
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
        result = {"U": U, "V": V, "W": W}
    return result
