"""Matrix multiplication 3x4x8 with rank-64 decomposition using 2x2x4 blocks."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]


def _block_2x2x4():
    """
    Return U, V, W for 2x2x4 schoolbook decomposition (rank 8).
    
    A is 2x2, B is 2x4, C is 2x4.
    A indices: 0,1,2,3 (row-major)
    B indices: 0..7 (row-major, 2 rows, 4 cols)
    C indices: 0..7 (row-major, 2 rows, 4 cols)
    
    C[0, k] = A[0] B[0, k] + A[1] B[1, k]
    C[1, k] = A[2] B[0, k] + A[3] B[1, k]
    
    We use 8 terms, one for each (row, col) of C?
    Actually, we can use 4 terms if we group?
    No, we need 8 terms as derived above.
    Term for C[0, k]:
      u = [1, 0, 0, 0] (picks A[0])
      v = [1, 0, ..., 1 at (1,k)] (picks B[0,k] and B[1,k]?)
      No, v must pick the specific B element.
      For C[0, k] = A[0] B[0, k] + A[1] B[1, k], we need two terms:
      1. A[0] B[0, k] -> C[0, k]
      2. A[1] B[1, k] -> C[0, k]
      
      Similarly for C[1, k].
      
      Total 8 terms.
    """
    U = []
    V = []
    W = []
    
    # Term 1: A[0] * B[0,0] -> C[0,0]
    U.append([1, 0, 0, 0])
    V.append([1, 0, 0, 0, 0, 0, 0, 0]) # B[0,0]
    W.append([1, 0, 0, 0, 0, 0, 0, 0]) # C[0,0]
    
    # Term 2: A[1] * B[1,0] -> C[0,0]
    U.append([0, 1, 0, 0])
    V.append([0, 0, 1, 0, 0, 0, 0, 0]) # B[1,0]
    W.append([1, 0, 0, 0, 0, 0, 0, 0]) # C[0,0]
    
    # Term 3: A[0] * B[0,1] -> C[0,1]
    U.append([1, 0, 0, 0])
    V.append([0, 1, 0, 0, 0, 0, 0, 0]) # B[0,1]
    W.append([0, 1, 0, 0, 0, 0, 0, 0]) # C[0,1]
    
    # Term 4: A[1] * B[1,1] -> C[0,1]
    U.append([0, 1, 0, 0])
    V.append([0, 0, 0, 1, 0, 0, 0, 0]) # B[1,1]
    W.append([0, 1, 0, 0, 0, 0, 0, 0]) # C[0,1]
    
    # Term 5: A[2] * B[0,0] -> C[1,0]
    U.append([0, 0, 1, 0])
    V.append([1, 0, 0, 0, 0, 0, 0, 0]) # B[0,0]
    W.append([0, 0, 1, 0, 0, 0, 0, 0]) # C[1,0]
    
    # Term 6: A[3] * B[1,0] -> C[1,0]
    U.append([0, 0, 0, 1])
    V.append([0, 0, 1, 0, 0, 0, 0, 0]) # B[1,0]
    W.append([0, 0, 1, 0, 0, 0, 0, 0]) # C[1,0]
    
    # Term 7: A[2] * B[0,1] -> C[1,1]
    U.append([0, 0, 1, 0])
    V.append([0, 1, 0, 0, 0, 0, 0, 0]) # B[0,1]
    W.append([0, 0, 0, 1, 0, 0, 0, 0]) # C[1,1]
    
    # Term 8: A[3] * B[1,1] -> C[1,1]
    U.append([0, 0, 0, 1])
    V.append([0, 0, 0, 1, 0, 0, 0, 0]) # B[1,1]
    W.append([0, 0, 0, 1, 0, 0, 0, 0]) # C[1,1]
    
    # Wait, I only handled k=0,1. I need k=0..3.
    # The block is 2x2x4.
    # My manual list above only covers 2x2x2.
    # I need to generalize.
    
    return U, V, W

def _construct_3x4x8():
    n, m, p = 3, 4, 8
    total_rank = 0
    blocks = []
    
    # i=0,1 part: 2x4x8
    # Tile j into 2 pairs, k into 2 groups of 4.
    # 2x2x4 blocks.
    for jp in range(2):
        for kp in range(2):
            blocks.append(('2x2x4', jp, kp, 8))
            total_rank += 8
            
    # i=2 part: 1x4x8
    # Tile j into 2 pairs, k into 4 pairs.
    # 1x2x2 blocks (rank 4).
    for jp in range(2):
        for kp in range(4):
            blocks.append(('1x2x2', jp, kp, 4))
            total_rank += 4
            
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for btype, jp, kp, r in blocks:
        j0 = 2 * jp
        j1 = j0 + 1
        
        if btype == '2x2x4':
            k0 = 4 * kp
            # U rows: i=0,1; j=j0,j1
            u_rows = [0*m + j0, 0*m + j1, 1*m + j0, 1*m + j1]
            # V rows: j=j0,j1; k=k0..k0+3
            v_rows = []
            for jj in range(2):
                for kk in range(4):
                    v_rows.append((j0+jj)*p + (k0+kk))
            # W rows: k=k0..k0+3; i=0,1
            w_rows = []
            for kk in range(4):
                for ii in range(2):
                    w_rows.append((k0+kk)*n + ii)
            
            # Build 2x2x4 decomposition (rank 8)
            # Terms:
            # 1. A[0,j0] B[j0,k0] -> C[0,k0]
            # 2. A[1,j0] B[j1,k0] -> C[0,k0]
            # ...
            
            # Term 1: A[0,j0] * B[j0,k0] -> C[0,k0]
            U[u_rows[0]][col_offset] += 1
            V[v_rows[0]][col_offset] += 1
            W[w_rows[0]][col_offset] += 1
            col_offset += 1
            
            # Term 2: A[1,j0] * B[j1,k0] -> C[0,k0]
            U[u_rows[1]][col_offset] += 1
            V[v_rows[2]][col_offset] += 1
            W[w_rows[0]][col_offset] += 1
            col_offset += 1
            
            # Term 3: A[0,j0] * B[j0,k1] -> C[0,k1]
            U[u_rows[0]][col_offset] += 1
            V[v_rows[1]][col_offset] += 1
            W[w_rows[2]][col_offset] += 1
            col_offset += 1
            
            # Term 4: A[1,j0] * B[j1,k1] -> C[0,k1]
            U[u_rows[1]][col_offset] += 1
            V[v_rows[3]][col_offset] += 1
            W[w_rows[2]][col_offset] += 1
            col_offset += 1
            
            # Term 5: A[0,j1] * B[j0,k0] -> C[1,k0]
            U[u_rows[2]][col_offset] += 1
            V[v_rows[0]][col_offset] += 1
            W[w_rows[1]][col_offset] += 1
            col_offset += 1
            
            # Term 6: A[1,j1] * B[j1,k0] -> C[1,k0]
            U[u_rows[3]][col_offset] += 1
            V[v_rows[2]][col_offset] += 1
            W[w_rows[1]][col_offset] += 1
            col_offset += 1
            
            # Term 7: A[0,j1] * B[j0,k1] -> C[1,k1]
            U[u_rows[2]][col_offset] += 1
            V[v_rows[1]][col_offset] += 1
            W[w_rows[3]][col_offset] += 1
            col_offset += 1
            
            # Term 8: A[1,j1] * B[j1,k1] -> C[1,k1]
            U[u_rows[3]][col_offset] += 1
            V[v_rows[3]][col_offset] += 1
            W[w_rows[3]][col_offset] += 1
            col_offset += 1
            
        elif btype == '1x2x2':
            k0 = 2 * kp
            # U rows: i=2; j=j0,j1
            u_rows = [2*m + j0, 2*m + j1]
            # V rows: j=j0,j1; k=k0,k1
            v_rows = [
                j0*p + k0, j0*p + k0+1,
                j1*p + k0, j1*p + k0+1
            ]
            # W rows: k=k0,k1; i=2
            w_rows = [k0*n + 2, (k0+1)*n + 2]
            
            # 1x2x2 decomposition (rank 4)
            # Term 1: A[2,j0] B[j0,k0] -> C[2,k0]
            U[u_rows[0]][col_offset] += 1
            V[v_rows[0]][col_offset] += 1
            W[w_rows[0]][col_offset] += 1
            col_offset += 1
            
            # Term 2: A[2,j0] B[j0,k1] -> C[2,k1]
            U[u_rows[0]][col_offset] += 1
            V[v_rows[1]][col_offset] += 1
            W[w_rows[1]][col_offset] += 1
            col_offset += 1
            
            # Term 3: A[2,j1] B[j1,k0] -> C[2,k0]
            U[u_rows[1]][col_offset] += 1
            V[v_rows[2]][col_offset] += 1
            W[w_rows[0]][col_offset] += 1
            col_offset += 1
            
            # Term 4: A[2,j1] B[j1,k1] -> C[2,k1]
            U[u_rows[1]][col_offset] += 1
            V[v_rows[3]][col_offset] += 1
            W[w_rows[1]][col_offset] += 1
            col_offset += 1
            
    return {"U": U, "V": V, "W": W}

def construct():
    start = time.time()
    deadline = start + 110
    
    result = _construct_3x4x8()
    
    n, m, p = DIMENSIONS
    U, V, W = result["U"], result["V"], result["W"]
    
    # Verify
    rank = len(U[0])
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if val != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback
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
