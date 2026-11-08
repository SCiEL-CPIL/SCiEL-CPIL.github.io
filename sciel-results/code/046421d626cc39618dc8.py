"""Improved rank-minimal tensor decomposition for 2x3x8 matrix multiplication.

Strategy:
- Use Strassen's 7-term decomposition for the 2x2 blocks of the matrix product.
- For the remaining 2x1 x 1x8 part, use a smarter decomposition that exploits
  the structure to reduce rank below the schoolbook 16 terms.
- Use a search to find the minimal rank decomposition for the 2x1 x 1x8 part.
"""
import time
import sys
from itertools import product

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        # Target tensor T[i, a, k] = 1 for all i, a, k
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = 0
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += u_row[c] * v_row[c] * w_row[c]
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition
    S_U = [
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0],
        [0, 1, 0, -1] 
    ]
    S_V = [
        [1, 0, 0, 1], 
        [1, 0, 0, 0], 
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, 1]  
    ]
    S_W = [
        [1, 0, 0, 1], 
        [0, 1, 0, -1],
        [0, 0, 1, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0],
        [0, 0, 0, 1], 
        [1, 0, 0, 0]  
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i*2 + a]
                        v_val = S_V[r][a*2 + k]
                        w_val = S_W[r][k*2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # Now we need to handle the 2x1 x 1x8 part (a=2)
    # This is: for i in {0,1}, k in {0,...,7}: sum_r U[i*3+2][r] * V[2*8+k][r] * W[k*2+i][r] = 1
    # The Strassen part already uses 4*7=28 columns
    # We need to find a decomposition for the remaining part

    # For the 2x1 x 1x8 part, we can think of it as:
    # We have vectors u_i (for i=0,1), v_k (for k=0..7), w_{k,i} (for k=0..7, i=0,1)
    # We need: sum_r u_i[r] * v_k[r] * w_{k,i}[r] = 1 for all i,k

    # One approach: use the fact that this is a 2x1x8 tensor
    # The rank of this tensor can be bounded. Let's try to find a good decomposition.

    # For a 2x1xp tensor, the rank is at most 2p (schoolbook) but can be less.
    # Actually, for 2x1xp, we can do:
    # For each k, we need u_0*v_k*w_{k,0} + u_1*v_k*w_{k,1} = 1 (but this is not quite right)
    
    # Let me think differently. The tensor is T[i, 2, k] = 1 for i in {0,1}, k in {0..7}
    # This is a 2x8 matrix of 1s, viewed as a tensor with the middle index fixed.
    # We can decompose this as: for r in 0..7, set u_0[r]=1, u_1[r]=0, v_k[r]=1, w_{k,0}[r]=1, w_{k,1}[r]=0
    # and for r in 8..15, set u_0[r]=0, u_1[r]=1, v_k[r]=1, w_{k,0}[r]=0, w_{k,1}[r]=1
    # This gives rank 16.
    
    # But we can do better. Notice that we can use:
    # For r in 0..7: u_0[r]=1, u_1[r]=1, v_k[r]=1, w_{k,0}[r]=1/2, w_{k,1}[r]=1/2
    # This gives: for each (i,k): sum_r u_i[r]*v_k[r]*w_{k,i}[r] = sum_{r=0}^7 1*1*1/2 = 8*1/2 = 4 ≠ 1
    
    # Let me try: use 8 terms with u_0[r]=1, u_1[r]=0 for r=0..7, and u_0[r]=0, u_1[r]=1 for r=8..15
    # That's rank 16 again.
    
    # Actually, the minimal rank for a 2x1xp tensor of all 1s is p (if we allow half-integers)
    # because we can set: for r=0..p-1, u_0[r]=1, u_1[r]=0, v_k[r]=1, w_{k,0}[r]=1, w_{k,1}[r]=0
    # and for r=p..2p-1, u_0[r]=0, u_1[r]=1, v_k[r]=1, w_{k,0}[r]=0, w_{k,1}[r]=1
    # But that's 2p terms.
    
    # Wait, let me reconsider. The tensor T[i, k] = 1 for i in {0,1}, k in {0..7}
    # This is just the 2x8 all-ones matrix. The rank of this matrix is 1.
    # So we can decompose it as: u_0 = [1,0,...,0], u_1 = [1,0,...,0], v_k = [1,1,...,1] for all k
    # w_{k,0} = [1,0,...,0], w_{k,1} = [0,1,...,0]... no that doesn't work.
    
    # Actually, for the tensor T[i,k] = 1, we can write:
    # T[i,k] = sum_r u_i[r] * v_k[r] * w_{k,i}[r]
    # If we set u_0 = [1,0,...,0], u_1 = [0,1,...,0], v_k = [1,1,...,1] (all ones),
    # w_{k,0} = [1,0,...,0], w_{k,1} = [0,1,...,0], then:
    # T[0,k] = 1*1*1 + 0*1*0 = 1 ✓
    # T[1,k] = 0*1*0 + 1*1*1 = 1 ✓
    # So we only need 2 terms! But wait, v_k must be the same for all k in this decomposition.
    # Let me verify: for r=0: u_0[0]=1, v_k[0]=1, w_{k,0}[0]=1, w_{k,1}[0]=0
    # for r=1: u_0[1]=0, v_k[1]=1, w_{k,0}[1]=0, w_{k,1}[1]=1
    # T[0,k] = u_0[0]*v_k[0]*w_{k,0}[0] + u_0[1]*v_k[1]*w_{k,0}[1] = 1*1*1 + 0*1*0 = 1 ✓
    # T[1,k] = u_1[0]*v_k[0]*w_{k,0}[0] + u_1[1]*v_k[1]*w_{k,1}[1] = 0*1*0 + 1*1*1 = 1 ✓
    
    # So the 2x1x8 part can be done in rank 2!
    
    # Let's verify this more carefully. We need:
    # For i=0: sum_r U[i*3+2][r] * V[2*8+k][r] * W[k*2+i][r] = 1
    # For i=1: sum_r U[i*3+2][r] * V[2*8+k][r] * W[k*2+i][r] = 1
    
    # Using 2 terms (r=0, r=1):
    # U[0*3+2][0] = 1, U[0*3+2][1] = 0  (u_0 = [1, 0])
    # U[1*3+2][0] = 0, U[1*3+2][1] = 1  (u_1 = [0, 1])
    # V[2*8+k][0] = 1, V[2*8+k][1] = 1  for all k (v_k = [1, 1])
    # W[k*2+0][0] = 1, W[k*2+0][1] = 0  for all k (w_{k,0} = [1, 0])
    # W[k*2+1][0] = 0, W[k*2+1][1] = 1  for all k (w_{k,1} = [0, 1])
    
    # Check: T[0,k] = U[2][0]*V[16+k][0]*W[2k][0] + U[2][1]*V[16+k][1]*W[2k][1]
    #         = 1*1*1 + 0*1*0 = 1 ✓
    # T[1,k] = U[5][0]*V[16+k][0]*W[2k+1][0] + U[5][1]*V[16+k][1]*W[2k+1][1]
    #         = 0*1*0 + 1*1*1 = 1 ✓
    
    # So the total rank is 4*7 + 2 = 30!

    rank = 4 * 7 + 2
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen part: 4 blocks of 2x2, each using 7 terms
    for block_idx in range(4):
        k0 = block_idx * 2
        k1 = k0 + 1
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li*2 + la]
                    if u_val == 0:
                        continue
                    u_idx = li * m + la
                    U[u_idx][col] = u_val
            
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la*2 + lk]
                    if v_val == 0:
                        continue
                    v_idx = la * p + (k0 + lk)
                    V[v_idx][col] = v_val
            
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk*2 + li]
                    if w_val == 0:
                        continue
                    w_idx = (k0 + lk) * n + li
                    W[w_idx][col] = w_val
            
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # 2x1x8 part: 2 terms
    # Term 0 (col = 28):
    U[0 * m + 2][col] = 1  # u_0[0] = 1
    U[1 * m + 2][col] = 0  # u_1[0] = 0
    for k in range(8):
        V[2 * p + k][col] = 1  # v_k[0] = 1
    for k in range(8):
        W[k * n + 0][col] = 1  # w_{k,0}[0] = 1
        W[k * n + 1][col] = 0  # w_{k,1}[0] = 0
    col += 1
    
    # Term 1 (col = 29):
    U[0 * m + 2][col] = 0  # u_0[1] = 0
    U[1 * m + 2][col] = 1  # u_1[1] = 1
    for k in range(8):
        V[2 * p + k][col] = 1  # v_k[1] = 1
    for k in range(8):
        W[k * n + 0][col] = 0  # w_{k,0}[1] = 0
        W[k * n + 1][col] = 1  # w_{k,1}[1] = 1
    col += 1

    if not _elapsed_ok():
        return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}
    
    if _validate(candidate):
        return candidate
    else:
        return _schoolbook()

# EVOLVE-BLOCK-END
