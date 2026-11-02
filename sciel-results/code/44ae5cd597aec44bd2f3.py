"""Matrix multiplication 2x2 x 2x16 via Strassen decomposition."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check valid terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check no spurious terms
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
                
    # Optimization: Only check a sample of invalid terms if rank is high, 
    # but for rank 56 and small dims, full check is fast enough.
    # n*m = 4, m*p = 32, p*n = 32. Total 4*32*32 = 4096 terms.
    # 56 rank. 4096 * 56 ops ~ 230k ops. Very fast.
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if (a, b, c) not in valid:
                    val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                    if abs(val) > 1e-9:
                        return False
    return True

def _time_remaining():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x16 matrix.
    
    Uses Strassen's rank-7 decomposition for the 2x2 core, applied
    independently to each of the 8 column-pairs of the 2x16 matrix.
    Total rank = 7 * 8 = 56.
    
    The 2x16 matrix B has columns indexed 0..15. We group them into
    8 pairs: (col 2t, col 2t+1) for t=0..7.
    
    For each pair t, we treat B[:, 2t:2t+2] as a 2x2 matrix and apply
    Strassen. The output C is 2x16, where C[:, 2t:2t+2] = A * B[:, 2t:2t+2].
    
    U shape: (n*m, rank) = (4, 56)
    V shape: (m*p, rank) = (32, 56)
    W shape: (p*n, rank) = (32, 56)
    
    Indexing conventions:
    - U row index: i*m + j where (i,j) indexes A (i=0,1; j=0,1)
      Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    - V row index: j*p + k where (j,k) indexes B (j=0,1; k=0..15)
      Indices: 0..15 for j=0, 16..31 for j=1
    - W row index: k*n + i where (k,i) indexes C output (k=0..15; i=0,1)
      Per the trace convention W[k*n+i, r]
      Indices: 2k, 2k+1 for each k.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    # Strassen vectors for 2x2 matrix multiplication
    # U vectors (for A)
    # M1 = (A00 + A11)(B00 + B11) -> u=[1,0,0,1]
    # M2 = (A10 + A11)B00       -> u=[0,0,1,1]
    # M3 = A00(B01 - B11)       -> u=[1,0,0,0]
    # M4 = A11(B10 - B00)       -> u=[0,0,0,1]
    # M5 = (A00 + A01)B11       -> u=[1,1,0,0]
    # M6 = (A10 - A00)(B00 + B01)-> u=[-1,0,1,0]
    # M7 = (A01 - A11)(B10 + B11)-> u=[0,1,0,-1]
    
    u_vecs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    # V vectors (for B)
    # M1: B00 + B11 -> v=[1,0,0,1] (indices 0,1,2,3 in local 2x2)
    #    Local indices: 0:B00, 1:B01, 2:B10, 3:B11
    #    Global V indices for block t:
    #    B00 -> j=0, k=2t -> 2t
    #    B01 -> j=0, k=2t+1 -> 2t+1
    #    B10 -> j=1, k=2t -> 16+2t
    #    B11 -> j=1, k=2t+1 -> 16+2t+1
    #    So v=[1,0,0,1] maps to V[2t]=1, V[16+2t+1]=1
    
    # M2: B00 -> v=[1,0,0,0]
    # M3: B01 - B11 -> v=[0,1,0,-1]
    # M4: B10 - B00 -> v=[-1,0,1,0]
    # M5: B11 -> v=[0,0,0,1]
    # M6: B00 + B01 -> v=[1,1,0,0]
    # M7: B10 + B11 -> v=[0,0,1,1]
    
    v_vecs = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # W vectors (for C)
    # C = [[C00, C01], [C10, C11]]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Local W indices: 0:C00, 1:C01, 2:C10, 3:C11
    # Global W indices for block t:
    # C00 -> k=2t, i=0 -> 2t*2 + 0 = 4t
    # C01 -> k=2t+1, i=0 -> (2t+1)*2 + 0 = 4t+2
    # C10 -> k=2t, i=1 -> 2t*2 + 1 = 4t+1
    # C11 -> k=2t+1, i=1 -> (2t+1)*2 + 1 = 4t+3
    
    # M1: C00 + C11 -> w=[1,0,0,1]
    # M2: C10 - C11 -> w=[0,0,1,-1]
    # M3: C01 + C11 -> w=[0,1,0,1]
    # M4: C00 + C10 -> w=[1,0,1,0]
    # M5: -C00 + C01 -> w=[-1,1,0,0]
    # M6: C11       -> w=[0,0,0,1]
    # M7: C00       -> w=[1,0,0,0]
    
    w_vecs = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    num_blocks = p // 2  # 8 blocks
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(num_blocks):
        if _time_remaining() < 1.0:
            # If time is running out, we might break, but construction is fast.
            # If we break, we return incomplete, which is invalid.
            # Better to just finish, it's < 1ms.
            pass
            
        base_col = 7 * t
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # V global indices for this block
        # B00: 2t
        # B01: 2t+1
        # B10: 16+2t
        # B11: 16+2t+1
        v_indices = [2*t, 2*t+1, 16+2*t, 16+2*t+1]
        
        # W global indices for this block
        # C00: 4t
        # C01: 4t+2
        # C10: 4t+1
        # C11: 4t+3
        w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
        
        for s in range(7):
            col = base_col + s
            
            # Fill U
            U[0][col] = u_vecs[s][0]
            U[1][col] = u_vecs[s][1]
            U[2][col] = u_vecs[s][2]
            U[3][col] = u_vecs[s][3]
            
            # Fill V
            V[v_indices[0]][col] = v_vecs[s][0]
            V[v_indices[1]][col] = v_vecs[s][1]
            V[v_indices[2]][col] = v_vecs[s][2]
            V[v_indices[3]][col] = v_vecs[s][3]
            
            # Fill W
            W[w_indices[0]][col] = w_vecs[s][0]
            W[w_indices[1]][col] = w_vecs[s][1]
            W[w_indices[2]][col] = w_vecs[s][2]
            W[w_indices[3]][col] = w_vecs[s][3]
            
    # Verify to be safe
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][col] = 1
                    V_s[j*p+k][col] = 1
                    W_s[k*n+i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}

# EVOLVE-BLOCK-END
