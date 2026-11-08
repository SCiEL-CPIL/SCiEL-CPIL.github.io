import time
import numpy as np

_DEADLINE = 120.0
_start_time = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _start_time)

def construct():
    n, m, p = 2, 3, 9
    
    # We will use a block decomposition.
    # Split the multiplication into:
    # 1. A_top (2x2) * B_top (2x8)  -> 2x8 output
    # 2. A_right (2x1) * B_bottom (3x1) -> 2x1 output? No, B is 3x9.
    # Let's split B columns: B[:, 0:8] and B[:, 8].
    # And A columns: A[:, 0:2] and A[:, 2].
    
    # C = A * B
    # C = [A[:,0:2] * B[:,0:8] + A[:,0:2] * B[:,8:9] + A[:,2:3] * B[:,0:8] + A[:,2:3] * B[:,8:9]]
    # This is getting complicated. Let's stick to the previous successful strategy which yielded rank 50.
    # The previous strategy:
    # - 4 blocks of Strassen (2x2x2) for columns 0-7. Rank 4*7 = 28.
    # - Naive for column 8 of B with A_top (2x2). Rank 4.
    # - Naive for A_right (col 2) with B_bottom (row 2) for all 9 columns. Rank 2*9 = 18.
    # Total 50.
    
    # Is there a better way?
    # What if we group columns 0-2, 3-5, 6-8?
    # 3 blocks of 2x3x3.
    # If we can find rank 16 for 2x3x3, total 48.
    # Let's try to construct a rank 16 for 2x3x3.
    # M_{2,3,3}: A(2x3) * B(3x3).
    # Split B into B1(3x2) and B2(3x1).
    # C = A*B1 + A*B2.
    # A*B2 is 2x3x1. Rank 6 (naive).
    # A*B1 is 2x3x2. 
    # Split A into A1(2x2) and A2(2x1).
    # A1*B1 is 2x2x2 (Strassen, rank 7).
    # A2*B1 is 2x1x2. Rank 4 (naive).
    # Total for 2x3x3: 7 + 4 + 6 = 17.
    # 3 * 17 = 51. Worse than 50.
    
    # What if we use Strassen on 2x2x3?
    # A(2x2) * B(2x3).
    # Split B into B1(2x2) and B2(2x1).
    # A*B1 is 2x2x2 (Rank 7).
    # A*B2 is 2x2x1 (Rank 4).
    # Total 11.
    # Then 2x3x3 = (2x2 part) + (2x1 part).
    # A(2x3) = [A1(2x2), A2(2x1)].
    # B(3x3).
    # C = A1*B[:,0:2] + A1*B[:,2:3] + A2*B.
    # A1*B[:,0:2] is 2x2x2 (Rank 7).
    # A1*B[:,2:3] is 2x2x1 (Rank 4).
    # A2*B is 2x1x3 (Rank 6).
    # Total 17. Same.
    
    # So 50 seems to be the best simple construction.
    # Let's verify the rank 50 construction is correct and robust.
    
    rank = 50
    
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    col = 0
    
    # Part 1: Strassen on 4 blocks of 2x2 (cols 0-7 of B, rows 0-1)
    # A_top is 2x2 (indices 0,1,3,4 in U which is flattened A[i*m+j])
    # U indices:
    # A[0,0]=0, A[0,1]=1
    # A[1,0]=3, A[1,1]=4
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # V indices for B[0, k0], B[0, k1], B[1, k0], B[1, k1]
        # V is flattened B[i*p+j]
        v_e = 0 * p + k0  # B[0, k0]
        v_f = 0 * p + k1  # B[0, k1]
        v_g = 1 * p + k0  # B[1, k0]
        v_h = 1 * p + k1  # B[1, k1]
        
        # W indices for C[0, k0], C[0, k1], C[1, k0], C[1, k1]
        # W is flattened C[i*n+j]? No, task says W[k*n+i, r].
        # So W index for C[i, j] is j*n + i.
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        # Strassen terms for A * B_block
        # A = [[a, b], [c, d]]
        # B_block = [[e, f], [g, h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # Reconstruction:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Term 1: M1
        U[0, col] += 1  # a
        U[4, col] += 1  # d
        V[v_e, col] += 1 # e
        V[v_h, col] += 1 # h
        W[w_00, col] += 1
        W[w_11, col] += 1
        
        # Term 2: M2
        U[3, col+1] += 1 # c
        U[4, col+1] += 1 # d
        V[v_e, col+1] += 1 # e
        W[w_10, col+1] += 1
        W[w_11, col+1] -= 1
        
        # Term 3: M3
        U[0, col+2] += 1 # a
        V[v_f, col+2] += 1 # f
        V[v_h, col+2] -= 1 # -h
        W[w_01, col+2] += 1
        W[w_11, col+2] += 1
        
        # Term 4: M4
        U[4, col+3] += 1 # d
        V[v_g, col+3] += 1 # g
        V[v_e, col+3] -= 1 # -e
        W[w_00, col+3] += 1
        W[w_10, col+3] += 1
        
        # Term 5: M5
        U[0, col+4] += 1 # a
        U[1, col+4] += 1 # b
        V[v_h, col+4] += 1 # h
        W[w_00, col+4] -= 1
        W[w_01, col+4] += 1
        
        # Term 6: M6
        U[3, col+5] += 1 # c
        U[0, col+5] -= 1 # -a
        V[v_e, col+5] += 1 # e
        V[v_f, col+5] += 1 # f
        W[w_11, col+5] += 1
        
        # Term 7: M7
        U[1, col+6] += 1 # b
        U[4, col+6] -= 1 # -d
        V[v_g, col+6] += 1 # g
        V[v_h, col+6] += 1 # h
        W[w_00, col+6] += 1
        
        col += 7
        
    # Part 2: Naive for B col 8, rows 0-1 (A_top)
    k8 = 8
    v_e8 = 0 * p + k8  # B[0, 8]
    v_g8 = 1 * p + k8  # B[1, 8]
    w_08 = k8 * n + 0  # C[0, 8]
    w_18 = k8 * n + 1  # C[1, 8]
    
    # P1 = a * B[0,8] -> C[0,8]
    U[0, col] += 1
    V[v_e8, col] += 1
    W[w_08, col] += 1
    
    # P2 = b * B[1,8] -> C[0,8]
    U[1, col+1] += 1
    V[v_g8, col+1] += 1
    W[w_08, col+1] += 1
    
    # P3 = c * B[0,8] -> C[1,8]
    U[3, col+2] += 1
    V[v_e8, col+2] += 1
    W[w_18, col+2] += 1
    
    # P4 = d * B[1,8] -> C[1,8]
    U[4, col+3] += 1
    V[v_g8, col+3] += 1
    W[w_18, col+3] += 1
    
    col += 4
    
    # Part 3: A_right (col 2) x B_bottom (row 2)
    # A[0,2] is index 0*m+2 = 2
    # A[1,2] is index 1*m+2 = 5
    # B[2, k] is index 2*p+k
    # C[0, k] is index k*n+0
    # C[1, k] is index k*n+1
    
    u_ar0 = 2  # A[0,2]
    u_ar1 = 5  # A[1,2]
    
    for k in range(p):
        v_bk = 2 * p + k
        w_0k = k * n + 0
        w_1k = k * n + 1
        
        # A[0,2] * B[2,k] -> C[0,k]
        U[u_ar0, col] += 1
        V[v_bk, col] += 1
        W[w_0k, col] += 1
        
        # A[1,2] * B[2,k] -> C[1,k]
        U[u_ar1, col+1] += 1
        V[v_bk, col+1] += 1
        W[w_1k, col+1] += 1
        
        col += 2
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

BET: {mechanism: "Re-implementation of the known rank-50 solution with explicit verification of indices and structure to ensure validity and potentially catch subtle bugs in the previous version.", q: 0.2, delta: [0, 0.05]}
