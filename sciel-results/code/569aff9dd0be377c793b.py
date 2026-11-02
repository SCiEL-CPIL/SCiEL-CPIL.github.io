import time
import numpy as np

def construct():
    """
    Rank-39 exact tensor decomposition for 2x2 * 2x11 matrix multiplication.
    
    Strategy:
      - 5 blocks of 2 columns each (k=0..9) using Strassen (rank 7 each) = 35
      - 1 tail column (k=10) using schoolbook (rank 4) = 4
      - Total rank = 39
    
    Index conventions:
      A is 2x2: A[i, j] where i in {0,1}, j in {0,1}
      B is 2x11: B[j, k] where j in {0,1}, k in {0..10}
      C is 2x11: C[i, k] where i in {0,1}, k in {0..10}
      
      U has shape (n*m, rank) = (4, 39): indices 0..3 for A[0,0], A[0,1], A[1,0], A[1,1]
      V has shape (m*p, rank) = (22, 39): indices 0..10 for B[0,k], 11..21 for B[1,k]
      W has shape (p*n, rank) = (22, 39): indices k*n+i for C[i,k] (official trace convention)
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline (10s safety margin)
    
    n, m, p = 2, 2, 11
    rank = 39
    
    # Initialize factor matrices as lists of lists of zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0  # current rank index
    
    # Process 5 blocks of 2 columns each (k=0,1; 2,3; 4,5; 6,7; 8,9)
    for block in range(5):
        if time.time() > deadline:
            break
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # B indices
        v_e = k0          # B[0, k0]
        v_f = k1          # B[0, k1]
        v_g = p + k0      # B[1, k0]
        v_h = p + k1      # B[1, k1]
        
        # W indices: C[i, k] -> W[k*n + i]
        w_c00 = k0 * n + 0   # C[0, k0]
        w_c10 = k0 * n + 1   # C[1, k0]
        w_c01 = k1 * n + 0   # C[0, k1]
        w_c11 = k1 * n + 1   # C[1, k1]
        
        # Strassen's 7 multiplications for 2x2 * 2x2:
        # A = [[a, b], [c, d]], B_block = [[e, f], [g, h]]
        # C_block = [[c00, c01], [c10, c11]]
        
        # m1 = (a+d)(e+h)
        U[0][r] = 1    # a
        U[3][r] = 1    # d
        V[v_e][r] = 1  # e
        V[v_h][r] = 1  # h
        W[w_c00][r] = 1   # c00 += m1
        W[w_c11][r] = 1   # c11 += m1
        r += 1
        
        # m2 = (c+d)e
        U[2][r] = 1    # c
        U[3][r] = 1    # d
        V[v_e][r] = 1  # e
        W[w_c10][r] = 1   # c10 += m2
        W[w_c11][r] = -1  # c11 -= m2
        r += 1
        
        # m3 = a(f-h)
        U[0][r] = 1    # a
        V[v_f][r] = 1  # f
        V[v_h][r] = -1 # h
        W[w_c01][r] = 1   # c01 += m3
        W[w_c11][r] = 1   # c11 += m3
        r += 1
        
        # m4 = d(g-e)
        U[3][r] = 1    # d
        V[v_g][r] = 1  # g
        V[v_e][r] = -1 # e
        W[w_c00][r] = 1   # c00 += m4
        W[w_c10][r] = 1   # c10 += m4
        r += 1
        
        # m5 = (a+b)h
        U[0][r] = 1    # a
        U[1][r] = 1    # b
        V[v_h][r] = 1  # h
        W[w_c00][r] = -1  # c00 -= m5
        W[w_c01][r] = 1   # c01 += m5
        r += 1
        
        # m6 = (c-a)(e+f)
        U[2][r] = 1    # c
        U[0][r] = -1   # -a
        V[v_e][r] = 1  # e
        V[v_f][r] = 1  # f
        W[w_c11][r] = 1   # c11 += m6
        r += 1
        
        # m7 = (b-d)(g+h)
        U[1][r] = 1    # b
        U[3][r] = -1   # -d
        V[v_g][r] = 1  # g
        V[v_h][r] = 1  # h
        W[w_c00][r] = 1   # c00 += m7
        r += 1
    
    # Tail column k=10: schoolbook (4 multiplications)
    # C[0,10] = a*B[0,10] + b*B[1,10]
    # C[1,10] = c*B[0,10] + d*B[1,10]
    
    if time.time() > deadline:
        # If we ran out of time, return what we have
        pass
    else:
        k_tail = 10
        v_et = k_tail          # B[0, 10]
        v_gt = p + k_tail      # B[1, 10]
        w_c0t = k_tail * n + 0  # C[0, 10]
        w_c1t = k_tail * n + 1  # C[1, 10]
        
        # a * B[0,10]
        U[0][r] = 1
        V[v_et][r] = 1
        W[w_c0t][r] = 1
        r += 1
        
        # b * B[1,10]
        U[1][r] = 1
        V[v_gt][r] = 1
        W[w_c0t][r] = 1
        r += 1
        
        # c * B[0,10]
        U[2][r] = 1
        V[v_et][r] = 1
        W[w_c1t][r] = 1
        r += 1
        
        # d * B[1,10]
        U[3][r] = 1
        V[v_gt][r] = 1
        W[w_c1t][r] = 1
        r += 1
    
    # If we completed all blocks, verify rank
    if r == rank:
        return {"U": U, "V": V, "W": W}
    else:
        # Return whatever we have (shouldn't happen in practice)
        return {"U": U, "V": V, "W": W}
