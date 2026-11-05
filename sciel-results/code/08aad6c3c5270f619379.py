import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    deadline = start_time + 110
    
    # Block structure: 6x6 = 3x3 blocks of 2x2
    # Outer: Schoolbook (rank 9) for 3x3 block multiplication
    # Inner: Strassen (rank 7) for 2x2 matrix multiplication
    # Total rank: 9 * 7 = 63
    
    block_size = 2
    num_blocks = 3
    dim = 6
    inner_rank = 7
    total_rank = 9 * inner_rank
    
    U = np.zeros((dim * dim, total_rank), dtype=np.float64)
    V = np.zeros((dim * dim, total_rank), dtype=np.float64)
    W = np.zeros((dim * dim, total_rank), dtype=np.float64)
    
    # Strassen for 2x2 with full C contributions:
    # M1 = (A00+A11)*(B00+B11) -> C00+=1, C11+=1
    # M2 = (A10+A11)*B00        -> C10+=1, C11+=-1
    # M3 = A00*(B01-B11)        -> C01+=1, C11+=1
    # M4 = A11*(B10-B00)        -> C00+=1, C10+=1
    # M5 = (A00+A01)*B11        -> C00+=-1, C01+=1
    # M6 = (A10-A00)*(B00+B01)  -> C11+=1
    # M7 = (A01-A11)*(B10+B11)  -> C00+=1
    
    strassen_steps_full = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    col = 0
    bs = block_size
    nb = num_blocks
    
    # Outer: schoolbook for 3x3 block mult
    # C[i][k] += A[i][j] * B[j][k] for i,j,k in 0..2
    for i in range(nb):
        for j in range(nb):
            for k in range(nb):
                for step_idx, (x_list, y_list, c_list) in enumerate(strassen_steps_full):
                    for r in range(bs):
                        for s in range(bs):
                            # U: A[i*bs+ar, j*bs+ac]
                            for (ar, ac, x_coeff) in x_list:
                                u_idx = (i * bs + ar) * dim + (j * bs + ac)
                                U[u_idx, col] += x_coeff
                            
                            # V: B[j*bs+br, k*bs+bc]
                            for (br, bc, y_coeff) in y_list:
                                v_idx = (j * bs + br) * dim + (k * bs + bc)
                                V[v_idx, col] += y_coeff
                            
                            # W: C[i*bs+cr, k*bs+cs] with trace convention W[k*n+i, r]
                            # row index in W = (k*bs+cs)*dim + (i*bs+cr)
                            for (cr, cs, c_coeff) in c_list:
                                w_idx = (k * bs + cs) * dim + (i * bs + cr)
                                W[w_idx, col] += c_coeff
                            
                            col += 1
                            
                            if time.time() > deadline:
                                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
