"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 120.0

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 10
    nm, mp, pn = n * m, m * p, p * n  # 4, 20, 20
    
    # Strassen's rank-7 decomposition for 2x2x2 matrix multiplication
    # A indices: [A00, A01, A10, A11] (row-major, i*m+j)
    # B indices: [B00, B01, B10, B11] (row-major, j*2+k for local 2x2)
    # C indices: [C00, C10, C01, C11] (W convention: k*n+i)
    #   C[0][0] -> idx 0, C[1][0] -> idx 1, C[0][1] -> idx 2, C[1][1] -> idx 3
    
    # Strassen terms: each is (u_vec, v_vec, w_vec) of length 4
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00       -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11)       -> C01 += M3, C11 += M3
    # M4 = A11(B10-B00)       -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11       -> C01 += M5, C00 -= M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    
    strassen_terms = [
        # (u_vec[4], v_vec[4], w_vec[4])
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),  # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),  # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),  # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # M7
    ]
    
    num_blocks = p // 2  # 5 blocks of 2 columns each
    rank = num_blocks * 7  # 35
    
    U = [[0] * rank for _ in range(nm)]   # 4 x 35
    V = [[0] * rank for _ in range(mp)]   # 20 x 35
    W = [[0] * rank for _ in range(pn)]   # 20 x 35
    
    col = 0
    for b in range(num_blocks):
        for u_vec, v_vec, w_vec in strassen_terms:
            # U side: A[i,j] with index i*m+j
            for idx in range(nm):
                U[idx][col] = u_vec[idx]
            
            # V side: B[j,k] with index j*p+k
            # v_vec indices: 0=B00, 1=B01, 2=B10, 3=B11
            # B[0, 2b] -> V[0*p + 2b]
            # B[0, 2b+1] -> V[0*p + 2b + 1]
            # B[1, 2b] -> V[1*p + 2b]
            # B[1, 2b+1] -> V[1*p + 2b + 1]
            V[0 * p + 2 * b][col] = v_vec[0]
            V[0 * p + 2 * b + 1][col] = v_vec[1]
            V[1 * p + 2 * b][col] = v_vec[2]
            V[1 * p + 2 * b + 1][col] = v_vec[3]
            
            # W side: C[i,k] with index k*n+i (trace convention)
            # w_vec indices: 0=C[0][0], 1=C[1][0], 2=C[0][1], 3=C[1][1]
            # C[0, 2b]   -> W[(2b)*n + 0]
            # C[1, 2b]   -> W[(2b)*n + 1]
            # C[0, 2b+1] -> W[(2b+1)*n + 0]
            # C[1, 2b+1] -> W[(2b+1)*n + 1]
            W[(2 * b) * n + 0][col] = w_vec[0]
            W[(2 * b) * n + 1][col] = w_vec[1]
            W[(2 * b + 1) * n + 0][col] = w_vec[2]
            W[(2 * b + 1) * n + 1][col] = w_vec[3]
            
            col += 1
            
            if time.time() > _DEADLINE - 10:
                break
        if time.time() > _DEADLINE - 10:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
