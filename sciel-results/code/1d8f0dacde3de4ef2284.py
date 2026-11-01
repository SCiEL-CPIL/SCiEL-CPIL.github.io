import time

DIMENSIONS = [2, 2, 8]
_DEADLINE = 145.0  # Safety margin before 150s kill
_START_TIME = time.time()

def _check_deadline():
    """Return True if we should stop and return best result."""
    return (time.time() - _START_TIME) > _DEADLINE

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-28 tensor decomposition for 2x2 * 2x8 matrix multiplication
    using Strassen's algorithm for four 2x2 blocks.
    """
    if _check_deadline():
        # Return the known good rank-28 solution immediately
        pass
    n, m, p = DIMENSIONS
    # n=2, m=2, p=8
    
    # Strassen 2x2 decomposition parameters
    # U_str: 4x7, V_str: 4x7, W_str: 4x7
    # Rows of U: 0:00, 1:01, 2:10, 3:11
    # Rows of V: 0:00, 1:01, 2:10, 3:11
    # Rows of W: 0:00, 1:10, 2:01, 3:11 (k*n+i)
    
    U_str = [
        [1, 0, 1, 0, 1, -1, 0], # a00
        [0, 0, 0, 0, 1, 0, 1],  # a01
        [0, 1, 0, 0, 0, 1, 0],  # a10
        [1, 1, 0, 1, 0, 0, -1]  # a11
    ]
    
    V_str = [
        [1, 1, 0, -1, 0, 1, 0], # b00
        [0, 0, 1, 0, 0, 1, 0],  # b01
        [0, 0, 0, 1, 0, 0, 1],  # b10
        [1, 0, -1, 0, 1, 0, 1]  # b11
    ]
    
    W_str = [
        [1, 0, 0, 1, -1, 0, 1], # c00 (k=0, i=0)
        [0, 1, 0, 1, 0, 0, 0],  # c10 (k=0, i=1)
        [0, 0, 1, 0, 1, 0, 0],  # c01 (k=1, i=0)
        [1, -1, 1, 0, 0, 1, 0]  # c11 (k=1, i=1)
    ]
    
    # Total rank
    num_blocks = p // 2 # 4 blocks of 2x2
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill U
    # U is the same for all blocks because A is shared.
    # However, in the tensor product, U[i*m+j, col] is the coefficient for A[i,j] in the col-th product.
    # Since the products for different blocks are independent, we can just repeat the U_str pattern.
    for t in range(num_blocks):
        for i in range(n):
            for j in range(m):
                row_u = i * m + j
                for c in range(7):
                    col_u = t * 7 + c
                    U[row_u][col_u] = U_str[i * m + j][c]
                    
    # Fill V
    # V[j*p+k, col] is coefficient for B[j,k] in col-th product.
    # For block t, B block is B_t. Local index k' in {0,1}. Global k = 2*t + k'.
    # V row index: j*p + (2*t + k')
    for t in range(num_blocks):
        for j in range(m):
            for kp in range(2): # local k'
                global_k = 2 * t + kp
                row_v = j * p + global_k
                for c in range(7):
                    col_v = t * 7 + c
                    # V_str row index is j*2 + kp (since m=2, local p=2)
                    V[row_v][col_v] = V_str[j * 2 + kp][c]
                    
    # Fill W
    # W[k*n+i, col] is coefficient for C[i,k] ? No, C[i, k] is not right.
    # C is n x p. C[i, k].
    # The task says W[k*n+i, r].
    # This implies the output index is mapped to k*n+i.
    # Usually C[i, j] where i is row, j is col.
    # Here output is 2 x 8.
    # So i in {0,1}, k in {0..7}.
    # Index in W is k*n + i.
    # For block t, local output k' in {0,1}. Global k = 2*t + k'.
    # W row index: (2*t + k')*n + i
    for t in range(num_blocks):
        for kp in range(2): # local k'
            global_k = 2 * t + kp
            for i in range(n):
                row_w = global_k * n + i
                for c in range(7):
                    col_w = t * 7 + c
                    # W_str row index is kp*2 + i (since n=2, local p=2)
                    W[row_w][col_w] = W_str[kp * 2 + i][c]
                    
    if _check_deadline():
        # We're close to the deadline; return what we have (which is complete)
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
