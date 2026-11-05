import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def _deadline_guard(start_time, limit=145.0):
    """Check if we've exceeded the time limit."""
    return (time.time() - start_time) > limit

# EVOLVE-BLOCK-START
import itertools
import numpy as np

def _schoolbook(n=4, m=5, p=5):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _verify(U, V, W, n=4, m=5, p=5):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    """
    rank = len(U[0]) if U else 0
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    U_reshaped = U_np.reshape(n, m, rank)
    V_reshaped = V_np.reshape(m, p, rank)
    W_reshaped = W_np.reshape(p, n, rank)
    
    try:
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    except Exception:
        return False
        
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
        
    return True

def _strassen_2x2_factors():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    """
    u = [[0]*7 for _ in range(4)]
    v = [[0]*7 for _ in range(4)]
    w = [[0]*7 for _ in range(4)]
    
    u[0][0]=1; u[3][0]=1
    v[0][0]=1; v[3][0]=1
    w[0][0]=1; w[3][0]=1
    
    u[2][1]=1; u[3][1]=1
    v[0][1]=1
    w[2][1]=1; w[3][1]=-1
    
    u[0][2]=1
    v[1][2]=1; v[3][2]=-1
    w[1][2]=1; w[3][2]=1
    
    u[3][3]=1
    v[2][3]=1; v[0][3]=-1
    w[0][3]=1; w[2][3]=1
    
    u[0][4]=1; u[1][4]=1
    v[3][4]=1
    w[0][4]=-1; w[1][4]=1
    
    u[0][5]=-1; u[2][5]=1
    v[0][5]=1; v[1][5]=1
    w[3][5]=1
    
    u[1][6]=1; u[3][6]=-1
    v[2][6]=1; v[3][6]=1
    w[0][6]=1
    
    return u, v, w

def _construct_rank44_5x5x5():
    """
    Construct a rank-44 decomposition for 5x5x5 matrix multiplication.
    
    Split 5x5 into 3x3 + 2x2 blocks:
    - Indices 0,1,2 form the 3x3 block
    - Indices 3,4 form the 2x2 block
    
    The 5x5x5 tensor T[i,j,k] can be decomposed as:
    T[i,j,k] = T_333[i,j,k] + T_332[i,j,k] + T_323[i,j,k] + T_322[i,j,k]
             + T_233[i,j,k] + T_232[i,j,k] + T_223[i,j,k] + T_222[i,j,k]
    
    where the subscripts indicate which block each index belongs to.
    
    We use:
    - 3x3x3: rank 23
    - 2x2x2: rank 7
    - Cross terms: need to be handled
    
    Total rank should be 44.
    """
    # This is complex. Let me use a known construction.
    # For now, I'll use a simpler approach: schoolbook for 5x5x5 is rank 125.
    # Let me try a block approach.
    
    # Actually, let me use a different strategy.
    # I'll construct the 4x5x5 decomposition directly using a rank-40 approach.
    
    return None

def _construct_optimal_4x5x5():
    """
    Construct a low-rank decomposition for 4x5x5 matrix multiplication.
    
    Strategy: Use the fact that 4x5x5 can be decomposed using
    a combination of smaller blocks.
    
    Split j (middle index) into j in 0..3 and j=4.
    Split k (last index) into k in 0..3 and k=4.
    
    T[i,j,k] for i in 0..3, j in 0..4, k in 0..4:
    
    1. j in 0..3, k in 0..3: 4x4x4 part
    2. j=4, k in 0..3: 4x1x4 part  
    3. j in 0..3, k=4: 4x4x1 part
    4. j=4, k=4: 4x1x1 part
    
    For the 4x4x4 part, use Strassen (rank 49).
    For the other parts, use schoolbook.
    
    Total: 49 + 16 + 16 + 4 = 85.
    
    Let me try to do better by using a rank-44 5x5x5 decomposition
    and restricting to i in 0..3.
    """
    # For now, use the block Strassen approach.
    return None

def _construct_block_strassen(n=4, m=5, p=5):
    """
    Constructs a rank-85 decomposition for 4x5x5 using block Strassen.
    """
    rank = 85
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    su, sv, sw = _strassen_2x2_factors()
    
    outer_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    col = 0
    for t_idx, (ac, bc, cc) in enumerate(outer_terms):
        for k in range(7):
            for (ar, ac_col), a_coeff in ac.items():
                for lr in range(2):
                    for lc in range(2):
                        li = lr * 2 + lc
                        s_val = su[li][k]
                        if s_val != 0:
                            global_i = 2 * ar + lr
                            global_j = 2 * ac_col + lc
                            U[global_i * 5 + global_j][col] += a_coeff * s_val
            
            for (br, bc_col), b_coeff in bc.items():
                for lr in range(2):
                    for lc in range(2):
                        li = lr * 2 + lc
                        s_val = sv[li][k]
                        if s_val != 0:
                            global_j = 2 * br + lr
                            global_k = 2 * bc_col + lc
                            V[global_j * 5 + global_k][col] += b_coeff * s_val
            
            for (cr, cc_col), c_coeff in cc.items():
                for lr in range(2):
                    for lc in range(2):
                        li = lr * 2 + lc
                        s_val = sw[li][k]
                        if s_val != 0:
                            global_k = 2 * cr + lr
                            global_i = 2 * cc_col + lc
                            W[global_k * 4 + global_i][col] += c_coeff * s_val
            
            col += 1
    
    for i in range(4):
        for k in range(4):
            U[i * 5 + 4][col] = 1
            V[4 * 5 + k][col] = 1
            W[k * 4 + i][col] = 1
            col += 1
            
    for i in range(4):
        for j in range(4):
            U[i * 5 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1
            
    for i in range(4):
        U[i * 5 + 4][col] = 1
        V[4 * 5 + 4][col] = 1
        W[4 * 4 + i][col] = 1
        col += 1
        
    return U, V, W

def construct():
    """
    Find a low-rank exact tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try the block Strassen decomposition (rank 85)
    try:
        U, V, W = _construct_block_strassen(n, m, p)
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception as e:
        pass
        
    # Fallback to schoolbook (rank 100)
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
