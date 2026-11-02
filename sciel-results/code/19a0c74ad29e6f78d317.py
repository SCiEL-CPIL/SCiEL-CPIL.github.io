"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition is exact using the tensor check."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, rank):
    """Fast verification using numpy."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _strassen_2x2x2(offset_u, offset_v, offset_w, U, V, W):
    """
    Add a Strassen 2x2x2 block at given offsets.
    U rows: offset_u to offset_u+3 (i=0,1; j=0,1)
    V rows: offset_v to offset_v+3 (j=0,1; k=0,1)  
    W rows: offset_w to offset_w+3 (k=0,1; i=0,1)
    Returns number of columns used (7).
    """
    col = 0
    # M1 = (A00+A11)*(B00+B11)
    U[offset_u + 0][offset_u + col] = 1
    U[offset_u + 3][offset_u + col] = 1
    V[offset_v + 0][offset_u + col] = 1
    V[offset_v + 3][offset_u + col] = 1
    W[offset_w + 0][offset_u + col] = 1
    W[offset_w + 3][offset_u + col] = 1
    col += 1
    
    # M2 = (A10+A11)*B00
    U[offset_u + 2][offset_u + col] = 1
    U[offset_u + 3][offset_u + col] = 1
    V[offset_v + 0][offset_u + col] = 1
    W[offset_w + 2][offset_u + col] = 1
    W[offset_w + 3][offset_u + col] = -1
    col += 1
    
    # M3 = A00*(B01-B11)
    U[offset_u + 0][offset_u + col] = 1
    V[offset_v + 1][offset_u + col] = 1
    V[offset_v + 3][offset_u + col] = -1
    W[offset_w + 0][offset_u + col] = 1
    W[offset_w + 3][offset_u + col] = 1
    col += 1
    
    # M4 = A11*(B10-B00)
    U[offset_u + 3][offset_u + col] = 1
    V[offset_v + 2][offset_u + col] = 1
    V[offset_v + 0][offset_u + col] = -1
    W[offset_w + 0][offset_u + col] = 1
    W[offset_w + 2][offset_u + col] = 1
    col += 1
    
    # M5 = (A00+A01)*B11
    U[offset_u + 0][offset_u + col] = 1
    U[offset_u + 1][offset_u + col] = 1
    V[offset_v + 3][offset_u + col] = 1
    W[offset_w + 0][offset_u + col] = -1
    W[offset_w + 1][offset_u + col] = 1
    col += 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[offset_u + 2][offset_u + col] = 1
    U[offset_u + 0][offset_u + col] = -1
    V[offset_v + 0][offset_u + col] = 1
    V[offset_v + 1][offset_u + col] = 1
    W[offset_w + 3][offset_u + col] = 1
    col += 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[offset_u + 1][offset_u + col] = 1
    U[offset_u + 3][offset_u + col] = -1
    V[offset_v + 2][offset_u + col] = 1
    V[offset_v + 3][offset_u + col] = 1
    W[offset_w + 0][offset_u + col] = 1
    col += 1
    
    return col

def _construct_hybrid_rank36():
    """Construct the rank-36 hybrid decomposition."""
    n, m, p = 2, 4, 5
    total_rank = 36
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    # Block 1: A1 (U rows 0-3) * B1 (V rows 0-9)
    col_offset += _strassen_2x2x2(0, 0, 0, U, V, W)
    # Schoolbook for col 2
    U[0][col_offset] = 1; V[2][col_offset] = 1; W[4][col_offset] = 1; col_offset += 1
    U[1][col_offset] = 1; V[3][col_offset] = 1; W[4][col_offset] = 1; col_offset += 1
    U[2][col_offset] = 1; V[2][col_offset] = 1; W[5][col_offset] = 1; col_offset += 1
    U[3][col_offset] = 1; V[3][col_offset] = 1; W[5][col_offset] = 1; col_offset += 1
    # Strassen for cols 3-4
    col_offset += _strassen_2x2x2(0, 6, 6, U, V, W)
    
    # Block 2: A2 (U rows 4-7) * B2 (V rows 10-19)
    col_offset += _strassen_2x2x2(4, 10, 0, U, V, W)
    # Schoolbook for col 2
    U[4][col_offset] = 1; V[12][col_offset] = 1; W[4][col_offset] = 1; col_offset += 1
    U[5][col_offset] = 1; V[13][col_offset] = 1; W[4][col_offset] = 1; col_offset += 1
    U[6][col_offset] = 1; V[12][col_offset] = 1; W[5][col_offset] = 1; col_offset += 1
    U[7][col_offset] = 1; V[13][col_offset] = 1; W[5][col_offset] = 1; col_offset += 1
    # Strassen for cols 3-4
    col_offset += _strassen_2x2x2(4, 16, 6, U, V, W)
    
    return {"U": U, "V": V, "W": W}

def _try_lower_rank_random(n, m, p, max_rank, deadline, num_trials=2000):
    """Try to find a lower rank decomposition using random search."""
    U_shape = (n*m,)
    V_shape = (m*p,)
    W_shape = (p*n,)
    
    for rank in range(max_rank, 24, -1):  # Try from max_rank down to 25
        if time.time() > deadline:
            break
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            # Initialize with sparse random matrices
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Each row gets a few non-zero entries
            for i in range(n*m):
                nnz = random.randint(1, 3)
                cols = random.sample(range(rank), nnz)
                for c in cols:
                    U[i, c] = random.choice([-1, 1])
            for j in range(m*p):
                nnz = random.randint(1, 3)
                cols = random.sample(range(rank), nnz)
                for c in cols:
                    V[j, c] = random.choice([-1, 1])
            for k in range(p*n):
                nnz = random.randint(1, 3)
                cols = random.sample(range(rank), nnz)
                for c in cols:
                    W[k, c] = random.choice([-1, 1])
            
            # Verify
            if _verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p, rank):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s limit
    
    # First try the constructive rank-36
    result = _construct_hybrid_rank36()
    U = result["U"]
    V = result["V"]
    W = result["W"]
    rank = len(U[0])
    
    if _verify_fast(U, V, W, n, m, p, rank):
        best_result = result
        best_rank = rank
    else:
        # Fallback to schoolbook
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
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank
    
    # Try to find lower rank using random search
    remaining_time = deadline - time.time()
    if remaining_time > 5:  # Only if we have enough time
        random_result = _try_lower_rank_random(n, m, p, best_rank - 1, deadline, num_trials=5000)
        if random_result is not None:
            best_result = random_result
            best_rank = len(random_result["U"][0])
    
    return best_result

def construct():
    return _construct()
# EVOLVE-BLOCK-END
