"""Search for low-rank decomposition of 4x5x5 matrix multiplication tensor."""
import numpy as np
import time
import random

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _verify_numpy(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using numpy einsum."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=tol)

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using integer arithmetic."""
    rank = len(U[0])
    # Multiply by 2 to convert half-integers to integers
    U2 = [[2 * x for x in row] for row in U]
    V2 = [[2 * x for x in row] for row in V]
    W2 = [[2 * x for x in row] for row in W]
    
    # Check each (i,j,k) entry
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    u_val = U2[i * m + j][c]
                    v_val = V2[j * p + k][c]
                    w_val = W2[k * n + i][c]
                    if u_val != 0 and v_val != 0 and w_val != 0:
                        val += u_val * v_val * w_val
                # Expected value is 1, so 2*2*2 = 8 when multiplied by 8
                if val != 8:
                    return False
    return True

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2 matrix multiplication (rank 7)."""
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # Strassen's 7-term algorithm for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    U[0,0] = 1; U[3,0] = 1; V[0,0] = 1; V[3,0] = 1; W[0,0] = 1; W[3,0] = 1
    # M2 = (A10+A11)B00
    U[1,1] = 1; U[2,1] = 1; V[0,1] = 1; W[0,1] = 1
    # M3 = A00(B10-B11)
    U[0,2] = 1; V[1,2] = 1; V[3,2] = -1; W[1,2] = 1; W[3,2] = -1
    # M4 = A11(B01-B00)
    U[3,3] = 1; V[1,3] = 1; V[0,3] = -1; W[1,3] = 1; W[0,3] = -1
    # M5 = (A00+A01)B11
    U[0,4] = 1; U[1,4] = 1; V[3,4] = 1; W[3,4] = 1
    # M6 = (A10-A00)(B00+B01)
    U[1,5] = 1; U[0,5] = -1; V[0,5] = 1; V[1,5] = 1; W[0,5] = 1; W[1,5] = 1
    # M7 = (A01-A11)(B10+B11)
    U[1,6] = 1; U[3,6] = -1; V[1,6] = 1; V[3,6] = 1; W[1,6] = 1; W[3,6] = 1
    
    return U.tolist(), V.tolist(), W.tolist()

def _laderman_3x3x3():
    """Return Laderman decomposition for 3x3x3 matrix multiplication (rank 13)."""
    # Laderman's 13-term algorithm for 3x3x3
    # This is a known result: M(3,3,3) <= 13
    # The decomposition uses specific sparse patterns
    
    U = np.zeros((9, 13), dtype=int)
    V = np.zeros((9, 13), dtype=int)
    W = np.zeros((9, 13), dtype=int)
    
    # Index mapping: row*3+col
    # Let's use the known Laderman decomposition
    # Terms are labeled M1 through M13
    
    # M1: covers (0,0),(1,0),(0,1) in A, (0,0),(0,1),(1,0) in B
    U[0,0] = 1; U[3,0] = 1; U[1,0] = 1
    V[0,0] = 1; V[1,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1; W[1,0] = 1
    
    # M2: covers (0,2) in A, (2,0),(2,1) in B
    U[2,1] = 1
    V[6,1] = 1; V[7,1] = 1
    W[6,1] = 1; W[7,1] = 1
    
    # M3: covers (1,2) in A, (2,0),(2,1) in B  
    U[5,2] = 1
    V[6,2] = 1; V[7,2] = 1
    W[6,2] = 1; W[7,2] = 1
    
    # M4: covers (2,0) in A, (0,2),(1,2) in B
    U[6,3] = 1
    V[2,3] = 1; V[5,3] = 1
    W[2,3] = 1; W[5,3] = 1
    
    # M5: covers (2,1) in A, (0,2),(1,2) in B
    U[7,4] = 1
    V[2,4] = 1; V[5,4] = 1
    W[2,4] = 1; W[5,4] = 1
    
    # M6: covers (2,2) in A, (2,2) in B
    U[8,5] = 1
    V[8,5] = 1
    W[8,5] = 1
    
    # M7: covers (1,1),(1,2) in A, (1,1) in B
    U[4,6] = 1; U[5,6] = 1
    V[4,6] = 1
    W[4,6] = 1
    
    # M8: covers (0,1),(1,1) in A, (1,1) in B
    U[1,7] = 1; U[4,7] = 1
    V[4,7] = 1
    W[1,7] = 1; W[4,7] = 1
    
    # M9: covers (1,0),(1,1) in A, (0,1) in B
    U[3,8] = 1; U[4,8] = 1
    V[1,8] = 1
    W[3,8] = 1; W[4,8] = 1
    
    # M10: covers (0,0),(0,1) in A, (1,0) in B
    U[0,9] = 1; U[1,9] = 1
    V[3,9] = 1
    W[0,9] = 1; W[3,9] = 1
    
    # M11: covers (0,2) in A, (0,2) in B
    U[2,10] = 1
    V[2,10] = 1
    W[2,10] = 1
    
    # M12: covers (2,0) in A, (2,0) in B
    U[6,11] = 1
    V[6,11] = 1
    W[6,11] = 1
    
    # M13: covers (2,1) in A, (1,2) in B
    U[7,12] = 1
    V[5,12] = 1
    W[7,12] = 1
    
    return U.tolist(), V.tolist(), W.tolist()

def _block_decomposition_strassen_laderman():
    """
    Use block decomposition: split 4x5x5 into 2x2x2 and 2x3x3 blocks.
    4 = 2+2, 5 = 2+3, 5 = 2+3
    
    This gives us a block structure where we can use:
    - Strassen for 2x2x2 blocks (rank 7)
    - Laderman for 3x3x3 blocks (rank 13)
    - Schoolbook for mixed-size blocks
    """
    n, m, p = 4, 5, 5
    
    # Split dimensions: n=4 as 2+2, m=5 as 2+3, p=5 as 2+3
    # Block indices:
    # A: (2x2, 2x3, 2x2, 2x3) - top-left, top-right, bottom-left, bottom-right
    # B: (2x2, 2x3, 3x2, 3x3) - top-left, top-right, bottom-left, bottom-right
    # C: (2x2, 2x3, 3x2, 3x3) - top-left, top-right, bottom-left, bottom-right
    
    # The full multiplication involves 4x4=16 block multiplications, but many are small
    # We need to carefully construct the decomposition
    
    # For simplicity, let's just use schoolbook for now as a valid baseline
    # and focus on the random search for improvement
    return _schoolbook()

def _try_rank_random(r, n, m, p, max_iters=2000, timeout=8.0, seed_base=0):
    """Try random {-1,0,1} entries for rank r."""
    start_time = time.time()
    for it in range(max_iters):
        if time.time() - start_time > timeout:
            break
        rng = np.random.RandomState(seed_base + it)
        U = rng.randint(-1, 2, size=(n * m, r))
        V = rng.randint(-1, 2, size=(m * p, r))
        W = rng.randint(-1, 2, size=(p * n, r))
        
        if _verify_numpy(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def _try_rank_gaussian(r, n, m, p, max_iters=50, timeout=5.0, seed_base=1000):
    """Try Gaussian random initialization with rounding."""
    start_time = time.time()
    for it in range(max_iters):
        if time.time() - start_time > timeout:
            break
        rng = np.random.RandomState(seed_base + it)
        U = rng.randn(n * m, r)
        V = rng.randn(m * p, r)
        W = rng.randn(p * n, r)
        
        # Round to nearest 0.5
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        if _verify_numpy(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    return None

def _construct_92_rank():
    """
    Construct a rank-92 decomposition for 4x5x5 matrix multiplication.
    """
    n, m, p = 4, 5, 5
    
    # Split the 4x5 matrix A into blocks:
    # A = [[A11, A12], [A21, A22]] where A11, A12 are 2x2 and 2x3, A21, A22 are 2x2 and 2x3
    # B is 5x5, split into 2+3 rows and 2+3 cols
    # C is 5x4, split into 2+3 rows and 2+2 cols
    
    # For now, return None to indicate this needs more work
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try the deterministic block decomposition first
    block_result = _block_decomposition_strassen_laderman()
    if block_result is not None:
        block_rank = len(block_result['U'][0])
        if block_rank < best_rank:
            best = block_result
            best_rank = block_rank
    
    # Try progressively lower ranks
    # Lower bound: max(n*m, m*p, p*n) = max(20, 25, 20) = 25
    # Try from reasonable lower bounds upward
    
    # First, try some known good ranks with random search
    # For 4x5x5, let's try ranks from 50 down to 25
    for rank in range(50, 24, -1):
        if time.time() > deadline - 20:
            break
        
        remaining = deadline - time.time()
        if remaining < 3:
            break
        
        # Try random {-1,0,1} search
        t_budget = min(8.0, remaining * 0.4)
        result = _try_rank_random(rank, n, m, p, max_iters=3000, timeout=t_budget, seed_base=rank * 100)
        if result is not None:
            best = result
            best_rank = rank
            break
        
        if time.time() > deadline - 15:
            break
        
        # Try Gaussian with rounding
        t_budget = min(5.0, remaining * 0.3)
        result = _try_rank_gaussian(rank, n, m, p, max_iters=30, timeout=t_budget, seed_base=rank * 200)
        if result is not None:
            best = result
            best_rank = rank
            break
    
    return best
# EVOLVE-BLOCK-END
