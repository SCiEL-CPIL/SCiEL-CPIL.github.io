import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def validate_exact(U, V, W, n, m, p):
    """Validate with exact fraction arithmetic."""
    r = len(W[0]) if W else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = Fraction(0, 1)
                for c in range(r):
                    val += Fraction(U[idx_u][c]) * Fraction(V[idx_v][c]) * Fraction(W[idx_w][c])
                if val != Fraction(1, 1):
                    return False
    return True

def calculate_defects(U, V, W, n, m, p):
    """Calculate the difference between the decomposed tensor and the target tensor (all 1s)."""
    r = len(W[0]) if W else 0
    defects = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = 0.0
                for c in range(r):
                    val += U[idx_u][c] * V[idx_v][c] * W[idx_w][c]
                defects.append(val - 1.0)
    return defects

def integer_local_search(n, m, p, rank, deadline, max_iter=5000):
    """
    Perform local search on integer/half-integer grid.
    """
    # Initialize with small random integers
    U = np.random.randint(-2, 3, size=(n * m, rank))
    V = np.random.randint(-2, 3, size=(m * p, rank))
    W = np.random.randint(-2, 3, size=(p * n, rank))
    
    current_defects = calculate_defects(U.tolist(), V.tolist(), W.tolist(), n, m, p)
    current_error = np.sum(np.square(current_defects))
    
    if current_error < 1e-9:
        return U.tolist(), V.tolist(), W.tolist()

    step_options = [0.5, -0.5, 1.0, -1.0, 1.5, -1.5]
    
    for _ in range(max_iter):
        if time.time() > deadline - 2:
            break
            
        mat_idx = np.random.randint(0, 3)
        if mat_idx == 0:
            mat = U
            rows = n * m
        elif mat_idx == 1:
            mat = V
            rows = m * p
        else:
            mat = W
            rows = p * n
            
        row = np.random.randint(0, rows)
        col = np.random.randint(0, rank)
        
        original_val = mat[row, col]
        step = random.choice(step_options)
        new_val = original_val + step
        
        if abs(new_val) > 5:
            continue
            
        mat[row, col] = new_val
        new_defects = calculate_defects(U.tolist(), V.tolist(), W.tolist(), n, m, p)
        new_error = np.sum(np.square(new_defects))
        
        if new_error < current_error:
            current_error = new_error
            current_defects = new_defects
            if current_error < 1e-9:
                return U.tolist(), V.tolist(), W.tolist()
        else:
            mat[row, col] = original_val
            
    return None

def schoolbook_decomposition(n, m, p):
    """Standard rank n*m*p decomposition."""
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
    return U, V, W

def strassen_2x2x2():
    """
    Return Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
    T_{2,2,2} has rank 7 (not 8).
    """
    # Strassen's 7 products
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U, V, W matrices for 2x2x2
    # U: (n*m, rank) = (4, 7)
    # V: (m*p, rank) = (4, 7)
    # W: (p*n, rank) = (4, 7)
    
    # Indexing: U[i*m+j], V[j*p+k], W[k*n+i]
    # n=2, m=2, p=2
    
    U = [
        [1, 0, 1, 0, 1, 0, 0],  # i=0,j=0: A00
        [0, 0, 0, 0, 0, 1, 0],  # i=0,j=1: A01
        [0, 1, 0, 0, 0, -1, 0], # i=1,j=0: A10
        [1, 1, 0, 1, 0, 0, -1]  # i=1,j=1: A11
    ]
    
    V = [
        [1, 1, 0, 0, 0, 1, 0],  # j=0,k=0: B00
        [0, 0, 1, -1, 0, 1, 0], # j=0,k=1: B01
        [1, 0, -1, 1, 1, 0, 1], # j=1,k=0: B10
        [1, 0, 0, 0, 1, 0, 1]   # j=1,k=1: B11
    ]
    
    W = [
        [0, 0, 0, 0, 0, 0, 1],  # k=0,i=0: C00
        [0, 0, 1, 0, 1, 0, 0],  # k=0,i=1: C01
        [1, 1, 0, 0, 0, 0, 0],  # k=1,i=0: C10
        [1, 0, 0, 1, 0, 0, 0]   # k=1,i=1: C11
    ]
    
    return U, V, W

def combine_decompositions(decomp_list, n, m, p):
    """
    Combine multiple tensor decompositions into one.
    Each decomp_list[i] is (U_i, V_i, W_i) for a sub-problem.
    We concatenate them along the rank dimension.
    """
    total_rank = sum(len(d[2][0]) for d in decomp_list)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    offset = 0
    for U_i, V_i, W_i in decomp_list:
        r_i = len(W_i[0])
        for i in range(len(U_i)):
            for c in range(r_i):
                U[i][offset + c] = U_i[i][c]
        for i in range(len(V_i)):
            for c in range(r_i):
                V[i][offset + c] = V_i[i][c]
        for i in range(len(W_i)):
            for c in range(r_i):
                W[i][offset + c] = W_i[i][c]
        offset += r_i
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s buffer
    
    # Start with schoolbook as fallback
    U, V, W = schoolbook_decomposition(n, m, p)
    best_result = (U, V, W)
    best_rank = len(W[0])
    
    # Try structured decomposition using Strassen for 2x2 blocks
    # Partition 5 = 3 + 2
    # We can decompose T_{2,5,5} into sub-tensors
    # For now, try a simpler approach: use Strassen on 2x2 sub-blocks
    
    # Strategy: Use Strassen's algorithm on the 2x2 parts
    # The 5x5 matrix can be viewed as having 2x2 blocks
    # We can apply Strassen to reduce rank
    
    # First, let's try a direct Strassen-based approach
    # For 2x5x5, we can partition the middle dimension (5) into 2+3
    # and the last dimension (5) into 2+3
    
    # Actually, let's try a more systematic approach:
    # Use the fact that T_{2,5,5} can be decomposed using
    # known optimal decompositions for smaller cases
    
    # Try rank 46 (which was achieved in another solution using Strassen)
    # and then try to improve further
    
    candidate_ranks = [46, 45, 44, 43, 42, 41, 40]
    
    for rank in candidate_ranks:
        if rank >= best_rank:
            continue
        if time.time() > deadline - 10:
            break
            
        # Try multiple random starts for local search
        found = False
        for attempt in range(15):
            if time.time() > deadline - 5:
                break
                
            result = integer_local_search(n, m, p, rank, deadline, max_iter=3000)
            if result:
                U, V, W = result
                best_result = (U, V, W)
                best_rank = rank
                found = True
                break
        
        if found:
            pass

    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
