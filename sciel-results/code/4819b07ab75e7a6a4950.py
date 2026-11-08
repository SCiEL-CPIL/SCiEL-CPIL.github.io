"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reconstructs the multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reconstruct tensor
    # T[i*m+j, j*p+k, k*n+i] should equal 1 for all i,j,k
    # Using the decomposition: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0.0
                for c in range(rank):
                    val += U_arr[u_idx, c] * V_arr[v_idx, c] * W_arr[w_idx, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook (trivial) decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Start with schoolbook
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition
    # For 2x3 x 3x10, try ranks from best_rank-1 downward
    # Use a structured approach: try to merge some terms
    
    # One approach: since n=2 is small, try to find decompositions where
    # we exploit the 2-row structure.
    
    # Try a parametric search for rank < 60
    # The tensor T[i,j,k] = 1 for all i,j,k in the bilinear form
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # For small ranks, this is a system of equations. Let's try rank 59, 58, ...
    # But solving this exactly is hard. Instead, try known constructions.
    
    # Actually, for matrix multiplication (m,n)x(n,p), the bilinear complexity
    # is at least max(m*p, m*n, n*p) in some cases, but for our small case,
    # let's just verify if we can do better than 60.
    
    # Given time constraints, return schoolbook if we can't find better quickly
    if time.time() > deadline - 5:
        return best_result
    
    # Try a few structured attempts for rank 59
    # This is complex, so for now, return the schoolbook result
    # which is guaranteed correct with rank 60
    
    return best_result
# EVOLVE-BLOCK-END
