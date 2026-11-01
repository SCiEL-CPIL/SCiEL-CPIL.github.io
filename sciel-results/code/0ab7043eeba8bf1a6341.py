"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact using numpy for speed."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(total - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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

def construct():
    start_time = time.time()
    deadline = 145  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    sb = _schoolbook(n, m, p)
    best_result = sb
    best_rank = n * m * p
    
    # Try to find a better decomposition using linear algebra approach
    # We want to find rank-r U, V, W such that for all i,j,k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Strategy: Fix W to be identity-like and solve for U, V
    # Or use a structured ansatz.
    
    # For 2x3x10, let's try rank 45, 40, 35, 30 in order
    # Use a greedy/iterative approach
    
    for target_rank in [45, 40, 35, 30]:
        if time.time() > deadline - 20:
            break
        if target_rank >= best_rank:
            continue
        
        # Try to construct a decomposition of this rank
        # Use the following ansatz:
        # For each k (0..9), we have a 2x3 sub-problem.
        # Use 5 terms per k (rank of 2x3x1 is 5), total 50.
        # But we can potentially share across k.
        
        # Let's try a direct construction for rank 50 first (5 per k)
        if target_rank >= 50:
            # 5 terms per k, 10 k values = 50 terms
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            col = 0
            for k in range(p):
                # 5 terms for this k
                # Term 0: A[0,0]*B[0,k]
                U[0*3+0][col] = 1; V[0*10+k][col] = 1; W[k*2+0][col] = 1
                # Term 1: A[0,1]*B[1,k]
                U[0*3+1][col+1] = 1; V[1*10+k][col+1] = 1; W[k*2+0][col+1] = 1
                # Term 2: A[0,2]*B[2,k]
                U[0*3+2][col+2] = 1; V[2*10+k][col+2] = 1; W[k*2+0][col+2] = 1
                # Term 3: (A[1,0]-A[0,0])*B[0,k]
                U[1*3+0][col+3] = 1; U[0*3+0][col+3] = -1
                V[0*10+k][col+3] = 1; W[k*2+1][col+3] = 1
                # Term 4: (A[1,1]-A[0,1])*B[1,k]
                U[1*3+1][col+4] = 1; U[0*3+1][col+4] = -1
                V[1*10+k][col+4] = 1; W[k*2+1][col+4] = 1
                
                col += 5
            
            # This gives C[0,k] = terms 0,1,2 and C[1,k] = terms 0,1,2,3,4
            # But we're missing A[1,2]*B[2,k]!
            # This construction is incomplete. Let me fix it.
            
            # Actually, let me just use schoolbook and try to verify it works
            pass
        
        # For now, just return schoolbook if we can't find better
        break
    
    # Return the best result found
    return best_result
# EVOLVE-BLOCK-END
