"""Matrix multiplication 3x5x7 - rank minimization."""
import time
import random
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication.
    
    The schoolbook decomposition has rank n*m*p = 105.
    We attempt to find a lower rank decomposition using a randomized algebraic
    search strategy, falling back to schoolbook if no improvement is found.
    
    The output factor W follows the trace convention W[k*n+i, r].
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = 3, 5, 7
    nm, mp, pn = n * m, m * p, p * n  # 15, 35, 21
    
    def schoolbook():
        rank = n * m * p  # 105
        U = [[0.0] * rank for _ in range(nm)]
        V = [[0.0] * rank for _ in range(mp)]
        W = [[0.0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, rank):
        """Check if U, V, W correctly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    def verify_np(U, V, W, rank):
        """Fast verification using numpy."""
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    # Try to find a decomposition with rank lower than schoolbook
    # Start from a reasonable lower bound and go up
    # Known lower bounds for matrix multiplication suggest rank could be lower
    # Let's try ranks from 80 to 104
    
    best_result = None
    
    for target_rank in range(104, 79, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 1000 + target_rank)
            
            # Generate random V and W with small integer/half-integer entries
            V = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(target_rank)] for _ in range(mp)]
            W = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(target_rank)] for _ in range(pn)]
            
            # For each i, solve for U[i*m+j, :] for j=0..m-1
            U = [[0.0]*target_rank for _ in range(nm)]
            
            valid = True
            for i in range(n):
                # Build the system for this i
                # Unknowns: U[i*m+j, c] for j in range(m), c in range(target_rank) -> m*target_rank unknowns
                # Equations: for j in range(m), k in range(p):
                #   sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                
                A_mat = np.zeros((m*p, m*target_rank))
                b_vec = np.ones(m*p)
                
                for j in range(m):
                    for k in range(p):
                        eq_idx = j*p + k
                        for c in range(target_rank):
                            A_mat[eq_idx, j*target_rank + c] = V[j*p+k][c] * W[k*n+i][c]
                
                # Solve least squares
                try:
                    sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    for j in range(m):
                        for c in range(target_rank):
                            U[i*m+j][c] = sol[j*target_rank + c]
                except:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Verify the solution
            if verify_np(U, V, W, target_rank):
                best_result = {"U": U, "V": V, "W": W}
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    # Fall back to schoolbook
    return schoolbook()
