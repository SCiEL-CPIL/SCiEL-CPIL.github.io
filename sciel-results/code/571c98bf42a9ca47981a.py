import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    base_rank = n * m * p  # 80

    # Schoolbook decomposition as baseline
    U = [[0] * base_rank for _ in range(n * m)]
    V = [[0] * base_rank for _ in range(m * p)]
    W = [[0] * base_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best_rank = base_rank

    # Try to find lower rank decompositions
    # The key insight: for fixed U and V, W can be solved linearly
    # For each (i,k), we need to solve a system of equations
    
    # Let's try ranks from just below base_rank down
    for target_rank in range(base_rank - 1, 20, -1):
        if time.time() > deadline - 30:
            break
        
        found = False
        # Multiple random attempts for this rank
        max_attempts = 30
        for attempt in range(max_attempts):
            if time.time() > deadline - 30:
                break
            
            # Generate random U and V with some structure
            U_cand = np.random.randn(n * m, target_rank) * 0.5
            V_cand = np.random.randn(m * p, target_rank) * 0.5
            
            # Try to solve for W
            W_cand = np.zeros((p * n, target_rank))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build the linear system for this (i,k)
                    # Unknowns: W[k*n+i, r] for r in [0, target_rank)
                    # Equations: for all (a, b, b', c) in [0,n)x[0,m)x[0,m)x[0,p)
                    # sum_r U[a*m+b, r] * V[b'*p+c, r] * W[k*n+i, r] = delta(a,i)*delta(b,b')*delta(c,k)
                    
                    num_eq = n * m * m * p  # 4*4*4*5 = 320
                    A_sys = np.zeros((num_eq, target_rank))
                    b_vec = np.zeros(num_eq)
                    
                    eq_idx = 0
                    for a in range(n):
                        for b in range(m):
                            for bp in range(m):
                                for c in range(p):
                                    for r in range(target_rank):
                                        A_sys[eq_idx, r] = U_cand[a*m+b, r] * V_cand[bp*p+c, r]
                                    b_vec[eq_idx] = 1.0 if (a == i and b == bp and c == k) else 0.0
                                    eq_idx += 1
                    
                    # Solve the overdetermined system
                    try:
                        w_sol, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_vec, rcond=None)
                        # Check if the solution is good enough
                        if len(residuals) > 0 and residuals[0] > 1e-5:
                            valid = False
                            break
                        W_cand[k*n+i, :] = w_sol
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid:
                # Verify the decomposition
                if _verify(U_cand.tolist(), V_cand.tolist(), W_cand.tolist()):
                    best_U = U_cand.tolist()
                    best_V = V_cand.tolist()
                    best_W = W_cand.tolist()
                    best_rank = target_rank
                    found = True
                    break
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _verify(U, V, W, n=4, m=4, p=5, trials=10):
    """Verify that the decomposition correctly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    # Compute sum_a,b U[a*m+b, r] * A[a, b]
                    u_dot_a = 0.0
                    for a in range(n):
                        for b in range(m):
                            u_dot_a += U[a*m+b, r] * A[a, b]
                    
                    # Compute sum_b,c V[b*p+c, r] * B[b, c]
                    v_dot_b = 0.0
                    for b in range(m):
                        for c in range(p):
                            v_dot_b += V[b*p+c, r] * B[b, c]
                    
                    val += u_dot_a * v_dot_b * W[k*n+i, r]
                C_computed[i, k] = val
        
        if not np.allclose(C_computed, C_expected, atol=1e-4):
            return False
    
    return True
