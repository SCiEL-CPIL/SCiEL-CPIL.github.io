import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin for 150s limit
    
    n, m, p = 3, 5, 7
    nm = n * m  # 15
    mp = m * p  # 35
    pn = p * n  # 21
    
    schoolbook_rank = n * m * p  # 105
    
    # Schoolbook solution as fallback
    U_school = [[0] * schoolbook_rank for _ in range(nm)]
    V_school = [[0] * schoolbook_rank for _ in range(mp)]
    W_school = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank
    
    def verify(U, V, W, rank):
        """Verify if the decomposition is exact with integer/half-integer entries."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def snap_to_half_integers(U, V, W):
        """Snap values to nearest half-integer."""
        U_s = [[round(x * 2) / 2 for x in row] for row in U]
        V_s = [[round(x * 2) / 2 for x in row] for row in V]
        W_s = [[round(x * 2) / 2 for x in row] for row in W]
        return U_s, V_s, W_s

    def alternating_minimization(U, V, W, rank, max_iters=200):
        """
        Run alternating least squares to minimize the error.
        Returns U, V, W, error
        """
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Fix V, W; solve for U
            # For each (i, j), we need U[i,j,:] such that sum_k U[i,j,:] * (V[j,:] * W[:,i]) = 1
            # This is a linear system: M_u[i,j] * u = 1
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # Vector of length rank: V[j*p + k, r] * W[k*n + i, r] for all k, r
                    # We want sum_k (V[j*p+k, r] * W[k*n+i, r]) * U[i*m+j, r] = 1
                    # Let A[r] = sum_k V[j*p+k, r] * W[k*n+i, r]
                    # Then sum_r A[r] * U[i*m+j, r] = 1
                    # This is a single equation with rank unknowns. Underdetermined if rank > 1.
                    # We want the minimum norm solution.
                    A_u = np.zeros(rank)
                    for k in range(p):
                        A_u += V[j*p+k, :] * W[k*n+i, :]
                    # Solve A_u . u = 1
                    # If A_u is zero, no solution, keep old
                    if np.linalg.norm(A_u) < 1e-12:
                        U_new[i*m+j, :] = U[i*m+j, :]
                    else:
                        # Minimum norm solution: u = A_u^T / ||A_u||^2
                        U_new[i*m+j, :] = A_u / np.dot(A_u, A_u)
            U = U_new
            
            # Fix U, W; solve for V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    # sum_i U[i*m+j, r] * W[k*n+i, r] * V[j*p+k, r] = 1
                    A_v = np.zeros(rank)
                    for i in range(n):
                        A_v += U[i*m+j, :] * W[k*n+i, :]
                    if np.linalg.norm(A_v) < 1e-12:
                        V_new[j*p+k, :] = V[j*p+k, :]
                    else:
                        V_new[j*p+k, :] = A_v / np.dot(A_v, A_v)
            V = V_new
            
            # Fix U, V; solve for W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    # sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                    A_w = np.zeros(rank)
                    for j in range(m):
                        A_w += U[i*m+j, :] * V[j*p+k, :]
                    if np.linalg.norm(A_w) < 1e-12:
                        W_new[k*n+i, :] = W[k*n+i, :]
                    else:
                        W_new[k*n+i, :] = A_w / np.dot(A_w, A_w)
            W = W_new
            
            # Calculate error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        err += (val - 1.0) ** 2
                        
            if err < 1e-12:
                break
                
        return U, V, W, err

    # Search for lower ranks
    # Lower bound is max(nm, mp, pn) = 35
    # We try ranks from 35 upwards
    # To save time, we limit the number of attempts per rank
    
    lower_bound = 35
    # We will try up to schoolbook_rank - 1
    # If we find a valid decomposition, we return it immediately as it's the lowest rank found so far
    # We must check ranks in increasing order
    
    for rank in range(lower_bound, schoolbook_rank):
        if time.time() > deadline - 10:
            break
            
        # Number of attempts decreases as rank increases? Or constant?
        # Let's try 50 attempts for low ranks, 20 for high ranks
        attempts = 50 if rank < 50 else 20
        
        found = False
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            # Scale initialization to help convergence
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # Run ALS
            U_f, V_f, W_f, err = alternating_minimization(U, V, W, rank, max_iters=100)
            
            if err < 1e-8:
                # Snap to half-integers
                U_s, V_s, W_s = snap_to_half_integers(U_f.tolist(), V_f.tolist(), W_f.tolist())
                
                if verify(U_s, V_s, W_s, rank):
                    best_result = {"U": U_s, "V": V_s, "W": W_s}
                    best_rank = rank
                    found = True
                    break
        
        if found:
            break
            
    return best_result
