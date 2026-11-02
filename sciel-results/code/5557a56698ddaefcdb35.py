import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p
    
    # Try to find a lower rank decomposition
    # Known rank for 2x3x4 is 18.
    # We will try ranks from 12 up to 23.
    
    best_rank = rank_school
    best_solution = None
    
    # Precompute indices
    # U: (n*m, R) -> index i*m + j
    # V: (m*p, R) -> index j*p + k
    # W: (p*n, R) -> index k*n + i
    
    # List of (i, j, k) tuples
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    # Helper to compute S_ijk and gradient
    def compute_error_and_grad(U, V, W):
        R = U.shape[1]
        S = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # S[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    S[i,j,k] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
        
        # Error
        error = np.sum((S - 1.0)**2)
        
        # Gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    diff = S[i,j,k] - 1.0
                    # dU[i*m+j, r] += 2 * diff * V[j*p+k, r] * W[k*n+i, r]
                    dU[i*m+j] += 2.0 * diff * (V[j*p+k] * W[k*n+i])
                    # dV[j*p+k, r] += 2 * diff * U[i*m+j, r] * W[k*n+i, r]
                    dV[j*p+k] += 2.0 * diff * (U[i*m+j] * W[k*n+i])
                    # dW[k*n+i, r] += 2 * diff * U[i*m+j, r] * V[j*p+k, r]
                    dW[k*n+i] += 2.0 * diff * (U[i*m+j] * V[j*p+k])
                    
        return error, dU, dV, dW

    def snap_to_half_int(mat):
        # Round to nearest 0.5
        return np.round(mat * 2.0) / 2.0

    def verify(U, V, W):
        # Check if exact
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        return False
        return True

    start_time = time.time()
    time_limit = 100.0 # Leave 20s margin
    
    # Try ranks from 12 to 23
    for R in range(12, rank_school):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random initializations for this rank
        attempts = 0
        max_attempts = 50 if R < 15 else 20
        
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize with small random integers
            # Scale matters. Let's try values in {-1, 0, 1}
            U = np.random.choice([-1, 0, 1], size=(n*m, R)).astype(float)
            V = np.random.choice([-1, 0, 1], size=(m*p, R)).astype(float)
            W = np.random.choice([-1, 0, 1], size=(p*n, R)).astype(float)
            
            # Gradient Descent
            lr = 0.01
            # Adaptive learning rate?
            
            for step in range(200):
                error, dU, dV, dW = compute_error_and_grad(U, V, W)
                if error < 1e-8:
                    break
                    
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Optional: periodic snap to keep values bounded?
                # No, let it run.
                
            # Snap to half-integers
            U_snap = snap_to_half_int(U)
            V_snap = snap_to_half_int(V)
            W_snap = snap_to_half_int(W)
            
            if verify(U_snap, V_snap, W_snap):
                best_rank = R
                best_solution = (U_snap, V_snap, W_snap)
                break # Found for this R, try lower R? No, R is increasing.
                # Wait, I am iterating R from 12 to 23.
                # So if I find one for R=12, I break the outer loop?
                # Yes, because I want the minimum rank.
                
        if best_solution is not None:
            break

    if best_solution is not None:
        U, V, W = best_solution
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook
        rank = rank_school
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
        return {
            "U": U,
            "V": V,
            "W": W
        }
