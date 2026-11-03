"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Strategy: Construct a rank-32 decomposition for 3x3x4 matrix multiplication
    # by combining a rank-23 decomposition for 3x3x3 with rank-9 for the extra column.
    
    # For 3x3x3: Use a known rank-23 decomposition based on Strassen-like methods
    # For the 4th column: Use schoolbook rank-9
    
    # Let's construct the 3x3x3 rank-23 decomposition first.
    # We'll use the following approach based on block partitioning:
    # A (3x3) = [[A11(2x2), A12(2x1)], [A21(1x2), A22(1x1)]]
    # B (3x3) = [[B11(2x2), B12(2x1)], [B21(1x2), B22(1x1)]]
    # C (3x3) = [[C11(2x2), C12(2x1)], [C21(1x2), C22(1x1)]]
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # For C11 (2x2 = 2x2*2x2 + 2x1*1x2):
    #   Use Strassen for 2x2*2x2 (rank 7)
    #   The 2x1*1x2 part: rank 4
    #   Total: 11
    
    # For C12 (2x1 = 2x2*2x1 + 2x1*1x1):
    #   2x2*2x1: rank 4 (schoolbook)
    #   2x1*1x1: rank 2
    #   Total: 6
    
    # For C21 (1x2 = 1x2*2x2 + 1x1*1x2):
    #   1x2*2x2: rank 4
    #   1x1*1x2: rank 2
    #   Total: 6
    
    # For C22 (1x1 = 1x2*2x1 + 1x1*1x1):
    #   1x2*2x1: rank 2
    #   1x1*1x1: rank 1
    #   Total: 3
    
    # Grand total: 11 + 6 + 6 + 3 = 26 (not 23, but let's try to optimize)
    
    # Actually, let me try a direct search approach with better initialization.
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try ranks from 30 to 36
    best_rank = n * m * p
    best_decomp = None
    
    for r in range(30, best_rank + 1):
        if time.time() > deadline:
            break
        
        found = False
        num_trials = 100 if r <= 33 else 50
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + trial * 7919 + r * 104729)
            
            # Initialize with small random values
            U = np.random.uniform(-0.5, 0.5, (n*m, r))
            V = np.random.uniform(-0.5, 0.5, (m*p, r))
            W = np.random.uniform(-0.5, 0.5, (p*n, r))
            
            # Use gradient descent to minimize the error
            lr = 0.01
            for step in range(500):
                if time.time() > deadline:
                    break
                
                # Compute gradients
                # T_decomp[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
                # dT/deU[a,t] = V[b,t] * W[c,t]
                # dT/deV[b,t] = U[a,t] * W[c,t]  
                # dT/deW[c,t] = U[a,t] * V[b,t]
                
                error = 0.0
                U_grad = np.zeros_like(U)
                V_grad = np.zeros_like(V)
                W_grad = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            
                            s = 0.0
                            for t in range(r):
                                s += U[a, t] * V[b, t] * W[c, t]
                            err = s - 1.0
                            error += err * err
                            
                            for t in range(r):
                                U_grad[a, t] += 2 * err * V[b, t] * W[c, t]
                                V_grad[b, t] += 2 * err * U[a, t] * W[c, t]
                                W_grad[c, t] += 2 * err * U[a, t] * V[b, t]
                
                # Update
                U -= lr * U_grad
                V -= lr * V_grad
                W -= lr * W_grad
                
                if error < 1e-8:
                    break
            
            # Check if we found an exact solution
            if error < 1e-6:
                # Snap to nearest half-integers
                U_snapped = np.round(U * 2) / 2
                V_snapped = np.round(V * 2) / 2
                W_snapped = np.round(W * 2) / 2
                
                # Verify
                T_check = np.einsum('at,bt,ct->abc', U_snapped, V_snapped, W_snapped)
                if np.allclose(T_check, T, atol=1e-6):
                    best_rank = r
                    best_decomp = (U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist())
                    found = True
                    break
        
        if found:
            break
    
    if best_decomp is not None:
        U, V, W = best_decomp
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
