import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    total = n * m * p
    
    # Target tensor: T[i,j,k] = 1 for all i,j,k
    # We need: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    
    start_time = time.time()
    deadline = start_time + 110  # 110 second deadline
    
    # Try ranks from low to high
    for rank in range(20, total + 1):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations
        for trial in range(50):
            if time.time() > deadline:
                break
                
            np.random.seed(trial * 1000 + rank)
            U = np.random.randn(nm, rank) * 0.1
            V = np.random.randn(mp, rank) * 0.1
            W = np.random.randn(pn, rank) * 0.1
            
            # Use gradient descent to minimize error
            lr = 0.01
            for step in range(1000):
                if time.time() > deadline:
                    break
                    
                # Compute current decomposition
                # For each (i,j,k), compute sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
                # This is expensive, so let's vectorize
                
                # Build the tensor from current U, V, W
                # T[i,j,k] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
                
                # Reshape for computation
                # U has shape (nm, rank) = (n*m, rank)
                # V has shape (mp, rank) = (m*p, rank)
                # W has shape (pn, rank) = (p*n, rank)
                
                # For fixed t, the contribution is U[:,t] ⊗ V[:,t] ⊗ W[:,t]
                # But we need to align the indices.
                
                # Let's compute the error directly
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for t in range(rank):
                                val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                            error += (val - 1.0) ** 2
                            
                if error < 1e-6:
                    # Found a solution, round to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Verify
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0.0
                                for t in range(rank):
                                    val += U_rounded[i*m+j, t] * V_rounded[j*p+k, t] * W_rounded[k*n+i, t]
                                if abs(val - 1.0) > 1e-9:
                                    valid = False
                                    break
                        if not valid:
                            break
                    if valid:
                        return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                
                # Compute gradients (approximate)
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for t in range(rank):
                                val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                            err = val - 1.0
                            for t in range(rank):
                                dU[i*m+j, t] += 2 * err * V[j*p+k, t] * W[k*n+i, t]
                                dV[j*p+k, t] += 2 * err * U[i*m+j, t] * W[k*n+i, t]
                                dW[k*n+i, t] += 2 * err * U[i*m+j, t] * V[j*p+k, t]
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
            # After gradient descent, try rounding
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for t in range(rank):
                            val += U_rounded[i*m+j, t] * V_rounded[j*p+k, t] * W_rounded[k*n+i, t]
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                if not valid:
                    break
            if valid:
                return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
    
    # Fallback: schoolbook
    rank = total
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
