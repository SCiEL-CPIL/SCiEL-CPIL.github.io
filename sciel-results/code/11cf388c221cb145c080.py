"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    deadline = time.time() + 140  # safety margin
    
    # The 3x3x6 multiplication tensor is the direct sum of two 3x3x3 tensors
    # (for k=0,1,2 and k=3,4,5). So rank = 2 * rank(3x3x3) = 2*21 = 42
    # We construct rank-21 decompositions for 3x3x3 and concatenate.
    
    # Try to find a rank-21 decomposition for 3x3x3
    # Use a known explicit construction or numerical search
    
    def find_3x3x3_rank21():
        """Find a rank-21 decomposition for 3x3x3 matrix multiplication."""
        # Target: T[i,j,k] = 1 for all i,j,k in {0,1,2}
        # Need U[9,r], V[9,r], W[9,r] with sum_r U[i*3+j,r]*V[j*3+k,r]*W[k*3+i,r] = 1
        
        r = 21
        # Initialize with schoolbook (rank 27) and try to reduce
        # Use a random initialization and iterative refinement
        
        best_result = None
        
        # Try multiple random restarts
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt)
            U = np.random.randn(9, r) * 0.1
            V = np.random.randn(9, r) * 0.1
            W = np.random.randn(9, r) * 0.1
            
            # Iterative refinement: minimize sum of (sum_r U*V*W - 1)^2
            for iteration in range(200):
                if time.time() > deadline:
                    break
                
                # Compute current tensor values
                # T_comp[i,j,k] = sum_r U[i*3+j,r]*V[j*3+k,r]*W[k*3+i,r]
                T_comp = np.zeros((3,3,3))
                for rr in range(r):
                    u = U[:, rr].reshape(3,3)
                    v = V[:, rr].reshape(3,3)
                    w = W[:, rr].reshape(3,3)
                    for i in range(3):
                        for j in range(3):
                            for k in range(3):
                                T_comp[i,j,k] += u[i,j] * v[j,k] * w[k,i]
                
                error = T_comp - 1.0
                err_norm = np.linalg.norm(error)
                if err_norm < 1e-10:
                    return U, V, W
                
                # Update U: gradient of error w.r.t. U[i*3+j,rr] = V[j*3+k,rr]*W[k*3+i,rr]*error[i,j,k]
                # Sum over k for each (i,j)
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for rr in range(r):
                    u = U[:, rr].reshape(3,3)
                    v = V[:, rr].reshape(3,3)
                    w = W[:, rr].reshape(3,3)
                    for i in range(3):
                        for j in range(3):
                            # d/dU[i,j,rr] = sum_k V[j,k,rr]*W[k,i,rr]*error[i,j,k]
                            sU = 0
                            sV = 0
                            for k in range(3):
                                sU += v[j,k] * w[k,i] * error[i,j,k]
                                sV += u[i,j] * w[k,i] * error[i,j,k]
                                # d/dW[k,i,rr] = u[i,j]*v[j,k]*error[i,j,k]
                            grad_U[i*3+j, rr] = sU
                            # d/dV[j,k,rr] = sum_i u[i,j]*w[k,i]*error[i,j,k]
                            for k in range(3):
                                sV_k = 0
                                for i in range(3):
                                    sV_k += u[i,j] * w[k,i] * error[i,j,k]
                                grad_V[j*3+k, rr] = sV_k
                            # d/dW[k,i,rr] = sum_j u[i,j]*v[j,k]*error[i,j,k]
                            for k in range(3):
                                sW_k = 0
                                for j in range(3):
                                    sW_k += u[i,j] * v[j,k] * error[i,j,k]
                                grad_W[k*3+i, rr] = sW_k
                
                # Gradient descent
                lr = 0.01
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                if err_norm < 1e-10:
                    break
            
            if err_norm < 1e-10:
                best_result = (U, V, W)
                break
        
        if best_result:
            return best_result
        
        # Fallback: return schoolbook rank-27
        U = np.zeros((9, 27))
        V = np.zeros((9, 27))
        W = np.zeros((9, 27))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j, col] = 1
                    V[j*3+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1
        return U, V, W
    
    # Get rank-21 (or fallback) decomposition for 3x3x3
    U1, V1, W1 = find_3x3x3_rank21()
    r1 = U1.shape[1]
    
    if time.time() > deadline:
        # Return schoolbook
        rank = 54
        U = np.zeros((9, rank))
        V = np.zeros((18, rank))
        W = np.zeros((18, rank))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(6):
                    U[i*3+j, col] = 1
                    V[j*6+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build 3x3x6 decomposition by concatenating two 3x3x3 decompositions
    # First half (k=0,1,2): use U1, V1 (first 3 cols of V), W1
    # Second half (k=3,4,5): use U1, V1 (last 3 cols of V), W1 (shifted)
    
    total_rank = 2 * r1
    U_full = np.zeros((9, total_rank))
    V_full = np.zeros((18, total_rank))
    W_full = np.zeros((18, total_rank))
    
    # First 3x3x3 block (k=0,1,2)
    U_full[:, :r1] = U1
    V_full[:9, :r1] = V1  # j*3+k for k=0,1,2 -> rows 0-8
    W_full[:9, :r1] = W1  # k*3+i for k=0,1,2 -> rows 0-8
    
    # Second 3x3x3 block (k=3,4,5)
    U_full[:, r1:] = U1
    V_full[9:, r1:] = V1  # j*6+k for k=3,4,5 -> rows 9-17 (j*3+(k-3))
    W_full[9:, r1:] = W1  # k*3+i for k=3,4,5 -> rows 9-17 ((k-3)*3+i)
    
    # Verify the decomposition
    # T[i,j,k] should = 1 for all i in [0,3), j in [0,3), k in [0,6)
    # sum_r U[i*3+j,r] * V[j*6+k,r] * W[k*3+i,r]
    
    valid = True
    for i in range(3):
        for j in range(3):
            for k in range(6):
                s = 0
                for rr in range(total_rank):
                    s += U_full[i*3+j, rr] * V_full[j*6+k, rr] * W_full[k*3+i, rr]
                if abs(s - 1.0) > 1e-8:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        # Round to half-integers
        U_r = np.round(U_full * 2) / 2
        V_r = np.round(V_full * 2) / 2
        W_r = np.round(W_full * 2) / 2
        
        # Re-verify after rounding
        valid2 = True
        for i in range(3):
            for j in range(3):
                for k in range(6):
                    s = 0
                    for rr in range(total_rank):
                        s += U_r[i*3+j, rr] * V_r[j*6+k, rr] * W_r[k*3+i, rr]
                    if abs(s - 1.0) > 1e-10:
                        valid2 = False
                        break
                if not valid2:
                    break
            if not valid2:
                break
        
        if valid2:
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Fallback to schoolbook
    rank = 54
    U = np.zeros((9, rank))
    V = np.zeros((18, rank))
    W = np.zeros((18, rank))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(6):
                U[i*3+j, col] = 1
                V[j*6+k, col] = 1
                W[k*3+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
