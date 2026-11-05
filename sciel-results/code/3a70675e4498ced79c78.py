"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # safety margin
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    
    # Try to find lower rank decompositions
    # The tensor condition: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should reconstruct
    # Actually, the condition is that the bilinear map is preserved.
    
    # Let me set up the problem properly.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We want: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_l V[j*p+l, r] * B[j,l]) * W[k*n+i, r]
    # This doesn't look right.
    
    # Correct formulation from tensor decomposition:
    # The multiplication tensor T has T[i,j,k] = 1 (for the product term)
    # Rank decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # But T[i,j,k] = 1 for all i,j,k, so we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a system of n*m*p = 54 equations.
    
    # For a given rank r, we need to find U (9×r), V (18×r), W (18×r) such that
    # for all i in [0,3), j in [0,3), k in [0,6):
    #   sum_{r=0}^{rank-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a trilinear system. Hard to solve directly.
    
    # Alternative: fix W and solve for U,V (bilinear). Or use iterative methods.
    
    # Let me try a simple approach: random initialization + gradient descent on the loss
    # L = sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
    
    best_result = None
    best_rank = rank_schoolbook
    
    for target_rank in range(46, rank_schoolbook):  # try 46, 47, ..., 53
        if time.time() > deadline - 30:
            break
        
        # Try a few random restarts
        for restart in range(5):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(42 + target_rank * 100 + restart)
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Gradient descent
            lr = 0.01
            for step in range(5000):
                if time.time() > deadline - 30:
                    break
                
                # Compute gradients
                # L = sum_{i,j,k} (S[i,j,k] - 1)^2 where S[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
                
                U_grad = np.zeros_like(U)
                V_grad = np.zeros_like(V)
                W_grad = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # S[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                            S = 0
                            for r in range(target_rank):
                                S += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            diff = S - 1
                            for r in range(target_rank):
                                U_grad[i*m+j, r] += 2 * diff * V[j*p+k, r] * W[k*n+i, r]
                                V_grad[j*p+k, r] += 2 * diff * U[i*m+j, r] * W[k*n+i, r]
                                W_grad[k*n+i, r] += 2 * diff * U[i*m+j, r] * V[j*p+k, r]
                
                U -= lr * U_grad
                V -= lr * V_grad
                W -= lr * W_grad
                
                if step % 500 == 0:
                    # Check loss
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                S = 0
                                for r in range(target_rank):
                                    S += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                                max_err = max(max_err, abs(S - 1))
                    if max_err < 1e-6:
                        # Round to half-integers and verify
                        U_rounded = np.round(U * 2) / 2
                        V_rounded = np.round(V * 2) / 2
                        W_rounded = np.round(W * 2) / 2
                        
                        # Verify
                        valid = True
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    S = 0
                                    for r in range(target_rank):
                                        S += U_rounded[i*m+j, r] * V_rounded[j*p+k, r] * W_rounded[k*n+i, r]
                                    if abs(S - 1) > 1e-9:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                break
                        if valid:
                            # Convert to lists
                            U_list = [[float(x) for x in U_rounded[i]] for i in range(n*m)]
                            V_list = [[float(x) for x in V_rounded[i]] for i in range(m*p)]
                            W_list = [[float(x) for x in W_rounded[i]] for i in range(p*n)]
                            best_result = {"U": U_list, "V": V_list, "W": W_list}
                            best_rank = target_rank
                            break
                        # If rounding didn't work, continue
                    if max_err < 1e-8:
                        break
                
                if time.time() > deadline - 30:
                    break
            
            if best_rank <= target_rank:
                break
        if best_rank <= target_rank:
            break
    
    if best_result is not None:
        return best_result
    else:
        return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
