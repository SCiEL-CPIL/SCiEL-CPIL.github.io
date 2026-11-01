"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin before 150s limit
    
    # Start with schoolbook rank 36
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by 1: rank 35
    target_rank = rank - 1
    if time.time() < deadline:
        # Random initialization for rank 35
        rng = np.random.RandomState(42)
        
        # Initialize with a perturbation of schoolbook
        U_new = np.zeros((n * m, target_rank), dtype=np.float64)
        V_new = np.zeros((m * p, target_rank), dtype=np.float64)
        W_new = np.zeros((p * n, target_rank), dtype=np.float64)
        
        # Use first target_rank columns of schoolbook as starting point
        U_new[:, :target_rank] = U[:, :target_rank]
        V_new[:, :target_rank] = V[:, :target_rank]
        W_new[:, :target_rank] = W[:, :target_rank]
        
        # Add small random perturbation
        U_new += rng.randn(n * m, target_rank) * 0.01
        V_new += rng.randn(m * p, target_rank) * 0.01
        W_new += rng.randn(p * n, target_rank) * 0.01
        
        # Gradient descent to minimize tensor reconstruction error
        # The tensor T[i,j,k] should equal sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # (with appropriate index mapping)
        
        lr = 0.01
        for step in range(10000):
            if time.time() > deadline - 5:
                break
            
            # Compute gradient numerically or analytically
            # For each (i,j,k), the contribution is sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            # Target is 1 if i,j,k form a valid multiplication term, else 0
            # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k
            # (since (A*B)[i,k] = sum_j A[i,j]*B[j,k])
            
            # Let's use a simpler approach: check if the decomposition is valid
            # by verifying that the reconstructed tensor matches the target
            
            # Compute the residual
            residual = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Target: 1.0
                        # Reconstruction: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                        recon = 0.0
                        for c in range(target_rank):
                            recon += U_new[i * m + j, c] * V_new[j * p + k, c] * W_new[k * n + i, c]
                        residual += (recon - 1.0) ** 2
            
            if residual < 1e-10:
                best_rank = target_rank
                best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
                break
            
            # Simple gradient step
            # d/dU[i,j,c] = 2 * (recon_ij - 1) * V[j,k,c] * W[k,i,c] summed over k
            # This is complex; use finite differences or a simpler heuristic
            
            # For now, just do a random search with occasional acceptance
            if step % 10 == 0:
                # Try a random perturbation
                U_trial = U_new + rng.randn(n * m, target_rank) * 0.05
                V_trial = V_new + rng.randn(m * p, target_rank) * 0.05
                W_trial = W_new + rng.randn(p * n, target_rank) * 0.05
                
                residual_trial = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            recon = 0.0
                        for c in range(target_rank):
                            recon += U_trial[i * m + j, c] * V_trial[j * p + k, c] * W_trial[k * n + i, c]
                        residual_trial += (recon - 1.0) ** 2
                
                if residual_trial < residual:
                    U_new, V_new, W_new = U_trial, V_trial, W_trial
                    residual = residual_trial
        
        # Final check
        if time.time() < deadline:
            residual_final = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        recon = 0.0
                        for c in range(target_rank):
                            recon += U_new[i * m + j, c] * V_new[j * p + k, c] * W_new[k * n + i, c]
                        residual_final += (recon - 1.0) ** 2
            
            if residual_final < 1e-10:
                best_rank = target_rank
                best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
    
    # Convert to lists and round to nearest integer/half-integer if close
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    # Round to nearest 0.5 if within tolerance
    for matrix in [U_out, V_out, W_out]:
        for row in range(len(matrix)):
            for col in range(len(matrix[row])):
                val = matrix[row][col]
                rounded = round(val * 2) / 2
                if abs(val - rounded) < 1e-6:
                    matrix[row][col] = rounded
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
