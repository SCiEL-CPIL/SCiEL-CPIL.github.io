"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for 3x3 by 3x8 matrix multiplication.
    
    Uses iterative hard thresholding (IHT) approach to minimize rank by
    exploiting the block-sparse structure of the matrix multiplication tensor.
    """
    import numpy as np
    import itertools
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 9, 24, 24
    
    # Build the target tensor T
    # T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a, b, c] = 1 iff j==j' and k==k' and i==i'
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def reconstruct(U, V, W, active_indices):
        """Reconstruct tensor from factor matrices using active columns."""
        T_recon = np.zeros((nm, mp, pn))
        for r in active_indices:
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            T_recon += np.einsum('i,j,k->ijk', u, v, w)
        return T_recon
    
    def error_norm(recon):
        """Compute Frobenius norm of error."""
        return np.linalg.norm(recon - T)
    
    def is_valid(recon, atol=1e-10):
        """Check if reconstruction is valid."""
        return np.allclose(recon, T, atol=atol)
    
    def solve_subsystem(U, V, W, active_indices, max_iters=50, tol=1e-12):
        """
        Alternating least squares to solve for factor matrices
        given active column indices.
        """
        for iteration in range(max_iters):
            # Solve for U
            for a in range(nm):
                # U[a, r] = sum_{b,c} T[a,b,c] * V[b,r] * W[c,r] / (sum V[b,r]^2 * W[c,r]^2)
                numerator = np.zeros(len(active_indices))
                denominator = np.zeros(len(active_indices))
                for r_idx, r in enumerate(active_indices):
                    for b in range(mp):
                        for c in range(pn):
                            if T[a, b, c] != 0:
                                numerator[r_idx] += T[a, b, c] * V[b, r] * W[c, r]
                                denominator[r_idx] += V[b, r] ** 2 * W[c, r] ** 2
                for r_idx, r in enumerate(active_indices):
                    if denominator[r_idx] > 1e-15:
                        U[a, r] = numerator[r_idx] / denominator[r_idx]
            
            # Solve for V
            for b in range(mp):
                numerator = np.zeros(len(active_indices))
                denominator = np.zeros(len(active_indices))
                for r_idx, r in enumerate(active_indices):
                    for a in range(nm):
                        for c in range(pn):
                            if T[a, b, c] != 0:
                                numerator[r_idx] += T[a, b, c] * U[a, r] * W[c, r]
                                denominator[r_idx] += U[a, r] ** 2 * W[c, r] ** 2
                for r_idx, r in enumerate(active_indices):
                    if denominator[r_idx] > 1e-15:
                        V[b, r] = numerator[r_idx] / denominator[r_idx]
            
            # Solve for W
            for c in range(pn):
                numerator = np.zeros(len(active_indices))
                denominator = np.zeros(len(active_indices))
                for r_idx, r in enumerate(active_indices):
                    for a in range(nm):
                        for b in range(mp):
                            if T[a, b, c] != 0:
                                numerator[r_idx] += T[a, b, c] * U[a, r] * V[b, r]
                                denominator[r_idx] += U[a, r] ** 2 * V[b, r] ** 2
                for r_idx, r in enumerate(active_indices):
                    if denominator[r_idx] > 1e-15:
                        W[c, r] = numerator[r_idx] / denominator[r_idx]
            
            # Check convergence
            recon = reconstruct(U, V, W, active_indices)
            err = error_norm(recon)
            if err < tol:
                break
        
        return U, V, W
    
    # Start with schoolbook decomposition
    school_rank = n * m * p  # 72
    U = np.zeros((nm, school_rank))
    V = np.zeros((mp, school_rank))
    W = np.zeros((pn, school_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Try to reduce rank using IHT
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = school_rank
    best_err = 0.0
    
    # Try various target ranks from lower bound upward
    lower_bound = max(nm, mp, pn)  # 24
    upper_bound = school_rank
    
    for target_rank in range(lower_bound, upper_bound + 1):
        if target_rank >= best_rank:
            break
        
        # Compute energy of each column
        energies = []
        for r in range(school_rank):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            energy = np.linalg.norm(u) * np.linalg.norm(v) * np.linalg.norm(w)
            energies.append((energy, r))
        
        # Sort by energy descending
        energies.sort(reverse=True)
        active = [col_idx for _, col_idx in energies[:target_rank]]
        
        # Solve subsystem
        U_trial, V_trial, W_trial = U.copy(), V.copy(), W.copy()
        U_trial, V_trial, W_trial = solve_subsystem(U_trial, V_trial, W_trial, active)
        
        # Check validity
        recon = reconstruct(U_trial, V_trial, W_trial, active)
        err = error_norm(recon)
        
        if is_valid(recon):
            best_U, best_V, best_W = U_trial.copy(), V_trial.copy(), W_trial.copy()
            best_rank = target_rank
            break
        
        # Update best if improved
        if err < best_err:
            best_err = err
    
    # Final validation
    final_recon = reconstruct(best_U, best_V, best_W, list(range(best_rank)))
    if not is_valid(final_recon):
        # Fall back to schoolbook
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = school_rank
    
    # Convert to lists with integer/half-integer entries
    U_list = [[round(float(x) * 2) / 2 for x in row] for row in best_U[:, :best_rank]]
    V_list = [[round(float(x) * 2) / 2 for x in row] for row in best_V[:, :best_rank]]
    W_list = [[round(float(x) * 2) / 2 for x in row] for row in best_W[:, :best_rank]]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
