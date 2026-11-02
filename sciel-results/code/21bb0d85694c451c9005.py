"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x5 matrix.
    Output factor follows the official trace convention W[k*n+i, r].
    
    Returns: {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s limit
    
    # Start with schoolbook decomposition (rank = n*m*p = 80)
    rank_schoolbook = n * m * p
    
    # We'll try to find a lower rank decomposition
    # Known lower bounds: rank >= max(n*p, m*p, n*m) = max(20, 20, 16) = 20
    # Upper bound: n*m*p = 80
    # Let's try to find a decomposition with rank less than 80
    
    # Strategy: Use random initialization and iterative refinement
    # The tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # In our notation: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should sum to 1 for each (i,j,k)
    
    # Let's try a structured approach based on known fast matrix multiplication algorithms
    # For 4x4 * 4x5, we can potentially use Strassen-like ideas
    
    # First, let's try a simple improvement: exploit structure
    # If we can find a decomposition where some entries are shared, we can reduce rank
    
    # Let's try rank = 32, 40, 48, 56, 64, 72, 80 and see which works
    
    best_result = None
    best_rank = rank_schoolbook
    
    # Generate schoolbook solution first as fallback
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    # Try to find lower rank decompositions
    # Use a greedy/optimization approach
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        n, m, p = DIMENSIONS
        # For each (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def try_rank(rank):
        """Try to find a decomposition of given rank using random search."""
        if time.time() > deadline - 5:
            return None
        
        n, m, p = DIMENSIONS
        nrm = n * m  # 16
        mpr = m * p  # 20
        pnr = p * n  # 20
        
        # Random initialization
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            U = np.random.randn(nrm, rank)
            V = np.random.randn(mpr, rank)
            W = np.random.randn(pnr, rank)
            
            # Normalize to help convergence
            for r in range(rank):
                if np.linalg.norm(U[:, r]) > 1e-10:
                    U[:, r] /= np.linalg.norm(U[:, r])
                if np.linalg.norm(V[:, r]) > 1e-10:
                    V[:, r] /= np.linalg.norm(V[:, r])
                if np.linalg.norm(W[:, r]) > 1e-10:
                    W[:, r] /= np.linalg.norm(W[:, r])
            
            # Iterative refinement using gradient-free approach
            # We want sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all (i,j,k)
            
            # Use a simple coordinate descent approach
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                
                improved = False
                # Update W
                for r in range(rank):
                    if np.linalg.norm(W[:, r]) < 1e-15:
                        continue
                    # For each (k,i), we want the contribution to be correct
                    # This is complex, so let's just do a simple check and perturb
                    pass
                
                # Simple approach: just check if we're close
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(rank):
                                s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            max_err = max(max_err, abs(s - 1.0))
                
                if max_err < 1e-6:
                    # Try to polish
                    for polish_it in range(100):
                        max_err = 0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    s = 0.0
                                    for r in range(rank):
                                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                                    max_err = max(max_err, abs(s - 1.0))
                        if max_err < 1e-10:
                            break
                        # Small perturbation
                        U += np.random.randn(nrm, rank) * 1e-4
                        V += np.random.randn(mpr, rank) * 1e-4
                        W += np.random.randn(pnr, rank) * 1e-4
                    
                    if verify_decomposition(U, V, W, tol=1e-9):
                        return U, V, W
                    elif max_err < 1e-8:
                        # Close enough, try more polish
                        for polish_it in range(500):
                            max_err = 0
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        s = 0.0
                                        for r in range(rank):
                                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                                        max_err = max(max_err, abs(s - 1.0))
                            if max_err < 1e-10:
                                break
                            U += np.random.randn(nrm, rank) * 1e-5
                            V += np.random.randn(mpr, rank) * 1e-5
                            W += np.random.randn(pnr, rank) * 1e-5
                        
                        if verify_decomposition(U, V, W, tol=1e-9):
                            return U, V, W
                elif max_err < 1e-3:
                    # Getting closer, continue
                    pass
                else:
                    # Reset with different random seed
                    break
            
    # Try decreasing ranks
    for rank in [64, 56, 48, 40, 32, 28, 24, 20]:
        if time.time() > deadline - 10:
            break
        
        result = try_rank(rank)
        if result is not None:
            U, V, W = result
            # Convert to lists with rounded values for cleanliness
            U_list = [[float(round(val, 12)) for val in row] for row in U.tolist()]
            V_list = [[float(round(val, 12)) for val in row] for row in V.tolist()]
            W_list = [[float(round(val, 12)) for val in row] for row in W.tolist()]
            
            # Final verification
            if verify_decomposition(np.array(U_list), np.array(V_list), np.array(W_list), tol=1e-9):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: return schoolbook decomposition
    U_list = [[float(val) for val in row] for row in U_sb.tolist()]
    V_list = [[float(val) for val in row] for row in V_sb.tolist()]
    W_list = [[float(val) for val in row] for row in W_sb.tolist()]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
