"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Returns a rank decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    The schoolbook rank is 63. We attempt to find a lower rank decomposition
    using structured search with a strict time deadline.
    
    The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    is decomposed as sum_r u_r[i] * v_r[j] * w_r[k].
    
    Output format:
    U: (n*m) x rank, row index = i*m + j
    V: (m*p) x rank, row index = j*p + k
    W: (p*n) x rank, row index = k*n + i
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 63
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    def verify_decomposition(U, V, W):
        """Verify that the decomposition exactly computes the all-ones tensor."""
        rank = U.shape[1]
        # Reconstruct T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T_reconstructed = np.zeros((n, m, p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_reconstructed[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T_expected = np.ones((n, m, p))
        return np.allclose(T_reconstructed, T_expected, atol=1e-10)
    
    def schoolbook_decomposition():
        """Return the standard schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
                    
        return U, V, W
    
    # Start with schoolbook as baseline
    U_best, V_best, W_best = schoolbook_decomposition()
    best_rank = schoolbook_rank
    
    # Strategy: Try to reduce rank by finding structured decompositions
    # For small tensors, we can attempt random searches with verification
    
    np.random.seed(42)
    
    # Try to find decompositions of rank < 63
    # We'll try a few target ranks and use random initialization + local search
    
    for target_rank in range(62, 45, -1):
        if time.time() > deadline - 20:
            break
            
        # Try multiple random initializations
        for trial in range(30):
            if time.time() > deadline - 10:
                    break
                
            # Initialize with random values, scaled appropriately
            U_try = np.random.randn(n*m, target_rank) * 0.5
            V_try = np.random.randn(m*p, target_rank) * 0.5
            W_try = np.random.randn(p*n, target_rank) * 0.5
            
            # Verify this random initialization (unlikely to work, but check)
            if verify_decomposition(U_try, V_try, W_try):
                best_rank = target_rank
                U_best, V_best, W_best = U_try.copy(), V_try.copy(), W_try.copy()
                break
            
            # Try a structured approach: start from schoolbook and perturb
            # Only valid if target_rank <= schoolbook_rank
            if target_rank <= schoolbook_rank:
                U_try2 = np.zeros((n*m, target_rank))
                V_try2 = np.zeros((m*p, target_rank))
                W_try2 = np.zeros((p*n, target_rank))
                
                # Copy first target_rank terms from schoolbook
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col < target_rank:
                                U_try2[i*m+j, col] = 1
                                V_try2[j*p+k, col] = 1
                                W_try2[k*n+i, col] = 1
                            col += 1
                            if col >= target_rank:
                                break
                        if col >= target_rank:
                            break
                    if col >= target_rank:
                        break
                
                # Add small random perturbations to try to "merge" terms
                U_try2 += np.random.randn(n*m, target_rank) * 0.1
                V_try2 += np.random.randn(m*p, target_rank) * 0.1
                W_try2 += np.random.randn(p*n, target_rank) * 0.1
                
                if verify_decomposition(U_try2, V_try2, W_try2):
                    best_rank = target_rank
                    U_best, V_best, W_best = U_try2.copy(), V_try2.copy(), W_try2.copy()
                    break
        
        if best_rank < schoolbook_rank:
            break
    
    # Convert to lists for the expected output format
    U_list = U_best.tolist()
    V_list = V_best.tolist()
    W_list = W_best.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
