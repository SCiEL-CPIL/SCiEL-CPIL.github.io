"""
Heuristic Maximum Cut solver using SDP Relaxation via Projected Gradient Ascent.
Strategy:
1. Formulate Maximum Cut as an SDP relaxation over correlation matrix X.
2. Optimize X using projected gradient ascent (keeping X symmetric PSD).
3. Extract an initial bit vector from the leading eigenvector of the optimized X.
4. Polish the result with a fast vectorized Best-Improvement Local Search.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    """
    Solve Maximum Cut using SDP Relaxation and Local Search.
    Returns a 0/1 label per vertex.
    """
    start_time = time.time()
    DEADLINE = 115.0  # Leave 5s safety margin
    
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array
    # Input is a bool matrix. We treat it as 0/1.
    # Ensure symmetry
    adj = np.array(adjacency, dtype=np.float64)
    adj = (adj + adj.T) / 2.0
    
    # Precompute degrees
    degrees = adj.sum(axis=1)
    total_degree = degrees.sum()
    
    # SDP Relaxation:
    # Maximize 0.5 * (sum(degrees) - s^T A s) subject to s_i^2 = 1, s_i in {-1, 1}
    # Relaxation: Maximize 0.5 * (sum(degrees) - Tr(A X)) subject to X_ii = 1, X >= 0
    # where X = s s^T.
    # Objective: Maximize -0.5 * Tr(A X) is equivalent to Minimizing Tr(A X).
    # But standard MaxCut SDP maximizes Tr(A X) with diagonal 1?
    # Let's check:
    # Cut = 1/2 * sum_{ij} A_ij (1 - s_i s_j) = 1/2 * (SumDeg - sum A_ij s_i s_j)
    # So we want to MINIMIZE sum A_ij s_i s_j.
    # In SDP, X = s s^T. sum A_ij X_ij = Tr(A X).
    # So we want to minimize Tr(A X) subject to X_ii=1, X >= 0.
    # Alternatively, maximize -Tr(A X).
    
    # Gradient of -Tr(A X) w.r.t X is -A.
    # We want to ascend in the direction of -A? No, we want to minimize Tr(A X).
    # So we descend in direction A.
    # Update: X <- X - eta * A?
    # Wait, the gradient of Tr(A X) is A. So to minimize, we do X <- X - eta * A.
    # But we also need to project to the feasible set (PSD, Diag=1).
    
    # Let's use the formulation: Maximize Tr(A X) is wrong for MaxCut if A has positive weights.
    # MaxCut wants edges to cross. s_i != s_j.
    # s_i s_j = -1 if different, 1 if same.
    # We want s_i s_j to be -1. So we want to minimize sum A_ij s_i s_j.
    # Yes, minimize Tr(A X).
    
    # Algorithm:
    # 1. Initialize X = I (identity matrix). This corresponds to s_i = 0? No, X_ii=1, X_ij=0.
    #    This is a valid PSD matrix with unit diagonal.
    #    Actually, a better initialization is X = J (all ones) / n? No, X_ii must be 1.
    #    X = I is a safe start.
    
    X = np.eye(n)
    
    # Gradient step
    # We want to minimize f(X) = Tr(A X).
    # Grad = A.
    # Step: X_new = X - eta * A.
    # Then Project to {X: X >= 0, Diag(X) = 1}.
    
    # Projection onto PSD with unit diagonal:
    # 1. Eigen-decompose X_new = V D V^T.
    # 2. Clip negative eigenvalues to 0: D_clipped.
    # 3. X_psd = V D_clipped V^T.
    # 4. Normalize diagonal to 1: For each i, divide row i and col i by sqrt(X_psd[i,i]) if X_psd[i,i] > 0.
    #    If X_psd[i,i] is 0, set to 1? This might break PSD.
    #    A more robust projection for Diag=1 is:
    #    X_proj = X_psd / sqrt(outer(diag, diag)) ? No.
    #    Standard approach:
    #    X_psd = V D_clipped V^T
    #    d = diag(X_psd)
    #    If any d_i == 0, handle carefully.
    #    X_final[i,j] = X_psd[i,j] / sqrt(d_i d_j)
    #    This ensures diag is 1. Does it preserve PSD?
    #    Yes, scaling rows/cols by positive factors preserves PSD if the scaling is consistent (diagonal scaling).
    
    eta = 0.1  # Step size. Might need tuning.
    max_iter = 1000
    
    # We can use a simple gradient ascent on the dual or primal.
    # Let's try a few iterations of projected gradient descent.
    
    for _ in range(max_iter):
        if time.time() > DEADLINE - 5.0:
            break
            
        # Gradient is A
        X -= eta * adj
        
        # Project to PSD
        # Symmetrize
        X = (X + X.T) / 2.0
        
        # Eigen decomposition
        # For n=100, this is fast enough.
        eigvals, eigvecs = np.linalg.eigh(X)
        
        # Clip negative eigenvalues
        eigvals_clipped = np.maximum(eigvals, 0.0)
        
        # Reconstruct
        X = (eigvecs * eigvals_clipped) @ eigvecs.T
        
        # Normalize diagonal to 1
        diag = np.diag(X)
        # Avoid division by zero
        diag_safe = np.where(diag > 1e-10, diag, 1.0)
        scale = 1.0 / np.sqrt(diag_safe)
        
        # Apply scaling: X[i,j] *= scale[i] * scale[j]
        X = X * np.outer(scale, scale)
        
        # Enforce symmetry again after numerical errors
        X = (X + X.T) / 2.0
        
    # Extract solution
    # The optimal X is close to s s^T.
    # We can take the leading eigenvector of X.
    # Since X is PSD and has trace n (sum of eigenvalues = sum of diag = n),
    # and we want a rank-1 approximation, the largest eigenvalue corresponds to the "direction" of s.
    
    eigvals_X, eigvecs_X = np.linalg.eigh(X)
    # Largest eigenvalue is last
    s = eigvecs_X[:, -1]
    
    # Convert to binary labels
    # s is a vector of floats. We need 0/1.
    # Threshold at 0.
    labels = (s >= 0).astype(int)
    
    # Now, polish with Local Search
    # Convert labels to spins (-1, 1) for calculation
    # 0 -> -1, 1 -> 1
    s_spin = 2 * labels - 1
    
    def get_cut(s):
        return 0.5 * (total_degree - s @ (adj @ s))
    
    def best_first_local_search(s, max_iters=1000):
        best_s = s.copy()
        best_cut = get_cut(best_s)
        
        for _ in range(max_iters):
            if time.time() > DEADLINE - 2.0:
                break
                
            adj_s = adj @ best_s
            gains = best_s * adj_s
            
            max_gain = np.max(gains)
            
            if max_gain <= 0:
                break
            
            v = np.argmax(gains)
            best_s[v] *= -1
            best_cut += max_gain
            
        return best_s, best_cut
    
    s_final, cut_final = best_first_local_search(s_spin)
    
    # Convert back to 0/1
    final_labels = ((s_final + 1) // 2).astype(int)
    
    return final_labels
# EVOLVE-BLOCK-END
