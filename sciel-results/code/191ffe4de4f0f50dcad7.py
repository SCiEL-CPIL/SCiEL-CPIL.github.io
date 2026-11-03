"""Matrix multiplication 3x3 x 3x7 - rank-minimal tensor decomposition."""
DIMENSIONS = [3, 3, 7]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """Construct rank-minimal tensor decomposition for 3x3 * 3x7 multiplication."""
    n, m, p = 3, 3, 7
    deadline = time.time() + 140  # Leave 10s safety margin before 150s kill
    
    # Build the target tensor T of size (n*m) x (m*p) x (p*n) = 9 x 21 x 21
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i in [0,n), j in [m), k in [p)
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Schoolbook solution as fallback (rank 63)
    best_rank = n * m * p  # 63
    U_school = np.zeros((n*m, best_rank))
    V_school = np.zeros((m*p, best_rank))
    W_school = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Strategy: Try to find lower-rank decompositions using gradient descent
    # with smart initialization and rounding to half-integers
    
    np.random.seed(42)
    
    # Key insight: We can decompose this as block operations.
    # C[i,k] = sum_j A[i,j] * B[j,k] for k=0..6
    # 
    # One approach: treat the 3x7 output as 3 separate 3x3 outputs (first 3 cols)
    # and a 3x4 output (last 4 cols). But that's not quite right.
    #
    # Better: use the fact that matrix multiplication tensor has structure.
    # We can try ranks from 20 up to 62 and see if we can find valid decompositions.
    
    # For efficiency, we'll use a fast reconstruction check
    def compute_reconstruction(U, V, W, rank):
        """Compute sum_r U[:,r] outer V[:,r] outer W[:,r]"""
        R = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(rank):
            R += np.outer(np.outer(U[:,r], V[:,r]), W[:,r])
        return R
    
    def verify_solution(U, V, W, rank, tol=1e-6):
        """Check if the decomposition exactly reconstructs T"""
        R = compute_reconstruction(U, V, W, rank)
        return np.linalg.norm(R - T) < tol
    
    # Try different rank values
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # Schoolbook: 63
    # Best known for 3x3x3 is 23, so 3x3x7 might be around 46-55
    
    target_ranks = list(range(25, 63))  # Try ranks from 25 to 62
    
    for target_rank in target_ranks:
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        found = False
        for attempt in range(8):
            if time.time() > deadline - 15:
                break
            
            # Initialize with small random values
            scale = 0.5
            U = np.random.randn(n*m, target_rank) * scale
            V = np.random.randn(m*p, target_rank) * scale
            W = np.random.randn(p*n, target_rank) * scale
            
            # Gradient descent with adaptive learning rate
            lr = 0.05
            best_err = np.inf
            
            for step in range(300):
                if time.time() > deadline - 10:
                    break
                
                # Compute reconstruction
                R = compute_reconstruction(U, V, W, target_rank)
                err = R - T
                err_norm = np.linalg.norm(err)
                
                if err_norm < best_err:
                    best_err = err_norm
                
                if err_norm < 1e-8:
                    break
                
                # Compute gradients
                # grad_U[a,r] = 2 * sum_{b,c} err[a,b,c] * V[b,r] * W[c,r]
                grad_U = 2.0 * np.einsum('abc,br,cr->ar', err, V, W)
                grad_V = 2.0 * np.einsum('abc,ar,cr->br', err, U, W)
                grad_W = 2.0 * np.einsum('abc,ar,br->cr', err, U, V)
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Normalize each rank component to prevent divergence
                for r in range(target_rank):
                    nu = np.linalg.norm(U[:,r])
                    nv = np.linalg.norm(V[:,r])
                    nw = np.linalg.norm(W[:,r])
                    if nu > 1e-10 and nv > 1e-10 and nw > 1e-10:
                        U[:,r] /= nu
                        V[:,r] /= nv
                        W[:,r] /= nw
                
                # Adaptive learning rate
                if step > 0 and err_norm < best_err * 0.9:
                    lr *= 1.1
                elif err_norm > best_err * 1.1:
                    lr *= 0.9
                    lr = max(lr, 0.001)
            
            # Check final error
            R = compute_reconstruction(U, V, W, target_rank)
            err_norm = np.linalg.norm(R - T)
            
            if err_norm < 1e-5:
                # Try to round to half-integers
                # Try various scales
                for scale in [1.0, 2.0, 0.5, 4.0, 0.25, 8.0, 0.125]:
                    U_r = np.round(U * scale) / scale
                    V_r = np.round(V * scale) / scale
                    W_r = np.round(W * scale) / scale
                    
                    if verify_solution(U_r, V_r, W_r, target_rank, tol=1e-6):
                        best_rank = target_rank
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        found = True
                        break
                
                if found:
                    break
            
            # Also try without scaling (just round to nearest 0.5)
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            if verify_solution(U_r, V_r, W_r, target_rank, tol=1e-6):
                best_rank = target_rank
                best_U = U_r
                best_V = V_r
                best_W = W_r
                found = True
                break
        
        if found:
            # Found a solution, try to see if we can go lower
            # But continue trying lower ranks for a bit
            if target_rank <= 35:
                continue
            else:
                break
    
    # Convert to lists with integer/half-integer entries
    def to_list(mat):
        """Convert numpy array to list of lists, ensuring half-integer entries"""
        mat = np.round(mat * 2) / 2
        return [[float(val) for val in row] for row in mat]
    
    # Final verification
    if not verify_solution(best_U, best_V, best_W, best_rank, tol=1e-5):
        # Fall back to schoolbook
        best_U = U_school
        best_V = V_school
        best_W = W_school
        best_rank = n * m * p
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
# EVOLVE-BLOCK-END
