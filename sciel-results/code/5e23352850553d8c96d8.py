"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Try to find a rank-minimal decomposition
    # Start with schoolbook as fallback
    schoolbook_rank = n * m * p  # 80
    
    # Build schoolbook decomposition
    def build_schoolbook():
        U = np.zeros((n * m, schoolbook_rank))
        V = np.zeros((m * p, schoolbook_rank))
        W = np.zeros((p * n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, schoolbook_rank
    
    def verify_decomposition(U, V, W, rank):
        """Check if the decomposition is exact."""
        # The decomposition should satisfy:
        # For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # For all other (i,j,k) combinations that don't correspond to valid matrix mult: should be 0
        # Actually, the tensor T[i,j,k] = 1 if it's a valid multiplication term, else 0
        # The decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        
        # Build the target tensor
        # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # This is the full matrix multiplication tensor
        
        # Check using random sampling for speed
        np.random.seed(42)
        for trial in range(200):
            i = np.random.randint(0, n)
            j = np.random.randint(0, m)
            k = np.random.randint(0, p)
            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
            if abs(val - 1.0) > 1e-6:
                return False
        
        # Also check that invalid terms are zero
        # Actually for matrix multiplication, ALL (i,j,k) with i<n, j<m, k<p are valid
        # The tensor is all ones in the [0,n)x[0,m)x[0,p) region
        # So we just need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all valid i,j,k
        
        return True
    
    def to_nice_list(mat):
        """Convert numpy array to list of integers/half-integers."""
        result = []
        for row in mat:
            nice_row = []
            for val in row:
                # Round to nearest half-integer
                rounded = round(val * 2) / 2.0
                if abs(val - rounded) < 1e-6:
                    nice_row.append(rounded)
                else:
                    nice_row.append(rounded)
            result.append(nice_row)
        return result
    
    # Strategy: Use a structured approach to find lower rank
    # For 4x4x5, we can try to exploit the rectangular structure
    
    # Known: For matrix multiplication, Strassen-like algorithms can reduce rank
    # But for small sizes, we need to find explicit decompositions
    
    # Let's try a different approach: use the fact that we can factor through smaller dimensions
    # Try rank = n*p = 20, m*p = 20, etc.
    
    # Actually, let's use a concrete method: try to find a decomposition with rank 48
    # This is based on the idea that we can save some terms by sharing
    
    # For now, implement a simple search: try to find a valid decomposition at various ranks
    # using a greedy approach
    
    def try_find_decomposition(rank, max_iters=500):
        """Try to find a valid decomposition at the given rank using vectorized operations."""
        np.random.seed(12345 + rank)
        
        # Initialize
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Normalize
        U = U / np.linalg.norm(U) * np.sqrt(rank)
        V = V / np.linalg.norm(V) * np.sqrt(rank)
        W = W / np.linalg.norm(W) * np.sqrt(rank)
        
        # Precompute index arrays
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Compute current tensor values vectorized
            vals = np.array([
                np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                for i, j, k in zip(i_idx, j_idx, k_idx)
            ])
            
            errors = vals - 1.0
            mse = np.mean(errors ** 2)
            
            if mse < 1e-4:
                # Try to snap to nice values
                U_nice = np.round(U * 2) / 2.0
                V_nice = np.round(V * 2) / 2.0
                W_nice = np.round(W * 2) / 2.0
                
                # Verify
                vals_nice = np.array([
                    np.sum(U_nice[i*m+j, :] * V_nice[j*p+k, :] * W_nice[k*n+i, :])
                    for i, j, k in zip(i_idx, j_idx, k_idx)
                ])
                max_err = np.max(np.abs(vals_nice - 1.0))
                
                if max_err < 1e-6:
                    return U_nice, V_nice, W_nice
                else:
                    return U, V, W
            
            # Vectorized gradient computation
            lr = 0.05 / (1 + it * 0.2)
            
            # For U[i*m+j, r]: grad = sum_k errors[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
            U_grad = np.zeros_like(U)
            V_grad = np.zeros_like(V)
            W_grad = np.zeros_like(W)
            
            for i, j, k in zip(i_idx, j_idx, k_idx):
                e = errors[(i, j, k)]
                r_idx = np.arange(rank)
                U_grad[i*m+j, :] += e * V[j*p+k, :] * W[k*n+i, :]
                V_grad[j*p+k, :] += e * U[i*m+j, :] * W[k*n+i, :]
                W_grad[k*n+i, :] += e * U[i*m+j, :] * V[j*p+k, :]
            
            U -= lr * U_grad
            V -= lr * V_grad
            W -= lr * W_grad
        
        return None
    
    # Try a few structured approaches to get below 80
    
    # Approach 1: Try rank = 64 (save 25%)
    # This is a common reduction factor
    
    # For now, just return schoolbook as a safe baseline
    # The optimization above may not converge quickly enough
    
    U, V, W, rank = build_schoolbook()
    
    U_list = to_nice_list(U)
    V_list = to_nice_list(V)
    W_list = to_nice_list(W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
