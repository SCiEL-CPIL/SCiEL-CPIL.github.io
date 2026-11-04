"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]
import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 4x5 by 5x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 100
    
    deadline = time.time() + 140  # Leave 10s safety margin
    
    # Build the matrix multiplication tensor T[i,j,k] = delta_{i, k} * delta_{j, j}
    # Actually T[i,j,k] = 1 if we're computing C[i,k] += A[i,j]*B[j,k]
    # So the tensor has T[i,j,k] = 1 for all valid (i,j,k), and we want:
    # C[i,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] ... no
    # Standard: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor form: T[i,j,k] where the multiplication tensor satisfies
    # C[i,k] = sum_{j,r} A[i,j] * B[j,k] * T[i,j,k,r]
    # Actually the standard formulation: we want rank-r decomposition such that
    # sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] = delta_{j,j} (always 1)
    # Wait, let me reconsider.
    # 
    # For matrix multiplication C = A*B where A is n×m, B is m×p, C is n×p:
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # 
    # Tensor decomposition: We want rank-r factors U, V, W such that:
    # sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] = delta (which is 1 for all valid indices)
    # 
    # Actually the standard is:
    # The multiplication tensor T has T[i,j,k] = 1 for all i,j,k
    # We want: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # 
    # Wait, let me re-read the task. The output factor follows W[k*n+i, r].
    # So W is indexed by (k*n+i) where k is from B's column and i is from C's row.
    # 
    # The decomposition should satisfy:
    # For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # (since A[i,j]*B[j,k] contributes to C[i,k])
    # 
    # Actually no. The tensor T[i,j,k] represents the coefficient in the bilinear form.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor form, this is: C[i,k] = sum_{j,r} A[i,j] B[j,k] T_{i,j,k,r}
    # where T_{i,j,k,r} should make this equal to sum_j A[i,j] B[j,k]
    # So T_{i,j,k,r} summed over r should give delta_{j,j} = 1... 
    # 
    # Let me think differently. The schoolbook has rank n*m*p with:
    # U[i*m+j, (i,j,k)] = 1, V[j*p+k, (i,j,k)] = 1, W[k*n+i, (i,j,k)] = 1
    # Then sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 if r=(i,j,k) else 0
    # So the sum equals 1 only when we pick the right r.
    # 
    # The condition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # This means the "tensor" T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1
    # 
    # So we need: for all (i,j,k) in [0,n)×[0,m)×[0,p):
    #   sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Start with schoolbook as baseline
    best_rank = schoolbook_rank
    best_U = [[0] * schoolbook_rank for _ in range(n * m)]
    best_V = [[0] * schoolbook_rank for _ in range(m * p)]
    best_W = [[0] * schoolbook_rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Try to reduce rank by attempting to merge columns
    # Strategy: for each pair of columns, check if they can be merged
    # This is complex, so use a simpler approach: random perturbation + rank reduction
    
    # Alternative: use the fact that for small matrices, known bounds exist
    # For 4x5x5, the upper bound is 100 (schoolbook). 
    # Let's try a greedy approach: try to find a rank-99 decomposition by
    # attempting to eliminate one column.
    
    # Actually, let's try a more systematic approach:
    # 1. Start with schoolbook
    # 2. Try to reduce rank by checking if any column can be expressed as a linear combination
    #    of others in a way that preserves the tensor equation
    
    # For time efficiency, let's try a limited search
    start_time = time.time()
    
    # Try random sparse decompositions with rank < schoolbook
    # Use a simple iterative method
    
    for target_rank in range(schoolbook_rank - 1, max(schoolbook_rank - 20, 0), -1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        max_attempts = 5 if (schoolbook_rank - target_rank) <= 5 else 2
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Initialize with random sparse entries
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Seed with schoolbook-like structure but with random signs
            for r in range(target_rank):
                # Pick random (i,j,k) pattern
                i = r % n
                j = (r // n) % m
                k = (r // (n*m)) % p
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
            
            # Iterate: try to fix the tensor equation
            converged = False
            for it in range(50):
                if time.time() > deadline:
                    break
                
                # Compute error
                max_err = 0
                err_i, err_j, err_k = 0, 0, 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for r in range(target_rank):
                                s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            e = abs(s - 1)
                            if e > max_err:
                                max_err = e
                                err_i, err_j, err_k = i, j, k
                
                if max_err < 1e-9:
                    converged = True
                    break
                
                # Try to fix the worst error by adjusting one factor
                # This is a heuristic approach
                i, j, k = err_i, err_j, err_k
                current = 0
                for r in range(target_rank):
                    current += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                
                # Find the column that contributes most and adjust
                best_r = 0
                best_contrib = 0
                for r in range(target_rank):
                    contrib = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(contrib) > abs(best_contrib):
                        best_contrib = contrib
                        best_r = r
                
                # Perturb slightly
                if U[i*m+j, best_r] != 0:
                    U[i*m+j, best_r] += 0.1 * (1 - current) / max(abs(U[i*m+j, best_r]), 1e-6)
                elif V[j*p+k, best_r] != 0:
                    V[j*p+k, best_r] += 0.1 * (1 - current) / max(abs(V[j*p+k, best_r]), 1e-6)
                elif W[k*n+i, best_r] != 0:
                    W[k*n+i, best_r] += 0.1 * (1 - current) / max(abs(W[k*n+i, best_r]), 1e-6)
            
            if converged:
                # Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                # Verify with exact arithmetic
                if verify(U_list, V_list, W_list, target_rank):
                    best_rank = target_rank
                    best_U = U_list
                    best_V = V_list
                    best_W = W_list
                    break
        else:
            continue
        break
    
    # Return the best decomposition found
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
