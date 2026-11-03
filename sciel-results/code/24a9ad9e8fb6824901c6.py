"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Build the target tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    def build_tensors(U, V, W):
        """Check if U,V,W give exact decomposition."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        # Compute C[i,k] = sum_j A[i,j]*B[j,k] where
        # A[i,j] = U[i*m+j, :], B[j,k] = V[j*p+k, :], and the product is weighted by W
        # Actually: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # We need this to equal 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find a lower-rank decomposition
    # Strategy: random initialization + local search
    
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # Generate schoolbook as fallback
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    # Try ranks from low to high
    for rank in range(15, schoolbook_rank + 1):
        if time.time() > deadline - 5:
            break
        
        attempts = 0
        max_attempts = 50
        
        while attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            
            # Random initialization
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Try to refine using iterative method
            # Fix two, solve for third
            for iteration in range(20):
                if time.time() > deadline - 5:
                    break
                    
                # Solve for W: for each (k,i), we have
                # sum_c W[k*n+i, c] * (U[i*m+j, c] * V[j*p+k, c]) = 1 for all j
                # This is a system: for fixed (k,i), the unknowns are W[k*n+i, :]
                # And the equations are: for each j, sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1
                # Let M_jk_i[c] = U[i*m+c_idx, c] * V[j*p+k, c] ... this is getting complex
                
                # Alternative: just check if current is close to valid
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(rank):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            if abs(val - 1.0) > 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_rank = rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    break
                
                # Simple gradient-free update: perturb the worst entry
                # Find the (i,j,k) with largest error
                max_err = 0
                worst_ijk = (0,0,0)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(rank):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                                worst_ijk = (i,j,k)
                
                if max_err < 1e-6:
                    valid = True
                    break
                
                # Perturb randomly
                idx = random.randint(0, 2)
                if idx == 0:
                    U[random.randint(0, n*m-1), random.randint(0, rank-1)] += np.random.randn() * 0.1
                elif idx == 1:
                    V[random.randint(0, m*p-1), random.randint(0, rank-1)] += np.random.randn() * 0.1
                else:
                    W[random.randint(0, p*n-1), random.randint(0, rank-1)] += np.random.randn() * 0.1
            
            if best_rank == rank:
                break
        
        if best_rank == rank:
            break
    
    # Return the best found
    if best_U is not None:
        # Convert to integer/half-integer if possible
        # Check if values are close to integers or half-integers
        U_out = [[int(round(U_out_val * 2)) / 2 for U_out_val in row] for row in best_U]
        V_out = [[int(round(V_out_val * 2)) / 2 for V_out_val in row] for row in best_V]
        W_out = [[int(round(W_out_val * 2)) / 2 for W_out_val in row] for row in best_W]
        
        # Verify
        n, m, p = DIMENSIONS
        rank = len(U_out[0])
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U_out[i*m+j][c] * V_out[j*p+k][c] * W_out[k*n+i][c]
                    if abs(val - 1.0) > 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback to schoolbook
    U_out = [[float(x) for x in row] for row in U_sb]
    V_out = [[float(x) for x in row] for row in V_sb]
    W_out = [[float(x) for x in row] for row in W_sb]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
