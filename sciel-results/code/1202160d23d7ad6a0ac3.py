"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 2*3*9 = 54
    
    def build_tensors(U, V, W):
        """Check if U,V,W give exact decomposition."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
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
    
    # Strategy: Try to find a lower-rank decomposition using structured random search
    # Key insight: For matrix multiplication, the rank is known to be at most n*m*p
    # but can be lower. For 2x3 * 3x9, let's try various approaches.
    
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # Approach 1: Try to find structured decompositions
    # For matrix multiplication, we can use the fact that:
    # (AB)_{ik} = sum_j A_{ij} B_{jk}
    # This can be decomposed as a sum of rank-1 terms.
    
    # Let's try a systematic search with better initialization
    # Use the structure of matrix multiplication to guide the search
    
    # First, let's try ranks that are known to be achievable
    # The rank of 2x3 * 3x9 matrix multiplication is at most 54
    # but we can try to find lower rank decompositions
    
    # Try a grid search with structured initialization
    for rank in range(10, min(schoolbook_rank + 1, 60)):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        max_attempts = 10
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with structure
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Try to refine using alternating least squares
            for iteration in range(30):
                if time.time() > deadline - 10:
                    break
                
                # Check if current decomposition is valid
                valid = True
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(rank):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                            if err > 1e-6:
                                valid = False
                
                if valid:
                    best_rank = rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    break
                
                if max_err < 1e-4:
                    # Close, try to fine-tune
                    for _ in range(50):
                        idx = random.randint(0, 2)
                        if idx == 0:
                            U[random.randint(0, n*m-1), random.randint(0, rank-1)] += np.random.randn() * 0.01
                        elif idx == 1:
                            V[random.randint(0, m*p-1), random.randint(0, rank-1)] += np.random.randn() * 0.01
                        else:
                            W[random.randint(0, p*n-1), random.randint(0, rank-1)] += np.random.randn() * 0.01
                else:
                    # Larger perturbation
                    idx = random.randint(0, 2)
                    if idx == 0:
                        U[random.randint(0, n*m-1), random.randint(0, rank-1)] += np.random.randn() * 0.5
                    elif idx == 1:
                        V[random.randint(0, m*p-1), random.randint(0, rank-1)] += np.random.randn() * 0.5
                    else:
                        W[random.randint(0, p*n-1), random.randint(0, rank-1)] += np.random.randn() * 0.5
            
            if best_rank == rank:
                break
        
        if best_rank == rank:
            break
    
    # Return the best found
    if best_U is not None:
        # Convert to integer/half-integer if possible
        U_out = [[int(round(float(U_out_val) * 2)) / 2 for U_out_val in row] for row in best_U]
        V_out = [[int(round(float(V_out_val) * 2)) / 2 for V_out_val in row] for row in best_V]
        W_out = [[int(round(float(W_out_val) * 2)) / 2 for W_out_val in row] for row in best_W]
        
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
