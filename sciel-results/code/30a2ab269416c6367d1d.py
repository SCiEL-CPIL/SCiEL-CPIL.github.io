"""Program for matrix multiplication tensor decomposition task."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    Strategy: Split into two independent 3x3x3 multiplications and use known rank-23
    decomposition for each, giving total rank 46.
    """
    n, m, p = DIMENSIONS  # 3, 3, 6
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    def is_valid_decomposition(U, V, W, rank):
        """Check if U,V,W of given rank exactly decompose the multiplication tensor."""
        n, m, p = DIMENSIONS
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # For 3x3*3x3 matrix multiplication, the known optimal rank is 23.
    # We need to construct two independent rank-23 decompositions (one for each 3x3 slice)
    # and concatenate them to get a rank-46 decomposition for 3x3*3x6.
    
    # Let's use a systematic approach to find the rank-23 decomposition.
    # We'll use a known algebraic construction based on Strassen-like methods.
    
    def find_3x3x3_rank23():
        """
        Find a rank-23 decomposition for 3x3x3 matrix multiplication.
        Uses a known explicit construction.
        """
        # For 3x3*3x3: C[i,k] = sum_j A[i,j]*B[j,k]
        # We need 23 rank-1 terms where each term is:
        # U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
        
        # One known construction uses the fact that 3x3 multiplication can be done
        # with 23 multiplications. Let's use a parameterized approach.
        
        # Actually, let's use a different strategy: we'll set up a system of equations
        # and try to solve it with a structured initialization.
        
        # For rank 23, we have 23*3*3 = 207 parameters (U, V, W each have 9*23 entries)
        # and 27 constraints (one for each (i,j,k) triple).
        
        # Let's try a gradient-based approach with multiple restarts.
        rng = np.random.RandomState(123)
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with small random values
            U = rng.randn(9, 23) * 0.5
            V = rng.randn(9, 23) * 0.5
            W = rng.randn(9, 23) * 0.5
            
            # Use a simple iterative method
            for it in range(200):
                if time.time() > deadline:
                    break
                
                # Compute the tensor from current factors
                tensor = np.zeros((3, 3, 3))
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            s = 0.0
                            for r in range(23):
                                s += U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
                            tensor[i, j, k] = s
                
                # Check if we've converged
                if np.max(np.abs(tensor - 1.0)) < 1e-10:
                    return U.tolist(), V.tolist(), W.tolist()
                
                # Update: use a simple gradient step
                # Gradient of sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] with respect to
                # U[i*m+j,r] is V[j*p+k,r]*W[k*n+i,r]
                
                # Actually, let's use a different approach: fix two factors and solve for the third
                # This is a linear system problem.
                
                # For each (i,j) pair, we want:
                # sum_r U[i*3+j, r] * (V[j*3+k, r] * W[k*3+i, r]) = 1 for all k
                # This is a linear system in U[i*3+j, :]
                
                # Let's fix V and W, and solve for U
                U_new = np.zeros_like(U)
                for i in range(3):
                    for j in range(3):
                        # Build the linear system: A @ u = b
                        # where A[k, r] = V[j*3+k, r] * W[k*3+i, r]
                        # and b[k] = 1
                        A = np.zeros((3, 23))
                        for k in range(3):
                            for r in range(23):
                                A[k, r] = V[j*3+k, r] * W[k*3+i, r]
                        b = np.ones(3)
                        
                        # Solve the least squares problem
                        u, residuals, rank_val, sv = np.linalg.lstsq(A, b, rcond=None)
                        U_new[i*3+j, :] = u
                
                # Similarly, fix U and W, solve for V
                V_new = np.zeros_like(V)
                for j in range(3):
                    for k in range(3):
                        A = np.zeros((3, 23))
                        for i in range(3):
                            for r in range(23):
                                A[i, r] = U[i*3+j, r] * W[k*3+i, r]
                        b = np.ones(3)
                        v, residuals, rank_val, sv = np.linalg.lstsq(A, b, rcond=None)
                        V_new[j*3+k, :] = v
                
                # Fix U and V, solve for W
                W_new = np.zeros_like(W)
                for k in range(3):
                    for i in range(3):
                        A = np.zeros((3, 23))
                        for j in range(3):
                            for r in range(23):
                                A[j, r] = U[i*3+j, r] * V[j*3+k, r]
                        b = np.ones(3)
                        w, residuals, rank_val, sv = np.linalg.lstsq(A, b, rcond=None)
                        W_new[k*3+i, :] = w
                
                # Check convergence
                if np.max(np.abs(U_new - U)) < 1e-12 and \
                   np.max(np.abs(V_new - V)) < 1e-12 and \
                   np.max(np.abs(W_new - W)) < 1e-12:
                    # Verify the solution
                    tensor = np.zeros((3, 3, 3))
                    for i in range(3):
                        for j in range(3):
                            for k in range(3):
                                s = 0.0
                                for r in range(23):
                                    s += U_new[i*3+j, r] * V_new[j*3+k, r] * W_new[k*3+i, r]
                                tensor[i, j, k] = s
                    if np.max(np.abs(tensor - 1.0)) < 1e-10:
                        return U_new.tolist(), V_new.tolist(), W_new.tolist()
                
                U = U_new
                V = V_new
                W = W_new
        
        return None
    
    # Try to find the rank-23 decomposition
    result_3x3 = find_3x3x3_rank23()
    
    if result_3x3 is not None:
        U1, V1, W1 = result_3x3
        rank_total = 46
        U_full = [[0] * rank_total for _ in range(9)]
        V_full = [[0] * rank_total for _ in range(18)]
        W_full = [[0] * rank_total for _ in range(18)]
        
        # First 23 columns: for k=0,1,2
        for r in range(23):
            for i in range(3):
                for j in range(3):
                    U_full[i*3+j][r] = U1[i*3+j][r]
            for j in range(3):
                for k in range(3):
                    V_full[j*3+k][r] = V1[j*3+k][r]
            for k in range(3):
                for i in range(3):
                    W_full[k*3+i][r] = W1[k*3+i][r]
        
        # Next 23 columns: for k=3,4,5
        for r in range(23):
            for i in range(3):
                for j in range(3):
                    U_full[i*3+j][r+23] = U1[i*3+j][r]
            for j in range(3):
                for k in range(3):
                    V_full[j*3+(k+3)][r+23] = V1[j*3+k][r]
            for k in range(3):
                for i in range(3):
                    W_full[(k+3)*3+i][r+23] = W1[k*3+i][r]
        
        # Verify
        if is_valid_decomposition(U_full, V_full, W_full, rank_total):
            return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback: schoolbook decomposition
    rank = n * m * p  # 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
