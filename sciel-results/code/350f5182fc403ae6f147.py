import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(4, 6, 6) = 6
    # But for matrix multiplication, Strassen-like bounds suggest rank >= n*m*p/(n+m+p-1) or similar
    # For 2x2x2, minimal rank is 7 (Strassen)
    # For 2x2x3, we'll search from rank 8 upward
    
    # First, let's try to find a rank-10 solution using structured parameterization
    # Then try rank-9, rank-8, etc.
    
    # The tensor equation: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where T[i,j,k] = 1 if (i,j,k) corresponds to a valid multiplication term, else 0
    # For 2x2 * 2x3: output C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k), and we need:
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_r (sum_j U[i,j,r]*V[j,k,r]) * W[k,i,r]
    # This means: sum_j U[i,j,r]*V[j,k,r] * W[k,i,r] must equal sum_j A[i,j]*B[j,k]
    # For this to hold for all A,B, we need:
    # For each r: the matrix M_r = U_r @ V_r (where U_r is n x m, V_r is m x p)
    # and the scalar w_r[k,i] = W[k*n+i, r]
    # Then: sum_r w_r[k,i] * (M_r)[i,k] = sum_j A[i,j]*B[j,k]
    
    # Actually, the standard formulation is:
    # C[i,k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    # Wait, let me re-read the convention.
    
    # The task says: "output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by [k*n + i, r] where k is the column of C, i is the row of C
    # U is indexed by [i*m + j, r] where i is row of A, j is column of A
    # V is indexed by [j*p + k, r] where j is row of B, k is column of B
    
    # The decomposition should satisfy:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
    
    # For this to be an exact bilinear decomposition, we need:
    # For all i, k, j: the coefficient of A[i,j]*B[j,k] in the RHS must be 1
    # The RHS expands to: sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r])
    # = sum_r sum_j W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r]
    
    # For this to equal sum_j A[i,j]*B[j,k] for all A, B, we need:
    # For each (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all other combinations (where the product A[i,j]*B[j,k] doesn't appear in C[i,k]):
    # Actually, every (i,j,k) triple with valid indices contributes to C[i,k] via the j-th term.
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
    
    # Let's define the constraint: for all i in [0,n), j in [0,m), k in [0,p):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of n*m*p = 12 equations in rank variables for each matrix.
    # Total unknowns: rank * (n*m + m*p + p*n) = rank * (4 + 6 + 6) = rank * 16
    # We need rank >= 12/16... no, that's not right.
    
    # Actually, each equation involves rank terms. We have 12 equations.
    # The unknowns are U (4*rank), V (6*rank), W (6*rank), total 16*rank unknowns.
    # But the equations are cubic in the unknowns (product of 3 variables).
    
    # Let's try a structured approach: assume a specific form that reduces the search space.
    
    # For 2x2x2, Strassen's algorithm uses rank 7 with specific structure.
    # For 2x2x3, we can try to extend this idea.
    
    # Let's start with a known valid decomposition (schoolbook rank 12) and try to reduce.
    
    # First, generate the schoolbook solution as a baseline
    def schoolbook_decomposition():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank
    
    # Verify a decomposition
    def verify_decomposition(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    # Try rank 10 first with a structured parameterization
    # We'll use the fact that we can fix some variables to reduce the search space
    
    # Strategy: Start from rank 12 and try to merge terms
    # Or: Use a parameterized family and solve for parameters
    
    # Let's try a different approach: use random initialization and Newton's method
    # to solve the nonlinear system
    
    def try_rank(rank):
        if time.time() > deadline:
            return None
        
        # Initialize with random values
        np.random.seed(42 + rank)
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            # Define the residual function
            def residuals(params):
                U_flat = params[:n*m*rank].reshape(n*m, rank)
                V_flat = params[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                W_flat = params[n*m*rank+m*p*rank:].reshape(p*n, rank)
                
                res = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for r in range(rank):
                                val += U_flat[i*m+j, r] * V_flat[j*p+k, r] * W_flat[k*n+i, r]
                            res.append(val - 1)
                return np.array(res)
            
            # Use scipy if available, otherwise implement simple Newton
            try:
                from scipy.optimize import fsolve
                x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                solution = fsolve(residuals, x0, full_output=True, maxfev=10000)
                x_sol, info, ier, msg = solution
                
                if ier == 1:
                    U_sol = x_sol[:n*m*rank].reshape(n*m, rank)
                    V_sol = x_sol[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                    W_sol = x_sol[n*m*rank+m*p*rank:].reshape(p*n, rank)
                    
                    # Convert to lists and check
                    U_list = U_sol.tolist()
                    V_list = V_sol.tolist()
                    W_list = W_sol.tolist()
                    
                    if verify_decomposition(U_list, V_list, W_list, rank):
                        return U_list, V_list, W_list, rank
            except Exception:
                pass
        
        return None
    
    # Try ranks from 10 down to 8
    for rank in [10, 9, 8]:
        if time.time() > deadline:
            break
        
        result = try_rank(rank)
        if result is not None:
            U, V, W, r = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W, rank = schoolbook_decomposition()
    return {"U": U, "V": V, "W": W}
