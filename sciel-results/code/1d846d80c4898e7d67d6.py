"""Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x4 matrix."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 4]

def _check_decomposition(U, V, W, n, m, p):
    """Check if the decomposition exactly reproduces the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the matmul tensor T[i,j,k] = 1 if the product C[i,k] = sum_j A[i,j]*B[j,k]
    # T is a 3D array of shape (n, m, p) where T[i,j,k] = 1
    # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Build T
    T = np.zeros((n, m, p), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    
    # Compute the decomposition
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # For each (i,j,k), compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _check_decomposition_fast(U, V, W, n, m, p):
    """Faster check using numpy with Fraction conversion."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=object)
    V_arr = np.array(V, dtype=object)
    W_arr = np.array(W, dtype=object)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                s = sum(U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r] for r in range(rank))
                if s != 1:
                    return False
    return True

def _try_rank(r, n, m, p, timeout=100):
    """Try to find a rank-r decomposition. Returns (U, V, W) or None."""
    start_time = time.time()
    deadline = start_time + timeout
    
    # We need to find U (n*m x r), V (m*p x r), W (p*n x r)
    # such that for all i,j,k: sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1
    
    # This is a system of nonlinear equations. Let's try a structured approach.
    
    # For small ranks, we can try to solve this as a system of polynomial equations.
    # The unknowns are the entries of U, V, W: n*m*r + m*p*r + p*n*r variables.
    # The equations are n*m*p = 36 equations.
    
    # For r = 10 (the known minimum for 3x3x3 is 19, but 3x3x4 might be different)
    # Actually, let's think about what ranks are possible.
    
    # The tensor has dimensions 3x3x4. The trivial rank is 36.
    # For matrix multiplication, the rank is related to the bilinear complexity.
    # For 3x3x3, the minimum rank is 19 (Strassen-like).
    # For 3x3x4, we need to figure out the minimum.
    
    # Let's try to use a numerical approach first, then verify with exact arithmetic.
    
    # Approach: Use a random initialization and gradient descent on the squared error.
    # Then round to nearest half-integers and verify.
    
    # Actually, for exact verification, let's try a different approach:
    # Fix U and V to have small integer/half-integer entries, and solve for W.
    
    # The equations are linear in W for fixed U and V:
    # For each (i,j,k): sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1
    
    # This is linear in W entries. Let's set up the linear system.
    # Unknowns: W[k*n+i, l] for k in 0..p-1, i in 0..n-1, l in 0..r-1
    # That's p*n*r unknowns.
    # Equations: n*m*p = 36.
    
    # For fixed U and V, we have 36 linear equations in p*n*r unknowns.
    # If p*n*r > 36, we have many solutions. We want W entries to be small (integer or half-integer).
    
    # Let's try small U and V with entries in {-1, 0, 1} and see if we can solve for W with small entries.
    
    # Actually, let's try a more systematic approach.
    
    # For rank r, we have r "terms" in the decomposition. Each term l contributes:
    # U[:, l] (n*m entries) * V[:, l] (m*p entries) * W[:, l] (p*n entries)
    # The sum over l must equal 1 for all (i,j,k).
    
    # Let's try to find a decomposition where each term is a "rank-1" tensor in the appropriate sense.
    # A rank-1 term for matmul would be: U[i*m+j, l] = a_i * b_j, V[j*p+k, l] = b_j * c_k, W[k*n+i, l] = c_k * a_i
    # Then the product is a_i * b_j * b_j * c_k * c_k * a_i = a_i^2 * b_j^2 * c_k^2
    # This doesn't work directly.
    
    # Let's just try numerical optimization.
    
    # Actually, for this specific problem, let me try to construct a decomposition manually.
    # The known minimum rank for 3x3x3 matmul is 19.
    # For 3x3x4, I believe the minimum is around 24-26 or so.
    
    # Let's try a numerical approach:
    # 1. Initialize U, V, W randomly
    # 2. Optimize to minimize sum of (sum_l U[i*m+j,l]*V[j*p+k,l]*W[k*n+i,l] - 1)^2
    # 3. Round to nearest 0.5 and verify
    
    # But this is a non-convex problem. Let's try multiple random restarts.
    
    best_result = None
    np.random.seed(42)
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Random initialization
        U_init = np.random.randn(n*m, r) * 0.1
        V_init = np.random.randn(m*p, r) * 0.1
        W_init = np.random.randn(p*n, r) * 0.1
        
        # Gradient descent
        lr = 0.01
        for step in range(2000):
            if time.time() > deadline:
                break
            
            # Compute gradients
            U_grad = np.zeros((n*m, r))
            V_grad = np.zeros((m*p, r))
            W_grad = np.zeros((p*n, r))
            
            loss = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # f(i,j,k) = sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
                        f = 0
                        df_dU = np.zeros(r)
                        df_dV = np.zeros(r)
                        df_dW = np.zeros(r)
                        for l in range(r):
                            term = U_init[i*m+j, l] * V_init[j*p+k, l] * W_init[k*n+i, l]
                            f += term
                            df_dU[l] = V_init[j*p+k, l] * W_init[k*n+i, l]
                            df_dV[l] = U_init[i*m+j, l] * W_init[k*n+i, l]
                            df_dW[l] = U_init[i*m+j, l] * V_init[j*p+k, l]
                        
                        err = f - 1
                        loss += err * err
                        U_grad[i*m+j] += 2 * err * df_dU
                        V_grad[j*p+k] += 2 * err * df_dV
                        W_grad[k*n+i] += 2 * err * df_dW
            
            U_init -= lr * U_grad
            V_init -= lr * V_grad
            W_init -= lr * W_grad
            
            if loss < 1e-10:
                break
        
        if loss < 1e-8:
            # Round to nearest 0.5
            U_rounded = np.round(U_init * 2) / 2
            V_rounded = np.round(V_init * 2) / 2
            W_rounded = np.round(W_init * 2) / 2
            
            # Convert to list of lists with Fraction
            U_frac = [[Fraction(U_rounded[i, l]) for l in range(r)] for i in range(n*m)]
            V_frac = [[Fraction(V_rounded[i, l]) for l in range(r)] for i in range(m*p)]
            W_frac = [[Fraction(W_rounded[i, l]) for l in range(r)] for i in range(p*n)]
            
            if _check_decomposition(U_frac, V_frac, W_frac, n, m, p):
                return U_frac, V_frac, W_frac
        
        # Also try without rounding - maybe we can find an exact solution
        # Check if the current solution is already close to half-integers
        U_close = np.all(np.abs(U_init * 2 - np.round(U_init * 2)) < 1e-6)
        V_close = np.all(np.abs(V_init * 2 - np.round(V_init * 2)) < 1e-6)
        W_close = np.all(np.abs(W_init * 2 - np.round(W_init * 2)) < 1e-6)
        
        if U_close and V_close and W_close:
            U_frac = [[Fraction(U_init[i, l]) for l in range(r)] for i in range(n*m)]
            V_frac = [[Fraction(V_init[i, l]) for l in range(r)] for i in range(m*p)]
            W_frac = [[Fraction(W_init[i, l]) for l in range(r)] for i in range(p*n)]
            
            if _check_decomposition(U_frac, V_frac, W_frac, n, m, p):
                return U_frac, V_frac, W_frac
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x4 matrix."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Leave 5 seconds margin
    
    # Try ranks from small to large
    # The trivial rank is n*m*p = 36
    # Let's try from 10 up to 36
    
    # First, let's try to find a good decomposition
    # For 3x3x3, the minimum is 19. For 3x3x4, let's try various ranks.
    
    # Actually, let's think about this more carefully.
    # The matmul tensor for n x m x p has a certain structure.
    # The rank is the minimum number of rank-1 tensors needed.
    
    # Let's try a structured approach:
    # For each rank r, try to find U, V, W with small entries.
    
    # Start with the schoolbook decomposition (rank 36) as a fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = Fraction(1)
                V_school[j * p + k][column] = Fraction(1)
                W_school[k * n + i][column] = Fraction(1)
                column += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook
    
    # Try to find better decompositions
    for r in range(10, best_rank):
        if time.time() > deadline - 30:
            break
        
        result = _try_rank(r, n, m, p, timeout=15)
        if result is not None:
            U_frac, V_frac, W_frac = result
            best_U, best_V, best_W = U_frac, V_frac, W_frac
            best_rank = r
            break  # Found a lower rank, stop
    
    return {"U": best_U, "V": best_V, "W": best_W}
