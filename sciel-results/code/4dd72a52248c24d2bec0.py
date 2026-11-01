"""Find a rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication."""
import time
import itertools
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [2, 3, 5]
DEADLINE_SECONDS = 110  # safety margin before 120s limit (plus some overhead)

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication, T[i,j,k] should be 1 if the (i,j) entry of A times (j,k) of B contributes to C[i,k]
    # Actually, the standard decomposition: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 for all valid i,j,k (it's the multiplication tensor)
    # Wait, let me reconsider. The matrix multiplication tensor T has T[i,j,k] = 1 if and only if... 
    # Actually in the standard formulation, the tensor T is defined such that
    # C[i,k] = sum_{j} A[i,j] B[j,k]
    # This can be written as: for each (i,j,k), the coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
    # So the tensor T[i,j,k] = 1 for all i,j,k.
    
    # Check: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication."""
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 30
    
    # Start with schoolbook decomposition
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Try to reduce rank by removing columns that are linearly dependent
    # The key insight: if a column can be expressed as a linear combination of other columns
    # in a way that preserves the tensor decomposition, we can remove it.
    
    # Actually, let's think about this differently. The tensor decomposition requires:
    # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # 
    # This is a system of n*m*p equations in rank * (n*m + m*p + p*n) variables (with the constraint
    # that we have 'rank' columns).
    
    # For rank r, we need to find U (n*m x r), V (m*p x r), W (p*n x r) such that
    # for all (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of multilinear equations. For small ranks, we can try to solve it.
    
    # Let's try ranks from some lower bound up to the schoolbook rank.
    # The minimum possible rank for matrix multiplication is at least max(n*m, m*p, p*n) = max(6, 15, 10) = 15
    # Actually, there are better lower bounds. For n x m times m x p, the rank is at least max(n, m, p) * something.
    # A simple lower bound is max(n*m, m*p, p*n) but that's not tight.
    # Actually, the trivial lower bound is max(n, m, p) for the rank of the "slice" but that's not right either.
    
    # Let's just try to find a decomposition with rank less than 30.
    # We'll use a randomized approach: start with random initializations and try to solve the system.
    
    # For a given rank r, the system is:
    # For each (i,j,k) in range(n) x range(m) x range(p):
    #   sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*m*p = 30 equations.
    # Unknowns: r * (n*m + m*p + p*n) = r * (6 + 15 + 10) = 31r variables.
    
    # For r >= 2, we have more unknowns than equations, so solutions should exist (generically).
    # The question is whether we can find integer/half-integer solutions.
    
    # Let's try small ranks. Start from the theoretical minimum.
    # The rank of the matrix multiplication tensor for n x m * m x p is known to be at least
    # max(n, m, p) for some cases, but for 2x3*3x5, let's just try various ranks.
    
    # Actually, let's think about this more carefully. The standard result is that the
    # tensor rank of the matrix multiplication tensor T_{n,m,p} is at most n*m*p (schoolbook)
    # and at least max(n*m, m*p, p*n) / min(n, m, p) ... no, that's not right.
    
    # A known lower bound is that the rank is at least max(n, m, p) for the "border rank" but
    # for actual rank, a simple bound is that rank >= max(n*m, m*p, p*n) is NOT correct.
    
    # Let me just try to find decompositions for ranks from 15 up to 29.
    
    # Strategy: For each candidate rank r, try to find U, V, W with the required property.
    # We can fix V and W to be random (or structured) and solve for U, or use other approaches.
    
    # Actually, a cleaner approach: fix two of the matrices and solve for the third.
    # If we fix V and W, then for each (i,j,k):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This is linear in U! For each i, we have m*p equations (over j,k) in m unknowns (U[i*m+0..m-1, c] for all c).
    # Wait, no. For fixed i, the unknowns are U[i*m+j, c] for j in 0..m-1 and c in 0..r-1, so m*r unknowns.
    # The equations are: for each (j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # That's m*p equations in m*r unknowns.
    
    # If m*r > m*p, i.e., r > p, we have more unknowns than equations and can find a solution.
    # For our case, p=5, so if r > 5, we can solve for U given V and W.
    
    # Similarly, if we fix U and W, we can solve for V if r > n (n=2, so r>2).
    # If we fix U and V, we can solve for W if r > m (m=3, so r>3).
    
    # So for r >= 6 (which is > max(n,m,p) = 5), we should be able to find a solution.
    
    # Let's try to find a decomposition with small rank.
    # We'll iterate over candidate ranks and try random initializations.
    
    def try_rank(r, num_trials=50):
        """Try to find a decomposition of rank r."""
        for trial in range(num_trials):
            # Initialize V and W randomly
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # For each i, solve for U[i*m: (i+1)*m, :]
            U = np.zeros((n * m, r))
            feasible = True
            for i in range(n):
                # Equations: for each (j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # Unknowns: U[i*m+0, :], ..., U[i*m+(m-1), :] -> m*r unknowns
                # Build the linear system A * u = b where u is the vectorized U[i*m: (i+1)*m, :]
                A = np.zeros((m * p, m * r))
                b = np.ones(m * p)
                for j in range(m):
                    for k in range(p):
                        eq_idx = j * p + k
                        for c in range(r):
                            col_idx = j * r + c
                            A[eq_idx, col_idx] = V[j * p + k, c] * W[k * n + i, c]
                
                # Solve A * u = b
                try:
                    u, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    # Check if the solution is close to exact
                    if np.max(np.abs(A @ u - b)) < 1e-8:
                        U[i * m: (i + 1) * m, :] = u.reshape(m, r)
                    else:
                        feasible = False
                        break
                except:
                    feasible = False
                    break
            
            if not feasible:
                continue
            
            # Verify the full decomposition
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
                # Try to round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-8):
                    return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
                else:
                    # Return the unrounded version (should still be valid)
                    return U.tolist(), V.tolist(), W.tolist()
        
        return None
    
    # Try ranks from a reasonable lower bound up to the schoolbook rank
    # Lower bound: the rank must be at least max(n, m, p) = 5, but more realistically
    # for matrix multiplication, it's known that rank(T_{n,m,p}) >= max(n*m, m*p, p*n) / max(n, m, p)
    # For 2x3*3x5: max(6,15,10)/5 = 3, which is weak.
    # Let's just try from rank 10 up.
    
    for r in range(10, best_rank):
        if time.time() - start_time > DEADLINE_SECONDS - 5:
            break
        result = try_rank(r, num_trials=100)
        if result is not None:
            U_sol, V_sol, W_sol = result
            # Verify once more
            if _verify_decomposition(U_sol, V_sol, W_sol, n, m, p, tol=1e-8):
                best_rank = r
                best_U, best_V, best_W = U_sol, V_sol, W_sol
                # If we found a good rank, we can stop early
                if r <= 15:
                    break
    
    # If we didn't find anything better, return the schoolbook
    if best_rank < rank_schoolbook:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
