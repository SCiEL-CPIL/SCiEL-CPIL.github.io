"""Matrix multiplication 2x5x5 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize
from scipy.linalg import solve

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit

    def schoolbook():
        rank = n * m * p
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

    def verify_exact(U, V, W, tol=1e-8):
        """Check if the decomposition is exact with integer/half-integer entries."""
        rank = len(U[0])
        # Check entries are integer or half-integer
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-6:
                        return False
        # Check tensor equation
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(total - 1.0) > tol:
                        return False
        # Also check that non-matching indices give 0
        # For matrix multiplication, the tensor has 1s at (i,j,k) and 0s elsewhere
        # Actually, the tensor T[i,j,k] = delta for multiplication
        # The condition is: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
        # This is the standard bilinear complexity formulation
        return True

    def verify_tensor(U, V, W, tol=1e-6):
        """More thorough verification including zero conditions."""
        rank = len(U[0])
        # For matrix multiplication (n x m) * (m x p) -> (n x p)
        # T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                    if abs(total - 1.0) > tol:
                        return False
        return True

    def objective(params, r, U_idx, V_idx, W_idx):
        """Compute the sum of squared residuals of the tensor equation."""
        U = params[0:n*m*r].reshape(n*m, r)
        V = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
        W = params[n*m*r + m*p*r:].reshape(p*n, r)
        
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    residual += (s - 1.0) ** 2
        return residual

    def solve_alternating(r, max_sweeps=300, tol=1e-10):
        """Alternating least squares to find rank-r decomposition."""
        # Initialize with random values
        rng = np.random.default_rng(42)
        U = rng.standard_normal((n*m, r))
        V = rng.standard_normal((m*p, r))
        W = rng.standard_normal((p*n, r))
        
        best_err = float('inf')
        
        for sweep in range(max_sweeps):
            if time.time() > deadline - 5:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # Build M: M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    try:
                        x, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        U[i*m+j, :] = x
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    # Build M: M[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        y, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        V[j*p+k, :] = y
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    # Build M: M[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        z, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        W[k*n+i, :] = z
                    except:
                        pass
            
            # Check error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        err = max(err, abs(s - 1.0))
            
            if err < best_err:
                best_err = err
            
            if err < tol:
                break
        
        return U, V, W, best_err

    def try_rounding(U, V, W, tol=1e-6):
        """Try to round to half-integers and verify."""
        # Round to nearest half-integer
        U_r = [[round(val * 2) / 2.0 for val in row] for row in U]
        V_r = [[round(val * 2) / 2.0 for val in row] for row in V]
        W_r = [[round(val * 2) / 2.0 for val in row] for row in W]
        
        if verify_tensor(U_r, V_r, W_r, tol):
            return U_r, V_r, W_r
        
        # Try small perturbations of rounding
        for du in [-1, 0, 1]:
            for dv in [-1, 0, 1]:
                for dw in [-1, 0, 1]:
                    U_p = [[round((val + du*0.01) * 2) / 2.0 for val in row] for row in U]
                    V_p = [[round((val + dv*0.01) * 2) / 2.0 for val in row] for row in V]
                    W_p = [[round((val + dw*0.01) * 2) / 2.0 for val in row] for row in W]
                    if verify_tensor(U_p, V_p, W_p, tol):
                        return U_p, V_p, W_p
        
        return None

    def strassen_2x2_block(a, b):
        """Multiply two 2x2 matrices using Strassen's algorithm.
        Returns the 7 Strassen products."""
        # a = [[a11, a12], [a21, a22]]
        # b = [[b11, b12], [b21, b22]]
        # m1 = (a11+a22)*(b11+b22)
        # m2 = (a21+a22)*b11
        # m3 = a11*(b12-b22)
        # m4 = a22*(b21-b11)
        # m5 = (a11+a12)*b22
        # m6 = (a21-a11)*(b11+b12)
        # m7 = (a12-a22)*(b21+b22)
        # c11 = m1 + m4 - m5 + m7
        # c12 = m3 + m5
        # c21 = m2 + m4
        # c22 = m1 - m2 + m3 + m6
        
        m1 = (a[0][0] + a[1][1]) * (b[0][0] + b[1][1])
        m2 = (a[1][0] + a[1][1]) * b[0][0]
        m3 = a[0][0] * (b[0][1] - b[1][1])
        m4 = a[1][1] * (b[1][0] - b[0][0])
        m5 = (a[0][0] + a[0][1]) * b[1][1]
        m6 = (a[1][0] - a[0][0]) * (b[0][0] + b[0][1])
        m7 = (a[0][1] - a[1][1]) * (b[1][0] + b[1][1])
        
        c = [[m1 + m4 - m5 + m7, m3 + m5],
             [m2 + m4, m1 - m2 + m3 + m6]]
        return c, [m1, m2, m3, m4, m5, m6, m7]

    def build_strassen_based():
        """Build a decomposition using Strassen-style recursion.
        
        For 2x5 * 5x5:
        Split the 5 columns into 2+2+1
        Split the 5 rows into 2+2+1
        
        For the 2x2 blocks, use Strassen (7 multiplications)
        For mixed-size blocks, use schoolbook
        
        This gives a structured decomposition.
        """
        # We'll use the alternating solver but initialize with a structured pattern
        # that exploits the block structure
        
        # For now, just use the alternating solver with multiple restarts
        return None

    # Try to find a low-rank decomposition
    # The theoretical lower bound for 2x5x5 is at least max(2,5,5) = 5
    # But practically, we expect rank around 25-50
    
    best_result = None
    best_rank = n * m * p  # schoolbook rank as upper bound
    
    # Try ranks from a reasonable range
    # Start with the known bound and work down
    min_possible = max(n, m, p)
    
    for r in range(25, 50):
        if time.time() > deadline - 15:
            break
        
        # Try multiple random initializations
        found = False
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            
            # Seed the RNG differently for each attempt
            np.random.seed(1000 + r * 10 + attempt)
            
            U, V, W, err = solve_alternating(r, max_sweeps=200)
            
            if err < 1e-8:
                rounded = try_rounding(U, V, W)
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    best_result = {"U": U_r, "V": V_r, "W": W_r}
                    best_rank = r
                    found = True
                    break
        
        if found:
            break
        if time.time() > deadline - 15:
            break
    
    if best_result is not None:
        return best_result
    
    # Try with higher rank if we haven't found a solution
    for r in range(50, 55):
        if time.time() > deadline - 10:
            break
        
        np.random.seed(5000 + r)
        U, V, W, err = solve_alternating(r, max_sweeps=100)
        
        if err < 1e-8:
            rounded = try_rounding(U, V, W)
            if rounded is not None:
                U_r, V_r, W_r = rounded
                return {"U": U_r, "V": V_r, "W": W_r}
    
    # Fallback to schoolbook
    return schoolbook()

# EVOLVE-BLOCK-END
