import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def _verify_np(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using numpy."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U[i*m+j, :]
                v_row = V[j*p+k, :]
                w_row = W[k*n+i, :]
                val = np.sum(u_row * v_row * w_row)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_fraction(U, V, W, n, m, p):
    """Exact verification using fractions."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if val != 1:
                    return False
    return True

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = 3, 5, 7
    N = n * m  # 15
    M = m * p  # 35
    P = p * n  # 21
    total = N + M + P  # 71
    target_rank = 104
    schoolbook_rank = n * m * p  # 105

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    def try_rank(r, num_attempts=50):
        """Try to find a decomposition of rank r using random V,W and solving for U."""
        if time.time() > deadline:
            return None
        
        # Precompute the structure of equations
        # For each i in range(n):
        #   We have m*p equations.
        #   Unknowns: U[i*m+j, c] for j in range(m), c in range(r).
        #   Equation (j, k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None
            
            # Random V and W with small integer entries
            # Using {-1, 0, 1} is common but might be too sparse?
            # Let's try {-1, 0, 1, 2} or just {-1, 1}?
            # Random integers in [-1, 1] often work well for linear algebra over reals.
            V = np.array([[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(M)], dtype=float)
            W = np.array([[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(P)], dtype=float)
            
            U = np.zeros((N, r), dtype=float)
            
            valid = True
            for i in range(n):
                # Build the system for this i
                # A_mat has shape (m*p, m*r)
                # Row (j, k) corresponds to equation index j*p + k
                # Column (j, c) corresponds to unknown index j*r + c
                # A[j*p+k, j*r+c] = V[j*p+k, c] * W[k*n+i, c]
                
                A_mat = np.zeros((m*p, m*r))
                b_vec = np.ones(m*p)
                
                for j in range(m):
                    for k in range(p):
                        eq_idx = j*p + k
                        for c in range(r):
                            A_mat[eq_idx, j*r + c] = V[j*p+k, c] * W[k*n+i, c]
                
                # Solve least squares
                try:
                    # Check if system is consistent (rank full?)
                    # If not full rank, there might be multiple solutions.
                    # lstsq finds the minimum norm solution.
                    sol, residuals, rank_A, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except:
                    valid = False
                    break
                
                # Check if solution is close to integer/half-integer
                sol_rounded = np.round(sol * 2) / 2
                diff = np.max(np.abs(sol - sol_rounded))
                if diff > 1e-6:
                    # Maybe try to snap to nearest integer if close?
                    # Or reject.
                    valid = False
                    break
                
                U[i*m:(i+1)*m, :] = sol_rounded
            
            if not valid:
                continue
            
            # Verify the full solution
            if _verify_np(U, V, W, n, m, p):
                # Convert to exact fractions
                # Ensure we use Fraction for exactness
                U_frac = [[Fraction(U[i][j]).limit_denominator(2) for j in range(r)] for i in range(N)]
                V_frac = [[Fraction(V[i][j]).limit_denominator(2) for j in range(r)] for i in range(M)]
                W_frac = [[Fraction(W[i][j]).limit_denominator(2) for j in range(r)] for i in range(P)]
                
                # Double check with fractions
                if _verify_fraction(U_frac, V_frac, W_frac, n, m, p):
                    return {
                        "U": [[float(x) for x in row] for row in U_frac],
                        "V": [[float(x) for x in row] for row in V_frac],
                        "W": [[float(x) for x in row] for row in W_frac]
                    }
        
        return None

    # Try to find a rank 104 decomposition
    # The schoolbook rank is 105.
    # Is 104 possible?
    # Lower bound for rank of matmul is max(n*m, m*p, p*n) = 35? No.
    # Strassen-like improvements?
    # For small dimensions, exact minimal rank is often close to n*m*p.
    # Let's try 104 first.
    
    # Strategy:
    # 1. Try random search for rank 104.
    # 2. If found, return it.
    # 3. If not, try rank 103? (Less likely)
    # 4. If not, return schoolbook (105).
    
    # To increase chances, we can run multiple attempts.
    # Time budget: ~140s.
    # Each attempt is fast (linear algebra on small matrices).
    # Let's allocate time for 104.
    
    result = None
    # Use a larger number of attempts if time permits
    # Estimate time per attempt:
    # 3 systems of size (35, 35*104) -> wait, A_mat is (35, 5*104) = (35, 520).
    # lstsq on (35, 520) is very fast.
    # Verification is O(105 * 104) ~ 10k ops.
    # So we can do thousands of attempts.
    
    # Let's run a loop with a deadline check.
    t_start = time.time()
    attempts = 0
    while time.time() < deadline - 10:
        if attempts % 100 == 0:
            # Every 100 attempts, check if we should switch rank?
            pass
        result = try_rank(104, num_attempts=100)
        if result is not None:
            return result
        attempts += 100
        if time.time() > deadline - 5:
            break
    
    # If 104 failed, try 103?
    # Unlikely, but if we have time.
    if time.time() < deadline - 2:
        result = try_rank(103, num_attempts=50)
        if result is not None:
            return result

    return schoolbook()
