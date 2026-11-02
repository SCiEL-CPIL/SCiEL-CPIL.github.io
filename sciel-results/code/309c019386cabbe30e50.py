import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using fractions."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if val != 1:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using numpy."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = 3, 5, 7
    nm, mp, pn = n*m, m*p, p*n  # 15, 35, 21
    
    def schoolbook():
        """Standard schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def try_rank(r, num_attempts=500):
        """
        Try to find a decomposition of rank r.
        Randomly initialize V and W with small integers, solve for U,
        and check if U has integer/half-integer entries.
        """
        if time.time() > deadline:
            return None
        
        # We want to find integer/half-integer solutions.
        # Using V, W in {-1, 1} is a good heuristic.
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None
            
            # Use a deterministic seed for reproducibility in debugging, 
            # but random is fine for search.
            # random.seed(12345 + r * 100 + attempt) # Uncomment for debugging
            
            # Initialize V and W
            # Using integers in {-1, 1}
            V = [[random.choice([-1, 1]) for _ in range(r)] for _ in range(mp)]
            W = [[random.choice([-1, 1]) for _ in range(r)] for _ in range(pn)]
            
            U = [[0.0] * r for _ in range(nm)]
            valid = True
            
            # Solve for U block by block (for each i)
            for i in range(n):
                # System: A * u_i = b
                # u_i is a vector of length m*r (flattened U rows for fixed i)
                # A is m*p x m*r
                A_mat = np.zeros((m*p, m*r))
                b_vec = np.ones(m*p)
                
                for j in range(m):
                    for k in range(p):
                        eq_idx = j * p + k
                        for c in range(r):
                            A_mat[eq_idx, j*r + c] = float(V[j*p+k][c] * W[k*n+i][c])
                
                try:
                    # Use least squares to find a solution
                    # Since we want exact integers, we check if the solution is close to them
                    sol, residuals, rank_A, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Check if solution is close to integers or half-integers
                # Try integers
                sol_int = np.round(sol)
                diff_int = np.max(np.abs(sol - sol_int))
                
                # Try half-integers
                sol_half = np.round(sol * 2.0) / 2.0
                diff_half = np.max(np.abs(sol - sol_half))
                
                if diff_int < diff_half:
                    sol_rounded = sol_int
                    diff = diff_int
                else:
                    sol_rounded = sol_half
                    diff = diff_half
                
                # Threshold for "close enough" to be considered exact after rounding
                # Given the linear system with integer coefficients, if a solution exists,
                # it will be rational. If the denominator is small, the float representation
                # should be very close to the rational value.
                if diff > 1e-8:
                    valid = False
                    break
                
                for j in range(m):
                    for c in range(r):
                        U[i*m+j][c] = float(sol_rounded[j*r + c])
            
            if not valid:
                continue
            
            # Convert to Fractions for exact verification
            # We assume the rounded values are the exact solution
            U_frac = [[Fraction(int(x)) if x == int(x) else Fraction(int(x*2), 2) for x in row] for row in U]
            # V and W are already integers
            V_frac = [[Fraction(x) for x in row] for row in V]
            W_frac = [[Fraction(x) for x in row] for row in W]
            
            if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
                return {
                    "U": [[float(x) for x in row] for row in U_frac],
                    "V": [[float(x) for x in row] for row in V_frac],
                    "W": [[float(x) for x in row] for row in W_frac]
                }
        
        return None

    def try_rank_circulant(r, num_attempts=100):
        """
        Attempt to find a decomposition using a circulant structure for V or W.
        This imposes structure that might allow lower rank solutions.
        We fix W to be block-circulant based on a small kernel.
        """
        if time.time() > deadline:
            return None
        
        # Try to construct W with circulant structure
        # W has shape (p*n, r) = (21, r)
        # We can view W as p blocks of n rows.
        # Let's try making each block a circulant shift of a base vector.
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None
            
            # Random base vector for the circulant part
            base_w = [random.choice([-1, 1]) for _ in range(n)]
            
            W = [[0] * r for _ in range(pn)]
            for k in range(p):
                for i in range(n):
                    # Cyclic shift
                    idx = (i - k) % n
                    W[k*n + i] = [base_w[idx]] * r # This is too simple, let's mix
            
            # Actually, let's just use random V and W but with a constraint on W's structure
            # Or better: Use the standard try_rank but with a different initialization strategy
            # The "circulant" idea is hard to apply directly to the rectangular matrices.
            # Let's stick to the robust random search but maybe try different integer ranges.
            
            pass # Placeholder for specific circulant logic if it were straightforward
            
        return None

    best_result = None
    best_rank = n * m * p
    
    # Try ranks from a lower bound upwards
    # Lower bound is 35. 
    # We start at 40 to allow some slack, but we should try 35, 36... if we have time.
    # However, 35 is very tight. 
    # Let's try from 40 to 105.
    
    # To maximize chance, we should spend more time on lower ranks.
    # But if lower ranks are impossible, we waste time.
    # Given the "random V, W" heuristic, success probability might be low for small r.
    
    for r in range(40, 106):
        if time.time() > deadline - 5:
            break
        
        # Allocate attempts based on remaining time and rank priority
        # Simpler: fixed number of attempts, but break early if time is low
        attempts = 100
        if r < 50:
            attempts = 200 # Try harder for lower ranks
        
        result = try_rank(r, num_attempts=attempts)
        if result is not None:
            best_result = result
            best_rank = r
            break

    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return schoolbook()
