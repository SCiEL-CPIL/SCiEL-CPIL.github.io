"""Matrix multiplication 3x5x7 - rank minimization via structured search and linear solve."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

def _verify_np(U, V, W, n, m, p):
    """Fast verification using numpy."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                u_row = U[i*m+j, :]
                v_row = V[j*p+k, :]
                w_row = W[k*n+i, :]
                val = np.sum(u_row * v_row * w_row)
                if abs(val - expected) > 1e-10:
                    return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using fractions."""
    from fractions import Fraction
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = Fraction(1)
                val = Fraction(0)
                for c in range(rank):
                    u_val = Fraction(U[i*m+j][c])
                    v_val = Fraction(V[j*p+k][c])
                    w_val = Fraction(W[k*n+i][c])
                    val += u_val * v_val * w_val
                if val != target:
                    return False
    return True

def _round_to_half_integer(x):
    """Round x to nearest half-integer."""
    return round(x * 2) / 2.0

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = 3, 5, 7
    nm, mp, pn = n*m, m*p, p*n  # 15, 35, 21
    
    # Schoolbook baseline: rank 105
    def schoolbook():
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def _try_rank_fixed_VW(V, W, r):
        """Given fixed V and W of shape (mp, r) and (pn, r), solve for U.
        Returns U if a valid half-integer solution exists, else None."""
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        
        U = [[0]*r for _ in range(nm)]
        
        for i in range(n):
            if time.time() > deadline:
                return None
            
            # Unknowns: U[i*m+j, c] for j in range(m), c in range(r) -> m*r unknowns
            # Equations: for j in range(m), k in range(p):
            #   sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            
            A_mat = np.zeros((m*p, m*r))
            b_vec = np.ones(m*p)
            
            for j in range(m):
                for k in range(p):
                    eq_idx = j*p + k
                    for c in range(r):
                        A_mat[eq_idx, j*r + c] = V_np[j*p+k, c] * W_np[k*n+i, c]
            
            # Solve least squares
            try:
                sol, residuals, rank_A, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except:
                return None
            
            # Check if solution is close to half-integer
            sol_rounded = np.round(sol * 2) / 2.0
            max_err = np.max(np.abs(sol - sol_rounded))
            
            if max_err > 1e-8:
                return None
            
            for j in range(m):
                for c in range(r):
                    U[i*m+j][c] = float(sol_rounded[j*r + c])
        
        # Verify the full solution
        if _verify_np(U, V, W, n, m, p):
            if _verify_exact(U, V, W, n, m, p):
                return U
        return None

    def try_rank(r, num_trials=100):
        """Try to find a decomposition of rank r."""
        nonlocal best_result, best_rank
        if r >= best_rank:
            return False
        
        random.seed(42 + r)
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return False
            
            # Generate random V and W with small integer entries
            # Use values from {-2, -1, 0, 1, 2}
            V = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(r)] for _ in range(mp)]
            W = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(r)] for _ in range(pn)]
            
            U = _try_rank_fixed_VW(V, W, r)
            if U is not None:
                best_result = {"U": U, "V": V, "W": W}
                best_rank = r
                return True
        
        return False

    # Try structured approaches first: exploit block structure
    # Strategy: partition the 7 columns of B into groups and use block decomposition
    # For 3x5x7, we can view it as 3 rows of A each multiplying 5x7 B
    # Try to find shared structure across the 3 rows
    
    # First, try a more structured random search with bias towards sparse patterns
    # Use a mix of dense and sparse random matrices
    def try_rank_structured(r, num_trials=50):
        """Try rank r with structured initialization."""
        nonlocal best_result, best_rank
        if r >= best_rank:
            return False
        
        random.seed(1000 + r)
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return False
            
            # Create V and W with some structure
            # Use blocks of non-zero entries
            V = [[0]*r for _ in range(mp)]
            W = [[0]*r for _ in range(pn)]
            
            # Randomly assign non-zero entries
            for row in range(mp):
                # Each row has 2-4 non-zero entries
                num_nz = random.randint(2, min(4, r))
                cols = random.sample(range(r), num_nz)
                for c in cols:
                    V[row][c] = random.choice([-2, -1, 1, 2])
            
            for row in range(pn):
                num_nz = random.randint(2, min(4, r))
                cols = random.sample(range(r), num_nz)
                for c in cols:
                    W[row][c] = random.choice([-2, -1, 1, 2])
            
            U = _try_rank_fixed_VW(V, W, r)
            if U is not None:
                best_result = {"U": U, "V": V, "W": W}
                best_rank = r
                return True
        
        return False

    # Try ranks from lower to higher
    # Start with structured search for higher ranks (more likely to succeed)
    for r in range(90, 105):
        if time.time() > deadline:
            break
        if try_rank_structured(r, num_trials=30):
            break
    
    # Then try unstructured search
    for r in range(70, 105):
        if time.time() > deadline:
            break
        if try_rank(r, num_trials=50):
            break
    
    # If we found something better, try even lower ranks
    if best_rank < 105:
        for r in range(40, best_rank):
            if time.time() > deadline:
                break
            if try_rank(r, num_trials=30):
                break
    
    return best_result
