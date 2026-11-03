"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = start_time + 140
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    nm, mp, pn = n * m, m * p, p * n  # 6, 18, 12
    N = n * m * p  # 36
    
    def verify(U, V, W, r, tol=1e-9):
        """Verify the decomposition exactly using rational arithmetic where possible."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def verify_np(U, V, W, r, tol=1e-9):
        """Vectorized verification using numpy."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(Ua[i*m+j], Va[j*p+k] * Wa[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def build_A_U(V, W, r):
        """Build linear system for U given V, W."""
        A = np.zeros((N, nm * r))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    e = (i * m + j) * p + k
                    for c in range(r):
                        A[e, (i * m + j) * r + c] = V[j * p + k][c] * W[k * n + i][c]
        return A
    
    def build_A_V(U, W, r):
        """Build linear system for V given U, W."""
        A = np.zeros((N, mp * r))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    e = (i * m + j) * p + k
                    for c in range(r):
                        A[e, (j * p + k) * r + c] = U[i * m + j][c] * W[k * n + i][c]
        return A
    
    def build_A_W(U, V, r):
        """Build linear system for W given U, V."""
        A = np.zeros((N, pn * r))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    e = (i * m + j) * p + k
                    for c in range(r):
                        A[e, (k * n + i) * r + c] = U[i * m + j][c] * V[j * p + k][c]
        return A
    
    def solve_factor(A, b):
        """Solve least squares."""
        try:
            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            return sol
        except Exception:
            return None
    
    def alternating_least_squares(r, U, V, W, max_iters=150, tol=1e-14):
        """Alternating least squares to find a rank-r decomposition."""
        b = np.ones(N)
        prev_err = float('inf')
        for iteration in range(max_iters):
            if time.time() > DEADLINE - 3:
                return None
            # Solve for U
            A_U = build_A_U(V, W, r)
            u_flat = solve_factor(A_U, b)
            if u_flat is None:
                return None
            U_new = u_flat.reshape(nm, r)
            
            # Solve for V
            A_V = build_A_V(U_new, W, r)
            v_flat = solve_factor(A_V, b)
            if v_flat is None:
                return None
            V_new = v_flat.reshape(mp, r)
            
            # Solve for W
            A_W = build_A_W(U_new, V_new, r)
            w_flat = solve_factor(A_W, b)
            if w_flat is None:
                return None
            W_new = w_flat.reshape(pn, r)
            
            # Verify
            Ua = U_new
            Va = V_new
            Wa = W_new
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(Ua[i*m+j], Va[j*p+k] * Wa[k*n+i])
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-10:
                return U_new.tolist(), V_new.tolist(), W_new.tolist(), max_err
            if abs(prev_err - max_err) < tol and max_err > 1e-10:
                return None
            prev_err = max_err
            U, V, W = U_new, V_new, W_new
        return None
    
    def try_round_and_verify(U, V, W, r):
        """Try to round to half-integers and verify."""
        for scale in [2, 4, 8, 16, 32]:
            U_r = np.round(np.array(U) * scale) / scale
            V_r = np.round(np.array(V) * scale) / scale
            W_r = np.round(np.array(W) * scale) / scale
            if verify_np(U_r.tolist(), V_r.tolist(), W_r.tolist(), r, tol=1e-8):
                return U_r.tolist(), V_r.tolist(), W_r.tolist()
        # Also try integers
        U_i = np.round(np.array(U))
        V_i = np.round(np.array(V))
        W_i = np.round(np.array(W))
        if verify_np(U_i.tolist(), V_i.tolist(), W_i.tolist(), r, tol=1e-8):
            return U_i.tolist(), V_i.tolist(), W_i.tolist()
        return None
    
    def make_structured_init(r, seed):
        """Create structured initializations based on known rank-20 structure."""
        np.random.seed(seed)
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        
        # For 2x3x6, use the following structure:
        # 20 terms. Group them:
        # Terms 0-5: for each k, one term handling j=0, i=0
        # Terms 6-11: for each k, one term handling j=0, i=1
        # Terms 12-17: for each k, one term handling j=1, i=0
        # Terms 18-19: handle j=2 with shared structure
        
        # Better: use a known construction pattern
        # Let's use random but structured initialization
        strategy = seed % 4
        
        if strategy == 0:
            # Dense random
            U = np.random.randn(nm, r) * 0.5
            V = np.random.randn(mp, r) * 0.3
            W = np.random.randn(pn, r) * 0.3
        elif strategy == 1:
            # Block structure: group by k
            for k in range(p):
                base = k * 3 + 2  # 3 terms per k roughly, but we have 20/6 ~ 3.3
            # Just use random with some zeros
            U = np.random.randn(nm, r) * 0.4
            V = np.random.randn(mp, r) * 0.3
            W = np.random.randn(pn, r) * 0.3
            # Make sparse
            mask = np.random.rand(nm, r) < 0.4
            U[mask] = 0
            mask = np.random.rand(mp, r) < 0.4
            V[mask] = 0
            mask = np.random.rand(pn, r) < 0.4
            W[mask] = 0
        elif strategy == 2:
            # Use the fact that for fixed j, the (i,k) submatrix has rank 2
            # So we can have 2 terms per j, but they need to be coordinated
            for j in range(m):
                for c in range(2):
                    col = j * 2 + c
                    if col < r:
                        U[j, col] = 1.0
                        U[j + n, col] = 1.0  # both i values
                        for k in range(p):
                            V[j * p + k, col] = np.random.randn() * 0.5
                        for i in range(n):
                            W[k * n + i, col] = 1.0 if i == c else 0.0
            # Fill remaining columns randomly
            for c in range(m * 2, r):
                U[:, c] = np.random.randn(nm) * 0.3
                V[:, c] = np.random.randn(mp) * 0.3
                W[:, c] = np.random.randn(pn) * 0.3
        else:
            # Very structured: use Kronecker-like
            U = np.random.randn(nm, r) * 0.3
            V = np.random.randn(mp, r) * 0.3
            W = np.random.randn(pn, r) * 0.3
            # Ensure each row has non-zero
            for row in range(nm):
                if np.all(np.abs(U[row]) < 0.01):
                    U[row, np.random.randint(0, r)] = 1.0
            for row in range(mp):
                if np.all(np.abs(V[row]) < 0.01):
                    V[row, np.random.randint(0, r)] = 1.0
            for row in range(pn):
                if np.all(np.abs(W[row]) < 0.01):
                    W[row, np.random.randint(0, r)] = 1.0
        
        return U, V, W
    
    def try_rank(r, num_attempts=20):
        """Try to find a rank-r decomposition."""
        if time.time() > DEADLINE - 10:
            return None
        
        best_result = None
        best_err = float('inf')
        
        for attempt in range(num_attempts):
            if time.time() > DEADLINE - 5:
                break
            
            seed = 1000 + r * 100 + attempt * 73
            U, V, W = make_structured_init(r, seed)
            
            result = alternating_least_squares(r, U, V, W, max_iters=100)
            if result is not None:
                U_sol, V_sol, W_sol, error = result
                if error < best_err:
                    best_err = error
                    best_result = (U_sol, V_sol, W_sol)
                
                rounded = try_round_and_verify(U_sol, V_sol, W_sol, r)
                if rounded is not None:
                    return rounded
        
        # Try to round the best result
        if best_result is not None and best_err < 1e-6:
            U_sol, V_sol, W_sol = best_result
            rounded = try_round_and_verify(U_sol, V_sol, W_sol, r)
            if rounded is not None:
                return rounded
        
        return None
    
    # Try ranks from known lower bound upward
    # Lower bound for 2x3x6 is 20 (known result)
    # Try 20 first, then 21, 22, ...
    for r in [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]:
        if time.time() > DEADLINE - 15:
            break
        
        attempts = 25 if r <= 22 else 15
        result = try_rank(r, num_attempts=attempts)
        if result is not None:
            U_r, V_r, W_r = result
            return {"U": U_r, "V": V_r, "W": W_r}
    
    # Fallback to schoolbook
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
# EVOLVE-BLOCK-END
