"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = start_time + 140  # 140s to leave margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    nm, mp, pn = n * m, m * p, p * n  # 6, 18, 12
    N = n * m * p  # 36 equations
    
    def verify_np(U, V, W, r):
        """Vectorized verification using numpy."""
        total_error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                    total_error += (val - 1.0) ** 2
        return total_error < 1e-9, total_error
    
    def build_A_U(V, W, r):
        """Build linear system for U given V, W."""
        A = np.zeros((N, nm * r))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    e = (i * m + j) * p + k
                    for c in range(r):
                        A[e, (i * m + j) * r + c] = V[j * p + k, c] * W[k * n + i, c]
        return A
    
    def build_A_V(U, W, r):
        """Build linear system for V given U, W."""
        A = np.zeros((N, mp * r))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    e = (i * m + j) * p + k
                    for c in range(r):
                        A[e, (j * p + k) * r + c] = U[i * m + j, c] * W[k * n + i, c]
        return A
    
    def build_A_W(U, V, r):
        """Build linear system for W given U, V."""
        A = np.zeros((N, pn * r))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    e = (i * m + j) * p + k
                    for c in range(r):
                        A[e, (k * n + i) * r + c] = U[i * m + j, c] * V[j * p + k, c]
        return A
    
    def solve_factor(A, b):
        """Solve least squares, return solution or None if too slow."""
        try:
            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            return sol
        except Exception:
            return None
    
    def alternating_least_squares(r, U, V, W, max_iters=100, tol=1e-14):
        """Alternating least squares to find a rank-r decomposition."""
        b = np.ones(N)
        prev_err = float('inf')
        for iteration in range(max_iters):
            if time.time() > DEADLINE - 2:
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
            
            valid, error = verify_np(U_new, V_new, W_new, r)
            if valid:
                return U_new, V_new, W_new, error
            if abs(prev_err - error) < tol and error > 1e-10:
                return None
            prev_err = error
            U, V, W = U_new, V_new, W_new
        return None
    
    def try_round_and_verify(U, V, W, r):
        """Try to round to half-integers and verify."""
        for scale in [2, 4, 8, 16]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            valid, error = verify_np(U_r, V_r, W_r, r)
            if valid:
                return U_r.tolist(), V_r.tolist(), W_r.tolist()
        # Also try integers
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        valid, error = verify_np(U_i, V_i, W_i, r)
        if valid:
            return U_i.tolist(), V_i.tolist(), W_i.tolist()
        return None
    
    def make_initialization(r, seed):
        """Create diverse initializations."""
        np.random.seed(seed)
        strategy = seed % 6
        if strategy == 0:
            U = np.random.randn(nm, r) * 0.5
            V = np.random.randn(mp, r) * 0.5
            W = np.random.randn(pn, r) * 0.5
        elif strategy == 1:
            # Sparse U
            U = np.zeros((nm, r))
            for row in range(nm):
                nnz = min(3, r)
                cols = np.random.choice(r, size=nnz, replace=False)
                for c in cols:
                    U[row, c] = np.random.choice([-1.0, 1.0])
            V = np.random.randn(mp, r) * 0.3
            W = np.random.randn(pn, r) * 0.3
        elif strategy == 2:
            # Structured: block diagonal-ish
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            for i in range(n):
                for j in range(m):
                    col = (i * m + j) % r
                    U[i * m + j, col] = 1.0
                    V[j * p + (j % p), col] = 1.0
                    W[(j % p) * n + i, col] = 1.0
            # Add some random noise
            U += np.random.randn(nm, r) * 0.1
            V += np.random.randn(mp, r) * 0.1
            W += np.random.randn(pn, r) * 0.1
        elif strategy == 3:
            # Kronecker-like structure
            U = np.random.randn(nm, r) * 0.3
            V = np.random.randn(mp, r) * 0.3
            W = np.random.randn(pn, r) * 0.3
            # Make some entries zero
            mask = np.random.rand(nm, r) < 0.3
            U[mask] = 0
            mask = np.random.rand(mp, r) < 0.3
            V[mask] = 0
            mask = np.random.rand(pn, r) < 0.3
            W[mask] = 0
        elif strategy == 4:
            # Symmetric-ish initialization
            base = np.random.randn(r, r)
            U = np.kron(np.eye(nm, r)[:min(nm,r),:], np.ones((1,1))) * 0.3
            U = np.random.randn(nm, r) * 0.4
            V = np.random.randn(mp, r) * 0.4
            W = np.random.randn(pn, r) * 0.4
        else:
            # Very sparse
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            for c in range(r):
                # Pick random rows for each factor
                ui = np.random.randint(0, nm)
                vj = np.random.randint(0, mp)
                wk = np.random.randint(0, pn)
                U[ui, c] = np.random.choice([-1.0, 1.0])
                V[vj, c] = np.random.choice([-1.0, 1.0])
                W[wk, c] = np.random.choice([-1.0, 1.0])
            # Ensure each row has at least one non-zero
            for row in range(nm):
                if U[row].sum() == 0:
                    U[row, np.random.randint(0, r)] = 1.0
            for row in range(mp):
                if V[row].sum() == 0:
                    V[row, np.random.randint(0, r)] = 1.0
            for row in range(pn):
                if W[row].sum() == 0:
                    W[row, np.random.randint(0, r)] = 1.0
        return U, V, W
    
    def try_rank(r, num_attempts=8):
        """Try to find a rank-r decomposition with multiple initializations."""
        if time.time() > DEADLINE - 5:
            return None
        
        for attempt in range(num_attempts):
            if time.time() > DEADLINE - 5:
                break
            
            seed = 1000 + r * 100 + attempt * 73
            U, V, W = make_initialization(r, seed)
            
            result = alternating_least_squares(r, U, V, W, max_iters=80)
            if result is not None:
                U_sol, V_sol, W_sol, error = result
                rounded = try_round_and_verify(U_sol, V_sol, W_sol, r)
                if rounded is not None:
                    return rounded
        
        return None
    
    # Try ranks from lower bound upward
    # Lower bound is max(n*m, m*p, p*n) = 18
    for r in range(18, 37):
        if time.time() > DEADLINE - 10:
            break
        
        # More attempts for lower ranks since they're harder
        attempts = 12 if r <= 24 else 8
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
