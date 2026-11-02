import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Target tensor T[i, k, j, j'] = 1 if j == j' else 0
    # i in 0..n-1, k in 0..p-1, j in 0..m-1, j' in 0..m-1
    T = np.zeros((n, p, m, m))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                T[i, k, j, j] = 1.0

    def compute_residuals(x, R):
        # x is flattened U, V, W
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        U = x[:n*m*R].reshape(n, m, R)
        V = x[n*m*R : n*m*R + m*p*R].reshape(m, p, R)
        W = x[n*m*R + m*p*R :].reshape(p, n, R) # Note: W is indexed by k, i in task? 
        # Task says W shape (p*n, rank). Index k*n+i.
        # So W[k, i, r].
        # My reshape: W is (p, n, R). W[k, i, r].
        
        # Compute M[i, k, j, j'] = sum_r W[k, i, r] U[i, j, r] V[j, k, r]
        # Wait, V index is j*p + k. So V[j, k, r].
        # U index is i*m + j. So U[i, j, r].
        # W index is k*n + i. So W[k, i, r].
        
        # einsum: 'kir,ijr,jkr->ikjj'
        # W is (p, n, R) -> k, i, r
        # U is (n, m, R) -> i, j, r
        # V is (m, p, R) -> j, k, r
        # Result: i, k, j, j'
        
        M = np.einsum('kir,ijr,jkr->ikjj', W, U, V)
        residuals = (M - T).flatten()
        return residuals

    def check_solution(U, V, W, R):
        # U, V, W are lists of lists or arrays with integer/half-integer values
        # Convert to numpy arrays
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # Check if entries are close to integers or half-integers
        # The task says "integer/half-integer entries".
        # We will assume we rounded to these.
        
        # Verify the tensor equation
        # M[i, k, j, j'] = sum_r W[k*n+i, r] U[i*m+j, r] V[j*p+k, r]
        # We can use the vectorized check here too, but with exact arithmetic?
        # Numpy floats might have precision issues.
        # Since entries are small integers/half-integers, we can multiply by 2 to make them integers.
        
        U_int = np.round(U_arr * 2).astype(int)
        V_int = np.round(V_arr * 2).astype(int)
        W_int = np.round(W_arr * 2).astype(int)
        
        # M[i, k, j, j'] = sum_r (W_int[k*n+i, r] * U_int[i*m+j, r] * V_int[j*p+k, r]) / 8
        # Target is delta_{j, j'}.
        # So sum_r W_int * U_int * V_int should be 8 * delta_{j, j'}
        
        M_int = np.zeros((n, p, m, m), dtype=int)
        for r in range(R):
            # W_col = W_int[:, r] shape (p*n, ) -> reshape (p, n)
            # U_col = U_int[:, r] shape (n*m, ) -> reshape (n, m)
            # V_col = V_int[:, r] shape (m*p, ) -> reshape (m, p)
            
            w_vec = W_int[:, r].reshape(p, n)
            u_vec = U_int[:, r].reshape(n, m)
            v_vec = V_int[:, r].reshape(m, p)
            
            # M[i, k, j, j'] += w_vec[k, i] * u_vec[i, j] * v_vec[j, k]
            # Use einsum
            # w_vec: k, i
            # u_vec: i, j
            # v_vec: j, k
            # Result: i, k, j, j'? No.
            # We want M[i, k, j, j'].
            # Term: w[k, i] * u[i, j] * v[j', k]
            # So we need v[j', k].
            # v_vec is (m, p) -> j, k.
            # So v[j', k] is v_vec[j', k].
            
            # M[i, k, j, j'] += w[k, i] * u[i, j] * v[j', k]
            # einsum: 'ki,ij,j'k->ikjj'
            # Note: j' is the row index of v.
            
            term = np.einsum('ki,ij,jk->ikjj', w_vec, u_vec, v_vec) 
            # Wait, v is (j, k). So v[j, k].
            # The index in V is j*p + k.
            # In the equation, V is indexed by j' (row of B) and k (col of B).
            # So V[j', k].
            # My v_vec reshape is (m, p) which is (j, k).
            # So v_vec[j', k] is correct.
            
            M_int += term
        
        # Check if M_int == 8 * T_int
        T_int = np.zeros((n, p, m, m), dtype=int)
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    T_int[i, k, j, j] = 8
        
        if np.array_equal(M_int, T_int):
            return True
        return False

    def try_rank(R):
        nonlocal best_rank, best_U, best_V, best_W
        if R >= best_rank:
            return
        
        # Try a few random starts
        for attempt in range(100):
            if time.time() > deadline:
                return
            
            # Random start with small integers
            # Bias towards sparsity
            U0 = np.random.choice([-1, 0, 1], size=(n*m, R))
            V0 = np.random.choice([-1, 0, 1], size=(m*p, R))
            W0 = np.random.choice([-1, 0, 1], size=(p*n, R))
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()]).astype(float)
            
            # Optimize
            try:
                res = opt.least_squares(compute_residuals, x0, args=(R,), method='lm', max_nfev=1000)
            except Exception:
                continue
            
            x_opt = res.x
            U = x_opt[:n*m*R].reshape(n*m, R)
            V = x_opt[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
            W = x_opt[n*m*R + m*p*R :].reshape(p*n, R)
            
            # Try rounding to integers and half-integers
            for scale in [1, 2]:
                U_int = np.round(U * scale) / scale
                V_int = np.round(V * scale) / scale
                W_int = np.round(W * scale) / scale
                
                if check_solution(U_int, V_int, W_int, R):
                    best_rank = R
                    best_U = U_int.tolist()
                    best_V = V_int.tolist()
                    best_W = W_int.tolist()
                    return

    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    # Initialize best with schoolbook
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][column] = 1
                best_V[j * p + k][column] = 1
                best_W[k * n + i][column] = 1
                column += 1

    # Try ranks lower than schoolbook
    for R in range(14, 18):
        try_rank(R)
        if best_rank <= R:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
