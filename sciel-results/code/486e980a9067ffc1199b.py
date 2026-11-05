import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    DEADLINE = 145 # 5s margin
    
    best_h = np.ones(N)
    best_val = np.inf
    
    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12: return np.inf
        conv = np.convolve(h, h)
        val = abs(2.0 * N * np.max(conv) / (s * s))
        return val if np.isfinite(val) else np.inf

    # 1. Try SDP
    try:
        import cvxpy as cp
        # Build SDP
        X = cp.Variable((N, N), symmetric=True)
        t = cp.Variable()
        
        constraints = [X >> 0]
        
        # Constraint: sum(h) = 1 => trace(J X) = 1
        J = np.ones((N, N))
        constraints += [cp.trace(J * X) == 1]
        
        # Constraints on autocorrelation
        # c_k = sum_{i=0}^{N-1-k} h_i h_{i+k}
        # For k=0: c_0 = trace(I X)
        # For k>0: c_k = 0.5 * trace(A_k X) where A_k has 1s at (i, i+k) and (i+k, i)
        
        I = np.eye(N)
        constraints += [cp.trace(I * X) <= t]
        
        for k in range(1, N):
            # Create A_k
            # Efficient way: use indices
            # A_k[i, i+k] = 1, A_k[i+k, i] = 1
            # Instead of building full matrix, we can use sparse or just loop
            # For N=400, building 400 matrices of 400x400 is 400*160k = 64M ops, ok.
            # But cvxpy trace(A*X) might be slow.
            # Alternative: express in terms of X entries?
            # cvxpy doesn't allow direct indexing in constraints easily without slicing.
            # X[i, i+k] is an affine expression.
            # sum_{i=0}^{N-1-k} X[i, i+k] <= t
            
            # Constructing the sum
            # Note: X is symmetric, so X[i, i+k] = X[i+k, i]
            # The term c_k is sum_{i=0}^{N-1-k} X[i, i+k]
            # We need sum_{i=0}^{N-1-k} X[i, i+k] <= t
            
            # This creates N-1 constraints.
            # Total variables: 400*400/2 = 80k.
            # This might be heavy for cvxpy.
            
            # Optimization: Only check a subset of k?
            # The max is over all k. We must constrain all.
            
            # Let's try to build it.
            # If it's too slow, we catch exception.
            
            # We'll build the constraint using slicing if possible, or sum of entries.
            # cvxpy supports sum(X[i, i+k] for i in range(...))
            # This creates a long affine expression.
            
            # To avoid timeout in SDP, we might limit the size or skip if N is large.
            # N=400 is medium.
            
            # Let's try.
            c_k_expr = sum(X[i, i+k] for i in range(N - k))
            constraints += [c_k_expr <= t]
            
        objective = cp.Minimize(t)
        prob = cp.Problem(objective, constraints)
        
        # Solve
        # Use SCS or MOSEK if available
        try:
            prob.solve(solver=cp.MOSEK)
        except Exception:
            try:
                prob.solve(solver=cp.SCS)
            except Exception:
                prob.solve()
                
        if X.value is not None:
            # Extract h
            # X should be rank 1
            # Eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(X.value)
            # Take largest eigenvector
            idx = np.argmax(eigvals)
            h_sdp = eigvecs[:, idx]
            # Normalize to sum 1
            s = np.sum(h_sdp)
            if abs(s) > 1e-6:
                h_sdp = h_sdp / s
            else:
                h_sdp = np.ones(N)
                
            val_sdp = evaluate(h_sdp)
            if val_sdp < best_val:
                best_val = val_sdp
                best_h = h_sdp.copy()
                
    except Exception:
        pass

    # 2. Fourier Basis Optimization
    # This is a strong fallback and often finds the global optimum for such symmetric problems.
    
    def fourier_optimize(K):
        # h(t) = a0 + sum_{k=1}^K (a_k cos(2pi k t) + b_k sin(2pi k t))
        # t = i / N
        t = np.arange(N) / N
        # Basis functions
        # 1 constant, K cos, K sin
        basis = np.ones((1, N))
        for k in range(1, K + 1):
            basis = np.vstack([basis, np.cos(2 * np.pi * k * t), np.sin(2 * np.pi * k * t)])
            
        # Optimize coefficients
        from scipy.optimize import minimize
        
        def obj(coeffs):
            h = basis @ coeffs
            return evaluate(h)
            
        # Random starts
        rng = np.random.RandomState(42)
        best_local = np.inf
        best_h_local = None
        
        for _ in range(10):
            coeffs0 = rng.normal(0, 1, basis.shape[0])
            try:
                res = minimize(obj, coeffs0, method='L-BFGS-B', options={'maxiter': 200})
                if res.fun < best_local:
                    best_local = res.fun
                    best_h_local = basis @ res.x
            except:
                pass
                
        return best_h_local, best_local

    # Try different K
    for K in [5, 10, 20, 40]:
        if time.time() - start_time > DEADLINE - 10: break
        h_four, val_four = fourier_optimize(K)
        if h_four is not None and val_four < best_val:
            best_val = val_four
            best_h = h_four.copy()

    # 3. Polish with full space
    if time.time() - start_time < DEADLINE - 5:
        from scipy.optimize import minimize
        
        # Use the best h as start
        h_start = best_h.copy()
        
        # Smooth surrogate for polish
        def smooth_obj_grad(x):
            s = np.sum(x)
            if abs(s) < 1e-10: return 1e10, np.zeros_like(x)
            conv = np.convolve(x, x)
            max_c = np.max(conv)
            shifted = conv - max_c
            beta = 50.0
            scaled = beta * shifted
            # Clip for stability
            scaled = np.clip(scaled, -500, 500)
            exp_scaled = np.exp(scaled)
            sum_exp = np.sum(exp_scaled)
            if sum_exp < 1e-300: return 1e10, np.zeros_like(x)
            
            lse = max_c + np.log(sum_exp) / beta
            weights = exp_scaled / sum_exp
            
            # Grad LSE
            # d_lse/dh[i] = 2 * sum_m weights[i+m] h[m]
            # Using FFT for speed
            fft_size = 1
            while fft_size < 2*N - 1: fft_size *= 2
            H = np.fft.rfft(x, n=fft_size)
            W = np.fft.rfft(weights, n=fft_size)
            # cross correlation
            # grad_lse[i] = 2 * sum_{m=0}^{N-1} weights[i+m] h[m]
            # This is convolution of weights and h?
            # Let's use np.correlate or direct for small N?
            # N=400, direct is 160k ops, fast.
            # grad_lse = 2 * np.array([np.dot(weights[i:i+N], x) for i in range(N)])
            # Vectorized:
            # M[i, m] = weights[i+m]
            idx = np.arange(N)
            M = weights[idx[:, None] + idx[None, :]]
            grad_lse = 2.0 * (M @ x)
            
            f = 2.0 * N * lse / (s * s)
            df_dh = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)
            
            if f >= 0:
                grad = df_dh
            else:
                grad = -df_dh
            return abs(f), grad

        try:
            res = minimize(
                lambda x: smooth_obj_grad(x)[0],
                h_start,
                jac=lambda x: smooth_obj_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-8}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_h = res.x
        except:
            pass

    return best_h
