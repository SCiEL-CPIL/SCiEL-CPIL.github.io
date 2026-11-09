import numpy as np
import time
import scipy.optimize as optimize

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    timeout = 620  # safety margin before 630s limit
    
    N = 8

    # Precompute quadrature nodes
    c_vals, w_vals = np.polynomial.legendre.leggauss(64)
    c_vals = 0.5 * (c_vals + 1.0)
    w_vals = 0.5 * w_vals
    c_indices = np.arange(N) / N

    def compute_volume(X, Y):
        """
        Compute the union volume of the 64 tubes.
        X, Y are arrays of shape (8,) representing base x and y coordinates for each index.
        Tube (i, j) has base (X[i], Y[j]).
        At height c, x-interval is [X[i] + c*i/N, X[i] + c*i/N + 1/N]
        y-interval is [Y[j] + c*j/N, Y[j] + c*j/N + 1/N]
        """
        total_vol = 0.0
        for idx in range(len(c_vals)):
            c = c_vals[idx]
            w = w_vals[idx]
            
            # X intervals for all i
            x_starts = X + c * c_indices
            x_ends = x_starts + 1.0 / N
            
            # Y intervals for all j
            y_starts = Y + c * c_indices
            y_ends = y_starts + 1.0 / N
            
            # Merge X intervals
            x_order = np.argsort(x_starts)
            xs = x_starts[x_order]
            xe = x_ends[x_order]
            
            x_len = 0.0
            cur_s = xs[0]
            cur_e = xe[0]
            for k in range(1, N):
                if xs[k] <= cur_e:
                    if xe[k] > cur_e:
                        cur_e = xe[k]
                else:
                    x_len += cur_e - cur_s
                    cur_s = xs[k]
                    cur_e = xe[k]
            x_len += cur_e - cur_s
            
            # Merge Y intervals
            y_order = np.argsort(y_starts)
            ys = y_starts[y_order]
            ye = y_ends[y_order]
            
            y_len = 0.0
            cur_s = ys[0]
            cur_e = ye[0]
            for k in range(1, N):
                if ys[k] <= cur_e:
                    if ye[k] > cur_e:
                        cur_e = ye[k]
                else:
                    y_len += cur_e - cur_s
                    cur_s = ys[k]
                    cur_e = ye[k]
            y_len += cur_e - cur_s
            
            total_vol += x_len * y_len * w
        
        return total_vol

    def obj(params):
        """
        params: shape (16,) -> [X_1..X_7, Y_1..Y_7]
        X_0 = 0, Y_0 = 0 fixed.
        """
        X = np.empty(N)
        Y = np.empty(N)
        X[0] = 0.0
        X[1:] = params[:N-1]
        Y[0] = 0.0
        Y[1:] = params[N-1:]
        return compute_volume(X, Y)

    # Initial guess: linear shearing
    # X_i = -i/N, Y_j = -j/N
    X_init_lin = -np.arange(1, N) / N
    Y_init_lin = -np.arange(1, N) / N
    init_params = np.concatenate((X_init_lin, Y_init_lin))
    
    X_best = np.concatenate(([0.0], X_init_lin))
    Y_best = np.concatenate(([0.0], Y_init_lin))
    best_val = obj(init_params)

    rng = np.random.default_rng(42)
    
    # List of initializations to try
    inits = [
        init_params, # Linear
        np.zeros(2*N-2), # Zero
        np.concatenate((rng.uniform(-0.5, 0.5, N-1), rng.uniform(-0.5, 0.5, N-1))),
        np.concatenate((-np.arange(1, N) / (2.0 * N), np.arange(1, N) / (2.0 * N))), # Anti-symmetric
        np.concatenate((np.arange(1, N) / (2.0 * N), -np.arange(1, N) / (2.0 * N))), # Anti-symmetric
    ]
    
    # Add structured inits based on quadratic patterns
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        x_quad = -alpha * (np.arange(1, N) ** 2) / (N ** 2)
        y_quad = -alpha * (np.arange(1, N) ** 2) / (N ** 2)
        inits.append(np.concatenate((x_quad, y_quad)))
        
    # Add sinusoidal patterns (spectral low-frequency modes)
    for freq in [1, 2]:
        for amp in [0.5, 1.0]:
            x_sine = amp * np.sin(2 * np.pi * freq * np.arange(1, N) / N)
            y_sine = amp * np.cos(2 * np.pi * freq * np.arange(1, N) / N)
            inits.append(np.concatenate((x_sine, y_sine)))
            inits.append(np.concatenate((y_sine, x_sine))) # Swapped

    # Add more random starts with wider range
    for _ in range(50):
        inits.append(rng.uniform(-2.0, 2.0, 2*N-2))
    
    # Add inits that try to maximize overlap at specific heights
    for c_target in [0.25, 0.5, 0.75]:
        x_at_c = -c_target * np.arange(1, N) / N
        y_at_c = -c_target * np.arange(1, N) / N
        inits.append(np.concatenate((x_at_c, y_at_c)))
    
    for init in inits:
        if time.time() - start_time > timeout - 60:  # Leave more time for polish
            break
        
        # Powell's method is good for this low-dim non-smooth problem
        try:
            res = optimize.minimize(obj, init, method='Powell',
                                    options={'maxiter': 500, 'xtol': 1e-8, 'ftol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                # Update best X, Y
                X_best = np.empty(N)
                Y_best = np.empty(N)
                X_best[0] = 0.0
                X_best[1:] = res.x[:N-1]
                Y_best[0] = 0.0
                Y_best[1:] = res.x[N-1:]
        except Exception:
            pass

    # Final polish: Nelder-Mead followed by tighter Powell
    if time.time() - start_time < timeout - 15:
        try:
            res_nm = optimize.minimize(obj, np.concatenate((X_best[1:], Y_best[1:])), method='Nelder-Mead',
                                       options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                X_best = np.empty(N)
                Y_best = np.empty(N)
                X_best[0] = 0.0
                X_best[1:] = res_nm.x[:N-1]
                Y_best[0] = 0.0
                Y_best[1:] = res_nm.x[N-1:]
        except Exception:
            pass
    
    # Final high-precision Powell polish
    if time.time() - start_time < timeout - 5:
        try:
            res2 = optimize.minimize(obj, np.concatenate((X_best[1:], Y_best[1:])), method='Powell',
                                     options={'maxiter': 5000, 'xtol': 1e-12, 'ftol': 1e-14})
            if res2.fun < best_val:
                best_val = res2.fun
                X_best = np.empty(N)
                Y_best = np.empty(N)
                X_best[0] = 0.0
                X_best[1:] = res2.x[:N-1]
                Y_best[0] = 0.0
                Y_best[1:] = res2.x[N-1:]
        except Exception:
            pass
            
    # Extra polish if time remains
    if time.time() - start_time < timeout - 1:
        try:
            res3 = optimize.minimize(obj, np.concatenate((X_best[1:], Y_best[1:])), method='Powell',
                                     options={'maxiter': 2000, 'xtol': 1e-14, 'ftol': 1e-16})
            if res3.fun < best_val:
                best_val = res3.fun
                X_best = np.empty(N)
                Y_best = np.empty(N)
                X_best[0] = 0.0
                X_best[1:] = res3.x[:N-1]
                Y_best[0] = 0.0
                Y_best[1:] = res3.x[N-1:]
        except Exception:
            pass

    # Construct positions
    positions = np.zeros((N * N, 2))
    for k in range(N * N):
        i = k // N
        j = k % N
        positions[k, 0] = X_best[i]
        positions[k, 1] = Y_best[j]

    return positions
