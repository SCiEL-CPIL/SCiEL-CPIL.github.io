import numpy as np
import scipy.optimize as optimize
import time

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    timeout = 580  # Safety margin

    N = 8
    # Variables: X_1, ..., X_7. X_0 is fixed at 0.
    # This removes the translational symmetry.
    
    def get_X(params):
        X = np.zeros(N)
        X[1:] = params
        return X

    def union_length(X, c):
        # Intervals: [X_i + c*i/N, X_i + c*i/N + 1/N]
        starts = X + c * np.arange(N) / N
        ends = starts + 1.0 / N
        
        # Sort by start
        order = np.argsort(starts)
        s = starts[order]
        e = ends[order]
        
        # Merge
        total_len = 0.0
        cur_s = s[0]
        cur_e = e[0]
        for k in range(1, N):
            if s[k] <= cur_e:
                if e[k] > cur_e:
                    cur_e = e[k]
            else:
                total_len += cur_e - cur_s
                cur_s = s[k]
                cur_e = e[k]
        total_len += cur_e - cur_s
        return total_len

    def compute_volume(params):
        X = get_X(params)
        
        # Find critical points
        criticals = [0.0, 1.0]
        for i in range(N):
            for j in range(N):
                if i == j: continue
                # a_i = a_j
                # X_i + c i/N = X_j + c j/N
                # c (i-j)/N = X_j - X_i
                denom = (i - j) / N
                if abs(denom) > 1e-12:
                    c = (X[j] - X[i]) / denom
                    if 0 <= c <= 1:
                        criticals.append(c)
                # a_i = b_j
                # X_i + c i/N = X_j + c j/N + 1/N
                # c (i-j)/N = X_j - X_i + 1/N
                c = (X[j] - X[i] + 1.0/N) / denom
                if 0 <= c <= 1:
                    criticals.append(c)
                # b_i = a_j
                # X_i + c i/N + 1/N = X_j + c j/N
                # c (i-j)/N = X_j - X_i - 1/N
                c = (X[j] - X[i] - 1.0/N) / denom
                if 0 <= c <= 1:
                    criticals.append(c)
                # b_i = b_j
                # Same as a_i = a_j for c, but offset is 0? No.
                # X_i + c i/N + 1/N = X_j + c j/N + 1/N
                # c (i-j)/N = X_j - X_i
                # This is the same c as a_i = a_j.
                
        criticals = np.unique(np.array(criticals))
        criticals = criticals[(criticals >= 0) & (criticals <= 1)]
        
        total_vol = 0.0
        for k in range(len(criticals) - 1):
            c0 = criticals[k]
            c1 = criticals[k+1]
            if c1 - c0 < 1e-12:
                continue
                
            L0 = union_length(X, c0)
            L1 = union_length(X, c1)
            
            # L(c) is linear: L(c) = m c + b
            # L0 = m c0 + b
            # L1 = m c1 + b
            # m = (L1 - L0) / (c1 - c0)
            # b = L0 - m c0
            m = (L1 - L0) / (c1 - c0)
            b = L0 - m * c0
            
            # Integrate (m c + b)^2 from c0 to c1
            # = [ (m c + b)^3 / (3m) ] from c0 to c1
            # If m is close to 0, use L0^2 * (c1-c0)
            if abs(m) < 1e-12:
                total_vol += L0 * L0 * (c1 - c0)
            else:
                val1 = (m * c1 + b) ** 3
                val0 = (m * c0 + b) ** 3
                total_vol += (val1 - val0) / (3.0 * m)
                
        return total_vol

    # Initial guess: X_i = -i/8
    # X_0 = 0, X_i = -i/8 for i=1..7
    params_init = np.array([-i/N for i in range(1, N)])
    
    best_params = params_init.copy()
    best_vol = compute_volume(best_params)
    
    # Optimization
    if time.time() - start_time < timeout - 100:
        try:
            res = optimize.minimize(compute_volume, best_params, method='Powell',
                                    options={'maxiter': 1000, 'xtol': 1e-12, 'ftol': 1e-15})
            if res.fun < best_vol:
                best_vol = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Try Nelder-Mead
    if time.time() - start_time < timeout - 50:
        try:
            res = optimize.minimize(compute_volume, best_params, method='Nelder-Mead',
                                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-14})
            if res.fun < best_vol:
                best_vol = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
            
    # Final Polish
    if time.time() - start_time < timeout - 20:
        try:
            res = optimize.minimize(compute_volume, best_params, method='Powell',
                                    options={'maxiter': 2000, 'xtol': 1e-14, 'ftol': 1e-16})
            if res.fun < best_vol:
                best_vol = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Construct positions
    X_best = get_X(best_params)
    # Optional: Center X_best to keep numbers small?
    # The volume is invariant to translation, but the box size [X0, X1] depends on it.
    # The problem states V is estimated in the box defined by min/max.
    # Wait, the volume V is the *union volume*.
    # The Monte Carlo estimate V = fraction * BoxVolume.
    # BoxVolume = (X1 - X0) * (Y1 - Y0).
    # If we translate X, BoxVolume changes?
    # X0 = min x_k, X1 = max (x_k + (i+1)/8).
    # If we add constant C to all x_k:
    # min x_k becomes min x_k + C.
    # max (x_k + (i+1)/8) becomes max (x_k + (i+1)/8) + C.
    # So X1 - X0 is invariant.
    # Similarly for Y.
    # So translation does not affect the score.
    # However, numerical stability might be better if centered.
    X_best = X_best - np.mean(X_best)
    
    positions = np.zeros((N * N, 2))
    for k in range(N * N):
        i = k // N
        j = k % N
        positions[k, 0] = X_best[i]
        positions[k, 1] = X_best[j]

    return positions
