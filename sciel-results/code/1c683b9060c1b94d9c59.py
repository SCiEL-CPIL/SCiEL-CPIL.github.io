import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _max_overlap(h):
    """Compute max_k sum_i h[i] * (1 - h[(i-k) % N])."""
    # h is 1D array of length N
    # Compute autocorrelation
    # acf[k] = sum_i h[i] h[(i-k)%N]
    H = np.fft.fft(h)
    acf = np.fft.ifft(H * np.conj(H)).real
    # Overlap O_k = sum h[i] - acf[k] = N/2 - acf[k]
    # Note: sum h[i] is fixed to N/2
    overlaps = N / 2.0 - acf
    return np.max(overlaps)

def _project_to_feasible(h):
    """Project h to 0<=h<=1, sum(h)=N/2."""
    h = h.copy()
    h = np.clip(h, 0, 1)
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-12:
        # Simple iterative projection
        for _ in range(10):
            if diff > 0:
                room = 1.0 - h
                mask = room > 1e-9
                if np.sum(room[mask]) > 1e-9:
                    h[mask] += diff * room[mask] / np.sum(room[mask])
                else:
                    h += diff / N
            else:
                mass = h
                mask = mass > 1e-9
                if np.sum(mass[mask]) > 1e-9:
                    h[mask] += diff * mass[mask] / np.sum(mass[mask])
                else:
                    h += diff / N
            h = np.clip(h, 0, 1)
            diff = N / 2.0 - np.sum(h)
            if abs(diff) < 1e-9:
                break
    return h

def _binary_projection(h):
    """Snap to nearest binary sequence with sum N/2."""
    # Sort indices by h value, set top N/2 to 1, rest to 0
    # To break ties randomly or by index, we can just sort
    idx = np.argsort(-h)
    h_bin = np.zeros(N)
    h_bin[idx[:N//2]] = 1.0
    return h_bin

def _objective_epigraph(x):
    """x = [h, t]. Minimize t."""
    h = x[:N]
    t = x[N]
    return t

def _constraints_epigraph(x):
    """Constraints: t - O_k >= 0 for all k, sum(h) = N/2."""
    h = x[:N]
    t = x[N]
    overlaps = N / 2.0 - np.fft.ifft(np.fft.fft(h) * np.conj(np.fft.fft(h))).real
    # Inequality: t - overlaps >= 0
    ineq = t - overlaps
    # Equality: sum(h) - N/2 = 0
    eq = np.sum(h) - N / 2.0
    return ineq, eq

def _optimize_from_start(h0, time_limit=10.0):
    start_time = time.time()
    h0 = _project_to_feasible(h0)
    # Initial t: max overlap of h0
    t0 = _max_overlap(h0)
    x0 = np.concatenate([h0, [t0]])
    
    bounds = [(0, 1)] * N + [(0, N)]
    
    cons = [
        {'type': 'ineq', 'fun': lambda x: x[N] - (N/2.0 - np.fft.ifft(np.fft.fft(x[:N]) * np.conj(np.fft.fft(x[:N]))).real)},
        {'type': 'eq', 'fun': lambda x: np.sum(x[:N]) - N / 2.0}
    ]
    
    result = minimize(
        _objective_epigraph,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': 100, 'ftol': 1e-9, 'disp': False}
    )
    
    if time.time() - start_time > time_limit:
        return h0, _max_overlap(h0)
        
    h_opt = result.x[:N]
    h_opt = _project_to_feasible(h_opt)
    score = _max_overlap(h_opt)
    
    # Try binary projection
    h_bin = _binary_projection(h_opt)
    score_bin = _max_overlap(h_bin)
    
    if score_bin < score:
        # Try to optimize from the binary point
        h_bin_opt = h_bin
        t_bin = score_bin
        x_bin = np.concatenate([h_bin, [t_bin]])
        result_bin = minimize(
            _objective_epigraph,
            x_bin,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 50, 'ftol': 1e-9, 'disp': False}
        )
        if time.time() - start_time < time_limit:
            h_bin_opt = result_bin.x[:N]
            h_bin_opt = _project_to_feasible(h_bin_opt)
            score_bin_opt = _max_overlap(h_bin_opt)
            if score_bin_opt < score:
                return h_bin_opt, score_bin_opt
                
    return h_opt, score

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Generate starting points
    starts = []
    
    # 1. Uniform
    starts.append(np.full(N, 0.5))
    
    # 2. Random binary
    np.random.seed(42)
    for _ in range(20):
        h = np.zeros(N)
        mask = np.random.random(N) < 0.5
        h[mask] = 1.0
        # Ensure sum is exactly N/2
        # If sum != N/2, adjust
        s = np.sum(h)
        if s != N/2:
            # Flip random bits to fix
            ones = np.where(h == 1)[0]
            zeros = np.where(h == 0)[0]
            if s > N/2:
                for _ in range(int(s - N/2)):
                    if len(ones) == 0: break
                    idx = np.random.choice(ones)
                    h[idx] = 0
                    ones = np.where(h == 1)[0]
                    zeros = np.where(h == 0)[0]
            else:
                for _ in range(int(N/2 - s)):
                    if len(zeros) == 0: break
                    idx = np.random.choice(zeros)
                    h[idx] = 1
                    ones = np.where(h == 1)[0]
                    zeros = np.where(h == 0)[0]
        starts.append(h)
        
    # 3. Sinusoidal
    for freq in [1, 2, 3, 5, 10]:
        x = np.linspace(0, 2*np.pi*freq, N, endpoint=False)
        h = 0.5 + 0.5 * np.sin(x)
        h = _project_to_feasible(h)
        starts.append(h)
        
    # 4. Greedy construction?
    # Let's just use the random ones and uniform.
    
    # Evaluate and optimize
    scored_starts = []
    for h in starts:
        if time.time() > deadline - 20:
            break
        score = _max_overlap(h)
        scored_starts.append((score, h))
        
    scored_starts.sort(key=lambda x: x[0])
    
    # Optimize from best starts
    for score, h0 in scored_starts[:10]:
        if time.time() > deadline - 5:
            break
        h_opt, opt_score = _optimize_from_start(h0, time_limit=10.0)
        if opt_score < best_score:
            best_score = opt_score
            best_h = h_opt.copy()
            
    return best_h
