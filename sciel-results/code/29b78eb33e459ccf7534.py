import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 600
DEADLINE = 142
NFFT = 2048

def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    def conv(h):
        H = np.fft.rfft(h, n=NFFT)
        C = np.fft.irfft(H * H, n=NFFT)
        return C[:2 * N - 1]

    def exact_metric(h):
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        hn = h / s
        return np.max(conv(hn))

    def softmax_max(c, temp):
        c = c / temp
        c = c - np.max(c)
        return temp * (np.log(np.sum(np.exp(c))))

    # ---- Gradient-based optimization with smooth max surrogate ----
    def objective_and_grad(x, temp):
        h = np.maximum(x, 0.0)
        s = np.sum(h)
        if s < 1e-15:
            return 1e10, np.zeros(N)
        hn = h / s
        c = conv(hn)
        
        c_shifted = c / temp
        c_shifted = c_shifted - np.max(c_shifted)
        weights = np.exp(c_shifted)
        w_sum = np.sum(weights)
        w = weights / w_sum
        f = temp * np.log(w_sum)
        
        # Gradient computation via FFT cross-correlation
        W_full = np.zeros(NFFT)
        W_full[:2*N-1] = w
        H_full = np.zeros(NFFT)
        H_full[:N] = h
        
        FW = np.fft.rfft(W_full, n=NFFT)
        FH = np.fft.rfft(H_full, n=NFFT)
        
        corr1 = np.fft.irfft(FW * np.conj(FH), n=NFFT)[:2*N-1]
        corr2 = np.fft.irfft(FH * np.conj(FW), n=NFFT)[:2*N-1]
        
        grad_h = (corr1[:N] + corr2[:N]) / s
        mask = x > 0
        grad = np.where(mask, grad_h, 0.0)
        return f, grad

    def objective_only(x, temp):
        return objective_and_grad(x, temp)[0]

    best_val = np.inf
    best_h = np.ones(N)

    # ---- Candidate initialization ----
    candidates = []
    candidates.append(np.ones(N))
    x = np.linspace(0, np.pi, N)
    candidates.append(0.5 * (1 + np.cos(x)))
    n = np.arange(N)
    center = N / 2 - 0.5
    for w in [0.2, 0.3, 0.4]:
        sh = np.sinc((n - center) / (N * w))
        candidates.append(np.clip(sh, 0, None))
    xg = np.linspace(-3, 3, N)
    candidates.append(np.exp(-0.5 * xg**2))
    blackman = 0.42 - 0.5 * np.cos(2*np.pi*n/(N-1)) + 0.08*np.cos(4*np.pi*n/(N-1))
    candidates.append(np.clip(blackman, 0, None))
    hamming = 0.54 - 0.46*np.cos(2*np.pi*n/(N-1))
    candidates.append(np.clip(hamming, 0, None))
    hanning = 0.5 - 0.5*np.cos(2*np.pi*n/(N-1))
    candidates.append(np.clip(hanning, 0, None))
    flat_top = (0.21557895 + 0.41663158*np.cos(2*np.pi*n/(N-1)) +
                0.277263158*np.cos(4*np.pi*n/(N-1)) + 0.083578947*np.cos(6*np.pi*n/(N-1)) +
                0.006947368*np.cos(8*np.pi*n/(N-1)))
    candidates.append(np.clip(flat_top, 0, None))
    for a, b in [(0.5, 0.5), (0.3, 0.7), (0.7, 0.3), (0.2, 0.8), (0.8, 0.2)]:
        tp = np.ones(N)
        tp[N//4:3*N//4] = a
        tp[:N//4] = b
        tp[3*N//4:] = b
        candidates.append(tp)
    candidates.append(np.sin(np.pi * n / (N - 1)))
    for e in [0.1, 0.2]:
        pp = np.zeros(N)
        pp[int(e*N):N-int(e*N)] = 1.0
        candidates.append(pp)

    for h in candidates:
        if np.sum(h) < 1e-10:
            continue
        v = exact_metric(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    # ---- Low-rank spline parameterization (rank-3 cosine basis) ----
    # h_i = sum_k a_k * phi_k(i) where phi are cosine modes
    # This allows non-Gaussian, non-smooth shapes
    RANK = 3
    n_params = 2 * RANK  # coefficients and scales

    # Precompute basis functions
    basis = np.zeros((RANK, N))
    for k in range(RANK):
        freq = (k + 1)
        basis[k] = np.cos(np.pi * freq * (n / (N - 1)))
    
    # Add a constant term
    basis_const = np.ones(N)

    def lowrank_to_h(params):
        """Convert low-rank parameters to step heights."""
        # params: [a_0, a_1, a_2, s_0, s_1, s_2]
        a = params[:RANK]
        s = params[RANK:]
        
        h = np.zeros(N)
        for k in range(RANK):
            # Scale the basis function
            scale = max(s[k], 0.1)
            h += a[k] * basis[k] * scale
        
        # Add constant component
        h += 0.5  # baseline
        
        h = np.maximum(h, 0)
        s_total = np.sum(h)
        if s_total < 1e-15:
            return np.ones(N) / N
        return h / s_total

    def lowrank_objective(params):
        if elapsed() > DEADLINE:
            return 1e10
        h = lowrank_to_h(params)
        return exact_metric(h)

    # Multi-start differential evolution on low-rank parameters
    de_configs = [
        (30, 8, 42),
        (40, 10, 123),
    ]
    
    for maxiter, popsize, seed in de_configs:
        if elapsed() > DEADLINE - 50:
            break
        
        bounds = []
        for k in range(RANK):
            bounds.append((-2.0, 2.0))  # amplitude
        for k in range(RANK):
            bounds.append((0.1, 3.0))   # scale
        
        try:
            de_result = differential_evolution(
                lowrank_objective,
                bounds,
                maxiter=maxiter,
                popsize=popsize,
                tol=1e-5,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=seed,
                polish=True
            )
            
            if de_result.x is not None:
                h_de = lowrank_to_h(de_result.x)
                v_de = exact_metric(h_de)
                if v_de < best_val:
                    best_val = v_de
                    best_h = h_de.copy()
        except Exception:
            pass

    # ---- Smooth gradient-based optimization with annealing ----
    x0 = best_h.copy()
    x0 = x0 / np.sum(x0)

    temp_schedule = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    iter_schedule = [100, 200, 300, 400, 500, 600, 800, 1000]
    
    for idx, (temp, maxiter) in enumerate(zip(temp_schedule, iter_schedule)):
        if elapsed() > DEADLINE - 20:
            break
        try:
            res = minimize(
                lambda x: objective_only(x, temp),
                x0,
                jac=lambda x: objective_and_grad(x, temp)[1],
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-18, 'gtol': 1e-10, 'maxfun': maxiter * 3},
            )
            h_opt = np.maximum(res.x, 0.0)
            if np.sum(h_opt) > 1e-10:
                v = exact_metric(h_opt)
                if v < best_val:
                    best_val = v
                    best_h = h_opt.copy()
                    x0 = h_opt / np.sum(h_opt)
        except Exception:
            pass

    # ---- Coordinate descent polish ----
    current_h = best_h.copy()
    current_val = best_val
    np.random.seed(7)
    no_improve = 0
    max_no_improve = 8000
    mean_h = np.mean(current_h)
    
    step_factors = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
    
    while elapsed() < DEADLINE - 10 and no_improve < max_no_improve:
        idx = np.random.randint(N)
        orig = current_h[idx]
        step_sizes = step_factors * mean_h
        cand_vals = np.concatenate([orig + step_sizes, orig - step_sizes, [0.0, orig]])
        cand_vals = np.array([max(0.0, v) for v in cand_vals])

        best_local_val = current_val
        best_local_v = orig
        for cv in cand_vals:
            if abs(cv - orig) < 1e-12:
                continue
            current_h[idx] = cv
            if np.sum(current_h) < 1e-15:
                current_h[idx] = orig
                continue
            v = exact_metric(current_h)
            if v < best_local_val:
                best_local_val = v
                best_local_v = cv

        if best_local_val < current_val - 1e-14:
            current_h[idx] = best_local_v
            current_val = best_local_val
            no_improve = 0
            mean_h = np.mean(current_h)
            if current_val < best_val:
                best_val = current_val
                best_h = current_h.copy()
        else:
            current_h[idx] = orig
            no_improve += 1

        if no_improve % 150 == 0:
            strength = 0.05 + 0.1 * (no_improve // 150) % 3
            pert = current_h.copy()
            pert += strength * np.random.randn(N) * mean_h
            pert = np.clip(pert, 0, None)
            if pert.sum() > 0:
                pv = exact_metric(pert)
                if pv < current_val:
                    current_h = pert
                    current_val = pv
                    no_improve = 0
                    mean_h = np.mean(current_h)
                    if current_val < best_val:
                        best_val = current_val
                        best_h = current_h.copy()

    # ---- Two-coordinate fine tune ----
    if elapsed() < DEADLINE - 5:
        step_frac = 0.002
        mean_best = np.mean(best_h)
        for _ in range(5000):
            if elapsed() > DEADLINE - 2:
                break
            i, j = np.random.choice(N, 2, replace=False)
            delta = step_frac * np.random.uniform(-1, 1) * mean_best
            new_h = best_h.copy()
            new_h[i] += delta
            new_h[j] -= delta
            if new_h[i] >= 0 and new_h[j] >= 0 and np.sum(new_h) > 1e-15:
                v = exact_metric(new_h)
                if v < best_val:
                    best_val = v
                    best_h = new_h
                    mean_best = np.mean(best_h)

    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)

    # Final micro-polish
    if elapsed() < DEADLINE - 1:
        top_idx = np.argsort(best_h)[-50:]
        for idx in top_idx:
            if elapsed() > DEADLINE - 0.5:
                break
            orig = best_h[idx]
            for factor in [0.01, -0.01, 0.005, -0.005]:
                new_v = orig + factor * np.mean(best_h)
                if new_v < 0:
                    continue
                best_h[idx] = new_v
                v = exact_metric(best_h)
                if v < best_val:
                    best_val = v
                else:
                    best_h[idx] = orig
                    break

    return best_h.tolist()
