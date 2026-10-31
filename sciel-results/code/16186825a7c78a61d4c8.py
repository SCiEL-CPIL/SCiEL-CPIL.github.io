import numpy as np
import time
from scipy.optimize import minimize

N = 600
DEADLINE = 140  # safety margin before 150s hard limit

def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    # ---- FFT-based autoconvolution of a (sum-normalized) height vector ----
    nfft = 2048  # next power of 2 >= 2N-1 = 1199

    def conv(h):
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
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

    def smooth_metric(h, temp):
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        hn = h / s
        return softmax_max(conv(hn), temp)

    # ---- Evaluate structured candidates ----
    best_val = np.inf
    best_h = np.ones(N)

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
    # Two-plateau variants
    for a, b in [(0.5, 0.5), (0.3, 0.7), (0.7, 0.3), (0.2, 0.8), (0.8, 0.2)]:
        tp = np.ones(N)
        tp[N//4:3*N//4] = a
        tp[:N//4] = b
        tp[3*N//4:] = b
        candidates.append(tp)
    # Sine bump
    candidates.append(np.sin(np.pi * n / (N - 1)))
    # Plateau in middle, zero edges
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

    # ---- Legendre polynomial basis representation ----
    # h[n] = sum_{k=0}^{K} c_k * P_k(x_n), where x_n = linspace(-1, 1, N)
    # This gives a low-dimensional parameterization that can capture smooth shapes.
    
    K_legendre = 12  # number of Legendre polynomials (0 to K)
    x_leg = np.linspace(-1, 1, N)
    
    # Build Legendre basis matrix
    # P_k(x) computed via scipy or numpy
    from numpy.polynomial import legendre as npleg
    leg_basis = np.zeros((N, K_legendre + 1))
    for k in range(K_legendre + 1):
        leg_basis[:, k] = npleg.legval(x_leg, np.eye(K_legendre + 1)[:, k])
    
    def leg_to_heights(c):
        """Convert Legendre coefficients to height vector."""
        return leg_basis @ c
    
    def heights_to_leg(h):
        """Convert height vector to Legendre coefficients via least squares."""
        # Solve min ||leg_basis @ c - h||^2
        c, _, _, _ = np.linalg.lstsq(leg_basis, h, rcond=None)
        return c
    
    # ---- Optimize in Legendre coefficient space with soft-max objective ----
    def leg_objective_and_grad(c, temp):
        """Objective: smooth max of autoconvolution of normalized heights from Legendre coeffs.
        Gradient computed via chain rule."""
        h = leg_to_heights(c)
        h = np.maximum(h, 0.0)  # enforce non-negativity
        s = np.sum(h)
        if s < 1e-15:
            return 1e10, np.zeros(K_legendre + 1)
        hn = h / s
        c_conv = conv(hn)
        
        # Softmax max
        c_norm = c_conv / temp
        c_norm = c_norm - np.max(c_norm)
        w = np.exp(c_norm)
        w = w / np.sum(w)
        f = temp * (np.max(c_norm) + np.log(np.sum(w)))  # = temp * log(sum(exp(c_conv/temp)))
        
        # Gradient wrt h: (2/s) * (w conv h)
        W = np.fft.rfft(w, n=nfft)
        H = np.fft.rfft(h, n=nfft)
        g_h = np.fft.irfft(W * H, n=nfft)[:2*N-1]
        g_h = (2.0 / s) * g_h
        
        # Gradient wrt c: g_h @ leg_basis (since h = leg_basis @ c, dh/dc = leg_basis)
        # But we have the max(0) projection, so only where h > 0
        mask = h > 0
        g_h_proj = np.where(mask, g_h, 0.0)
        g_c = leg_basis.T @ g_h_proj
        
        return f, g_c

    def leg_objective_only(c, temp):
        return leg_objective_and_grad(c, temp)[0]

    # Start from best candidate
    c0 = heights_to_leg(best_h)
    
    for temp in [0.05, 0.02, 0.01, 0.005, 0.002]:
        if elapsed() > DEADLINE - 25:
            break
        try:
            res = minimize(
                lambda c: leg_objective_only(c, temp),
                c0,
                jac=lambda c: leg_objective_and_grad(c, temp)[1],
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-8, 'maxfun': 800},
            )
            c_opt = res.x
            h_opt = np.maximum(leg_to_heights(c_opt), 0.0)
            if np.sum(h_opt) > 1e-10:
                v = exact_metric(h_opt)
                if v < best_val:
                    best_val = v
                    best_h = h_opt.copy()
                    c0 = c_opt.copy()
        except Exception:
            pass

    # ---- Direct optimization in height space with smooth objective ----
    def objective_and_grad(x, temp):
        h = np.maximum(x, 0.0)
        s = np.sum(h)
        if s < 1e-15:
            return 1e10, np.zeros(N)
        hn = h / s
        c = conv(hn)
        c_norm = c / temp
        c_norm = c_norm - np.max(c_norm)
        w = np.exp(c_norm)
        w = w / np.sum(w)
        f = temp * (np.max(c_norm) + np.log(np.sum(w)))
        
        W = np.fft.rfft(w, n=nfft)
        H = np.fft.rfft(h, n=nfft)
        g = np.fft.irfft(W * H, n=nfft)[:2*N-1]
        grad = (2.0 / s) * g
        mask = x > 0
        grad = np.where(mask, grad, 0.0)
        return f, grad

    def objective_only(x, temp):
        return objective_and_grad(x, temp)[0]

    x0 = best_h.copy()
    x0 = x0 / np.sum(x0)

    for temp in [0.05, 0.02, 0.01, 0.005, 0.002]:
        if elapsed() > DEADLINE - 20:
            break
        try:
            res = minimize(
                lambda x: objective_only(x, temp),
                x0,
                jac=lambda x: objective_and_grad(x, temp)[1],
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-8, 'maxfun': 600},
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

    # ---- Final exact coordinate-descent polish ----
    current_h = best_h.copy()
    current_val = best_val
    np.random.seed(7)
    no_improve = 0
    max_no_improve = 4000
    mean_h = np.mean(current_h)

    while elapsed() < DEADLINE - 8 and no_improve < max_no_improve:
        idx = np.random.randint(N)
        orig = current_h[idx]
        step_factors = np.array([0.05, 0.1, 0.25, 0.5, 1.0])
        step_sizes = step_factors * mean_h
        cand_vals = np.concatenate([orig + step_sizes, orig - step_sizes, [0.0, orig]])
        cand_vals = np.array([max(0.0, v) for v in cand_vals])

        best_local_val = current_val
        best_local_v = orig
        for cv in cand_vals:
            if abs(cv - orig) < 1e-10:
                continue
            current_h[idx] = cv
            if np.sum(current_h) < 1e-15:
                current_h[idx] = orig
                continue
            v = exact_metric(current_h)
            if v < best_local_val:
                best_local_val = v
                best_local_v = cv

        if best_local_val < current_val - 1e-12:
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

        # periodic global perturbation
        if no_improve % 200 == 0:
            pert = current_h.copy()
            pert += 0.1 * np.random.randn(N) * mean_h
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
    if elapsed() < DEADLINE - 3:
        step_frac = 0.003
        for _ in range(1500):
            if elapsed() > DEADLINE - 1:
                break
            i, j = np.random.choice(N, 2, replace=False)
            delta = step_frac * np.random.uniform(-1, 1) * np.mean(best_h)
            new_h = best_h.copy()
            new_h[i] += delta
            new_h[j] -= delta
            if new_h[i] >= 0 and new_h[j] >= 0 and np.sum(new_h) > 1e-15:
                v = exact_metric(new_h)
                if v < best_val:
                    best_val = v
                    best_h = new_h

    # Ensure nonnegative and valid
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)

    return best_h.tolist()
