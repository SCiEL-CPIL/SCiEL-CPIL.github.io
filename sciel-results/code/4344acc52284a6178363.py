import numpy as np
import time

def construct():
    start = time.time()
    deadline = start + 145.0
    N = 24
    k_arr = np.arange(N)

    # Precompute FFT-based evaluation on dense grid
    FFT_SIZE = 8192

    def grs(n):
        a = np.zeros(n)
        a[0] = 1.0
        for i in range(1, n):
            kk = i >> 1
            if i & 1:
                a[i] = -a[kk]
            else:
                a[i] = a[kk]
        return a

    def eval_sup(c, fft_size=FFT_SIZE):
        """Evaluate max |P(e^{i theta})| using FFT-based dense sampling."""
        c_pad = np.zeros(fft_size, dtype=np.complex128)
        c_pad[:N] = c
        F = np.fft.fft(c_pad)
        return float(np.max(np.abs(F)))

    def golden_refine(c, theta_lo, theta_hi, n_iter=30):
        """Golden section search for max |P(e^{i theta})| in [theta_lo, theta_hi]."""
        inv_phi = (1 + 5**0.5) / 2
        phi = 1.0 / inv_phi
        a, b = theta_lo, theta_hi
        c1 = b - phi * (b - a)
        c2 = a + phi * (b - a)

        def f(theta):
            return float(np.abs(np.sum(c * np.exp(1j * theta * k_arr))))

        f1 = f(c1)
        f2 = f(c2)
        for _ in range(n_iter):
            if f1 < f2:
                a = c1
                c1 = c2
                f1 = f2
                c2 = a + phi * (b - a)
                f2 = f(c2)
            else:
                b = c2
                c2 = c1
                f2 = f1
                c1 = b - phi * (b - a)
                f1 = f(c1)
        return max(f1, f2)

    def parabolic_refine(c, theta_lo, theta_mid, theta_hi):
        """Parabolic interpolation for faster peak refinement."""
        x1, x2, x3 = theta_lo, theta_mid, theta_hi
        f1 = float(np.abs(np.sum(c * np.exp(1j * x1 * k_arr))))
        f2 = float(np.abs(np.sum(c * np.exp(1j * x2 * k_arr))))
        f3 = float(np.abs(np.sum(c * np.exp(1j * x3 * k_arr))))
        # Parabolic fit: x = x2 - 0.5 * ((x2-x1)^2*(f2-f3) - (x2-x3)^2*(f2-f1)) / ((x2-x1)*(f2-f3) - (x2-x3)*(f2-f1))
        denom = ((x2 - x1)**2 * (f2 - f3) - (x2 - x3)**2 * (f2 - f1))
        if abs(denom) < 1e-15:
            return x2, f2
        x_peak = x2 - 0.5 * ((x2 - x1)**2 * (f2 - f3) - (x2 - x3)**2 * (f2 - f1)) / denom
        # Clamp to [x1, x3]
        x_peak = max(x1, min(x3, x_peak))
        f_peak = float(np.abs(np.sum(c * np.exp(1j * x_peak * k_arr))))
        return x_peak, f_peak

    def find_peak_precise(c):
        """Find the true sup-norm peak using dense FFT grid + parabolic/golden refinement."""
        c_pad = np.zeros(FFT_SIZE, dtype=np.complex128)
        c_pad[:N] = c
        F = np.fft.fft(c_pad)
        abs_F = np.abs(F)
        peak_idx = int(np.argmax(abs_F))

        dtheta = 2 * np.pi / FFT_SIZE
        window = 5
        best_local = 0
        best_theta = 2 * np.pi * peak_idx / FFT_SIZE
        for di in range(-window, window + 1):
            idx = (peak_idx + di) % FFT_SIZE
            if abs_F[idx] > best_local:
                best_local = abs_F[idx]
                best_theta = 2 * np.pi * idx / FFT_SIZE

        lo = max(0, best_theta - dtheta)
        hi = min(2 * np.pi, best_theta + dtheta)
        # Use parabolic refinement first, then golden for final precision
        theta_p, val_p = parabolic_refine(c, lo, best_theta, hi)
        # Narrow window around parabolic result for golden section
        eps = dtheta * 0.5
        lo2 = max(0, theta_p - eps)
        hi2 = min(2 * np.pi, theta_p + eps)
        val_g = golden_refine(c, lo2, hi2, 20)
        return max(val_p, val_g)

    def local_search(start_c, time_limit):
        """Perform local search starting from start_c until time_limit."""
        best_c = start_c.copy()
        cur_val = find_peak_precise(best_c)
        improved = True
        passes = 0
        # Single bit flips
        while improved and passes < 300 and time.time() < time_limit:
            improved = False
            passes += 1
            for j in range(N):
                if time.time() > time_limit:
                    break
                trial = best_c.copy()
                trial[j] = -trial[j]
                quick = eval_sup(trial)
                if quick < cur_val * 1.002:
                    pv = find_peak_precise(trial)
                    if pv < cur_val - 1e-12:
                        best_c = trial
                        cur_val = pv
                        improved = True
        # Double bit flips
        improved = True
        passes = 0
        while improved and passes < 50 and time.time() < time_limit:
            improved = False
            passes += 1
            for j in range(N):
                for l in range(j + 1, N):
                    if time.time() > time_limit:
                        break
                    trial = best_c.copy()
                    trial[j] = -trial[j]
                    trial[l] = -trial[l]
                    quick = eval_sup(trial)
                    if quick < cur_val * 1.002:
                        pv = find_peak_precise(trial)
                        if pv < cur_val - 1e-12:
                            best_c = trial
                            cur_val = pv
                            improved = True
        return best_c, cur_val

    def simulated_annealing(start_c, time_limit):
        """Simulated annealing to escape local minima."""
        cur_c = start_c.copy()
        cur_val = find_peak_precise(cur_c)
        best_c = cur_c.copy()
        best_val = cur_val

        T0 = 0.01
        Tend = 1e-6
        n_steps = 5000
        rng = np.random.RandomState(12345)

        for step in range(n_steps):
            if time.time() > time_limit:
                break
            T = T0 * (Tend / T0) ** (step / n_steps)

            # Random single bit flip
            j = rng.randint(N)
            trial = cur_c.copy()
            trial[j] = -trial[j]
            trial_val = find_peak_precise(trial)

            delta = trial_val - cur_val
            if delta < 0 or rng.random() < np.exp(-delta / T):
                cur_c = trial
                cur_val = trial_val
                if cur_val < best_val - 1e-12:
                    best_val = cur_val
                    best_c = cur_c.copy()

        # Final local search polish
        if time.time() < time_limit - 5.0:
            best_c, best_val = local_search(best_c, time_limit)

        return best_c, best_val

    # Generate GRS
    g = grs(N)
    gr = g[::-1].copy()

    best_overall_c = None
    best_overall_val = float('inf')

    # More diverse starting candidates
    rng = np.random.RandomState(42)
    starts = [g, gr]
    for _ in range(8):
        starts.append(rng.choice([-1.0, 1.0], size=N))
    # Add some structured starts: alternating, blocks
    starts.append(np.tile([1.0, -1.0], N // 2))
    starts.append(np.tile([1.0, 1.0, -1.0, -1.0], N // 4))
    starts.append(np.concatenate([np.ones(12), -np.ones(12)]))

    # Allocate time budget: ~145s total
    # Use simulated annealing for some starts, local search for others
    n_starts = len(starts)
    time_per_start = (deadline - start - 15.0) / n_starts

    for i, s in enumerate(starts):
        if time.time() > deadline - 10.0:
            break
        t_limit = min(deadline - 10.0, time.time() + time_per_start)

        # Alternate between SA and local search for diversity
        if i % 3 == 0:
            c_res, v_res = simulated_annealing(s, t_limit)
        else:
            c_res, v_res = local_search(s, t_limit)

        if v_res < best_overall_val:
            best_overall_val = v_res
            best_overall_c = c_res

    # Final polish with remaining time
    if best_overall_c is not None and time.time() < deadline - 5.0:
        best_overall_c, best_overall_val = local_search(best_overall_c, deadline - 5.0)

    if best_overall_c is None:
        best_overall_c = g
        best_overall_val = find_peak_precise(best_overall_c)

    out = [1 if v >= 0 else -1 for v in best_overall_c]
    return out
