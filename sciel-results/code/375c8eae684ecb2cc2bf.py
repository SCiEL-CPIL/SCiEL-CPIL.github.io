"""Search for a low-autocorrelation signed step function (400 equal intervals).

Combines Iterated Local Search with smooth surrogate optimization (log-sum-exp)
to escape nonsmooth local minima, plus exact best-improvement polishing.
"""
import numpy as np
import time

N = 400


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    tm = np.zeros(n, dtype=np.int8)
    tm[0] = 0
    for i in range(1, n):
        tm[i] = tm[i // 2] if (i % 2 == 0) else 1 - tm[i // 2]
    return 1.0 - 2.0 * tm.astype(np.float64)


def _calc_ac(h):
    """Calculate aperiodic autocorrelation using FFT."""
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
    H = np.fft.fft(h, fft_size)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return ac


def _peak_sidelobe_from_ac(ac):
    """Calculate the peak aperiodic autocorrelation sidelobe."""
    if len(ac) <= 1:
        return 0.0
    return float(np.max(np.abs(ac[1:])))


def _smooth_sidelobe_from_ac(ac, beta=50.0):
    """Smooth surrogate: log-sum-exp of |ac[1:]|^2 scaled by beta."""
    if len(ac) <= 1:
        return 0.0
    vals = np.abs(ac[1:])
    # Use log-sum-exp for numerical stability
    max_val = np.max(vals)
    if max_val < 1e-12:
        return 0.0
    # log(sum(exp(beta * vals))) / beta approximates max for large beta
    # But we want it to approximate the peak, so use:
    # (1/beta) * log(sum(exp(beta * vals)))
    # For large beta this approaches max(vals)
    scaled = beta * vals
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return log_sum / beta


def _flip_delta(h, i, ac_buf):
    """
    Compute the delta to add to ac[1:] when flipping h[i].
    ac_buf is pre-allocated, size N.
    """
    ac_buf[:] = 0.0
    if i + 1 < N:
        ac_buf[1: N - i] = -2.0 * h[i] * h[i + 1: N]
    if i > 0:
        ac_buf[1: i + 1] += -2.0 * h[i] * h[i - 1: -1: -1]
    return ac_buf


def _fast_greedy_polish(h, ac, time_budget, start_time):
    """
    Fast greedy polish using 'first improvement' strategy.
    """
    h = h.copy()
    ac = ac.copy()
    cur_peak = _peak_sidelobe_from_ac(ac)
    best_h = h.copy()
    best_peak = cur_peak
    delta_buf = np.zeros(N, dtype=np.float64)

    while time.time() - start_time < time_budget:
        improved = False
        for i in range(N):
            old_h_i = h[i]
            h[i] = -old_h_i
            _flip_delta(h, i, delta_buf)
            delta_buf *= -1.0
            ac[1:] += delta_buf
            new_peak = _peak_sidelobe_from_ac(ac)
            if new_peak < cur_peak:
                cur_peak = new_peak
                if cur_peak < best_peak:
                    best_peak = cur_peak
                    best_h = h.copy()
                improved = True
                break
            else:
                h[i] = old_h_i
                ac[1:] -= delta_buf
        if not improved:
            break
    return best_h, best_peak


def _smooth_greedy_polish(h, ac, time_budget, start_time, beta=50.0):
    """
    Greedy polish using smooth surrogate objective.
    Helps escape nonsmooth local minima.
    """
    h = h.copy()
    ac = ac.copy()
    cur_val = _smooth_sidelobe_from_ac(ac, beta)
    best_h = h.copy()
    best_val = cur_val
    delta_buf = np.zeros(N, dtype=np.float64)

    while time.time() - start_time < time_budget:
        improved = False
        for i in range(N):
            old_h_i = h[i]
            h[i] = -old_h_i
            _flip_delta(h, i, delta_buf)
            delta_buf *= -1.0
            ac[1:] += delta_buf
            new_val = _smooth_sidelobe_from_ac(ac, beta)
            if new_val < cur_val:
                cur_val = new_val
                if cur_val < best_val:
                    best_val = cur_val
                    best_h = h.copy()
                improved = True
                break
            else:
                h[i] = old_h_i
                ac[1:] -= delta_buf
        if not improved:
            break
    return best_h, best_val


def _best_greedy_polish(h, ac, time_budget, start_time):
    """
    Best improvement greedy polish.
    """
    h = h.copy()
    ac = ac.copy()
    cur_peak = _peak_sidelobe_from_ac(ac)
    best_h = h.copy()
    best_peak = cur_peak
    delta_buf = np.zeros(N, dtype=np.float64)

    while time.time() - start_time < time_budget:
        best_i = -1
        best_new_peak = cur_peak

        for i in range(N):
            old_h_i = h[i]
            h[i] = -old_h_i
            _flip_delta(h, i, delta_buf)
            delta_buf *= -1.0
            ac[1:] += delta_buf
            new_peak = _peak_sidelobe_from_ac(ac)
            if new_peak < best_new_peak:
                best_new_peak = new_peak
                best_i = i
            h[i] = old_h_i
            ac[1:] -= delta_buf

        if best_i < 0:
            break

        i = best_i
        old_h_i = h[i]
        h[i] = -old_h_i
        _flip_delta(h, i, delta_buf)
        delta_buf *= -1.0
        ac[1:] += delta_buf
        cur_peak = best_new_peak

        if cur_peak < best_peak:
            best_peak = cur_peak
            best_h = h.copy()

    return best_h, best_peak


def _anneal_smooth(h, ac, time_budget, start_time, beta=30.0):
    """
    Simulated annealing using smooth surrogate.
    Explores more broadly than pure greedy.
    """
    h = h.copy()
    ac = ac.copy()
    cur_val = _smooth_sidelobe_from_ac(ac, beta)
    best_h = h.copy()
    best_val = cur_val
    best_exact = _peak_sidelobe_from_ac(ac)
    delta_buf = np.zeros(N, dtype=np.float64)
    rng = np.random.default_rng(123)

    T = 2.0
    T_min = 0.01
    cooling = 0.998

    while T > T_min and time.time() - start_time < time_budget:
        i = int(rng.integers(0, N))
        old_h_i = h[i]
        h[i] = -old_h_i
        _flip_delta(h, i, delta_buf)
        delta_buf *= -1.0
        ac[1:] += delta_buf
        new_val = _smooth_sidelobe_from_ac(ac, beta)

        d = new_val - cur_val
        if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            cur_val = new_val
            new_exact = _peak_sidelobe_from_ac(ac)
            if new_exact < best_exact:
                best_exact = new_exact
                best_h = h.copy()
                best_val = cur_val
        else:
            h[i] = old_h_i
            ac[1:] -= delta_buf
        T *= cooling

    return best_h, best_exact


def construct():
    try:
        hard_deadline = time.time() + 145.0
        start_time = time.time()

        # Multi-start initializations
        rng = np.random.default_rng(42)

        # Start 1: Thue-Morse
        h_init = _thue_morse(N)
        ac_init = _calc_ac(h_init)

        # Start 2: Random with TM bias
        h_rand = h_init.copy()
        flip_idx = rng.choice(N, size=50, replace=False)
        h_rand[flip_idx] *= -1.0
        ac_rand = _calc_ac(h_rand)

        # Pick better initial
        peak_tm = _peak_sidelobe_from_ac(ac_init)
        peak_rand = _peak_sidelobe_from_ac(ac_rand)
        if peak_rand < peak_tm:
            h_init, ac_init = h_rand, ac_rand
            peak_init = peak_rand
        else:
            peak_init = peak_tm

        # Phase 1: Fast greedy polish (exact objective)
        budget1 = min(15.0, hard_deadline - (time.time() - start_time) - 5.0)
        if budget1 > 1.0:
            h_best, peak_best = _fast_greedy_polish(h_init, ac_init, budget1, time.time())
        else:
            h_best = h_init
            peak_best = peak_init

        # Phase 2: Smooth surrogate optimization to escape local minima
        # Use multiple beta values for different smoothness levels
        elapsed = time.time() - start_time
        remaining_after_phase2 = hard_deadline - elapsed - 20.0  # reserve 20s for ILS + final

        if remaining_after_phase2 > 5.0:
            # Try smooth greedy with moderate beta
            beta1 = 30.0
            budget2a = min(10.0, remaining_after_phase2 / 3.0)
            if budget2a > 1.0:
                h_smooth1, val_smooth1 = _smooth_greedy_polish(h_best, _calc_ac(h_best), budget2a, time.time(), beta=beta1)
                peak_smooth1 = _peak_sidelobe_from_ac(_calc_ac(h_smooth1))
                if peak_smooth1 < peak_best:
                    h_best = h_smooth1
                    peak_best = peak_smooth1

            # Smooth anneal with lower beta (smoother landscape)
            beta2 = 15.0
            elapsed = time.time() - start_time
            remaining_after_phase2 = hard_deadline - elapsed - 20.0
            budget2b = min(10.0, remaining_after_phase2 / 2.0)
            if budget2b > 1.0:
                h_anneal, peak_anneal = _anneal_smooth(h_best, _calc_ac(h_best), budget2b, time.time(), beta=beta2)
                if peak_anneal < peak_best:
                    h_best = h_anneal
                    peak_best = peak_anneal

        # Phase 3: ILS with mixed smooth/exact polish
        elapsed = time.time() - start_time
        remaining_for_ils = hard_deadline - elapsed - 15.0  # reserve 15s for final polish

        if remaining_for_ils > 5.0:
            perturbation_size = 5
            ils_rng = np.random.default_rng(7)
            smooth_counter = 0

            while time.time() < hard_deadline - 15.0:
                # Perturb
                h_perturb = h_best.copy()
                indices = ils_rng.choice(N, size=perturbation_size, replace=False)
                h_perturb[indices] *= -1.0
                ac_perturb = _calc_ac(h_perturb)

                # Alternate between smooth and exact polish
                smooth_counter += 1
                use_smooth = (smooth_counter % 3 == 0)  # Every 3rd iteration use smooth

                budget_iter = min(4.0, hard_deadline - (time.time() - start_time) - 15.0)
                if budget_iter < 0.5:
                    break

                if use_smooth:
                    h_new, val_new = _smooth_greedy_polish(h_perturb, ac_perturb, budget_iter, time.time(), beta=25.0)
                    peak_new = _peak_sidelobe_from_ac(_calc_ac(h_new))
                else:
                    h_new, peak_new = _fast_greedy_polish(h_perturb, ac_perturb, budget_iter, time.time())

                if peak_new < peak_best:
                    peak_best = peak_new
                    h_best = h_new
                    perturbation_size = max(3, perturbation_size - 1)
                else:
                    if perturbation_size < 20:
                        perturbation_size += 2
                    else:
                        perturbation_size = int(ils_rng.integers(5, 20))

        # Phase 4: Final best-improvement polish (exact)
        remaining_time = hard_deadline - (time.time() - start_time) - 1.0
        if remaining_time > 1.0:
            ac_best = _calc_ac(h_best)
            h_final, peak_final = _best_greedy_polish(h_best, ac_best, remaining_time, time.time())
            if peak_final < peak_best:
                h_best = h_final
                peak_best = peak_final

        return np.ascontiguousarray(h_best, dtype=np.float64)

    except Exception:
        return np.ones(N)
