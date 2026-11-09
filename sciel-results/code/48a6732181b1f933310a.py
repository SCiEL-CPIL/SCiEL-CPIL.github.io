"""DFT-based spectral root-finding with safeguarded Newton/bisection fallback."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def find_root(f, x0):
    x0 = float(x0)
    T0 = time.time()
    DEADLINE = 145.0
    TOL = 1e-12
    HARD_TOL = 1e-8
    MAX_REFINE_ITERS = 300

    def elapsed_ok():
        return (time.time() - T0) < DEADLINE

    try:
        f0 = f(x0)
    except Exception:
        f0 = float('nan')

    best_x = x0
    best_val = abs(f0) if (f0 == f0 and math.isfinite(f0)) else float('inf')

    def track(x, fx):
        nonlocal best_x, best_val
        if fx == fx and math.isfinite(fx):
            av = abs(fx)
            if av < best_val:
                best_val = av
                best_x = x

    track(x0, f0)

    if f0 == f0 and math.isfinite(f0) and abs(f0) <= HARD_TOL:
        return x0

    # ------------------------------------------------------------------
    # 1) DFT-based spectral prediction of nearest zero crossing
    # ------------------------------------------------------------------
    spectral_candidates = []

    base = max(1.0, abs(x0))

    # Try multiple sampling windows to capture different frequency scales
    for window_scale in [1e-2, 1e-1, 1.0, 10.0, 100.0]:
        if not elapsed_ok():
            break

        window = base * window_scale
        if window < 1e-15:
            window = 1e-15

        # Number of samples: enough to resolve a few periods
        n_samples = 64
        left = x0 - window
        right = x0 + window

        xs = np.linspace(left, right, n_samples)
        fs = np.empty(n_samples)

        for i in range(n_samples):
            if not elapsed_ok():
                break
            try:
                fs[i] = f(float(xs[i]))
            except Exception:
                fs[i] = float('nan')
            track(float(xs[i]), fs[i])

        # Check if we have enough finite values
        finite_mask = np.isfinite(fs) & (fs == fs)
        if np.sum(finite_mask) < n_samples // 2:
            continue

        # Replace NaNs with interpolation for DFT
        fs_clean = fs.copy()
        if not np.all(finite_mask):
            idx = np.arange(n_samples)
            fs_clean[~finite_mask] = np.interp(idx[~finite_mask], idx[finite_mask], fs[finite_mask])

        # Check for sign changes in the sampled data (fast bracket)
        for i in range(n_samples - 1):
            if fs_clean[i] * fs_clean[i+1] < 0:
                a_cand = float(xs[i])
                b_cand = float(xs[i+1])
                fa_cand = float(fs_clean[i])
                fb_cand = float(fs_clean[i+1])
                spectral_candidates.append((a_cand, b_cand, fa_cand, fb_cand, window_scale))
                break

        # DFT to estimate dominant frequency
        # Remove mean
        fs_centered = fs_clean - np.mean(fs_clean)
        fft_vals = np.fft.rfft(fs_centered)

        # Find dominant frequency (skip DC)
        if len(fft_vals) > 1:
            magnitudes = np.abs(fft_vals)
            # Skip first few bins (low frequency noise) and last few (high freq noise)
            skip = max(1, n_samples // 32)
            if skip < len(magnitudes) - 1:
                search_mags = magnitudes[skip:-skip] if skip < len(magnitudes) - 1 else magnitudes[skip:]
                if len(search_mags) > 0:
                    peak_idx = int(np.argmax(search_mags)) + skip
                    peak_mag = magnitudes[peak_idx]

                    if peak_mag > 0.1 * np.max(magnitudes):  # Significant peak
                        # Frequency in cycles per sample
                        freq_cycles = peak_idx / n_samples
                        # Frequency in cycles per unit x
                        freq_hz = freq_cycles / (2.0 * window)  # window is half-range, total range is 2*window

                        if freq_hz > 0 and math.isfinite(freq_hz):
                            # Estimate phase from the complex FFT value
                            phase = np.angle(fft_vals[peak_idx])

                            # The fitted sinusoid: A * cos(2*pi*freq_hz*(x - x0) + phase)
                            # Zero when: 2*pi*freq_hz*(x - x0) + phase = pi/2 + k*pi
                            # x = x0 + (pi/2 + k*pi - phase) / (2*pi*freq_hz)

                            # Find nearest zero crossing
                            k_vals = []
                            # k can be any integer; find the one giving x closest to x0
                            base_angle = phase
                            # We want 2*pi*freq_hz*(x - x0) = pi/2 + k*pi - phase
                            # x - x0 = (pi/2 + k*pi - phase) / (2*pi*freq_hz)

                            # Find k that minimizes |x - x0|
                            # x - x0 = (pi/2 - phase + k*pi) / (2*pi*freq_hz)
                            # Minimize |pi/2 - phase + k*pi|
                            # k*pi ≈ -(pi/2 - phase) => k ≈ (phase - pi/2)/pi
                            k_float = (phase - math.pi / 2) / math.pi
                            k_near1 = int(math.floor(k_float))
                            k_near2 = k_near1 + 1

                            for k in [k_near1, k_near2]:
                                dx = (math.pi / 2 + k * math.pi - phase) / (2.0 * math.pi * freq_hz)
                                x_pred = x0 + dx
                                if math.isfinite(x_pred):
                                    spectral_candidates.append(('pred', x_pred, freq_hz, phase, window_scale))

    # ------------------------------------------------------------------
    # 2) Try to refine spectral candidates
    # ------------------------------------------------------------------
    # Sort candidates by proximity to x0
    def candidate_distance(c):
        if isinstance(c, tuple) and len(c) == 5:
            if c[0] == 'pred':
                return abs(c[1] - x0)
            else:
                # Bracket: use midpoint
                return abs(0.5 * (c[0] + c[1]) - x0)
        return float('inf')

    # Separate brackets and predictions
    brackets = [c for c in spectral_candidates if c[0] != 'pred']
    predictions = [c for c in spectral_candidates if c[0] == 'pred']

    # Try brackets first (more reliable)
    brackets.sort(key=lambda c: abs(0.5 * (c[0] + c[1]) - x0))
    predictions.sort(key=lambda c: abs(c[1] - x0))

    # First, try to find a bracket via standard outward scan if none found
    a = b = None
    fa = fb = None

    if brackets:
        a, b, fa, fb = brackets[0][0], brackets[0][1], brackets[0][2], brackets[0][3]

    if a is None:
        # Standard outward scan for bracket
        step = base * 1e-3
        lo = x0
        flo = f0
        found = False

        for i in range(60):
            if not elapsed_ok():
                break
            hi = lo + step
            try:
                fhi = f(hi)
            except Exception:
                fhi = float('nan')
            track(hi, fhi)
            if fhi == fhi and math.isfinite(fhi) and flo == flo and math.isfinite(flo) and flo * fhi < 0:
                a, b, fa, fb = lo, hi, flo, fhi
                found = True
                break
            lo = hi
            if fhi == fhi and math.isfinite(fhi):
                flo = fhi
            else:
                flo = float('nan')
            if i < 5:
                step = step * 1.5
            else:
                step = step * 2.0

        if not found:
            step = base * 1e-3
            hi = x0
            fhi = f0
            for i in range(60):
                if not elapsed_ok():
                    break
                lo = hi - step
                try:
                    flo = f(lo)
                except Exception:
                    flo = float('nan')
                track(lo, flo)
                if flo == flo and math.isfinite(flo) and fhi == fhi and math.isfinite(fhi) and flo * fhi < 0:
                    a, b, fa, fb = lo, hi, flo, fhi
                    found = True
                    break
                hi = lo
                if flo == flo and math.isfinite(flo):
                    fhi = flo
                else:
                    fhi = float('nan')
                if i < 5:
                    step = step * 1.5
                else:
                    step = step * 2.0

    # ------------------------------------------------------------------
    # 3) Refine bracket with safeguarded Newton + bisection
    # ------------------------------------------------------------------
    if a is not None:
        if fa is None or not math.isfinite(fa) or fa != fa:
            try:
                fa = f(a)
            except Exception:
                fa = float('nan')
        if fb is None or not math.isfinite(fb) or fb != fb:
            try:
                fb = f(b)
            except Exception:
                fb = float('nan')

        if abs(fa) <= abs(fb):
            x, fx = a, fa
            x2, f2 = b, fb
        else:
            x, fx = b, fb
            x2, f2 = a, fa

        for _ in range(MAX_REFINE_ITERS):
            if not elapsed_ok():
                break
            if fx == fx and math.isfinite(fx) and abs(fx) <= TOL:
                return x
            track(x, fx)

            mid = 0.5 * (x + x2)

            used_newton = False
            if fx == fx and f2 == f2 and math.isfinite(fx) and math.isfinite(f2) and f2 != fx:
                denom = (f2 - fx) / (x2 - x) if x2 != x else 0.0
                if denom != 0 and math.isfinite(denom):
                    xn = x - fx / denom
                    if (min(a, b) < xn < max(a, b)):
                        try:
                            fn = f(xn)
                        except Exception:
                            fn = float('nan')
                        track(xn, fn)
                        if fn == fn and math.isfinite(fn) and abs(fn) <= TOL:
                            return xn
                        if fn == fn and math.isfinite(fn):
                            if fx * fn < 0:
                                x2, f2 = x, fx
                                x, fx = xn, fn
                            else:
                                x, fx = xn, fn
                            used_newton = True

            if not used_newton:
                try:
                    fm = f(mid)
                except Exception:
                    fm = float('nan')
                track(mid, fm)
                if fm == fm and math.isfinite(fm) and abs(fm) <= TOL:
                    return mid
                if fm == fm and math.isfinite(fm):
                    if fx == fx and math.isfinite(fx) and fx * fm < 0:
                        x2, f2 = mid, fm
                    else:
                        x, fx = mid, fm
                else:
                    if fx == fx and math.isfinite(fx):
                        x2, f2 = mid, float('nan')
                    else:
                        x, fx = mid, float('nan')

        if best_val <= HARD_TOL:
            return best_x

    # ------------------------------------------------------------------
    # 4) Try spectral predictions as starting points for local refinement
    # ------------------------------------------------------------------
    for pred in predictions:
        if not elapsed_ok():
            break
        x_pred = pred[1]
        try:
            f_pred = f(x_pred)
        except Exception:
            continue
        track(x_pred, f_pred)
        if f_pred == f_pred and math.isfinite(f_pred) and abs(f_pred) <= HARD_TOL:
            return x_pred

        # Local Newton refinement from prediction
        x = x_pred
        fx = f_pred
        for _ in range(50):
            if not elapsed_ok():
                break
            if fx == fx and math.isfinite(fx) and abs(fx) <= TOL:
                return x
            h = 1e-7 * max(1.0, abs(x))
            try:
                fx2 = f(x + h)
            except Exception:
                break
            if fx2 != fx2 or not math.isfinite(fx2):
                break
            d = (fx2 - fx) / h
            if d == 0 or d != d or not math.isfinite(d):
                break
            xn = x - fx / d
            if xn != xn or not math.isfinite(xn):
                break
            try:
                fn = f(xn)
            except Exception:
                break
            track(xn, fn)
            if fn == fn and math.isfinite(fn) and abs(fn) <= TOL:
                return xn
            if abs(xn - x) > 1e6 * max(1.0, abs(x)):
                xn = 0.5 * (x + xn)
            x, fx = xn, fn

    # ------------------------------------------------------------------
    # 5) Fallback: local secant from x0
    # ------------------------------------------------------------------
    try:
        x = x0
        fx = f0
        if fx != fx or not math.isfinite(fx):
            for dx in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]:
                try:
                    f_try = f(x0 + dx)
                except Exception:
                    continue
                if f_try == f_try and math.isfinite(f_try):
                    x = x0 + dx
                    fx = f_try
                    track(x, fx)
                    break
                try:
                    f_try = f(x0 - dx)
                except Exception:
                    continue
                if f_try == f_try and math.isfinite(f_try):
                    x = x0 - dx
                    fx = f_try
                    track(x, fx)
                    break

        if fx != fx or not math.isfinite(fx):
            if best_x == best_x and math.isfinite(best_val):
                return best_x
            return x0

        for _ in range(80):
            if not elapsed_ok():
                break
            if fx == fx and math.isfinite(fx) and abs(fx) <= TOL:
                return x
            h = 1e-7 * max(1.0, abs(x))
            try:
                fx2 = f(x + h)
            except Exception:
                break
            if fx2 != fx2 or not math.isfinite(fx2):
                break
            d = (fx2 - fx) / h
            if d == 0 or d != d or not math.isfinite(d):
                break
            xn = x - fx / d
            if xn != xn or not math.isfinite(xn):
                break
            try:
                fn = f(xn)
            except Exception:
                break
            track(xn, fn)
            if fn == fn and math.isfinite(fn) and abs(fn) <= TOL:
                return xn
            if abs(xn - x) > 1e6 * max(1.0, abs(x)):
                xn = 0.5 * (x + xn)
            x, fx = xn, fn
    except Exception:
        pass

    if best_x == best_x and math.isfinite(best_val):
        return best_x
    return x0
# EVOLVE-BLOCK-END
