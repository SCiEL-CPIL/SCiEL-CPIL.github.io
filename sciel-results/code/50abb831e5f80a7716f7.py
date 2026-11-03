import numpy as np
import time

N = 400
_NFFT = 1 << (2 * N - 1).bit_length()  # next power of 2 >= 2N for fast FFT


def _autocorr_fast(h):
    # linear one-sided autocorrelation ac[k] = sum_i h[i] h[i+k], k = 0..N-1
    # Use real FFT trick: autocorr of real signal = IFFT(|FFT|^2)
    F = np.fft.rfft(h, n=_NFFT)
    ac = np.fft.irfft(F * np.conj(F), n=_NFFT)[:N]
    return ac


def _autocorr(h):
    return _autocorr_fast(h)


def _hard_score(h):
    # the actual metric: peak autoconvolution concentration (lower is better)
    ac = _autocorr(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    return float(np.max(np.abs(ac)) / dc)


def _continuous_optimize(h_init, max_iter=5000, lr_init=0.01, lr_decay=0.999):
    """
    Continuous gradient descent on L2 norm of autocorrelation sidelobes.
    This finds a low-autocorrelation continuous signal, which we then threshold
    to get a better binary seed than random.
    """
    h = h_init.astype(np.float64).copy()
    lr = lr_init
    
    for it in range(max_iter):
        # Compute autocorrelation
        ac = _autocorr_fast(h)
        dc = ac[0]
        if dc <= 1e-30:
            break
        
        # Objective: minimize sum of squared sidelobes (excluding DC)
        sidelobes = ac[1:] / dc
        
        # Gradient of sum(sidelobes^2) w.r.t. h
        # ac[k] = sum_i h[i] h[i+k]
        # d(ac[k])/d(h[i]) = h[i+k] + h[i-k] (for 0 <= i < N)
        # Using FFT: gradient can be computed efficiently
        F = np.fft.rfft(h, n=_NFFT)
        # Power spectrum
        P = F * np.conj(F)
        
        # Gradient of ||ac[1:]||^2 w.r.t. h
        # This is equivalent to: grad = 2 * IFFT(P * (1 - delta_0)) convolved appropriately
        # More directly: grad[i] = 2 * sum_{k=1}^{N-1} ac[k] * (h[i+k] + h[i-k])
        
        # Efficient computation using FFT
        # The gradient of sum_{k=1}^{N-1} ac[k]^2 w.r.t. h[i]
        # = 2 * sum_{k=1}^{N-1} ac[k] * (h[i+k] + h[i-k])
        
        # Create a signal g where g[k] = ac[k] for k=1..N-1, g[0]=0
        g = np.zeros(_NFFT)
        g[1:N] = ac[1:]
        # Also need negative frequencies for circular convolution
        g[_NFFT - 1:_NFFT - (N-1)] = ac[1:]
        
        G = np.fft.rfft(g, n=_NFFT)
        H = np.fft.rfft(h, n=_NFFT)
        
        # grad = 2 * IFFT(G * H) + 2 * IFFT(conj(G) * H) but we need to be careful
        # Actually: sum_k ac[k] * h[i+k] is circular correlation
        # Let's use: grad[i] = 2 * (correlation of ac and h)[i]
        
        # Circular correlation of ac_padded and h
        ac_pad = np.zeros(_NFFT)
        ac_pad[1:N] = ac[1:]
        ac_pad[_NFFT-1:_NFFT-(N-1)] = ac[1:]
        
        A = np.fft.rfft(ac_pad, n=_NFFT)
        H_full = np.fft.rfft(h, n=_NFFT)
        
        # Correlation: IFFT(conj(A) * H)
        corr = np.fft.irfft(np.conj(A) * H_full, n=_NFFT)[:N]
        
        grad = 2.0 * corr / (dc * dc)  # Normalize by dc^2
        
        # Gradient step
        h -= lr * grad
        
        # Keep h roughly normalized
        h_norm = np.linalg.norm(h)
        if h_norm > 1e-10:
            h /= h_norm * np.sqrt(N)
        
        lr *= lr_decay
    
    return h


def _smooth_score(h, beta):
    # smooth (log-sum-exp) surrogate of the peak sidelobe energy, guides SA
    ac = _autocorr(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    y = (ac[1:] / dc) ** 2
    m = float(np.max(y))
    z = beta * (y - m)
    return m + float(np.log(np.sum(np.exp(z)))) / beta


def _hard_score_approx(h):
    """Faster approximate score: uses a subset of lags to estimate the peak."""
    ac = _autocorr_fast(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    # Check all lags but use a quick max
    return float(np.max(np.abs(ac)) / dc)


class _FastAutocorr:
    """
    Maintains FFT of h and supports O(N) single-bit flip updates.
    
    For real h, flipping h[i] -> -h[i] changes FFT by:
    F_new[k] = F_old[k] - 2 * h[i] * exp(-2j*pi*k*i/NFFT)
    
    This is O(NFFT/2) = O(N) per flip, vs O(N log N) for full recompute.
    """
    def __init__(self, h):
        self.h = h.astype(np.float64).copy()
        self.N = len(self.h)
        self.NFFT = _NFFT
        self.F = np.fft.rfft(self.h, n=self.NFFT)
        # Precompute the twiddle factors for each position
        k = np.arange(self.NFFT // 2 + 1, dtype=np.float64)
        self.twiddle = np.exp(-2j * np.pi * k[:, None] * np.arange(self.N)[None, :] / self.NFFT)
        # self.twiddle[i, k] = exp(-2j*pi*k*i/NFFT)
    
    def flip(self, i):
        """Flip sign of h[i] and update FFT in O(NFFT/2)."""
        delta = -2.0 * self.h[i]
        self.F += delta * self.twiddle[i]
        self.h[i] *= -1.0
    
    def autocorr(self):
        """Compute autocorrelation from current FFT state."""
        ac = np.fft.irfft(self.F * np.conj(self.F), n=self.NFFT)[:self.N]
        return ac
    
    def hard_score(self):
        ac = self.autocorr()
        dc = ac[0]
        if dc <= 1e-30:
            return 1e9
        return float(np.max(np.abs(ac)) / dc)
    
    def smooth_score(self, beta):
        ac = self.autocorr()
        dc = ac[0]
        if dc <= 1e-30:
            return 1e9
        y = (ac[1:] / dc) ** 2
        m = float(np.max(y))
        z = beta * (y - m)
        return m + float(np.log(np.sum(np.exp(z)))) / beta
    
    def get_h(self):
        return self.h.copy()
    
    def set_h(self, h):
        """Reset to a new h, recomputing FFT."""
        self.h = h.astype(np.float64).copy()
        self.F = np.fft.rfft(self.h, n=self.NFFT)


def construct():
    try:
        t0 = time.time()
        deadline = t0 + 115.0          # safety margin inside the 120s limit
        rng = np.random.default_rng(20240517)

        # ---------- diverse initializations ----------
        idx = np.arange(N)
        inits = [
            np.ones(N),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            np.cos(2 * np.pi * idx ** 2 / (2 * N)),
            np.cos(2 * np.pi * idx ** 2 / (N + 1)),
            np.cos(2 * np.pi * idx ** 2 / (3 * N)),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            np.cos(2 * np.pi * idx / N),
            np.sin(2 * np.pi * idx / N),
            np.cos(2 * np.pi * 2 * idx / N),
            rng.choice(np.array([-1.0, 1.0]), size=N),
        ]
        
        # Add continuous optimization seeds
        for h_cont_init in inits[:5]:
            h_cont = _continuous_optimize(h_cont_init, max_iter=2000, lr_init=0.005, lr_decay=0.9995)
            # Threshold to binary
            h_bin = np.sign(h_cont)
            h_bin[h_bin == 0] = 1.0
            inits.append(h_bin)
        
        best_h = None
        best_hard = 1e9
        for h in inits:
            s = _hard_score(h)
            if s < best_hard:
                best_hard = s
                best_h = h.copy()

        # Use fast autocorr class for SA with incremental updates
        fa = _FastAutocorr(best_h)
        cur_hard = best_hard
        cur_smooth = fa.smooth_score(30.0)

        # ---------- simulated annealing with incremental FFT updates ----------
        T = 0.02
        T_min = 1e-4
        decay = 0.9997
        beta = 30.0
        sizes = np.array([1, 2, 3, 4, 8, 12, 16, 20, 24, 32, 48, 64])

        while time.time() < deadline - 5.0:
            if rng.random() < 0.40:
                # Single-bit flip: O(N) via incremental FFT
                i = int(rng.integers(0, N))
                old_val = fa.h[i]
                fa.flip(i)
                new_hard = fa.hard_score()
                new_smooth = fa.smooth_score(beta)
                dS = new_smooth - cur_smooth
                if new_hard < cur_hard or (dS <= 0 or rng.random() < np.exp(-dS / T)):
                    cur_hard = new_hard
                    cur_smooth = new_smooth
                    if new_hard < best_hard:
                        best_hard = new_hard
                        best_h = fa.get_h()
                else:
                    # Revert
                    fa.flip(i)
            elif rng.random() < 0.60:
                # Two-bit flip: O(N) via two incremental updates
                i1 = int(rng.integers(0, N))
                i2 = int(rng.integers(0, N))
                if i1 == i2:
                    i2 = (i2 + 1) % N
                fa.flip(i1)
                fa.flip(i2)
                new_hard = fa.hard_score()
                new_smooth = fa.smooth_score(beta)
                dS = new_smooth - cur_smooth
                if new_hard < cur_hard or (dS <= 0 or rng.random() < np.exp(-dS / T)):
                    cur_hard = new_hard
                    cur_smooth = new_smooth
                    if new_hard < best_hard:
                        best_hard = new_hard
                        best_h = fa.get_h()
                else:
                    fa.flip(i1)
                    fa.flip(i2)
            else:
                # Block flip: requires full recompute
                size = int(rng.choice(sizes))
                start = int(rng.integers(0, N - size + 1))
                sl = slice(start, start + size)
                old = fa.h[sl].copy()
                fa.h[sl] *= -1.0
                # Recompute FFT for block flip
                fa.F = np.fft.rfft(fa.h, n=fa.NFFT)
                new_hard = fa.hard_score()
                new_smooth = fa.smooth_score(beta)
                dS = new_smooth - cur_smooth
                if new_hard < cur_hard or (dS <= 0 or rng.random() < np.exp(-dS / T)):
                    cur_hard = new_hard
                    cur_smooth = new_smooth
                    if new_hard < best_hard:
                        best_hard = new_hard
                        best_h = fa.get_h()
                else:
                    fa.h[sl] = old
                    fa.F = np.fft.rfft(fa.h, n=fa.NFFT)
            T = max(T_min, T * decay)

        # ---------- greedy multi-scale local polish ----------
        h = best_h.copy()
        sizes_polish = [1, 2, 4, 8, 16, 32, 64]
        for _pass in range(60):
            if time.time() > deadline - 1.0:
                break
            any_improve = False
            for size in sizes_polish:
                for start in range(0, N - size + 1):
                    if time.time() > deadline - 1.0:
                        return best_h
                    sl = slice(start, start + size)
                    old = h[sl].copy()
                    h[sl] *= -1.0
                    new_hard = _hard_score(h)
                    if new_hard < best_hard - 1e-12:
                        best_hard = new_hard
                        best_h = h.copy()
                        any_improve = True
                    else:
                        h[sl] = old
            if not any_improve:
                break

        # ---------- final single-bit polish ----------
        h = best_h.copy()
        for _pass in range(200):
            if time.time() > deadline - 0.5:
                break
            any_improve = False
            for i in range(N):
                old = h[i]
                h[i] *= -1.0
                new_hard = _hard_score(h)
                if new_hard < best_hard - 1e-12:
                    best_hard = new_hard
                    best_h = h.copy()
                    any_improve = True
                else:
                    h[i] = old
            if not any_improve:
                break

        return best_h
    except Exception:
        return np.ones(N)
