import numpy as np
import time

N = 400
_NFFT = 1 << (2 * N - 1).bit_length()  # next power of 2 >= 2N for fast FFT

# Approximate score: use only first K lags for speed during SA
_K_APPROX = 200


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


def _hard_score_approx(ac):
    """Fast approximate score using only first _K_APPROX lags."""
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    # Use a subset of lags for speed
    k = min(_K_APPROX, len(ac))
    return float(np.max(np.abs(ac[:k])) / dc)


def _smooth_score(h, beta, ac=None):
    # smooth (log-sum-exp) surrogate of the peak sidelobe energy, guides SA
    if ac is None:
        ac = _autocorr(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    y = (ac[1:] / dc) ** 2
    m = float(np.max(y))
    z = beta * (y - m)
    # Clip z to avoid overflow in exp
    z_clipped = np.clip(z, -50, 50)
    return m + float(np.log(np.sum(np.exp(z_clipped)))) / beta


def _spectral_weighted_score(ac, beta, top_k=10):
    """Weighted smooth score that emphasizes the top-k dominant lags."""
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    y = (ac[1:] / dc) ** 2
    # Get top k lags by magnitude
    if len(y) <= top_k:
        top_lags = np.arange(len(y))
    else:
        top_lags = np.argsort(y)[-top_k:]
    
    # Create weight vector: higher weight for dominant lags
    weights = np.ones(len(y))
    weights[top_lags] = 3.0  # Emphasize dominant lags
    
    # Weighted log-sum-exp
    weighted_y = y * weights
    m = float(np.max(weighted_y))
    z = beta * (weighted_y - m)
    z_clipped = np.clip(z, -50, 50)
    return m + float(np.log(np.sum(np.exp(z_clipped)))) / beta


def _find_dominant_lags(ac, top_k=5):
    """Find indices of top-k dominant lags (excluding lag 0)."""
    if len(ac) <= 1:
        return np.array([], dtype=int)
    y = np.abs(ac[1:]) / max(abs(ac[0]), 1e-30)
    if len(y) <= top_k:
        return np.arange(1, len(ac))
    top_idx = np.argsort(y)[-top_k:]
    return top_idx + 1  # Offset by 1 since we excluded lag 0


class _FastAutocorr:
    """Fast autocorrelation with incremental updates for block flips."""
    
    def __init__(self, h):
        self.h = h.copy()
        self.F = np.fft.rfft(self.h, n=_NFFT)
        self._update_ac()
        self._dominant_lags = _find_dominant_lags(self.ac, top_k=5)
    
    def _update_ac(self):
        self.ac = np.fft.irfft(self.F * np.conj(self.F), n=_NFFT)[:N]
    
    def _update_dominant_lags(self):
        self._dominant_lags = _find_dominant_lags(self.ac, top_k=5)
    
    def flip_block(self, start, size):
        """Flip a block of signs and update autocorrelation."""
        sl = slice(start, start + size)
        self.h[sl] *= -1.0
        # Update FFT incrementally
        delta_h = np.zeros(_NFFT)
        delta_h[sl] = self.h[sl] * 2.0
        delta_F = np.fft.rfft(delta_h, n=_NFFT)
        self.F = self.F + delta_F
        self._update_ac()
    
    def unflip_block(self, start, size):
        """Undo a block flip."""
        self.flip_block(start, size)
    
    def flip_bit(self, i):
        """Flip a single bit."""
        self.flip_block(i, 1)
    
    def get_score(self):
        dc = self.ac[0]
        if dc <= 1e-30:
            return 1e9
        return float(np.max(np.abs(self.ac)) / dc)
    
    def get_score_approx(self):
        """Fast approximate score using subset of lags."""
        return _hard_score_approx(self.ac)
    
    def get_smooth_score(self, beta):
        dc = self.ac[0]
        if dc <= 1e-30:
            return 1e9
        y = (self.ac[1:] / dc) ** 2
        m = float(np.max(y))
        z = beta * (y - m)
        z_clipped = np.clip(z, -50, 50)
        return m + float(np.log(np.sum(np.exp(z_clipped)))) / beta
    
    def get_spectral_score(self, beta, top_k=10):
        """Spectral-weighted smooth score emphasizing dominant lags."""
        return _spectral_weighted_score(self.ac, beta, top_k)
    
    def get_dominant_lags(self):
        """Return current dominant lag indices."""
        return self._dominant_lags


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
            # Add more diverse inits
            np.sign(np.cos(2 * np.pi * 3 * idx / N)),
            np.sign(np.cos(2 * np.pi * 5 * idx / N)),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            rng.choice(np.array([-1.0, 1.0]), size=N),
        ]
        best_h = None
        best_hard = 1e9
        for h in inits:
            s = _hard_score(h)
            if s < best_hard:
                best_hard = s
                best_h = h.copy()

        h = best_h.copy()
        cur_hard = best_hard
        cur_smooth = _smooth_score(h, 30.0)
        cur_spectral = _spectral_weighted_score(_autocorr(h), 30.0, top_k=10)

        # ---------- simulated annealing with coordinated block sign-flips ----------
        T = 0.05  # Higher initial temperature for more exploration
        T_min = 1e-5
        decay = 0.9998  # Slower decay for longer exploration
        beta = 30.0
        sizes = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256])
        
        # Use FastAutocorr for efficient updates
        fa = _FastAutocorr(h)
        
        # Track dominant lags for spectral-guided mutations
        last_dominant_update = 0
        
        # Use approximate score for SA acceptance, exact for best tracking
        while time.time() < deadline - 8.0:
            # Periodically update dominant lags
            if time.time() - last_dominant_update > 0.5:
                fa._update_dominant_lags()
                last_dominant_update = time.time()
            
            if rng.random() < 0.55:
                # Block flip
                size = int(rng.choice(sizes))
                start = int(rng.integers(0, N - size + 1))
                fa.flip_block(start, size)
                # Use approximate score for speed
                new_hard_approx = fa.get_score_approx()
                new_smooth = fa.get_smooth_score(beta)
                new_spectral = fa.get_spectral_score(beta, top_k=10)
                dS = new_smooth - cur_smooth
                dS_spec = new_spectral - cur_spectral
                # Combine both scores for acceptance
                if dS <= 0 and dS_spec <= 0:
                    accept = True
                elif dS <= 0 or dS_spec <= 0:
                    accept = rng.random() < 0.7
                else:
                    accept = rng.random() < np.exp(-(dS + dS_spec) / T)
                
                if accept:
                    # Check with exact score before updating best
                    new_hard_exact = fa.get_score()
                    if new_hard_exact < cur_hard:
                        cur_hard = new_hard_exact
                    cur_smooth = new_smooth
                    cur_spectral = new_spectral
                    if new_hard_exact < best_hard:
                        best_hard = new_hard_exact
                        best_h = fa.h.copy()
                else:
                    fa.unflip_block(start, size)
            elif rng.random() < 0.5:
                # Two-bit flip
                i1 = int(rng.integers(0, N))
                i2 = int(rng.integers(0, N))
                if i1 == i2:
                    i2 = (i2 + 1) % N
                fa.flip_bit(i1)
                fa.flip_bit(i2)
                new_hard_approx = fa.get_score_approx()
                new_smooth = fa.get_smooth_score(beta)
                new_spectral = fa.get_spectral_score(beta, top_k=10)
                dS = new_smooth - cur_smooth
                dS_spec = new_spectral - cur_spectral
                if dS <= 0 and dS_spec <= 0:
                    accept = True
                elif dS <= 0 or dS_spec <= 0:
                    accept = rng.random() < 0.7
                else:
                    accept = rng.random() < np.exp(-(dS + dS_spec) / T)
                
                if accept:
                    new_hard_exact = fa.get_score()
                    if new_hard_exact < cur_hard:
                        cur_hard = new_hard_exact
                    cur_smooth = new_smooth
                    cur_spectral = new_spectral
                    if new_hard_exact < best_hard:
                        best_hard = new_hard_exact
                        best_h = fa.h.copy()
                else:
                    fa.flip_bit(i1)
                    fa.flip_bit(i2)
            else:
                # Single bit flip
                i = int(rng.integers(0, N))
                fa.flip_bit(i)
                new_hard_approx = fa.get_score_approx()
                new_smooth = fa.get_smooth_score(beta)
                new_spectral = fa.get_spectral_score(beta, top_k=10)
                dS = new_smooth - cur_smooth
                dS_spec = new_spectral - cur_spectral
                if dS <= 0 and dS_spec <= 0:
                    accept = True
                elif dS <= 0 or dS_spec <= 0:
                    accept = rng.random() < 0.7
                else:
                    accept = rng.random() < np.exp(-(dS + dS_spec) / T)
                
                if accept:
                    new_hard_exact = fa.get_score()
                    if new_hard_exact < cur_hard:
                        cur_hard = new_hard_exact
                    cur_smooth = new_smooth
                    cur_spectral = new_spectral
                    if new_hard_exact < best_hard:
                        best_hard = new_hard_exact
                        best_h = fa.h.copy()
                else:
                    fa.flip_bit(i)
            T = max(T_min, T * decay)

        # ---------- greedy multi-scale local polish (optimized) ----------
        h = best_h.copy()
        sizes_polish = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 96, 112, 128, 144, 160, 192, 224, 256]
        fa = _FastAutocorr(h)
        for _pass in range(80):
            if time.time() > deadline - 3.0:
                break
            any_improve = False
            for size in sizes_polish:
                for start in range(0, N - size + 1):
                    if time.time() > deadline - 3.0:
                        break
                    fa.flip_block(start, size)
                    new_hard = fa.get_score()
                    if new_hard < best_hard - 1e-10:
                        best_hard = new_hard
                        best_h = fa.h.copy()
                        any_improve = True
                    else:
                        fa.unflip_block(start, size)
                if time.time() > deadline - 3.0:
                    break
            if not any_improve:
                break
        # Final verification with exact score
        best_hard = _hard_score(best_h)

        # ---------- final single-bit polish (optimized) ----------
        h = best_h.copy()
        fa = _FastAutocorr(h)
        for _pass in range(300):
            if time.time() > deadline - 0.5:
                break
            any_improve = False
            for i in range(N):
                if time.time() > deadline - 0.5:
                    break
                fa.flip_bit(i)
                new_hard = fa.get_score()
                if new_hard < best_hard - 1e-10:
                    best_hard = new_hard
                    best_h = fa.h.copy()
                    any_improve = True
                else:
                    fa.flip_bit(i)
            if not any_improve:
                break
        # Final verification with exact score
        best_hard = _hard_score(best_h)

        # ---------- final two-bit polish ----------
        h = best_h.copy()
        fa = _FastAutocorr(h)
        for _pass in range(80):
            if time.time() > deadline - 0.1:
                break
            any_improve = False
            for i in range(N):
                if time.time() > deadline - 0.1:
                    break
                for j in range(i+1, min(i+30, N)):  # Local pairs only
                    fa.flip_bit(i)
                    fa.flip_bit(j)
                    new_hard = fa.get_score()
                    if new_hard < best_hard - 1e-10:
                        best_hard = new_hard
                        best_h = fa.h.copy()
                        any_improve = True
                        fa.flip_bit(i)  # Undo j flip to continue
                        break
                    else:
                        fa.flip_bit(j)
                        fa.flip_bit(i)
                if time.time() > deadline - 0.1:
                    break
            if not any_improve:
                break
        # Final verification with exact score
        best_hard = _hard_score(best_h)

        # ---------- final three-bit polish on local neighborhoods ----------
        h = best_h.copy()
        fa = _FastAutocorr(h)
        for _pass in range(40):
            if time.time() > deadline - 0.05:
                break
            any_improve = False
            for i in range(N - 2):
                if time.time() > deadline - 0.05:
                    break
                # Try flipping triples (i, i+1, i+2)
                fa.flip_bit(i)
                fa.flip_bit(i + 1)
                fa.flip_bit(i + 2)
                new_hard = fa.get_score()
                if new_hard < best_hard - 1e-10:
                    best_hard = new_hard
                    best_h = fa.h.copy()
                    any_improve = True
                    fa.flip_bit(i)
                    fa.flip_bit(i + 1)
                    fa.flip_bit(i + 2)
                else:
                    fa.flip_bit(i)
                    fa.flip_bit(i + 1)
                    fa.flip_bit(i + 2)
            if not any_improve:
                break
        # Final verification with exact score
        best_hard = _hard_score(best_h)

        return best_h
    except Exception:
        return np.ones(N)
