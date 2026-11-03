"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve


def _fft_size_for(n):
    """Return next power of 2 >= n."""
    size = 1
    while size < n:
        size <<= 1
    return size


def _autoconv_fft(s):
    """Compute full autoconvolution via FFT."""
    n = len(s)
    size = _fft_size_for(2 * n - 1)
    f = np.fft.rfft(s, size)
    ac = np.fft.irfft(f * f, size)[:2 * n - 1]
    return ac


def _score_sequence(s):
    """True score: max|autoconv| / (sum s)^2."""
    sq_int = (np.sum(s)) ** 2
    if sq_int < 1e-10:
        return np.inf
    ac = _autoconv_fft(s)
    max_ac = np.max(np.abs(ac))
    return max_ac / sq_int


def _score_batch(s_batch):
    """Score a batch of sequences. Returns array of scores."""
    scores = np.zeros(len(s_batch))
    for i, s in enumerate(s_batch):
        scores[i] = _score_sequence(s)
    return scores


def construct():
    start_time = time.time()
    deadline = start_time + 143.0  # 7 s safety margin before the 150 s kill

    def _now():
        return time.time()

    best_score = np.inf
    best_s = None
    np.random.seed(42)

    # ---------------- Phase 1: Generate diverse candidate sequences ----------------
    candidates = []

    # All ones
    candidates.append(np.ones(N))

    # Random binary sequences
    for _ in range(50):
        if _now() > deadline - 40:
            break
        s = np.random.choice([-1.0, 1.0], size=N)
        candidates.append(s)

    # Low-frequency sinusoidal patterns
    for freq in np.linspace(0.01, 0.5, 20):
        if _now() > deadline - 40:
            break
        t = np.arange(N)
        s = np.sign(np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2 * np.pi)))
        s[s == 0] = 1.0
        candidates.append(s)

    # Multi-frequency combinations
    for _ in range(30):
        if _now() > deadline - 40:
            break
        t = np.arange(N)
        f1 = np.random.uniform(0.01, 0.3)
        f2 = np.random.uniform(0.01, 0.3)
        phase1 = np.random.uniform(0, 2 * np.pi)
        phase2 = np.random.uniform(0, 2 * np.pi)
        s = np.sign(np.sin(2 * np.pi * f1 * t + phase1) + np.sin(2 * np.pi * f2 * t + phase2))
        s[s == 0] = 1.0
        candidates.append(s)

    # Chirp signals
    for _ in range(20):
        if _now() > deadline - 40:
            break
        t = np.arange(N)
        f0 = np.random.uniform(0.01, 0.1)
        f1 = np.random.uniform(0.1, 0.5)
        phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * N))
        s = np.sign(np.sin(phase))
        s[s == 0] = 1.0
        candidates.append(s)

    # Evaluate all candidates
    for s in candidates:
        if _now() > deadline - 30:
            break
        sc = _score_sequence(s)
        if sc < best_score:
            best_score = sc
            best_s = s.copy()

    # ---------------- Phase 2: Local search on best candidate ----------------
    if best_s is not None:
        cur = best_s.copy()
        cur_ac = _autoconv_fft(cur)
        cur_sum = float(np.sum(cur))
        cur_score = best_score

        def _try_flip(idx):
            """Return (new_ac, new_sum, new_score) after flipping cur[idx]."""
            val = cur[idx]
            delta = -2.0 * val
            new_ac = cur_ac.copy()
            start = idx
            end = idx + N
            new_ac[start:end] += -4.0 * val * cur
            new_ac[idx] += 4.0 * val * val
            new_sum = cur_sum + delta
            nsq = new_sum ** 2
            if nsq < 1e-10:
                return new_ac, new_sum, np.inf
            return new_ac, new_sum, float(np.max(np.abs(new_ac)) / nsq)

        # Greedy local search with multiple passes
        max_passes = 200
        for _ in range(max_passes):
            if _now() > deadline - 20:
                break

            best_j = -1
            best_js = cur_score
            best_new_ac = None
            best_new_sum = cur_sum

            for j in range(N):
                new_ac, new_sum, new_score = _try_flip(j)
                if new_score < best_js:
                    best_js = new_score
                    best_j = j
                    best_new_ac = new_ac
                    best_new_sum = new_sum

            if best_j == -1 or best_js >= cur_score - 1e-12:
                break

            cur[best_j] = -cur[best_j]
            cur_ac = best_new_ac
            cur_sum = best_new_sum
            cur_score = best_js

            if cur_score < best_score:
                best_score = cur_score
                best_s = cur.copy()

        # Simulated annealing
        T_start = 0.03
        sa_steps = 3000
        for it in range(sa_steps):
            if _now() > deadline - 15:
                break
            idx = int(np.random.randint(N))
            new_ac, new_sum, new_score = _try_flip(idx)
            if new_score == np.inf:
                continue
            d = new_score - cur_score
            T = max(T_start * (0.995 ** it), 1e-6)
            if d < 0 or np.random.rand() < np.exp(-d / T):
                cur[idx] = -cur[idx]
                cur_ac, cur_sum, cur_score = new_ac, new_sum, new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_s = cur.copy()

    # If we didn't find anything, return all ones
    if best_s is None:
        return np.ones(N)

    return best_s
# EVOLVE-BLOCK-END
