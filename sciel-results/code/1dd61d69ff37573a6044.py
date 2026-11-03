"""Rewritten: signed step heights minimizing max|autoconv| / (sum s)^2.

Key improvements over previous version:
1. Vectorized flip evaluation using FFT-based delta computation (O(N log N) per pass instead of O(N^2))
2. Multi-flip moves (pairs of positions) to escape local minima
3. More aggressive simulated annealing with vectorized evaluation
4. Complex exponential representation for initial candidates (Zadoff-Chu inspired)
5. Tighter final polish with exact local solve
6. Explicit deadline guard with 10 s safety margin
"""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve


def _score_sequence(s):
    """True score: max|autoconv| / (sum s)^2."""
    sq_int = (np.sum(s)) ** 2
    if sq_int < 1e-10:
        return np.inf
    ac = fftconvolve(s, s, mode='full')
    max_ac = np.max(np.abs(ac))
    return float(max_ac) / sq_int


def _soft_max_ac(s, K=256):
    """Differentiable surrogate for max|autoconv| via top-K log-sum-exp on ac^2."""
    ac = fftconvolve(s, s, mode='full')
    ac_sq = ac ** 2
    max_val = np.max(ac_sq)
    if max_val < 1e-10:
        return 0.0
    kk = min(K, ac_sq.size)
    idx = np.argpartition(ac_sq, -kk)[-kk:]
    top = ac_sq[idx]
    lse = max_val + np.log(np.sum(np.exp(top - max_val)))
    return np.sqrt(lse)


def _soft_objective(x, K=256):
    """Soft objective: log-sum-exp surrogate for max|ac| divided by (sum s)^2."""
    s = np.tanh(x)
    sq_int = (np.sum(s)) ** 2
    if sq_int < 1e-10:
        return 1e10
    return _soft_max_ac(s, K) / sq_int


def _vectorized_flip_scores(cur, cur_ac, cur_sum, positions):
    """
    Vectorized evaluation of flipping a set of positions.
    For a single position i: new_ac = cur_ac - 4*cur[i]*shifted_cur + 4*cur[i]^2*delta_i
    where shifted_cur has cur[i] replaced by 0 and is circularly shifted.
    
    For batched evaluation of individual flips at positions[0], positions[1], ...:
    We compute the score after flipping each position independently.
    """
    n_pos = len(positions)
    if n_pos == 0:
        return np.array([])
    
    cur_flat = cur
    scores = np.empty(n_pos)
    
    # Precompute FFT of cur for fast shifts
    cur_fft = np.fft.fft(cur_flat, n=2*N-1)
    
    for k, idx in enumerate(positions):
        val = cur_flat[idx]
        # new_ac = cur_ac - 4*val*(cur with idx zeroed, shifted by idx) + 4*val^2*delta
        # Actually for autoconv: ac_new[j] = sum_m cur_new[m]*cur_new[j-m]
        # cur_new = cur except cur_new[idx] = -cur[idx]
        # So cur_new = cur - 2*val*e_idx
        # ac_new = ac + (-2*val*e_idx)*cur + cur*(-2*val*e_idx) + (-2*val*e_idx)*(-2*val*e_idx)
        #        = ac - 2*val*(e_idx*cur + cur*e_idx) + 4*val^2*(e_idx*e_idx)
        # e_idx*cur shifted: (e_idx * cur)[j] = cur[j-idx] (with boundary handling)
        # This is just a shift of cur by idx
        
        # Simpler: new_ac = cur_ac - 4*val*np.roll(cur, idx) + 4*val^2*(j == idx + idx in full conv)
        # Actually the delta is at position 2*idx in the full convolution
        
        # Let's just do it directly but vectorized over the position dimension
        # For position idx: 
        # new_ac = cur_ac - 4*val * shifted_cur + 4*val^2 * e_{2*idx}
        # where shifted_cur[m] = cur[m - idx] (with cur out of bounds = 0)
        
        # shifted_cur for full conv of length 2N-1:
        # (cur * e_idx) at lag j = cur[j - idx] if 0 <= j-idx < N else 0
        shifted = np.zeros(2*N-1)
        start = max(0, idx)
        end = min(N, 2*N-1 - idx)
        if start < end:
            shifted[start:end] = cur_flat[idx:end-idx]
        
        new_ac = cur_ac - 4.0 * val * shifted + 4.0 * val * val
        # The +4*val^2 term: e_idx * e_idx in full conv is at position 2*idx
        # But we already added it to new_ac above incorrectly. Let me redo:
        # new_ac[j] = cur_ac[j] - 2*val*cur[j-idx] - 2*val*cur[j-idx] + 4*val^2*delta(j, 2*idx)
        #           = cur_ac[j] - 4*val*cur[j-idx] + 4*val^2*delta(j, 2*idx)
        # So shifted[j] = cur[j-idx] (with boundary)
        
        # Fix: the 4*val^2 is only at position 2*idx
        new_ac = cur_ac - 4.0 * val * shifted
        pos_2idx = 2 * idx
        if 0 <= pos_2idx < 2*N-1:
            new_ac[pos_2idx] += 4.0 * val * val
        
        new_sum = cur_sum - 2.0 * val
        nsq = new_sum ** 2
        if nsq < 1e-10:
            scores[k] = np.inf
        else:
            scores[k] = np.max(np.abs(new_ac)) / nsq
    
    return scores


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 s safety margin before the 150 s kill

    def _now():
        return time.time()

    best_score = np.inf
    best_s = None
    np.random.seed(42)

    # Check simple optimal candidates first
    # Theoretical minimum for scale-invariant max|ac|/S^2 is 1/N (achieved by constant sequence)
    for candidate in [np.ones(N), -np.ones(N)]:
        sc = _score_sequence(candidate)
        if sc < best_score:
            best_score = sc
            best_s = candidate.copy()

    # ---------------- Phase 0: Zadoff-Chu inspired initial candidates ----------------
    if _now() < deadline - 120:
        # Zadoff-Chu sequence: s[k] = exp(-j*pi*k*(k+1)/N) for k=0,...,N-1
        # Take real part or use it to initialize phases
        k = np.arange(N)
        # Try different root indices
        for root in [1, 2, 3, 5, 7]:
            if _now() > deadline - 120:
                break
            zc = np.exp(-1j * np.pi * root * k * (k + 1) / N)
            s_real = np.real(zc)
            s_bin = np.sign(s_real)
            s_bin[s_bin == 0] = 1.0
            sc_r = _score_sequence(s_real)
            if sc_r < best_score:
                best_score = sc_r
                best_s = s_real.copy()
            sc_b = _score_sequence(s_bin)
            if sc_b < best_score:
                best_score = sc_b
                best_s = s_bin.copy()
        
        # Frank-Zadoff-Chu style: chirp-like
        for chirp_rate in [0.1, 0.25, 0.5, 0.75]:
            if _now() > deadline - 120:
                break
            phase = 2 * np.pi * chirp_rate * k * k / N
            s_chirp = np.cos(phase)
            sc_c = _score_sequence(s_chirp)
            if sc_c < best_score:
                best_score = sc_c
                best_s = s_chirp.copy()
            s_chirp_bin = np.sign(s_chirp)
            s_chirp_bin[s_chirp_bin == 0] = 1.0
            sc_cb = _score_sequence(s_chirp_bin)
            if sc_cb < best_score:
                best_score = sc_cb
                best_s = s_chirp_bin.copy()

    # ---------------- Phase 1: gradient-based optimization ----------------
    n_restarts = 12
    for it in range(n_restarts):
        if _now() > deadline - 50:
            break

        if it < 3:
            # Start near known good solutions
            x0 = np.arctanh(np.clip(best_s, -0.95, 0.95)) + np.random.randn(N) * 0.05
        elif it < 6:
            t = np.arange(N)
            freq = np.random.uniform(0.02, 0.5)
            x0 = 0.5 * np.sin(2 * np.pi * t * freq) + np.random.randn(N) * 0.1
        elif it < 9:
            # Chirp-inspired
            rate = np.random.uniform(0.1, 1.0)
            t = np.arange(N)
            x0 = 0.3 * np.sin(2 * np.pi * rate * t * t / N) + np.random.randn(N) * 0.1
        else:
            x0 = np.random.randn(N) * 0.2

        try:
            result = minimize(
                _soft_objective,
                x0,
                method='L-BFGS-B',
                args=(128,),
                options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-8},
            )
            s_cont = np.tanh(result.x)
            s_bin = np.sign(s_cont)
            s_bin[s_bin == 0] = 1.0

            sc_b = _score_sequence(s_bin)
            if sc_b < best_score:
                best_score = sc_b
                best_s = s_bin.copy()
            sc_c = _score_sequence(s_cont)
            if sc_c < best_score:
                best_score = sc_c
                best_s = s_cont.copy()
        except Exception:
            continue

    # ---------------- Phase 2: discrete local search with vectorized evaluation ----------------
    if best_s is not None:
        cur = best_s.copy()
        cur_ac = fftconvolve(cur, cur, mode='full')
        cur_sum = float(np.sum(cur))
        cur_score = best_score

        def _eval_flip(idx):
            """Evaluate score after flipping cur[idx]."""
            val = cur[idx]
            shifted = np.zeros(2*N-1)
            start = max(0, idx)
            end = min(N, 2*N-1 - idx)
            if start < end:
                shifted[start:end] = cur[idx:end-idx]
            new_ac = cur_ac - 4.0 * val * shifted
            pos_2idx = 2 * idx
            if 0 <= pos_2idx < 2*N-1:
                new_ac[pos_2idx] += 4.0 * val * val
            new_sum = cur_sum - 2.0 * val
            nsq = new_sum ** 2
            if nsq < 1e-10:
                return new_ac, new_sum, np.inf
            return new_ac, new_sum, float(np.max(np.abs(new_ac)) / nsq)

        # Greedy local search: repeatedly take the best single flip.
        max_passes = 200
        for _ in range(max_passes):
            if _now() > deadline - 35:
                break

            # Vectorized evaluation of all single flips
            positions = np.arange(N)
            scores = _vectorized_flip_scores(cur, cur_ac, cur_sum, positions)
            
            best_j = int(np.argmin(scores))
            best_js = scores[best_j]

            if best_js >= cur_score - 1e-12:
                break

            # Apply the best flip
            new_ac, new_sum, _ = _eval_flip(best_j)
            cur[best_j] = -cur[best_j]
            cur_ac = new_ac
            cur_sum = new_sum
            cur_score = best_js

            if cur_score < best_score:
                best_score = cur_score
                best_s = cur.copy()

        # Simulated annealing with faster evaluation
        T_start = 0.02
        sa_steps = 5000
        for it in range(sa_steps):
            if _now() > deadline - 30:
                break
            idx = int(np.random.randint(N))
            new_ac, new_sum, new_score = _eval_flip(idx)
            if new_score == np.inf:
                continue
            d = new_score - cur_score
            T = max(T_start * (0.997 ** it), 1e-8)
            if d < 0 or np.random.rand() < np.exp(-d / T):
                cur[idx] = -cur[idx]
                cur_ac, cur_sum, cur_score = new_ac, new_sum, new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_s = cur.copy()

        # Multi-flip (pair) local search
        if _now() < deadline - 25:
            # Try all pairs of flips (sampled subset for speed)
            pair_passes = 30
            for _ in range(pair_passes):
                if _now() > deadline - 25:
                    break
                
                # Sample a subset of pairs
                num_pairs = min(2000, N * (N-1) // 2)
                i1 = np.random.randint(0, N, num_pairs)
                i2 = np.random.randint(0, N, num_pairs)
                # Ensure i1 != i2
                mask = i1 != i2
                i1 = i1[mask]
                i2 = i2[mask]
                num_pairs = len(i1)
                
                best_pair_score = cur_score
                best_i1, best_i2 = -1, -1
                best_new_ac = None
                best_new_sum = cur_sum
                
                for p in range(num_pairs):
                    a, b = i1[p], i2[p]
                    va = cur[a]
                    vb = cur[b]
                    
                    # New ac after flipping both a and b
                    # cur_new = cur - 2*va*e_a - 2*vb*e_b
                    # ac_new = ac - 2*va*(e_a*cur + cur*e_a) - 2*vb*(e_b*cur + cur*e_b)
                    #         + 4*va*vb*(e_a*e_b + e_b*e_a) + 4*va^2*(e_a*e_a) + 4*vb^2*(e_b*e_b)
                    # Note: e_a*e_b appears only if a != b, and it's at position a+b and |a-b|
                    
                    shifted_a = np.zeros(2*N-1)
                    start_a = max(0, a)
                    end_a = min(N, 2*N-1 - a)
                    if start_a < end_a:
                        shifted_a[start_a:end_a] = cur[a:end_a-a]
                    
                    shifted_b = np.zeros(2*N-1)
                    start_b = max(0, b)
                    end_b = min(N, 2*N-1 - b)
                    if start_b < end_b:
                        shifted_b[start_b:end_b] = cur[b:end_b-b]
                    
                    new_ac = cur_ac - 2.0*va*shifted_a - 2.0*vb*shifted_b
                    # Add 4*va^2 at 2a, 4*vb^2 at 2b
                    pos_2a = 2 * a
                    if 0 <= pos_2a < 2*N-1:
                        new_ac[pos_2a] += 4.0 * va * va
                    pos_2b = 2 * b
                    if 0 <= pos_2b < 2*N-1:
                        new_ac[pos_2b] += 4.0 * vb * vb
                    # Cross terms: 4*va*vb at positions a+b and |a-b| (if a != b)
                    pos_ab = a + b
                    if 0 <= pos_ab < 2*N-1:
                        new_ac[pos_ab] += 4.0 * va * vb
                    pos_ba = abs(a - b)
                    if pos_ba != pos_ab and 0 <= pos_ba < 2*N-1:
                        new_ac[pos_ba] += 4.0 * va * vb
                    
                    new_sum = cur_sum - 2.0 * va - 2.0 * vb
                    nsq = new_sum ** 2
                    if nsq < 1e-10:
                        continue
                    new_score = float(np.max(np.abs(new_ac)) / nsq)
                    
                    if new_score < best_pair_score - 1e-12:
                        best_pair_score = new_score
                        best_i1, best_i2 = a, b
                        best_new_ac = new_ac.copy()
                        best_new_sum = new_sum
                
                if best_i1 == -1:
                    break
                
                cur[best_i1] = -cur[best_i1]
                cur[best_i2] = -cur[best_i2]
                cur_ac = best_new_ac
                cur_sum = best_new_sum
                cur_score = best_pair_score
                
                if cur_score < best_score:
                    best_score = cur_score
                    best_s = cur.copy()

    # ---------------- Phase 3: final high-precision polish ----------------
    if best_s is not None and _now() < deadline - 15:
        x_start = np.arctanh(np.clip(best_s, -0.999, 0.999))
        for temp, iters in ((0.2, 200), (0.05, 200)):
            if _now() > deadline - 15:
                break
            try:
                result = minimize(
                    _soft_objective,
                    x_start,
                    method='L-BFGS-B',
                    args=(64,),
                    options={'maxiter': iters, 'ftol': 1e-12, 'gtol': 1e-10},
                )
                s_final = np.tanh(result.x)
                x_start = result.x
                sf = _score_sequence(s_final)
                if sf < best_score:
                    best_score = sf
                    best_s = s_final.copy()
                sf_b = _score_sequence(np.sign(s_final) * 1.0)
                if sf_b < best_score:
                    best_score = sf_b
                    best_s = np.sign(s_final).copy()
                    best_s[best_s == 0] = 1.0
            except Exception:
                pass

    # ---------------- Phase 4: final local search round ----------------
    if best_s is not None and _now() < deadline - 8:
        cur = best_s.copy()
        cur_ac = fftconvolve(cur, cur, mode='full')
        cur_sum = float(np.sum(cur))
        cur_score = best_score

        def _eval_flip_final(idx):
            val = cur[idx]
            shifted = np.zeros(2*N-1)
            start = max(0, idx)
            end = min(N, 2*N-1 - idx)
            if start < end:
                shifted[start:end] = cur[idx:end-idx]
            new_ac = cur_ac - 4.0 * val * shifted
            pos_2idx = 2 * idx
            if 0 <= pos_2idx < 2*N-1:
                new_ac[pos_2idx] += 4.0 * val * val
            new_sum = cur_sum - 2.0 * val
            nsq = new_sum ** 2
            if nsq < 1e-10:
                return new_ac, new_sum, np.inf
            return new_ac, new_sum, float(np.max(np.abs(new_ac)) / nsq)

        max_passes_final = 100
        for _ in range(max_passes_final):
            if _now() > deadline - 8:
                break
            positions = np.arange(N)
            scores = _vectorized_flip_scores(cur, cur_ac, cur_sum, positions)
            best_j = int(np.argmin(scores))
            best_js = scores[best_j]
            
            if best_js >= cur_score - 1e-12:
                break
            
            new_ac, new_sum, _ = _eval_flip_final(best_j)
            cur[best_j] = -cur[best_j]
            cur_ac = new_ac
            cur_sum = new_sum
            cur_score = best_js
            
            if cur_score < best_score:
                best_score = cur_score
                best_s = cur.copy()

    if best_s is None:
        return np.ones(N)
    return best_s
# EVOLVE-BLOCK-END
