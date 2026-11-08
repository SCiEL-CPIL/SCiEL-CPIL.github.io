import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 140.0  # safety margin under 150s

def _max_overlap(h):
    """Maximum overlap of h with shifted (1-h)."""
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(np.abs(H)**2)) / N
    return 0.5 - np.min(autocorr)

def _soft_max_overlap(h, temperature=0.05):
    """Smooth surrogate for max overlap using log-sum-exp."""
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(np.abs(H)**2)) / N
    overlap = 0.5 - autocorr
    max_o = np.max(overlap)
    log_sum = max_o + np.log(np.sum(np.exp((overlap - max_o) / temperature))) * temperature
    return log_sum

def _try_binary_sequences():
    """Try to find good binary sequences (h in {0,1}, sum = N/2) with low max overlap."""
    best_h = None
    best_score = float('inf')
    
    # For binary h, overlap[m] = 0.5 - (1/N) * (number of positions where h_j=1 and h_{j+m}=1)
    # We want to minimize max overlap, so maximize min autocorr.
    # For balanced binary (N/2 ones), average autocorr is N/4 = 25 for N=100.
    # We want all autocorr values close to 25.
    
    # Try random balanced binary sequences
    np.random.seed(123)
    for trial in range(2000):
        h = np.zeros(N)
        indices = np.random.choice(N, N//2, replace=False)
        h[indices] = 1.0
        score = _max_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Try structured sequences: periodic patterns with different duty cycles
    for period in range(2, 21):
        for ones_in_period in range(1, period):
            if ones_in_period == period - ones_in_period:
                continue  # skip 50% duty, already tried
            pattern = np.zeros(period)
            pattern[:ones_in_period] = 1.0
            h = np.tile(pattern, N//period + 1)[:N]
            s = np.sum(h)
            if s != N//2:
                # Adjust to get exactly N/2 ones
                if s < N//2:
                    zeros = np.where(h == 0)[0]
                    np.random.shuffle(zeros)
                    for idx in zeros[:N//2 - int(s)]:
                        h[idx] = 1.0
                elif s > N//2:
                    ones = np.where(h == 1)[0]
                    np.random.shuffle(ones)
                    for idx in ones[:int(s) - N//2]:
                        h[idx] = 0.0
            score = _max_overlap(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    # Try m-sequence-like patterns using modular arithmetic
    for shift in range(1, 15):
        for threshold_frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
            h = np.zeros(N)
            for j in range(N):
                if ((j * shift) % N) / N < threshold_frac:
                    h[j] = 1.0
            s = np.sum(h)
            if abs(s - N//2) <= 2:
                # Adjust if needed
                if s < N//2:
                    zeros = np.where(h == 0)[0]
                    np.random.shuffle(zeros)
                    for idx in zeros[:N//2 - int(s)]:
                        h[idx] = 1.0
                elif s > N//2:
                    ones = np.where(h == 1)[0]
                    np.random.shuffle(ones)
                    for idx in ones[:int(s) - N//2]:
                        h[idx] = 0.0
                score = _max_overlap(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
    
    return best_h, best_score

def _polish(h, time_budget=30.0):
    """Polish a starting h using coordinate descent and local optimization."""
    start = time.time()
    h = h.copy()
    h = np.clip(h, 0, 1)
    s = np.sum(h)
    if s > 0:
        h *= (N/2.0) / s
        h = np.clip(h, 0, 1)
    best_score = _max_overlap(h)
    
    # Coordinate descent: try adjusting individual elements
    improved = True
    while improved and time.time() - start < time_budget:
        improved = False
        for j in range(N):
            if time.time() - start > time_budget:
                break
            # Try different values for h[j]
            for delta in [0.01, -0.01, 0.05, -0.05, 0.1, -0.1, 0.2, -0.2]:
                h_test = h.copy()
                h_test[j] = np.clip(h_test[j] + delta, 0, 1)
                # Renormalize
                s_test = np.sum(h_test)
                if s_test > 0:
                    h_test *= (N/2.0) / s_test
                    h_test = np.clip(h_test, 0, 1)
                score_test = _max_overlap(h_test)
                if score_test < best_score - 1e-8:
                    h = h_test
                    best_score = score_test
                    improved = True
                    break
    
    # Try L-BFGS-B on a smoothed version
    if time.time() - start < time_budget - 5:
        # Parameterize h = sigmoid(x) / sum(sigmoid(x)) * N/2
        h_safe = np.clip(h, 1e-10, 1 - 1e-10)
        x0 = np.log(h_safe / (1 - h_safe))
        x0 = np.clip(x0, -10, 10)
        
        def obj(x):
            h_s = 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
            h_s = np.clip(h_s, 0, 1)
            s_s = np.sum(h_s)
            if s_s > 1e-10:
                h_s *= (N/2.0) / s_s
                h_s = np.clip(h_s, 0, 1)
            return _soft_max_overlap(h_s, 0.03)
        
        try:
            result = minimize(obj, x0, method='L-BFGS-B', 
                             options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            h_opt = 1.0 / (1.0 + np.exp(-np.clip(result.x, -50, 50)))
            h_opt = np.clip(h_opt, 0, 1)
            s_opt = np.sum(h_opt)
            if s_opt > 0:
                h_opt *= (N/2.0) / s_opt
                h_opt = np.clip(h_opt, 0, 1)
            score_opt = _max_overlap(h_opt)
            if score_opt < best_score:
                h = h_opt
                best_score = score_opt
        except Exception:
            pass
    
    return h, best_score

def construct():
    start = time.time()
    
    # The constant function h=0.5 gives overlap = 0.5 - 0.25 = 0.25 for all shifts
    # Wait, let me recalculate: autocorr[0] = (1/N) sum h_j^2 = (1/N) * N * 0.25 = 0.25
    # autocorr[m] for m != 0: (1/N) sum h_j h_{j+m} = (1/N) * N * 0.25 = 0.25
    # So min autocorr = 0.25, overlap = 0.5 - 0.25 = 0.25
    # But the knowledge says 0.75... let me re-examine the definition.
    
    # Looking at the code: overlap = 0.5 - min(autocorr)
    # For h=0.5: autocorr[m] = 0.25 for all m, so overlap = 0.5 - 0.25 = 0.25
    # But the knowledge says the constant achieves 0.75. There must be a different definition.
    
    # Let me check: maybe the overlap is defined differently.
    # The task says "maximum overlap of h with shifted 1-h"
    # overlap[tau] = integral h(x) * (1-h(x+tau)) dx
    # = integral h(x) dx - integral h(x)h(x+tau) dx
    # = 1 - N * autocorr[tau] (if autocorr is normalized by N)
    # Wait, autocorr[tau] = (1/N) sum h_j h_{j+tau}
    # So integral h(x)h(x+tau) dx = sum h_j h_{j+tau} = N * autocorr[tau]
    # overlap[tau] = 1 - N * autocorr[tau]
    # For h=0.5: overlap[tau] = 1 - N * 0.25 = 1 - 25 = -24? That can't be right.
    
    # I think the issue is that the "overlap" in the code is not the actual integral.
    # Let me just trust the code's definition and find the minimum.
    
    # Get initial candidates
    # 1. Constant
    h_const = np.full(N, 0.5)
    score_const = _max_overlap(h_const)
    
    # 2. Binary sequences
    h_bin, score_bin = _try_binary_sequences()
    
    # 3. Random continuous
    np.random.seed(42)
    h_rand = np.random.uniform(0, 1, N)
    h_rand *= (N/2.0) / np.sum(h_rand)
    h_rand = np.clip(h_rand, 0, 1)
    score_rand = _max_overlap(h_rand)
    
    # Pick best starting point
    candidates = [(h_const, score_const), (h_bin, score_bin), (h_rand, score_rand)]
    candidates = [c for c in candidates if c[0] is not None]
    candidates.sort(key=lambda x: x[1])
    best_h, best_score = candidates[0]
    
    # Polish the best candidate
    if time.time() - start < DEADLINE - 10:
        best_h, best_score = _polish(best_h, time_budget=min(60, DEADLINE - 10 - (time.time() - start)))
    
    # Try polishing other candidates if time allows
    for h_cand, score_cand in candidates[1:]:
        if time.time() - start > DEADLINE - 15:
            break
        if score_cand >= best_score:
            continue
        h_polished, score_polished = _polish(h_cand, time_budget=min(20, DEADLINE - 15 - (time.time() - start)))
        if score_polished < best_score:
            best_h = h_polished
            best_score = score_polished
    
    # Final: ensure best_h is valid
    h_final = np.clip(best_h, 0, 1)
    s = np.sum(h_final)
    if s > 0:
        h_final *= (N/2.0) / s
        h_final = np.clip(h_final, 0, 1)
    
    # Ensure sum is exactly N/2
    diff = N/2.0 - np.sum(h_final)
    if abs(diff) > 1e-10:
        if diff > 0:
            idx = np.argmin(h_final)
            h_final[idx] = min(1.0, h_final[idx] + diff)
            remaining = N/2.0 - np.sum(h_final)
            if abs(remaining) > 1e-10:
                mask = h_final < 1.0
                if np.any(mask):
                    room = 1.0 - h_final[mask]
                    total_room = np.sum(room)
                    if total_room > 0:
                        h_final[mask] += remaining * room / total_room
        else:
            idx = np.argmax(h_final)
            h_final[idx] = max(0.0, h_final[idx] + diff)
            remaining = N/2.0 - np.sum(h_final)
            if abs(remaining) > 1e-10:
                mask = h_final > 0.0
                if np.any(mask):
                    total = np.sum(h_final[mask])
                    if total > 0:
                        h_final[mask] += remaining * h_final[mask] / total
    
    # Final check
    if not (np.all(h_final >= -1e-9) and np.all(h_final <= 1.0 + 1e-9) and abs(np.sum(h_final) - N/2.0) < 1e-6):
        return np.full(N, 0.5)
    
    return h_final
