import numpy as np
import time

N = 400

FFT_SIZE = 1
while FFT_SIZE < 2 * N - 1:
    FFT_SIZE *= 2

_k = np.arange(FFT_SIZE // 2 + 1)
EXP_FACTORS = np.exp(-2j * np.pi * np.outer(np.arange(N), _k) / FFT_SIZE)


def _autoconv(x):
    X = np.fft.rfft(x, FFT_SIZE)
    ac = np.fft.irfft(X * np.conj(X), FFT_SIZE)[:2 * N - 1]
    return ac


def _score(x):
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    ac = _autoconv(x)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)


def _ac_from_fft(X):
    return np.fft.irfft(X * np.conj(X), FFT_SIZE)[:2 * N - 1]


def construct():
    start_time = time.time()
    deadline = start_time + 138.0

    # ---------- Phase 0: Sparse random signed subsets ----------
    rng = np.random.default_rng(7)
    best_x = np.ones(N, dtype=np.float64)
    best_score = _score(best_x)

    # Try a range of densities; higher density gives smaller peak/sum^2
    densities = [0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.90, 0.88, 0.85]
    for p in densities:
        if time.time() > deadline - 100:
            break
        # Bernoulli mask with random signs, scaled to unit magnitude
        for _ in range(40):
            mask = rng.random(N) < p
            x = np.zeros(N, dtype=np.float64)
            if mask.any():
                x[mask] = rng.choice([-1.0, 1.0], size=int(mask.sum()))
            # Ensure non-zero sum
            if abs(np.sum(x)) < 1e-9:
                continue
            sc = _score(x)
            if sc < best_score:
                best_score = sc
                best_x = x.copy()

    # ---------- Phase 1: Local search on sparse set (add/remove/flip) ----------
    x = best_x.copy()
    cur_score = best_score
    X = np.fft.rfft(x, FFT_SIZE)
    s = np.sum(x)

    def _flip_score(x, i, X, s):
        old_val = x[i]
        if old_val == 0.0:
            # add a new ±1
            new_val = 1.0
            delta = new_val
        else:
            new_val = -old_val
            delta = -2.0 * old_val
        new_s = s + delta
        if abs(new_s) < 1e-12:
            return float('inf'), X, new_s
        X_new = X + delta * EXP_FACTORS[i]
        ac = _ac_from_fft(X_new)
        max_ac = np.max(np.abs(ac))
        return max_ac / (new_s * new_s), X_new, new_s

    # Simulated annealing over sparse moves
    rng2 = np.random.default_rng(99)
    T = 0.02
    T_min = 1e-7
    cooling = 0.9995
    no_improve = 0
    max_no_improve = 4000

    while time.time() < deadline - 25 and no_improve < max_no_improve:
        i = int(rng2.integers(0, N))
        old_val = x[i]
        # Decide move: flip sign or toggle presence
        if old_val == 0.0:
            x[i] = 1.0
            delta = 1.0
        else:
            # 70% flip, 30% remove
            if rng2.random() < 0.7:
                x[i] = -old_val
                delta = -2.0 * old_val
            else:
                x[i] = 0.0
                delta = -old_val
        new_s = s + delta
        if abs(new_s) < 1e-12:
            x[i] = old_val
            no_improve += 1
            T *= cooling
            continue
        X_new = X + delta * EXP_FACTORS[i]
        ac = _ac_from_fft(X_new)
        max_ac = np.max(np.abs(ac))
        new_score = max_ac / (new_s * new_s)

        if new_score < cur_score or rng2.random() < np.exp(-(new_score - cur_score) / max(T, 1e-12)):
            X = X_new
            s = new_s
            cur_score = new_score
            if cur_score < best_score:
                best_score = cur_score
                best_x = x.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            x[i] = old_val
            no_improve += 1

        T *= cooling
        if T < T_min:
            T = T_min

        if no_improve > 800:
            x = best_x.copy()
            X = np.fft.rfft(x, FFT_SIZE)
            s = np.sum(x)
            cur_score = best_score
            no_improve = 0
            T = max(T, 0.005)

    # ---------- Phase 2: Continuous refinement (subgradient on max|ac|) ----------
    x = best_x.astype(np.float64).copy()
    s = np.sum(x)
    if abs(s) > 1e-12:
        x = x / s  # normalize sum to 1

    best_cont_score = _score(x)
    best_cont_x = x.copy()

    step_size = 0.01
    no_improve_c = 0
    max_no_improve_c = 250

    for it in range(30000):
        if time.time() > deadline - 3:
            break

        ac = _autoconv(x)
        abs_ac = np.abs(ac)
        k_max = int(np.argmax(abs_ac))
        val_max = ac[k_max]

        s = np.sum(x)
        cur_sc = abs(val_max) / (s * s) if abs(s) > 1e-12 else float('inf')

        if cur_sc < best_cont_score:
            best_cont_score = cur_sc
            best_cont_x = x.copy()
            no_improve_c = 0
        else:
            no_improve_c += 1
            if no_improve_c > max_no_improve_c:
                step_size *= 0.7
                if step_size < 1e-7:
                    break
                no_improve_c = 0

        k = k_max
        grad = np.zeros(N)
        j_lo1 = max(0, k - N + 1)
        j_hi1 = min(N - 1, k)
        if j_lo1 <= j_hi1:
            js = np.arange(j_lo1, j_hi1 + 1)
            grad[js] += x[k - js]
        j_lo2 = max(0, k)
        j_hi2 = min(N - 1, k + N - 1)
        if j_lo2 <= j_hi2:
            js = np.arange(j_lo2, j_hi2 + 1)
            grad[js] += x[js - k]

        direction = -np.sign(val_max)
        x_new = x + step_size * direction * grad
        # Preserve sum = 1
        x_new = x_new - np.mean(x_new) + 1.0 / N
        x = x_new

    final_cont_score = _score(best_cont_x)
    if final_cont_score < best_score:
        best_score = final_cont_score
        best_x = best_cont_x

    return best_x
