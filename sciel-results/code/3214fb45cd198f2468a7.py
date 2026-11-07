"""Erdos minimum overlap - combined control-point + soft-max + epigraph.

Structural idea: represent the 100 step heights as a piecewise-linear function
of K control points (interpolated to N samples, then sum-projected). This
low-dimensional smooth parameterization lets the optimizer explore global
shapes (bimodal, oscillatory) far more effectively than 100 independent vars.

Refinement machinery:
  Phase 1  global search (differential evolution) over control points
  Phase 2  ramping-lambda log-sum-exp soft-max surrogate + L-BFGS-B multi-start
  Phase 3  exact (nonsmooth) SLSQP on control points
  Phase 4  epigraph formulation: auxiliary t + smooth max-of-affine constraint
  Phase 5  pairwise-swap local polish (provably preserves sum)
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100
K = 20
DEADLINE = 142.0  # safety margin before 150s kill
BINARY_PHASE_TIME = 15.0  # time allocated for binary permutation search


# ---------------------------------------------------------------------------
# Core evaluation / feasibility
# ---------------------------------------------------------------------------
def _correlation_value(h):
    """The evaluator's value: max(corr)/N*2. Lower is better."""
    corr = np.correlate(h, 1 - h, mode="full")
    return float(np.max(corr)) / N * 2.0


def _correlation_array(h):
    return np.correlate(h, 1 - h, mode="full")


def _interp_from_control(c):
    """Linearly interpolate K control points (bin centers) to N samples."""
    c = np.asarray(c, dtype=float)
    cx = (np.arange(K) + 0.5) * (N / K)
    x = np.arange(N, dtype=float)
    return np.interp(x, cx, c)


def _enforce_sum(h, target=None):
    """Offset + clip + iterative re-correction to hit sum==target within tol."""
    if target is None:
        target = N / 2.0
    h = np.array(h, dtype=float)
    diff = target - np.sum(h)
    h = h + diff / N
    for _ in range(80):
        h = np.clip(h, 0.0, 1.0)
        diff = target - np.sum(h)
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h < 1.0 - 1e-12
        else:
            mask = h > 1e-12
        if not np.any(mask):
            break
        h[mask] += diff / np.sum(mask)
    return h


def _feasible_from_control(c, target=None):
    h = _interp_from_control(c)
    return _enforce_sum(h, target)


def _feasible_from_h(h, target=None):
    return _enforce_sum(h, target)


# ---------------------------------------------------------------------------
# Smooth surrogates
# ---------------------------------------------------------------------------
def _soft_max(corr, lam):
    """Numerically stable log-sum-exp soft-max."""
    m = np.max(corr)
    if lam <= 0:
        return m
    shifted = corr - m
    return m + np.log(np.sum(np.exp(lam * shifted)) / len(corr)) / lam


def _objective_soft(c, lam):
    h = _feasible_from_control(c)
    corr = _correlation_array(h)
    return _soft_max(corr, lam)


def _objective_exact(c):
    h = _feasible_from_control(c)
    return _correlation_value(h)


# ---------------------------------------------------------------------------
# Epigraph formulation: min t  s.t.  corr(h) <= t,  0<=h<=1,  sum(h)=N/2
# We use a smooth max-of-affine surrogate for the max constraint so a
# gradient method (SLSQP) can drive t down toward the true max.
# ---------------------------------------------------------------------------
def _epigraph_solve(h0, lam, maxiter=400):
    h0 = np.array(h0, dtype=float)
    corr0 = _correlation_array(h0)
    t0 = float(np.max(corr0)) + 1e-4

    def epig(z):
        h = z[:N]
        t = z[N]
        corr = np.correlate(h, 1 - h, mode="full")
        shifted = (corr - t)
        smax = np.log(np.sum(np.exp(lam * shifted)) / len(corr)) / lam
        return t + 10.0 * np.log1p(np.exp(50.0 * smax))

    def epig_jac(z):
        h = z[:N]
        t = z[N]
        corr = np.correlate(h, 1 - h, mode="full")
        shifted = corr - t
        w = np.exp(lam * shifted)
        w = w / np.sum(w)
        # d smax / d corr_k = w[k]
        # d smax / d t = -1
        # d corr[k] / d h[i]: for full correlation, corr[k] = sum_m h[m]*(1-h[m-(k-(N-1))])
        # Let L = k - (N-1)
        # d corr[k]/d h[i] = (1-h[i-L]) - h[i+L]  (with boundary handling)
        
        grad = np.zeros(N + 1)
        # grad[t] = 1 + 10 * (50 * exp(50*smax) / (1+exp(50*smax))) * (-1)
        smax = np.log(np.sum(np.exp(lam * shifted)) / len(corr)) / lam
        sig = 1.0 / (1.0 + np.exp(-50.0 * smax))
        grad[N] = 1.0 - 10.0 * 50.0 * sig * (1.0 - sig)
        
        # For each h[i], compute gradient contribution
        # d obj / d h[i] = sum_k w[k] * d corr[k] / d h[i] * 10 * 50 * sig * (1-sig)
        coeff = 10.0 * 50.0 * sig * (1.0 - sig)
        
        for i in range(N):
            g_i = 0.0
            # corr[k] involves h[i] when i is in range for the sum
            # L ranges from -(N-1) to (N-1), k = L + (N-1)
            # d corr[k]/d h[i] = (1-h[i-L]) if 0<=i-L<N, minus h[i+L] if 0<=i+L<N
            # We need to sum over all k where this is nonzero
            L_min = max(-i, i - N + 1, -(N-1))
            L_max = min(N - 1 - i, i, N - 1)
            if L_min <= L_max:
                Ls = np.arange(L_min, L_max + 1)
                k_indices = Ls + (N - 1)
                # term1: (1-h[i-L])
                idx1 = i - Ls
                mask1 = (idx1 >= 0) & (idx1 < N)
                t1 = np.zeros(len(Ls))
                t1[mask1] = 1.0 - h[idx1[mask1]]
                # term2: -h[i+L]
                idx2 = i + Ls
                mask2 = (idx2 >= 0) & (idx2 < N)
                t2 = np.zeros(len(Ls))
                t2[mask2] = -h[idx2[mask2]]
                dcorr_di = t1 + t2
                g_i = np.dot(w[k_indices], dcorr_di)
            grad[i] = coeff * g_i
        
        return grad

    bounds = [(0.0, 1.0)] * N + [(0.0, float(np.max(corr0)) + 0.5)]
    cons = [{'type': 'eq', 'fun': lambda z: np.sum(z[:N]) - N / 2.0,
             'jac': lambda z: np.concatenate([np.ones(N), [0.0]])}]
    try:
        res = minimize(epig, np.concatenate([h0, [t0]]), method='SLSQP',
                       bounds=bounds, constraints=cons, jac=epig_jac,
                       options={'maxiter': maxiter, 'ftol': 1e-14})
        h_opt = _feasible_from_h(res.x[:N])
        return h_opt
    except Exception:
        return h0


# ---------------------------------------------------------------------------
# Local polish: pairwise swaps preserve sum exactly.
# ---------------------------------------------------------------------------
def _local_polish_h(h, max_iters=25, step_sizes=None):
    """Fast local polish using pairwise swaps that preserve sum exactly.
    Uses incremental correlation updates for speed."""
    best_h = np.array(h, dtype=float)
    best_val = _correlation_value(best_h)
    if step_sizes is None:
        step_sizes = [0.01, 0.02, 0.05, 0.1]
    n = len(best_h)
    corr = np.correlate(best_h, 1 - best_h, mode="full")
    max_val = float(np.max(corr))
    
    def _swap_delta_corr(h, i, j, a, b):
        """Compute the change in each correlation lag when h[i]+=a, h[j]+=b.
        Returns array of length 2n-1."""
        delta = np.zeros(2 * n - 1)
        # corr[k] = sum_m h[m] * (1-h[m+k-n+1]) for full mode
        # When h[i] changes by a: corr[k] changes by a*(1-h[i+k-n+1]) if valid
        # When h[j] changes by b: corr[k] changes by b*(1-h[j-k+n-1]) if valid
        # More precisely for full mode:
        # corr_full[k] = sum_{m} h[m] * (1-h[m-(k-(n-1))])
        # Let lag = k - (n-1)
        # corr_full[k] = sum_m h[m] * (1 - h[m - lag])
        # d/dh[i]: contributes a * (1-h[i-lag]) for each lag where i-lag is in [0,n-1]
        # d/dh[j]: contributes b * (1-h[j-lag]) ... wait, need to be careful
        
        # Actually: corr_full[k] = sum_m h[m] * (1-h[m-(k-(n-1))])
        # Let L = k-(n-1)
        # corr_full[k] = sum_m h[m] * (1-h[m-L])
        # If h[i] += a: delta[k] += a * (1-h[i-L]) for m=i
        # If h[j] += b: delta[k] += b * (1-h[j-L]) for m=j
        # Also, (1-h[m-L]) term: if m-L = i, i.e., m = i+L, then delta[k] += h[i+L] * (-a)
        # Wait no: corr = sum h[m]*(1-h[m-L]). 
        # d corr / d h[i] = (1-h[i-L]) + sum_{m: m-L=i} h[m] * (-1) = (1-h[i-L]) - h[i+L]
        # where terms are 0 if indices out of range.
        
        L_arr = np.arange(2*n-1) - (n-1)  # L from -(n-1) to (n-1)
        
        # For h[i] += a:
        # term1: (1-h[i-L]) valid when 0 <= i-L < n => L in [i-n+1, i]
        # term2: -h[i+L] valid when 0 <= i+L < n => L in [-i, n-1-i]
        
        L_min = max(-i, i-n+1, -(n-1))
        L_max = min(n-1-i, i, n-1)
        if L_min <= L_max:
            Ls = np.arange(L_min, L_max+1)
            # term1: 1-h[i-L]
            idx1 = i - Ls
            mask1 = (idx1 >= 0) & (idx1 < n)
            t1 = np.zeros(len(Ls))
            t1[mask1] = 1.0 - h[idx1[mask1]]
            # term2: -h[i+L]
            idx2 = i + Ls
            mask2 = (idx2 >= 0) & (idx2 < n)
            t2 = np.zeros(len(Ls))
            t2[mask2] = -h[idx2[mask2]]
            delta_vals = a * (t1 + t2)
            # Map Ls to k indices: k = L + (n-1)
            k_indices = Ls + (n-1)
            delta[k_indices] += delta_vals
        
        # For h[j] += b:
        L_min = max(-j, j-n+1, -(n-1))
        L_max = min(n-1-j, j, n-1)
        if L_min <= L_max:
            Ls = np.arange(L_min, L_max+1)
            idx1 = j - Ls
            mask1 = (idx1 >= 0) & (idx1 < n)
            t1 = np.zeros(len(Ls))
            t1[mask1] = 1.0 - h[idx1[mask1]]
            idx2 = j + Ls
            mask2 = (idx2 >= 0) & (idx2 < n)
            t2 = np.zeros(len(Ls))
            t2[mask2] = -h[idx2[mask2]]
            delta_vals = b * (t1 + t2)
            k_indices = Ls + (n-1)
            delta[k_indices] += delta_vals
        
        return delta
    
    for it in range(max_iters):
        improved = False
        max_idx = int(np.argmax(corr))
        
        # Try moving mass from high-h to low-h positions to reduce overlap
        for d in step_sizes:
            for sign in (+1, -1):
                best_pair_val = best_val
                best_pair = None
                best_i, best_j = -1, -1
                
                if n <= 30:
                    indices = list(range(n))
                else:
                    rng_local = np.random.default_rng(it * 1000 + int(d * 1000))
                    indices = rng_local.choice(n, size=min(n, 40), replace=False)
                
                for ii in range(len(indices)):
                    i = indices[ii]
                    for jj in range(ii + 1, len(indices)):
                        j = indices[jj]
                        ni = best_h[i] + sign * d
                        nj = best_h[j] - sign * d
                        if ni < 0.0 or ni > 1.0 or nj < 0.0 or nj > 1.0:
                            continue
                        # Quick estimate using incremental update
                        a = sign * d
                        b = -sign * d
                        delta = _swap_delta_corr(best_h, i, j, a, b)
                        new_corr = corr + delta
                        new_max = float(np.max(new_corr))
                        val = new_max / n * 2.0
                        if val < best_pair_val - 1e-14:
                            best_pair_val = val
                            best_pair = best_h.copy()
                            best_pair[i] = ni
                            best_pair[j] = nj
                            best_i, best_j = i, j
                            if val < best_val - 1e-13:
                                best_val = val
                                best_h = best_pair
                                corr = new_corr
                                improved = True
                                break
                    if improved:
                        break
            if improved:
                break
        if not improved:
            break
    return best_h


# ---------------------------------------------------------------------------
# Binary permutation search: simulated annealing on 0/1 arrangements
# ---------------------------------------------------------------------------
def _binary_permutation_search(start_time, deadline, initial_h=None):
    """Search over binary permutations of 50 ones and 50 zeros using simulated
    annealing with pairwise swaps. This satisfies sum and bound constraints
    inherently. Returns the best continuous h found (via projection)."""
    
    # Initialize from best continuous solution if available
    if initial_h is not None:
        # Threshold to binary: top 50 values become 1
        h_init = np.array(initial_h, dtype=float)
        threshold = np.sort(h_init)[50]
        binary = (h_init >= threshold).astype(float)
        # Ensure exactly 50 ones
        if binary.sum() != 50:
            # If we have more than 50 ones, set some to 0; if fewer, set some to 1
            ones = np.where(binary == 1)[0]
            zeros = np.where(binary == 0)[0]
            if len(ones) > 50:
                # Keep the 50 largest h values
                top50 = np.argsort(h_init)[-50:]
                binary = np.zeros(N)
                binary[top50] = 1.0
            elif len(ones) < 50:
                # Add the largest zero-valued h values
                need = 50 - len(ones)
                zero_indices = np.where(binary == 0)[0]
                add = np.argsort(h_init[zero_indices])[-need:]
                binary[zero_indices[add]] = 1.0
    else:
        # Start with alternating pattern
        binary = np.zeros(N)
        binary[::2] = 1.0
    
    # Compute initial value
    h_cont = _enforce_sum(binary)
    best_val = _correlation_value(h_cont)
    best_binary = binary.copy()
    
    # Simulated annealing schedule
    t_start = time.time()
    t_end = min(deadline, start_time + BINARY_PHASE_TIME)
    max_iters = int((t_end - t_start) / 0.001)  # estimate max iterations
    
    # Temperature schedule: exponential cooling
    T_init = 0.5
    T_min = 1e-6
    cooling = (T_min / T_init) ** (1.0 / max(max_iters, 1))
    
    current_binary = binary.copy()
    current_val = best_val
    T = T_init
    
    ones_positions = np.where(current_binary == 1)[0]
    zeros_positions = np.where(current_binary == 0)[0]
    
    rng = np.random.default_rng(42)
    
    for it in range(max_iters):
        if time.time() > t_end - 0.5:
            break
        
        # Pick a random 1 and a random 0 to swap
        i_idx = rng.integers(len(ones_positions))
        j_idx = rng.integers(len(zeros_positions))
        i = ones_positions[i_idx]
        j = zeros_positions[j_idx]
        
        # Perform swap
        current_binary[i] = 0.0
        current_binary[j] = 1.0
        
        # Update position lists
        ones_positions[i_idx] = j
        zeros_positions[j_idx] = i
        
        # Compute new value
        h_new = _enforce_sum(current_binary)
        new_val = _correlation_value(h_new)
        
        delta = new_val - current_val
        
        if delta < 0 or rng.random() < np.exp(-delta / max(T, T_min)):
            # Accept
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_binary = current_binary.copy()
        else:
            # Reject: undo swap
            current_binary[i] = 1.0
            current_binary[j] = 0.0
            ones_positions[i_idx] = i
            zeros_positions[j_idx] = j
        
        # Cool
        T *= cooling
    
    # Return best continuous projection
    return _enforce_sum(best_binary), best_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def construct():
    start_time = time.time()

    def _remaining():
        return DEADLINE - (time.time() - start_time)

    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)  # seed value

    # ---------------- Phase 0: binary permutation search (early) -------------
    if _remaining() > BINARY_PHASE_TIME + 30.0:
        try:
            h_bin, val_bin = _binary_permutation_search(
                start_time, start_time + BINARY_PHASE_TIME, initial_h=None)
            if val_bin < best_val:
                best_val = val_bin
                best_h = h_bin.copy()
        except Exception:
            pass

    # ---------------- Phase 1: differential evolution over control points ----
    if _remaining() > 40.0:
        de_deadline = start_time + min(40.0, _remaining() - 5.0)
        try:
            c_lo = np.full(K, 0.0)
            c_hi = np.full(K, 1.0)

            def de_obj(c):
                if time.time() > de_deadline:
                    raise TimeoutError()
                return _objective_soft(c, lam=25.0)

            res_de = differential_evolution(
                de_obj,
                bounds=list(zip(c_lo, c_hi)),
                maxiter=60, popsize=12, tol=1e-7,
                mutation=(0.4, 1.0), recombination=0.7,
                seed=7, polish=False, init='sobol')
            if res_de.x is not None:
                h_de = _feasible_from_control(res_de.x)
                v_de = _correlation_value(h_de)
                if v_de < best_val:
                    best_val = v_de
                    best_h = h_de.copy()
        except TimeoutError:
            pass
        except Exception:
            pass

    if _remaining() < 20.0:
        return _feasible_from_h(best_h)

    # ---------------- Build diverse control-point starts ---------------------
    rng = np.random.default_rng(2024)
    starts_c = []

    # bin averages of current best (smooth init)
    bin_avg = np.array([best_h[i * (N // K):(i + 1) * (N // K)].mean()
                        for i in range(K)])
    starts_c.append(bin_avg.copy())

    for _ in range(12):
        starts_c.append(rng.uniform(0.0, 1.0, K))

    for phase in (0, 1):
        starts_c.append(np.array([0.8 if (i + phase) % 2 == 0 else 0.2
                                  for i in range(K)]))
    c_bi = np.linspace(0.8, 0.2, K)
    starts_c.append(c_bi.copy())
    starts_c.append(c_bi[::-1].copy())

    # sine / oscillatory control shapes
    for freq in (1, 2, 3, 4, 5, 7, 10):
        amp = min(0.4, 0.5 / freq)
        starts_c.append(np.clip(0.5 + amp * np.sin(2 * np.pi * freq * np.arange(K) / K),
                                0.0, 1.0))
    # triangular / sawtooth
    for period in (2, 3, 4, 5, 10):
        tri = np.zeros(K)
        for i in range(K):
            t = (i % period) / period
            tri[i] = 2 * t if t < 0.5 else 2 - 2 * t
        starts_c.append(np.clip(tri, 0.0, 1.0))
    
    # Block patterns: high-low blocks of various sizes
    for block_size in (2, 3, 4, 5):
        block = np.array([0.8 if (i // block_size) % 2 == 0 else 0.2 for i in range(K)])
        starts_c.append(block.copy())

    # ---------------- Phase 2: ramping-lambda L-BFGS-B over control points ----
    slsqp_deadline = start_time + max(0.0, min(110.0, _remaining() - 15.0))
    for idx, c_start in enumerate(starts_c):
        if time.time() > slsqp_deadline - 10.0:
            break
        c_start = np.clip(np.asarray(c_start, dtype=float), 0.0, 1.0)
        current_c = c_start.copy()
        for lam in (15.0, 60.0, 300.0, 1000.0):
            if time.time() > slsqp_deadline - 4.0:
                break

            def soft_obj(c, lam=lam):
                return _objective_soft(c, lam)

            try:
                res = minimize(soft_obj, current_c, method='L-BFGS-B',
                               bounds=[(0.0, 1.0)] * K,
                               options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12})
                c_opt = np.clip(res.x, 0.0, 1.0)
                h_opt = _feasible_from_control(c_opt)
                val = _correlation_value(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
                current_c = c_opt
            except Exception:
                pass

        # ---------------- Phase 3: exact SLSQP on control points -------------
        if time.time() > slsqp_deadline - 3.0:
            break

        def exact_obj(c):
            return _objective_exact(c)

        try:
            res_exact = minimize(exact_obj, current_c, method='SLSQP',
                                 bounds=[(0.0, 1.0)] * K,
                                 options={'maxiter': 400, 'ftol': 1e-16})
            h_exact = _feasible_from_control(np.clip(res_exact.x, 0.0, 1.0))
            val_exact = _correlation_value(h_exact)
            if val_exact < best_val:
                best_val = val_exact
                best_h = h_exact.copy()
        except Exception:
            pass

    # ---------------- Phase 4: epigraph smooth solve from best ---------------
    if _remaining() > 30.0:
        epig_deadline = start_time + min(45.0, _remaining() - 15.0)
        for lam in (100.0, 500.0):
            if time.time() > epig_deadline:
                break
            try:
                h_ep = _epigraph_solve(best_h, lam, maxiter=200)
                v_ep = _correlation_value(h_ep)
                if v_ep < best_val:
                    best_val = v_ep
                    best_h = h_ep.copy()
            except Exception:
                pass

    # ---------------- Phase 5: targeted local polish on top lags -------------
    if _remaining() > 15.0:
        try:
            # Identify the top few correlation lags and do focused polish
            corr = np.correlate(best_h, 1 - best_h, mode="full")
            top_lags = np.argsort(corr)[-3:]  # 3 largest lags
            top_lags = sorted(top_lags.tolist())
            
            # For each top lag, try to reduce it via pairwise swaps
            for lag_idx in top_lags:
                if time.time() > start_time + DEADLINE - 8.0:
                    break
                lag = lag_idx - (N - 1)
                # Pairs (i, i+lag) contribute to this lag
                for d in [0.02, 0.05, 0.1]:
                    for sign in (+1, -1):
                        improved = False
                        for i in range(N - abs(lag)):
                            j = i + abs(lag)
                            ni = best_h[i] + sign * d
                            nj = best_h[j] - sign * d
                            if ni < 0.0 or ni > 1.0 or nj < 0.0 or nj > 1.0:
                                continue
                            new_h = best_h.copy()
                            new_h[i] = ni
                            new_h[j] = nj
                            val = _correlation_value(new_h)
                            if val < best_val - 1e-14:
                                best_val = val
                                best_h = new_h
                                improved = True
                        if not improved:
                            break
        except Exception:
            pass

    # ---------------- Phase 6: general pairwise polish (fast version) --------
    if _remaining() > 10.0:
        try:
            best_h = _local_polish_h(best_h, max_iters=30)
            best_val = _correlation_value(best_h)
        except Exception:
            pass

    # ---------------- Phase 7: binary refinement from best continuous --------
    if _remaining() > 8.0:
        try:
            h_bin2, val_bin2 = _binary_permutation_search(
                start_time, start_time + DEADLINE - 2.0, initial_h=best_h)
            if val_bin2 < best_val:
                best_val = val_bin2
                best_h = h_bin2.copy()
        except Exception:
            pass

    # Final projection + safety
    best_h = _feasible_from_h(best_h)
    final_val = _correlation_value(best_h)
    if final_val < best_val:
        best_val = final_val

    # Guard: ensure feasibility (finite, in range, sum within tolerance)
    if not np.all(np.isfinite(best_h)):
        best_h = np.full(N, 0.5)
    best_h = np.clip(best_h, 0.0, 1.0)
    if abs(np.sum(best_h) - N / 2.0) > 1e-9 * N:
        best_h = _enforce_sum(best_h)

    # Final micro-polish: try tiny adjustments to the most influential elements
    if _remaining() > 5.0:
        try:
            corr = np.correlate(best_h, 1 - best_h, mode="full")
            max_lag_idx = int(np.argmax(corr))
            max_lag = max_lag_idx - (N - 1)
            # Elements involved in the max lag
            involved = set()
            for i in range(N - abs(max_lag)):
                involved.add(i)
                involved.add(i + abs(max_lag))
            involved = sorted(involved)
            
            for d in [0.001, 0.003, 0.005]:
                for sign in (+1, -1):
                    for i in involved:
                        # Find a partner to swap with (prefer opposite extreme)
                        if best_h[i] > 0.5:
                            # Want to decrease h[i], find low h[j]
                            candidates = [j for j in range(N) if j != i and best_h[j] < 0.3]
                        else:
                            candidates = [j for j in range(N) if j != i and best_h[j] > 0.7]
                        for j in candidates[:5]:
                            ni = best_h[i] - sign * d  # move away from extreme
                            nj = best_h[j] + sign * d
                            if ni < 0.0 or ni > 1.0 or nj < 0.0 or nj > 1.0:
                                continue
                            new_h = best_h.copy()
                            new_h[i] = ni
                            new_h[j] = nj
                            val = _correlation_value(new_h)
                            if val < best_val - 1e-15:
                                best_val = val
                                best_h = new_h
                                break
                        if time.time() > start_time + DEADLINE - 3.0:
                            break
                    if time.time() > start_time + DEADLINE - 3.0:
                        break
        except Exception:
            pass

    # Final feasibility check
    best_h = _feasible_from_h(best_h)
    if not np.all(np.isfinite(best_h)):
        best_h = np.full(N, 0.5)
    best_h = np.clip(best_h, 0.0, 1.0)
    if abs(np.sum(best_h) - N / 2.0) > 1e-9 * N:
        best_h = _enforce_sum(best_h)

    return best_h
