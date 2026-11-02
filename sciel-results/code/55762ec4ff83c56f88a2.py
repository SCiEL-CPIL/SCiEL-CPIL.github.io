import time
import numpy as np
from scipy.optimize import differential_evolution

def _score_set(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)."""
    if not U or len(U) < 2:
        return 0.0
    arr = np.array(U, dtype=np.int64)
    if len(arr) == 0:
        return 0.0
    if arr[0] != 0:
        arr = np.concatenate(([0], arr))
    arr = np.unique(arr)
    n = len(arr)
    if n < 2:
        return 0.0
    mx = int(arr.max())
    if mx == 0:
        return 0.0
    
    N = 2 * mx + 1
    
    sig = np.zeros(N, dtype=np.float64)
    sig[arr] = 1.0
    
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
        
    F = np.fft.rfft(sig, n=fft_size)
    
    S_sum = np.fft.irfft(F * F, n=fft_size)
    sumset_size = np.sum(S_sum[:N] > 1e-6)
    
    flip_sig = np.zeros(N, dtype=np.float64)
    flip_sig[:] = sig[::-1]
    
    F_flip = np.fft.rfft(flip_sig, n=fft_size)
    S_diff = np.fft.irfft(F * F_flip, n=fft_size)
    
    diffset_size = np.sum(S_diff > 1e-6)
    
    if sumset_size == 0:
        return 0.0
        
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
        
    denom = np.log(2 * mx + 1)
    if denom == 0:
        return 0.0
        
    return 1.0 + np.log(ratio) / denom


def _compute_sumset_diffset_sizes(D):
    """Compute |D+D| and |D-D| for a small set D containing 0."""
    D = list(D)
    n = len(D)
    sumset = set()
    diffset = set()
    for a in D:
        for b in D:
            sumset.add(a + b)
            diffset.add(a - b)
    return len(sumset), len(diffset)


def _analytic_score_for_product(D, K, B):
    """
    Analytically compute the score for U = {sum d_i * B^i : d_i in D, i=0..K-1}
    where B > 2*max(D) to ensure no carries.
    
    For such a product set:
    |U| = |D|^K
    max(U) = max(D) * (B^K - 1) / (B - 1)
    
    The sumset U+U has structure: each "digit position" contributes to the sum,
    but carries can occur. However, if B > 2*max(D), then the sum of two elements
    at position i is at most 2*max(D) < B, so no carries occur between positions.
    Therefore |U+U| = |D+D|^K and |U-U| = |D-D|^K.
    
    Score = 1 + K*log(|D-D|/|D+D|) / log(2 * max(D) * (B^K - 1)/(B-1) + 1)
    """
    if K < 1 or len(D) < 2:
        return 0.0
    if B <= 2 * max(D):
        return 0.0  # No-carry condition violated
    
    ssum, sdiff = _compute_sumset_diffset_sizes(D)
    if ssum == 0 or sdiff == 0:
        return 0.0
    
    ratio = sdiff / ssum
    if ratio <= 0:
        return 0.0
    
    M_max = max(D)
    # max(U) = M_max * (B^K - 1) / (B - 1)
    if K > 30:  # Avoid overflow
        # Use log form: log(2*M_max*(B^K-1)/(B-1) + 1) ≈ K*log(B) + log(M_max) for large K
        log_denom = K * np.log(B) + np.log(M_max)
    else:
        max_U = M_max * (B**K - 1) / (B - 1)
        if max_U > 1e18:
            log_denom = K * np.log(B) + np.log(M_max)
        else:
            log_denom = np.log(2 * max_U + 1)
    
    if log_denom <= 0:
        return 0.0
    
    return 1.0 + K * np.log(ratio) / log_denom


def _objective(params):
    """
    Objective function for differential evolution.
    params: [log_ratio, log_gap, log_K, log_base_extra]
    
    We parameterize:
    - D as a set derived from a geometric-like structure with tunable parameters
    - K as the number of levels
    - B as the base
    
    Strategy: D = {0} ∪ {floor(a * r^i) for i=1..m} where a, r, m are tuned.
    Actually, let's use a simpler parameterization:
    - D = {0, 1, ..., L-1} with some elements removed based on a pattern
    - Or: D = {0, s, 2s, ..., (L-1)s} ∪ {M} where M creates a gap
    
    Let's try: D = {0, 1, ..., L-1} \ {removed} ∪ {gap}
    where gap > L-1 creates a sparse structure.
    
    Parameters:
    p[0] = log(L) where L is the "dense part" size (L in [2, 20])
    p[1] = gap position relative to L (gap = L + floor(exp(p[1])))
    p[2] = log(K) where K is number of levels
    p[3] = log(base_extra) where B = 2*max(D) + 1 + floor(exp(p[3]))
    """
    try:
        L = int(np.exp(np.clip(params[0], np.log(2), np.log(25))))
        gap_offset = int(np.exp(np.clip(params[1], 0, np.log(50))))
        K = int(np.exp(np.clip(params[2], 0, np.log(15))))
        base_extra = int(np.exp(np.clip(params[3], 0, np.log(10))))
        
        # Construct D: {0, 1, ..., L-1} ∪ {L + gap_offset}
        # This creates a set with a gap after L-1
        D = list(range(L)) + [L + gap_offset]
        D.sort()
        
        # Ensure 0 is in D and D has at least 2 elements
        if 0 not in D:
            D = [0] + D
        D = sorted(set(D))
        
        if len(D) < 2:
            return 0.0
        
        max_D = max(D)
        B = 2 * max_D + 1 + base_extra
        
        # Check size constraint: |U| = |D|^K should be manageable
        if len(D) ** K > 200000:
            return 0.0
        
        score = _analytic_score_for_product(D, K, B)
        return score
    
    except Exception:
        return 0.0


def _construct_from_params(D, K, B):
    """Construct the actual set U from digit set D, K levels, base B."""
    U = [0]
    current_base = 1
    for _ in range(K):
        new_U = []
        for u in U:
            for d in D:
                new_U.append(u + d * current_base)
        U = new_U
        current_base *= B
    
    # Ensure 0 is included and sort
    U = sorted(set(U))
    if 0 not in U:
        U = [0] + U
    return U


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Phase 1: Differential evolution over continuous parameters
    bounds = [
        (np.log(2), np.log(25)),      # L (dense part size)
        (0, np.log(50)),              # gap offset
        (0, np.log(15)),              # K (levels)
        (0, np.log(10)),              # base extra
    ]
    
    best_score_de = 0.0
    best_params_de = None
    
    try:
        result = differential_evolution(
            _objective,
            bounds,
            maxiter=50,
            popsize=15,
            tol=0.01,
            seed=42,
            polish=False,
            callback=lambda x, convergence: None
        )
        
        if result.fun > best_score_de:
            best_score_de = result.fun
            best_params_de = result.x
        
        # Check time
        if time.time() > deadline - 30:
            pass  # Skip phase 2
    except Exception:
        pass
    
    # Phase 2: Local refinement around the best DE solution + try known good structures
    candidates = []
    
    # Add the DE result
    if best_params_de is not None:
        L = int(np.exp(np.clip(best_params_de[0], np.log(2), np.log(25))))
        gap_offset = int(np.exp(np.clip(best_params_de[1], 0, np.log(50))))
        K = int(np.exp(np.clip(best_params_de[2], 0, np.log(15))))
        base_extra = int(np.exp(np.clip(best_params_de[3], 0, np.log(10))))
        D = list(range(L)) + [L + gap_offset]
        D = sorted(set([0] + D))
        if len(D) >= 2:
            B = 2 * max(D) + 1 + base_extra
            if len(D) ** K <= 200000:
                candidates.append((D, K, B))
    
    # Try a grid of known good structures
    # Structure type 1: D = {0, 1, ..., L-1} ∪ {gap}
    for L in range(2, 15):
        for gap in range(L, L + 20):
            D = list(range(L)) + [gap]
            D = sorted(set(D))
            B = 2 * max(D) + 1
            for K in range(1, 12):
                if len(D) ** K > 100000:
                    break
                candidates.append((D, K, B))
    
    # Structure type 2: D = {0, s, 2s, ..., (L-1)s} ∪ {M} (scaled AP + outlier)
    for s in [1, 2, 3]:
        for L in range(2, 10):
            for M in range(L * s, L * s + 15):
                D = [i * s for i in range(L)] + [M]
                D = sorted(set(D))
                B = 2 * max(D) + 1
                for K in range(1, 10):
                    if len(D) ** K > 100000:
                        break
                    candidates.append((D, K, B))
    
    # Structure type 3: D = {0, a, b} (3-element sets)
    for a in range(1, 15):
        for b in range(a + 1, a + 20):
            D = [0, a, b]
            B = 2 * b + 1
            for K in range(1, 12):
                if len(D) ** K > 100000:
                    break
                candidates.append((D, K, B))
    
    # Structure type 4: D = {0, 1, M} (specific 3-element)
    for M in range(2, 30):
        D = [0, 1, M]
        B = 2 * M + 1
        for K in range(1, 12):
            if len(D) ** K > 100000:
                break
            candidates.append((D, K, B))
    
    # Structure type 5: D = {0, 1, 2, M}
    for M in range(3, 25):
        D = [0, 1, 2, M]
        B = 2 * M + 1
        for K in range(1, 10):
            if len(D) ** K > 100000:
                break
            candidates.append((D, K, B))
    
    # Structure type 6: D = {0, 1, M-1, M}
    for M in range(3, 25):
        D = [0, 1, M - 1, M]
        B = 2 * M + 1
        for K in range(1, 10):
            if len(D) ** K > 100000:
                break
            candidates.append((D, K, B))
    
    # Structure type 7: Mian-Chowla-like: D = {0, 1, 3, 7, 15, ...} (2^k - 1)
    for n_mc in range(3, 8):
        D = [2**k - 1 for k in range(n_mc)]
        D = [0] + D
        D = sorted(set(D))
        B = 2 * max(D) + 1
        for K in range(1, 8):
            if len(D) ** K > 100000:
                break
            candidates.append((D, K, B))
    
    # Evaluate all candidates using analytic score, pick top ones to actually construct
    scored_candidates = []
    for D, K, B in candidates:
        if time.time() > deadline - 20:
            break
        score = _analytic_score_for_product(D, K, B)
        if score > 0:
            scored_candidates.append((score, D, K, B))
    
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Construct the top candidates and measure actual score
    best_actual_score = 0.0
    best_U = None
    
    # Always include a fallback
    fallback_U = [0, 1, 3]
    best_actual_score = _score_set(fallback_U)
    best_U = fallback_U
    
    for score, D, K, B in scored_candidates[:30]:  # Try top 30
        if time.time() > deadline - 15:
            break
        
        try:
            U = _construct_from_params(D, K, B)
            if len(U) == 0:
                continue
            actual = _score_set(U)
            if actual > best_actual_score:
                best_actual_score = actual
                best_U = U
        except Exception:
            continue
    
    if best_U is None:
        return [0, 1, 3]
    
    return best_U


if __name__ == "__main__":
    s = construct()
    print(f"Result: {s[:10]}... Size: {len(s)}, Score: {_score_set(s)}")
