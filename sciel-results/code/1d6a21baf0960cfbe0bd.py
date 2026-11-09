"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_deadline = time.time() + 135  # Safety margin before 150s kill


def _simulate_bin_packing_fast(items, capacity, param_type, param_value):
    """
    Fast simulation using pre-allocated arrays and direct scoring.
    Returns: number of bins used
    """
    n = len(items)
    # Pre-allocate max possible bins
    max_bins = n + 1
    remaining = np.zeros(max_bins, dtype=np.float64)
    n_bins = 0
    
    for item in items:
        if n_bins == 0:
            remaining[0] = capacity - item
            n_bins = 1
            continue
        
        valid = remaining[:n_bins] >= item
        valid_count = np.count_nonzero(valid)
        
        if valid_count == 0:
            remaining[n_bins] = capacity - item
            n_bins += 1
            continue
        
        # Compute scores for valid bins only
        valid_remaining = remaining[:n_bins][valid]
        gap = valid_remaining - item
        
        if param_type == 'alpha':
            scores = -gap ** param_value
        elif param_type == 'target':
            scores = -((gap) - param_value) ** 2
        elif param_type == 'norm_alpha':
            safe_item = max(item, 1.0)
            ratio = gap / safe_item
            scores = -ratio ** param_value
        elif param_type == 'combined':
            alpha, beta = param_value
            fill_frac = valid_remaining / capacity
            scores = -(gap ** alpha) + beta * fill_frac
        elif param_type == 'content_bf':
            beta = param_value
            content = capacity - valid_remaining
            scores = -gap + beta * content
        elif param_type == 'next_fit':
            # Next-Item Fit: score = -((remaining - item) - target_gap)^2
            # where target_gap approximates the median future item size
            # param_value is (target_gap, weight)
            target_gap, weight = param_value
            scores = -weight * ((gap - target_gap) ** 2)
        elif param_type == 'next_fit_hybrid':
            # Hybrid: blend next-fit with best-fit
            # score = -alpha * gap - beta * (gap - target_gap)^2
            # param_value is (alpha, beta, target_gap)
            alpha, beta, target_gap = param_value
            scores = -alpha * gap - beta * ((gap - target_gap) ** 2)
        else:
            scores = -gap
        
        best_local = np.argmax(scores)
        valid_idx = np.where(valid)[0]
        best_global = valid_idx[best_local]
        remaining[best_global] -= item
    
    return n_bins


def _l1_lower_bound(items, capacity):
    total = np.sum(items)
    max_item = np.max(items)
    lb1 = int(np.ceil(total / capacity))
    lb2 = int(np.ceil(max_item))
    return max(lb1, lb2)


def _generate_sequences(n_sequences=5, n_items=1000, capacity=100):
    """Generate Weibull sequences for testing."""
    sequences = []
    for seed in range(n_sequences):
        rng = np.random.RandomState(seed)
        items = rng.weibull(2, size=n_items) * 20
        items = items[items <= capacity]
        while len(items) < n_items:
            new_items = rng.weibull(2, size=n_items) * 20
            new_items = new_items[new_items <= capacity]
            items = np.concatenate([items, new_items])[:n_items]
        sequences.append(items)
    return sequences


def _evaluate_params(sequences, capacity, param_type, param_value):
    """Evaluate a parameter set across all sequences. Returns mean excess %."""
    total_excess = 0.0
    for seq in sequences:
        bins_used = _simulate_bin_packing_fast(seq, capacity, param_type, param_value)
        lb = _l1_lower_bound(seq, capacity)
        excess = (bins_used - lb) / lb * 100
        total_excess += excess
    return total_excess / len(sequences)


def _fit_parameters():
    """
    Search over scoring function families with focus on Next-Item Fit strategies.
    """
    sequences = _generate_sequences()
    capacity = 100
    
    best_score = float('inf')
    best_params = ('next_fit', (15.0, 1.0))
    
    # Compute median item size from sequences to inform target_gap
    all_items = np.concatenate(sequences)
    median_item = np.median(all_items)
    
    # === Phase 1: Coarse grid search ===
    
    # Next-Fit family: score = -weight * ((gap - target_gap)^2)
    # target_gap should be around median item size
    target_gaps = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 70.0, 80.0]
    weights = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    for tg in target_gaps:
        if time.time() > _deadline - 30:
            break
        for w in weights:
            if time.time() > _deadline - 30:
                break
            pv = (tg, w)
            score = _evaluate_params(sequences, capacity, 'next_fit', pv)
            if score < best_score:
                best_score = score
                best_params = ('next_fit', pv)
    
    # Next-Fit Hybrid: score = -alpha*gap - beta*(gap - target_gap)^2
    for alpha in [0.1, 0.5, 1.0, 2.0, 5.0]:
        if time.time() > _deadline - 20:
            break
        for beta in [0.01, 0.05, 0.1, 0.5, 1.0]:
            if time.time() > _deadline - 20:
                break
            for tg in [10.0, 15.0, 20.0, 25.0, 30.0]:
                if time.time() > _deadline - 20:
                    break
                pv = (alpha, beta, tg)
                score = _evaluate_params(sequences, capacity, 'next_fit_hybrid', pv)
                if score < best_score:
                    best_score = score
                    best_params = ('next_fit_hybrid', pv)
    
    # Also check classic strategies for comparison
    alphas_coarse = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    for alpha in alphas_coarse:
        if time.time() > _deadline - 15:
            break
        score = _evaluate_params(sequences, capacity, 'alpha', alpha)
        if score < best_score:
            best_score = score
            best_params = ('alpha', alpha)
    
    targets_coarse = [10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]
    for t in targets_coarse:
        if time.time() > _deadline - 15:
            break
        score = _evaluate_params(sequences, capacity, 'target', t)
        if score < best_score:
            best_score = score
            best_params = ('target', t)
    
    # === Phase 2: Local refinement around best ===
    if time.time() < _deadline - 10:
        ptype, pval = best_params
        if ptype == 'next_fit':
            tg, w = pval
            # Refine target_gap
            for tg2 in [max(1.0, tg - 5), tg - 2.5, tg, tg + 2.5, min(90.0, tg + 5)]:
                if time.time() > _deadline - 5:
                    break
                for w2 in [max(0.05, w * 0.5), w * 0.75, w, w * 1.25, min(50.0, w * 1.5)]:
                    if time.time() > _deadline - 5:
                        break
                    pv = (tg2, w2)
                    score = _evaluate_params(sequences, capacity, 'next_fit', pv)
                    if score < best_score:
                        best_score = score
                        best_params = ('next_fit', pv)
        elif ptype == 'next_fit_hybrid':
            alpha, beta, tg = pval
            for da in [-0.5, -0.2, 0.0, 0.2, 0.5]:
                if time.time() > _deadline - 5:
                    break
                for db in [-0.2, -0.1, 0.0, 0.1, 0.2]:
                    if time.time() > _deadline - 5:
                        break
                    for dtg in [-3, -1, 0, 1, 3]:
                        if time.time() > _deadline - 5:
                            break
                        a2 = max(0.01, alpha + da)
                        b2 = max(0.001, beta + db)
                        tg2 = max(1.0, tg + dtg)
                        pv = (a2, b2, tg2)
                        score = _evaluate_params(sequences, capacity, 'next_fit_hybrid', pv)
                        if score < best_score:
                            best_score = score
                            best_params = ('next_fit_hybrid', pv)
        elif ptype == 'alpha':
            lo = max(0.1, pval * 0.5)
            hi = pval * 1.5 + 0.5
            n_steps = 10
            for i in range(n_steps + 1):
                if time.time() > _deadline - 5:
                    break
                a = lo + (hi - lo) * i / n_steps
                score = _evaluate_params(sequences, capacity, 'alpha', a)
                if score < best_score:
                    best_score = score
                    best_params = ('alpha', a)
        elif ptype == 'target':
            lo = max(0.0, pval - 10.0)
            hi = pval + 10.0
            n_steps = 10
            for i in range(n_steps + 1):
                if time.time() > _deadline - 5:
                    break
                t = lo + (hi - lo) * i / n_steps
                score = _evaluate_params(sequences, capacity, 'target', t)
                if score < best_score:
                    best_score = score
                    best_params = ('target', t)
    
    return best_params, best_score


# Try fitting at module load time, but with a time guard
_start_time = time.time()

try:
    _BEST_PARAMS, _BEST_SCORE = _fit_parameters()
except Exception:
    _BEST_PARAMS = ('next_fit', (15.0, 1.0))
    _BEST_SCORE = float('inf')

# Build the scoring function based on best params
_param_type, param_value = _BEST_PARAMS


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Returns an array of scores; the item goes to the highest-scoring bin.
    """
    bins = np.asarray(bins, dtype=np.float64)
    remaining = bins
    valid_mask = remaining >= item
    
    scores = np.full(len(bins), -np.inf, dtype=np.float64)
    
    if not np.any(valid_mask):
        return scores
    
    gap = remaining[valid_mask] - item
    
    if _param_type == 'alpha':
        scores[valid_mask] = -gap ** param_value
    elif _param_type == 'target':
        scores[valid_mask] = -((gap) - param_value) ** 2
    elif _param_type == 'norm_alpha':
        safe_item = max(item, 1.0)
        ratio = gap / safe_item
        scores[valid_mask] = -ratio ** param_value
    elif _param_type == 'combined':
        alpha, beta = param_value
        fill_frac = remaining[valid_mask] / 100.0
        scores[valid_mask] = -(gap ** alpha) + beta * fill_frac
    elif _param_type == 'content_bf':
        beta = param_value
        content = 100.0 - remaining[valid_mask]
        scores[valid_mask] = -gap + beta * content
    elif _param_type == 'next_fit':
        target_gap, weight = param_value
        scores[valid_mask] = -weight * ((gap - target_gap) ** 2)
    elif _param_type == 'next_fit_hybrid':
        alpha, beta, target_gap = param_value
        scores[valid_mask] = -alpha * gap - beta * ((gap - target_gap) ** 2)
    else:
        scores[valid_mask] = -gap
    
    return scores
# EVOLVE-BLOCK-END
