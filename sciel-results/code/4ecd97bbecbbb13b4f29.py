"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Pre-compute optimized parameters via simulation
def _simulate(items, capacity, scoring_func):
    """Simulate bin packing with given scoring function."""
    bins = []  # list of remaining capacities
    for item in items:
        if not bins:
            bins.append(capacity - item)
            continue
        # Find bins with enough capacity
        valid = [i for i, b in enumerate(bins) if b >= item]
        if not valid:
            bins.append(capacity - item)
            continue
        # Score valid bins
        scores = scoring_func(item, np.array(bins))
        # Only consider valid bins
        valid_scores = [(scores[i], i) for i in valid]
        # Pick highest score
        valid_scores.sort(key=lambda x: x[0], reverse=True)
        best_idx = valid_scores[0][1]
        bins[best_idx] -= item
    return len(bins)

def _simulate_fast(items, capacity, wt, we, wl):
    """Fast vectorized simulation for a fixed parameter triple."""
    bins = np.empty(2048, dtype=np.float64)
    n = 0
    for item in items:
        if n == 0:
            bins[0] = capacity - item
            n = 1
            continue
        # vectorized: remaining for all open bins
        rem = bins[:n] - item
        valid = rem >= 0
        if not np.any(valid):
            bins[n] = capacity - item
            n += 1
            continue
        # scores
        idx = np.arange(n, dtype=np.float64)
        score = -wt * rem - we * idx + wl * (item / 100.0)
        score[~valid] = -np.inf
        # argmax among valid
        best = int(np.argmax(score))
        bins[best] -= item
    return n

def _generate_weibull(n, capacity, scale, shape, seed):
    """Generate Weibull-distributed items scaled to capacity."""
    rng = np.random.default_rng(seed)
    # Generate Weibull and scale to [0, capacity]
    raw = rng.weibull(shape, size=n)
    # Normalize to [0, 1] then scale
    raw = raw / raw.max()
    items = raw * capacity * 0.9  # Keep below capacity
    return items

def _l1_lower_bound(items, capacity):
    """Compute L1 lower bound on number of bins."""
    total = sum(items)
    max_item = max(items)
    return max(np.ceil(total / capacity), np.ceil(max_item / capacity))

def _score_excess(n_bins, l1_bound):
    """Compute excess over L1 lower bound in percent."""
    if l1_bound == 0:
        return 0.0
    return (n_bins - l1_bound) / l1_bound * 100.0

def _optimize_parameters():
    """Optimize scoring parameters via simulation."""
    start = time.time()
    deadline = start + 120  # 120s budget
    
    # Parameter grid: (w_tight, w_early, w_large)
    # score = -w_tight * (bins - item) + w_early * (-index) + w_large * (item/capacity)
    
    best_score = float('inf')
    best_params = (1.0, 0.0, 0.0)
    
    # Generate test sequences
    sequences = []
    for s in range(5):
        items = _generate_weibull(1000, 100, 1.0, 1.5, seed=s)
        sequences.append(items)
    
    # Try different parameter combinations
    w_tight_vals = [0.5, 1.0, 1.5, 2.0]
    w_early_vals = [0.0, 0.01, 0.05, 0.1]
    w_large_vals = [0.0, 0.1, 0.5, 1.0]
    
    for wt in w_tight_vals:
        for we in w_early_vals:
            for wl in w_large_vals:
                if time.time() > deadline - 30:
                    break
                total_excess = 0
                for items in sequences:
                    def scoring(item, bins, wt=wt, we=we, wl=wl):
                        n = len(bins)
                        remaining = bins - item
                        # Only valid bins have remaining >= 0
                        valid_mask = remaining >= 0
                        # Score: -wt * remaining + we * (-index) + wl * (item/100)
                        # For invalid bins, set score to -inf
                        indices = np.arange(n)
                        score = -wt * remaining - we * indices + wl * (item / 100.0)
                        score[~valid_mask] = -np.inf
                        return score
                    
                    n_bins = _simulate(items, 100, scoring)
                    l1 = _l1_lower_bound(items, 100)
                    total_excess += _score_excess(n_bins, l1)
                
                avg_excess = total_excess / 5
                if avg_excess < best_score:
                    best_score = avg_excess
                    best_params = (wt, we, wl)
    
    return best_params

# Run optimization once at import time
_PARAMS = _optimize_parameters()
_WT, _WE, _WL = _PARAMS
_CAP = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Score bins: higher score = better choice."""
    remaining = bins - item
    valid_mask = remaining >= 0
    n = len(bins)
    indices = np.arange(n, dtype=np.float64)
    # score = -wt * remaining - we * index + wl * (item/capacity)
    score = -_WT * remaining - _WE * indices + _WL * (item / _CAP)
    score[~valid_mask] = -np.inf
    return score
# EVOLVE-BLOCK-END
