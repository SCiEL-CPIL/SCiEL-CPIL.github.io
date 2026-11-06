"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _generate_weibull(n, capacity, scale, shape, seed):
    """Generate Weibull-distributed items scaled to capacity."""
    rng = np.random.default_rng(seed)
    raw = rng.weibull(shape, size=n)
    raw = raw / raw.max()
    items = raw * capacity * 0.9
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

def _simulate_fast(items, capacity, w_tight, w_early):
    """
    Fast simulation for a specific (w_tight, w_early) pair.
    Score = -w_tight * remaining - w_early * index
    """
    bins = [] # list of remaining capacities
    for item in items:
        if not bins:
            bins.append(capacity - item)
            continue
        
        # Vectorize scoring
        bins_arr = np.array(bins)
        remaining = bins_arr - item
        valid_mask = remaining >= 0
        
        if not np.any(valid_mask):
            bins.append(capacity - item)
            continue
            
        # Calculate scores
        n = len(bins)
        indices = np.arange(n)
        # Only calculate for valid bins to avoid inf issues, or use masking
        # score = -w_tight * remaining - w_early * index
        # For invalid bins, set score to -inf
        scores = -w_tight * remaining - w_early * indices
        scores[~valid_mask] = -np.inf
        
        best_idx = np.argmax(scores)
        bins[best_idx] -= item
        
    return len(bins)

def _optimize_parameters():
    start = time.time()
    deadline = start + 110 # Safety margin
    
    # Generate sequences
    sequences = []
    for s in range(5):
        items = _generate_weibull(1000, 100, 1.0, 1.5, seed=s)
        sequences.append(items)
    
    l1_bounds = [_l1_lower_bound(seq, 100) for seq in sequences]
    
    best_score = float('inf')
    best_params = (1.0, 0.0) # Default: Best Fit
    
    # Grid search
    # w_tight: positive favors Best Fit, negative favors Worst Fit
    w_tight_vals = [-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    w_early_vals = [0.0, 0.01, 0.1, 0.5, 1.0, 5.0]
    
    for wt in w_tight_vals:
        for we in w_early_vals:
            if time.time() > deadline:
                break
                
            total_excess = 0
            for i, items in enumerate(sequences):
                n_bins = _simulate_fast(items, 100, wt, we)
                total_excess += _score_excess(n_bins, l1_bounds[i])
            
            avg_excess = total_excess / 5
            if avg_excess < best_score:
                best_score = avg_excess
                best_params = (wt, we)
                
    return best_params

# Run optimization once at import time
_PARAMS = _optimize_parameters()
_WT, _WE = _PARAMS

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Score bins: higher score = better choice."""
    remaining = bins - item
    valid_mask = remaining >= 0
    n = len(bins)
    indices = np.arange(n, dtype=float)
    # score = -_WT * remaining - _WE * indices
    score = -_WT * remaining - _WE * indices
    score[~valid_mask] = -np.inf
    return score
# EVOLVE-BLOCK-END
