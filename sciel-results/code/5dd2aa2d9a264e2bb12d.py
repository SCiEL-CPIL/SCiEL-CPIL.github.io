"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_deadline = time.time() + 140  # Safety margin before 150s kill


def _simulate_bin_packing(items, capacity, score_fn):
    """
    Simulate online bin packing with a given scoring function.
    items: array of item sizes
    capacity: bin capacity
    score_fn: callable(item, remaining_array) -> score_array
    Returns: number of bins used
    """
    bins = []  # list of remaining capacities
    for item in items:
        if not bins:
            bins.append(capacity - item)
            continue
        
        remaining = np.array(bins)
        valid = remaining >= item
        
        if not np.any(valid):
            bins.append(capacity - item)
            continue
        
        # Compute scores for valid bins
        scores = score_fn(item, remaining)
        # Among valid bins, pick the one with highest score
        valid_indices = np.where(valid)[0]
        valid_scores = scores[valid_indices]
        best_idx = valid_indices[np.argmax(valid_scores)]
        bins[best_idx] -= item
    
    return len(bins)


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


def _fit_parameters():
    """
    Search over a family of scoring functions and return the best parameters.
    
    Family: score = -(remaining - item)^alpha for alpha > 0
    - alpha=1: standard Best Fit
    - alpha<1: softer preference for tight fit
    - alpha>1: stronger preference for tight fit
    
    Also try: score = -w*(remaining - item) + (1-w)*remaining (weighted BF/WF)
    """
    sequences = _generate_sequences()
    
    best_score = float('inf')
    best_params = None
    
    # Test alpha family: score = -(remaining - item)^alpha
    alphas = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    
    for alpha in alphas:
        if time.time() > _deadline - 10:
            break
        total_excess = 0.0
        for seq in sequences:
            if time.time() > _deadline - 10:
                break
            def score_fn(item, remaining, alpha=alpha):
                valid = remaining >= item
                scores = np.full(len(remaining), -np.inf)
                if np.any(valid):
                    scores[valid] = -(remaining[valid] - item) ** alpha
                return scores
            
            bins_used = _simulate_bin_packing(seq, 100, score_fn)
            lb = _l1_lower_bound(seq, 100)
            excess = (bins_used - lb) / lb * 100
            total_excess += excess
        
        mean_excess = total_excess / len(sequences)
        if mean_excess < best_score:
            best_score = mean_excess
            best_params = ('alpha', alpha)
    
    # Test weighted family: score = -w*(remaining - item) + (1-w)*remaining
    # This is equivalent to: score = remaining - w*(remaining - item) = (1-w)*remaining + w*item
    # = w*item + (1-w)*remaining
    # For w=1: score = item (all equal, random)
    # For w=0: score = remaining (Worst Fit)
    # For w=0.5: score = 0.5*item + 0.5*remaining
    # We want to find w that balances BF and WF
    # Actually, let's reformulate: score = w*(item - remaining) + (1-w)*(remaining - item)
    # = w*item - w*remaining + (1-w)*remaining - (1-w)*item
    # = w*item - w*remaining + remaining - w*remaining - item + w*item
    # = 2*w*item - 2*w*remaining + remaining - item
    # = (2w-1)*item + (1-2w)*remaining
    # = (2w-1)*(item - remaining)
    # So it's just a scaled version of BF! That's not useful.
    
    # Let me try a different family:
    # score = -(remaining - item) + lambda * (capacity - remaining)
    # This adds a preference for bins that are already partially filled
    # (i.e., bins with less remaining capacity get a bonus from the second term)
    
    # Actually, let's try: score = -(remaining - item) + lambda * remaining
    # = -remaining + item + lambda * remaining
    # = item + (lambda - 1) * remaining
    # For lambda < 1: this is like BF (prefer lower remaining)
    # For lambda > 1: this is like WF (prefer higher remaining)
    # For lambda = 1: all equal
    
    # So this is just a linear combination. Let's parameterize by the coefficient of remaining.
    # score = c * remaining + d (constant doesn't matter)
    # The direction is determined by sign of c.
    # c < 0: BF-like
    # c > 0: WF-like
    
    # Let's try a non-linear family instead:
    # score = -a*(remaining - item) - b*(remaining - item)^2
    # This penalizes both small and large remaining, with minimum at some optimal point
    
    # Or: score = -(remaining - item - target)^2 where target is some fraction of capacity
    # This prefers bins where remaining is close to target after placement
    
    # Let's try: score = -((remaining - item) - t)^2 for various t
    # This prefers bins where (remaining - item) is close to t
    # i.e., remaining is close to item + t
    
    targets = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]
    for t in targets:
        if time.time() > _deadline - 10:
            break
        total_excess = 0.0
        for seq in sequences:
            if time.time() > _deadline - 10:
                break
            def score_fn(item, remaining, t=t):
                valid = remaining >= item
                scores = np.full(len(remaining), -np.inf)
                if np.any(valid):
                    scores[valid] = -((remaining[valid] - item) - t) ** 2
                return scores
            
            bins_used = _simulate_bin_packing(seq, 100, score_fn)
            lb = _l1_lower_bound(seq, 100)
            excess = (bins_used - lb) / lb * 100
            total_excess += excess
        
        mean_excess = total_excess / len(sequences)
        if mean_excess < best_score:
            best_score = mean_excess
            best_params = ('target', t)
    
    # Also try: score = -((remaining - item) / item)^alpha (normalized)
    for alpha in [0.5, 1.0, 2.0]:
        if time.time() > _deadline - 10:
            break
        total_excess = 0.0
        for seq in sequences:
            if time.time() > _deadline - 10:
                break
            def score_fn(item, remaining, alpha=alpha):
                valid = remaining >= item
                scores = np.full(len(remaining), -np.inf)
                if np.any(valid):
                    ratio = (remaining[valid] - item) / item
                    scores[valid] = -ratio ** alpha
                return scores
            
            bins_used = _simulate_bin_packing(seq, 100, score_fn)
            lb = _l1_lower_bound(seq, 100)
            excess = (bins_used - lb) / lb * 100
            total_excess += excess
        
        mean_excess = total_excess / len(sequences)
        if mean_excess < best_score:
            best_score = mean_excess
            best_params = ('norm_alpha', alpha)
    
    if best_params is None:
        best_params = ('alpha', 1.0)  # Default to standard BF
    
    return best_params, best_score


# Try fitting at module load time, but with a time guard
_start_time = time.time()

try:
    _BEST_PARAMS, _BEST_SCORE = _fit_parameters()
except Exception:
    _BEST_PARAMS = ('alpha', 1.0)
    _BEST_SCORE = float('inf')

# Build the scoring function based on best params
_param_type, param_value = _BEST_PARAMS

if _param_type == 'alpha':
    _ALPHA = param_value
    _USE_ALPHA = True
    _USE_TARGET = False
    _USE_NORM = False
elif _param_type == 'target':
    _TARGET = param_value
    _USE_ALPHA = False
    _USE_TARGET = True
    _USE_NORM = False
elif _param_type == 'norm_alpha':
    _NORM_ALPHA = param_value
    _USE_ALPHA = False
    _USE_TARGET = False
    _USE_NORM = True
else:
    _ALPHA = 1.0
    _USE_ALPHA = True
    _USE_TARGET = False
    _USE_NORM = False


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
    
    if _USE_ALPHA:
        scores[valid_mask] = -(remaining[valid_mask] - item) ** _ALPHA
    elif _USE_TARGET:
        scores[valid_mask] = -((remaining[valid_mask] - item) - _TARGET) ** 2
    elif _USE_NORM:
        ratio = (remaining[valid_mask] - item) / item
        scores[valid_mask] = -ratio ** _NORM_ALPHA
    
    return scores
# EVOLVE-BLOCK-END
