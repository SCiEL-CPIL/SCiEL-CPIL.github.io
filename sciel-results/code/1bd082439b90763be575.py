"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    Uses a vectorized greedy approach with multiple random restarts to find
    a large independent set in the conflict graph of candidate vectors.
    
    Strategy:
    1. Generate all integer vectors with squared norm in {2, 4, 6}
    2. For each norm class, compute pairwise dot products
    3. Use randomized greedy selection with vectorized distance checks
    4. Try multiple random orderings and keep the best result
    """
    candidates = []
    
    # Type 1: two ±1's, rest 0 (norm^2 = 2)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                candidates.append(v)
    
    # Type 2: one ±2, rest 0 (norm^2 = 4)
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s * 2
            candidates.append(v)
    
    # Type 3: four ±1's, rest 0 (norm^2 = 4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
    
    # Type 4: one ±2 and two ±1's, rest 0 (norm^2 = 6)
    for i in range(D):
        for si in (1, -1):
            for j, k in itertools.combinations([x for x in range(D) if x != i], 2):
                for sj in (1, -1):
                    for sk in (1, -1):
                        v = [0] * D
                        v[i] = si * 2
                        v[j] = sj
                        v[k] = sk
                        candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    # Compute squared norms
    norms_sq = np.sum(candidates**2, axis=1)
    max_norm_sq = int(np.max(norms_sq))
    
    # Compute all pairwise dot products (n x n matrix)
    dots = candidates @ candidates.T
    
    # We need: for selected pairs (i,j), norms_sq[i] + norms_sq[j] - 2*dots[i,j] >= max_norm_sq
    # i.e., dots[i,j] <= (norms_sq[i] + norms_sq[j] - max_norm_sq) / 2
    
    # Precompute threshold for each pair
    # threshold[i,j] = (norms_sq[i] + norms_sq[j] - max_norm_sq) // 2
    # A pair is compatible if dots[i,j] <= threshold[i,j]
    
    norm_sum = norms_sq[:, None] + norms_sq[None, :]
    thresholds = (norm_sum - max_norm_sq) // 2
    
    # Build compatibility matrix: compatible[i,j] = (dots[i,j] <= thresholds[i,j])
    # Note: we only care about i != j
    compatible = dots <= thresholds
    
    # Diagonal should be False (no self-pairs)
    np.fill_diagonal(compatible, False)
    
    def greedy_select(order):
        """Greedy selection given an ordering of candidate indices."""
        selected = []
        selected_set = set()
        
        for idx in order:
            if idx in selected_set:
                continue
            # Check if idx is compatible with all selected
            # selected is a list, so check against each
            is_compatible = True
            for sel in selected:
                if not compatible[idx, sel]:
                    is_compatible = False
                    break
            if is_compatible:
                selected.append(idx)
                selected_set.add(idx)
        
        return selected
    
    # Try multiple random orderings
    best_selected = []
    np.random.seed(12345)
    
    for trial in range(20):
        # Random ordering with bias towards lower-norm vectors first
        # (lower norm vectors are more "compatible" with others)
        order = np.random.permutation(n)
        # Sort by norm ascending to try smaller vectors first
        order_sorted = sorted(order, key=lambda x: norms_sq[x])
        
        selected = greedy_select(order_sorted)
        if len(selected) > len(best_selected):
            best_selected = selected
    
    # Also try pure norm-ascending order
    order_norm_asc = sorted(range(n), key=lambda x: norms_sq[x])
    selected = greedy_select(order_norm_asc)
    if len(selected) > len(best_selected):
        best_selected = selected
    
    # Try norm-descending order
    order_norm_desc = sorted(range(n), key=lambda x: -norms_sq[x])
    selected = greedy_select(order_norm_desc)
    if len(selected) > len(best_selected):
        best_selected = selected
    
    result = candidates[best_selected]
    return result
# EVOLVE-BLOCK-END
