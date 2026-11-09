"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition (each edge counted once)."""
    # Edge (i,j) crosses if labels[i] != labels[j] and adj[i,j] == 1
    # Sum over all pairs, divide by 2 since each edge is counted twice
    mask = (labels[:, None] != labels[None, :]) & adj
    return mask.sum() // 2


def _local_improvement(adj, labels, max_passes=10):
    """Greedy local improvement: flip vertex if it increases cut size."""
    n = len(labels)
    labels = labels.copy()
    
    for _ in range(max_passes):
        improved = False
        for v in range(n):
            # Count edges from v to same side and other side
            same_side = labels == labels[v]
            other_side = labels != labels[v]
            
            same_count = adj[v, same_side].sum()
            other_count = adj[v, other_side].sum()
            
            # Flipping v changes cut by (other_count - same_count)
            # Actually: current contribution of v to cut is other_count
            # After flip, contribution becomes same_count
            # So change = same_count - other_count
            if same_count > other_count:
                labels[v] = 1 - labels[v]
                improved = True
        if not improved:
            break
    
    return labels


def partition(adjacency):
    """
    Find a near-maximum cut using multiple restarts with local improvement.
    
    Strategy:
    1. Generate diverse initial partitions (alternating, random, perturbed)
    2. Apply greedy local improvement to each
    3. Return the best partition found
    
    Uses a time budget to ensure we return within the deadline.
    """
    adj = np.asarray(adjacency, dtype=bool)
    n = len(adj)
    
    # Deadline: use 120 seconds (with safety margin)
    start_time = time.time()
    deadline = start_time + 115  # 5 second safety margin
    
    best_labels = None
    best_cut = -1
    
    def time_left():
        return deadline - time.time()
    
    # Generate diverse initializations
    rng = np.random.RandomState(42)
    
    # Initialization 1: Alternating
    init_variants = []
    init_variants.append(np.arange(n) % 2)
    
    # Initialization 2: Random 50/50
    init_variants.append(rng.randint(0, 2, size=n))
    
    # Initialization 3: Perturbed alternating (flip ~10% randomly)
    alt = np.arange(n) % 2
    perturbed = alt.copy()
    flip_mask = rng.random(n) < 0.1
    perturbed[flip_mask] = 1 - perturbed[flip_mask]
    init_variants.append(perturbed)
    
    # Initialization 4: Random with bias (60/40)
    init_variants.append((rng.random(n) < 0.6).astype(int))
    
    # Initialization 5: Another random
    init_variants.append(rng.randint(0, 2, size=n))
    
    # Initialization 6: Perturbed alternating (flip ~20%)
    perturbed2 = alt.copy()
    flip_mask2 = rng.random(n) < 0.2
    perturbed2[flip_mask2] = 1 - perturbed2[flip_mask2]
    init_variants.append(perturbed2)
    
    # Initialization 7: Random with different bias
    init_variants.append((rng.random(n) < 0.4).astype(int))
    
    # Initialization 8: Another perturbed alternating
    perturbed3 = alt.copy()
    flip_mask3 = rng.random(n) < 0.3
    perturbed3[flip_mask3] = 1 - perturbed3[flip_mask3]
    init_variants.append(perturbed3)
    
    # Run local improvement on each initialization
    for i, init in enumerate(init_variants):
        if time_left() < 10:  # Need at least 10s for one more improvement
            break
        
        labels = _local_improvement(adj, init, max_passes=20)
        cut = _cut_size(adj, labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # If we still have time, do more random restarts
    additional_restarts = 0
    while time_left() > 15:
        additional_restarts += 1
        init = rng.randint(0, 2, size=n)
        labels = _local_improvement(adj, init, max_passes=20)
        cut = _cut_size(adj, labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    if best_labels is None:
        # Fallback: alternating
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
