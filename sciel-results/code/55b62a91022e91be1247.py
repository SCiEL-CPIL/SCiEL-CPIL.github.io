"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(A, labels):
    """Number of edges crossing the partition."""
    cross = (labels[:, None] != labels[None, :])
    return int((A * cross).sum() // 2)


def _vectorized_gains(A, labels):
    """Gain from flipping each vertex (positive => improvement)."""
    # For vertex v, gain = 2*(edges to opposite) - degree
    # edges to opposite = sum over u where labels[u] != labels[v] of A[v,u]
    same_mask = (labels[:, None] == labels[None, :])
    same_deg = (A * same_mask).sum(axis=1)
    deg = A.sum(axis=1)
    gain = 2.0 * (deg - same_deg) - deg
    return gain


def _local_search(A, labels, max_rounds=50, deadline=None):
    """Greedy single-flip local search, vectorized."""
    n = len(A)
    labels = labels.copy()
    for _ in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        gain = _vectorized_gains(A, labels)
        best = int(np.argmax(gain))
        if gain[best] <= 1e-9:
            break
        labels[best] = 1 - labels[best]
    return labels


def _pairwise_polish(A, labels, max_rounds=30, deadline=None):
    """Polish with pairwise flip moves to escape single-flip local optima."""
    n = len(A)
    labels = labels.copy()
    for _ in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Vectorized: for each pair (i,j), compute gain of flipping both
        # Gain of flipping i and j simultaneously:
        #   change = -gain_i - gain_j + 2*A[i,j]*2 if labels[i]!=labels[j] ... 
        # Actually simpler: just try all pairs where both have positive individual gain,
        # or use a direct computation.
        # Direct: flipping i and j changes cut by:
        #   delta = 2*A[i,j] if labels[i]==labels[j] else -2*A[i,j]
        #   plus the individual effects minus the double-counted edge
        # Let's just do a simpler approach: random pairs with positive combined gain
        gain = _vectorized_gains(A, labels)
        # Try flipping pairs where both vertices have decent gain
        candidates = np.where(gain > 0)[0]
        if len(candidates) < 2:
            break
        # Sample pairs from candidates
        npairs = min(200, len(candidates) * (len(candidates) - 1) // 2)
        for _p in range(npairs):
            if deadline is not None and time.time() > deadline:
                break
            i, j = np.random.choice(len(candidates), 2, replace=False)
            i, j = candidates[i], candidates[j]
            if i == j:
                continue
            # Compute gain of flipping both i and j
            # Before: contribution of edges from i and j to cut
            # After flipping both, edges between i and j change status if they were same side
            # Edges from i to others (except j): flip changes their crossing status
            # Edges from j to others (except i): flip changes their crossing status
            # Edge i-j: if same side before, it wasn't crossing; after flip still same side, still not crossing
            #            if different sides before, it was crossing; after flip different sides, still crossing
            # So edge i-j contribution unchanged!
            # The gain is just gain_i + gain_j (since edge i-j doesn't change)
            combined = gain[i] + gain[j]
            if combined > 1e-9:
                labels[i] = 1 - labels[i]
                labels[j] = 1 - labels[j]
                improved = True
        if not improved:
            break
    return labels


def _random_restarts(A, n, num_restarts=10, deadline=None):
    """Multiple random restarts with local search."""
    best_labels = None
    best_cut = -1
    for r in range(num_restarts):
        if deadline is not None and time.time() > deadline:
            break
        if r == 0:
            labels = np.arange(n) % 2
        else:
            labels = np.random.randint(0, 2, n)
        labels = _local_search(A, labels, deadline=deadline)
        labels = _pairwise_polish(A, labels, deadline=deadline)
        cut = _cut_size(A, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    return best_labels


def partition(adjacency):
    A = np.array(adjacency, dtype=np.float64)
    n = len(A)
    
    # Deadline: 120s limit, use 110s for safety
    deadline = time.time() + 110
    
    # Run random restarts
    best_labels = _random_restarts(A, n, num_restarts=20, deadline=deadline)
    
    # Final polish pass with more iterations if time remains
    if time.time() < deadline - 5:
        best_labels = _local_search(A, best_labels, max_rounds=100, deadline=deadline)
        best_labels = _pairwise_polish(A, best_labels, max_rounds=50, deadline=deadline)
    
    return best_labels
# EVOLVE-BLOCK-END
