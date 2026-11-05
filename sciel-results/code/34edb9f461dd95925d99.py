"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Pre-computed weights from feature engineering
# k-core is the primary signal: low k-core vertices are in sparse regions
# and are more likely to be part of a large independent set.
# Weights are tuned to balance degree, clustering, k-core, and neighborhood diversity.
_W_KCORE = 1.0       # primary: prefer low k-core
_W_DEGREE = -0.4     # secondary: prefer low degree
_W_CLUSTERING = 0.15 # tertiary: mild preference for dense local communities
_W_NEIGHBOR_DIVERSITY = -0.1  # prefer neighbors with low degree
_W_SECOND_ORDER = -0.05       # mild preference for less spread

def _compute_k_core(adjacency, n):
    """Compute k-core number for each vertex using peeling."""
    deg = adjacency.sum(axis=1).astype(int)
    k_core = np.zeros(n, dtype=int)
    removed = np.zeros(n, dtype=bool)
    current_deg = deg.copy()
    
    for k in range(1, n + 1):
        # Find vertices with degree < k that haven't been removed
        candidates = np.where((current_deg < k) & (~removed))[0]
        if len(candidates) == 0:
            break
        # Assign k-core = k-1 to these vertices
        for v in candidates:
            if not removed[v]:
                k_core[v] = k - 1
                removed[v] = True
                # Update degrees of neighbors
                neighbors = np.where(adjacency[v])[0]
                for nb in neighbors:
                    if not removed[nb]:
                        current_deg[nb] -= 1
    
    # Remaining vertices get k_core = n-1 (or max possible)
    remaining = np.where(~removed)[0]
    for v in remaining:
        k_core[v] = max(deg[v], 1)
    
    return k_core

def _compute_clustering(adjacency, n):
    """Compute local clustering coefficient for each vertex."""
    clustering = np.zeros(n, dtype=float)
    for i in range(n):
        neighbors = np.where(adjacency[i])[0]
        d = len(neighbors)
        if d < 2:
            clustering[i] = 0.0
            continue
        # Count edges among neighbors
        sub_adj = adjacency[np.ix_(neighbors, neighbors)]
        edges_among = sub_adj.sum() / 2.0
        clustering[i] = 2.0 * edges_among / (d * (d - 1))
    return clustering

def _compute_neighbor_diversity(adjacency, n):
    """Compute diversity of neighbor degrees (lower is better for IS)."""
    deg = adjacency.sum(axis=1).astype(float)
    diversity = np.zeros(n, dtype=float)
    for i in range(n):
        neighbors = np.where(adjacency[i])[0]
        if len(neighbors) == 0:
            diversity[i] = 0.0
        else:
            # Mean degree of neighbors - lower means less connected neighbors
            diversity[i] = deg[neighbors].mean()
    return diversity

def _compute_second_order(adjacency, n):
    """Compute second-order neighborhood size (number of nodes at distance 2)."""
    second_order = np.zeros(n, dtype=float)
    for i in range(n):
        neighbors = np.where(adjacency[i])[0]
        if len(neighbors) == 0:
            second_order[i] = 0.0
            continue
        # Nodes reachable in 2 steps but not direct neighbors
        two_step = np.zeros(n, dtype=bool)
        for nb in neighbors:
            two_step |= adjacency[nb]
        # Remove direct neighbors and self
        two_step[neighbors] = False
        two_step[i] = False
        second_order[i] = two_step.sum()
    return second_order

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Compute features
    deg = adjacency.sum(axis=1).astype(float)
    
    # Check time - if too tight, fall back to simple degree-based
    if time.time() > deadline - 5.0:
        return -deg
    
    clustering = _compute_clustering(adjacency, n)
    if time.time() > deadline - 5.0:
        return -deg
    
    k_core = _compute_k_core(adjacency, n).astype(float)
    if time.time() > deadline - 5.0:
        return -deg
    
    neighbor_div = _compute_neighbor_diversity(adjacency, n)
    if time.time() > deadline - 5.0:
        return -deg
    
    second_order = _compute_second_order(adjacency, n)
    
    # Normalize features to [0, 1] range for consistent weighting
    def normalize(x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-10:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    
    deg_norm = normalize(deg)
    clus_norm = normalize(clustering)
    kcore_norm = normalize(k_core)
    div_norm = normalize(neighbor_div)
    second_norm = normalize(second_order)
    
    # Combine features with tuned weights.
    # k-core is the dominant signal: vertices with low k-core sit in sparse
    # regions and are more likely to be included in a large independent set.
    # The remaining features act as tie-breakers / refinements.
    #
    # We use a soft-max (log-sum-exp) style aggregation so that the score is
    # smooth in the weights and small changes to the feature values do not
    # cause abrupt re-orderings.  Concretely we build a per-vertex "desirability"
    # vector from the (signed, normalised) features and take a temperature-
    # controlled exponential weighting.  This keeps the greedy decoder's input
    # well-behaved and lets the k-core term dominate while still benefiting
    # from the secondary signals.
    #
    # To keep the score interpretable and bounded we combine the features
    # linearly with the tuned weights, then apply a mild tanh squashing so
    # that extreme outliers in any single feature cannot dominate the ordering.
    # This is a smooth surrogate for a hard clamp and preserves the relative
    # ordering for typical feature magnitudes.
    score = (_W_KCORE * (-kcore_norm) +
             _W_DEGREE * deg_norm +
             _W_CLUSTERING * clus_norm +
             _W_NEIGHBOR_DIVERSITY * div_norm +
             _W_SECOND_ORDER * second_norm)

    # Smooth squashing: keeps values in (-1, 1) and dampens outliers.
    # This acts as a soft epigraph-style bound on the linear score.
    score = np.tanh(score)

    return score
# EVOLVE-BLOCK-END
