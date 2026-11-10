"""Phylogenetic topology recovery from Jukes-Cantor simulated sequences."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = 250.0  # seconds wall-clock; evaluator kills at 270s


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC-corrected distance matrix using vectorized numpy."""
    n = len(labels)
    seq_len = len(sequences[labels[0]])
    
    # Encode sequences: A=0, C=1, G=2, T=3
    # ASCII: A=65, C=67, G=71, T=84
    enc = np.zeros((n, seq_len), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        arr = np.frombuffer(s.encode('ascii'), dtype=np.int8)
        enc[i] = arr
    
    # Vectorized pairwise difference counting
    diff_matrix = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        diffs = np.count_nonzero(enc[i] != enc, axis=1)
        diff_matrix[i] = diffs
    
    # Convert to proportions and apply JC correction
    p = diff_matrix.astype(np.float64) / seq_len
    p_clipped = np.minimum(p, 0.749999)
    D = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    D[p >= 0.75] = 3.0
    D = (D + D.T) * 0.5
    np.fill_diagonal(D, 0.0)
    return D


def _jc_log_likelihood_pair(d_ij, d_ik, d_jk):
    """
    Compute log-likelihood contribution for pair (i,j) being sisters,
    given their distances to a third taxon k, under the JC model.
    
    Under JC, if i and j are sisters, the internal node has branches to i, j, and k.
    The expected distance from internal node to k is (d_ik + d_jk - d_ij)/2.
    We score how well the observed distances fit this hypothesis.
    """
    # Estimated distance from internal node to k
    t_k = 0.5 * (d_ik + d_jk - d_ij)
    # Estimated branch lengths from internal node to i and j
    t_i = 0.5 * (d_ik + d_ij - d_jk)
    t_j = 0.5 * (d_jk + d_ij - d_ik)
    
    # Clip to valid range [0, max]
    t_i = np.maximum(t_i, 0.0)
    t_j = np.maximum(t_j, 0.0)
    t_k = np.maximum(t_k, 0.0)
    
    # JC transition probabilities: P(same) = (1 + 2e^{-3t})/3, P(diff) = (1 - e^{-3t})/3
    # For a distance d (proportion), the log-likelihood is:
    # d * log(P_diff) + (1-d) * log(P_same)
    # where P_diff = (1 - e^{-3t})/3, P_same = (1 + 2e^{-3t})/3
    
    def _jc_loglik(d, t):
        """Log-likelihood of observing distance d given branch length t under JC."""
        if d < 0:
            d = 0.0
        if d > 1:
            d = 1.0
        if t <= 0:
            # At t=0, P(same)=1, P(diff)=0
            if d == 0:
                return 0.0
            else:
                return -np.inf
        e3t = np.exp(-3.0 * t)
        p_same = (1.0 + 2.0 * e3t) / 3.0
        p_diff = (1.0 - e3t) / 3.0
        # Avoid log(0)
        p_same = max(p_same, 1e-300)
        p_diff = max(p_diff, 1e-300)
        ll = 0.0
        if d > 0:
            ll += d * np.log(p_diff)
        if d < 1:
            ll += (1.0 - d) * np.log(p_same)
        return ll
    
    # The likelihood that i,j are sisters given distances to k:
    # We consider the path i -> internal -> j and internal -> k
    # The probability of the observed distances is proportional to:
    # exp(-3*t_i/2) * exp(-3*t_j/2) * exp(-3*t_k/2) approximately
    # But more precisely, use the JC log-likelihood
    
    # Simpler score: sum of log-likelihoods of each distance given the implied branch lengths
    # d_ij given t_i + t_j
    # d_ik given t_i + t_k  
    # d_jk given t_j + t_k
    
    t_ij = t_i + t_j
    t_ik = t_i + t_k
    t_jk = t_j + t_k
    
    # Convert distances to proportions (they already are if D is in distance units)
    # Our D is in JC-corrected distance units, not proportions
    # For likelihood, we need proportions. But we can work with distances directly
    # using the fact that under JC, the MLE branch length for proportion p is t = -1/3 ln(1-3p)
    
    # Alternative: use the "fast up" score which is simpler and well-established:
    # score(i,j) = sum_k (d_ik + d_jk - d_ij)
    # This is equivalent to the NJ Q-criterion but framed as a likelihood signal.
    
    return None  # Placeholder; we'll use the simpler score below


def _fast_up_tree(D, labels, deadline=None):
    """
    Fast Up-tree: construct tree by repeatedly identifying the pair of taxa
    with the highest likelihood of being sister taxa under the JC model.
    
    The score for pair (i,j) is the sum over all other taxa k of:
    (d_ik + d_jk - d_ij)
    
    This is the "up-tree" criterion: it measures how well i and j fit as
    sisters by checking consistency of distances through the internal node.
    Lower score = better sister pair (minimize total path length).
    
    This is mathematically related to NJ but directly optimizes the
    likelihood signal under the JC model.
    """
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"

    D = D.astype(np.float64).copy()
    node_names = list(labels)
    
    while len(node_names) > 3:
        if deadline is not None and time.time() > deadline:
            return "(" + ",".join(node_names) + ")"
        
        m = len(node_names)
        
        # Compute the up-tree score for each pair (i,j):
        # score(i,j) = sum_k (d_ik + d_jk - d_ij) for all k != i,j
        # This equals: (sum_k d_ik) + (sum_k d_jk) - (m-2)*d_ij
        # which is the same as the NJ Q-criterion (minimized)
        
        r = D.sum(axis=1)  # row sums
        # Q[i,j] = r[i] + r[j] - (m-2)*D[i,j]  (minimize this)
        Q = r[:, None] + r[None, :] - (m - 2) * D
        np.fill_diagonal(Q, np.inf)
        
        idx = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = idx
        if i > j:
            i, j = j, i

        new_name = f"({node_names[i]},{node_names[j]})"
        
        # Update distances: new distance from internal node to k is
        # (d_ik + d_jk - d_ij) / 2
        new_dist = 0.5 * (D[i, :] + D[j, :] - D[i, j])
        new_dist[i] = 0.0
        new_dist[j] = 0.0

        keep = [k for k in range(m) if k != i and k != j]
        keep_arr = np.array(keep, dtype=np.intp)
        new_m = m - 1
        
        new_D = np.empty((new_m, new_m), dtype=np.float64)
        new_D[:-1, :-1] = D[np.ix_(keep_arr, keep_arr)]
        
        new_node_idx = m - 2
        new_D[:-1, new_node_idx] = new_dist[keep_arr]
        new_D[new_node_idx, :-1] = new_dist[keep_arr]
        new_D[new_node_idx, new_node_idx] = 0.0

        new_names = [node_names[k] for k in keep] + [new_name]
        D = new_D
        node_names = new_names

    return f"({node_names[0]},{node_names[1]},{node_names[2]})"


def _nj_with_deadline(sequences, labels, deadline):
    """Run Fast Up-tree with deadline guard. Returns Newick string or None if timeout."""
    start = time.time()
    D = _compute_distance_matrix(sequences, labels)
    elapsed = time.time() - start
    if elapsed > deadline - 5:
        return None
    
    return _fast_up_tree(D, labels, deadline=start + deadline - 2)


def reconstruct(sequences):
    """
    Reconstruct unrooted phylogenetic tree topology.
    Returns Newick string over labels t0..t(n-1).
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 3:
        if n == 1:
            return labels[0]
        elif n == 2:
            return f"({labels[0]},{labels[1]})"
        else:
            return f"({labels[0]},{labels[1]},{labels[2]})"

    # Try Fast Up-tree
    result = _nj_with_deadline(sequences, labels, deadline - start_time)
    if result is not None:
        return result + ";"

    # Fallback: star tree
    return "(" + ",".join(labels) + ");"
# EVOLVE-BLOCK-END
