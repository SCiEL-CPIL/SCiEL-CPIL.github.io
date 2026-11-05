"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(labels, adj):
    """Compute cut size: number of edges crossing the partition."""
    # adj is a bool matrix; cut = sum over edges (i<j) where labels differ
    # Vectorized: for each i, count neighbors j>i with different label
    # Simpler: cut = (n - sum(labels*(1-labels))*... ) 
    # Actually: cut = 0.5 * sum_{i,j} adj[i,j] * (labels[i] != labels[j])
    # But since adj is symmetric, we can do:
    # For each i, sum adj[i, labels != labels[i]]
    # Let's use: cut = sum_i (adj[i] * (labels != labels[i])).sum() / 2
    # But that's O(n^2) per evaluation. For n=100 it's fine.
    diff = (labels[:, None] != labels[None, :])
    return int(np.sum(adj * diff)) // 2


def _local_search(labels, adj, max_iters=500, deadline=None):
    """Single-flip local search until convergence or deadline."""
    n = len(labels)
    labels = labels.copy()
    # Precompute current cut size
    cut = _cut_size(labels, adj)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for v in range(n):
            # Gain from flipping v: if v is in group g, flipping to 1-g
            # Current contribution of v to cut: number of neighbors in opposite group
            # After flip: number of neighbors in same group (which becomes opposite)
            same = adj[v][labels == labels[v]].sum()
            other = adj[v][labels != labels[v]].sum()
            # If same > other, flipping improves cut by (same - other)
            gain = same - other
            if gain > 0:
                labels[v] = 1 - labels[v]
                cut += gain
                improved = True
        if not improved:
            break
    return labels, cut


def _pairwise_local_search(labels, adj, max_iters=200, deadline=None):
    """Local search with pairwise flips to escape 2-vertex local optima."""
    n = len(labels)
    labels = labels.copy()
    cut = _cut_size(labels, adj)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Try single flips first
        for v in range(n):
            same = adj[v][labels == labels[v]].sum()
            other = adj[v][labels != labels[v]].sum()
            gain = same - other
            if gain > 0:
                labels[v] = 1 - labels[v]
                cut += gain
                improved = True
        if improved:
            continue
        
        # Try pairwise flips
        for i in range(n):
            for j in range(i + 1, n):
                # Compute gain of flipping both i and j
                li = labels[i]
                lj = labels[j]
                # Neighbors of i: same_i = adj[i][labels==li].sum(), other_i = adj[i][labels!=li].sum()
                # After flipping i: i's neighbors in same group become opposite, and vice versa
                # But j also flips, so we need to account for the edge (i,j)
                # Let's compute the change in cut directly.
                # Current cut contribution from edges incident to i or j:
                # This is complex; let's just compute cut before and after for the pair.
                # For n=100, this is 4950 pairs, each requiring O(n) work = ~500k ops per iter.
                # That's too slow for many iterations. Let's do a smarter computation.
                
                # Gain of flipping i alone: same_i - other_i
                # Gain of flipping j alone: same_j - other_j
                # But when both flip, the edge (i,j) changes status.
                # If labels[i] == labels[j]: edge (i,j) is NOT in cut. After both flip, still not in cut. No change.
                # If labels[i] != labels[j]: edge (i,j) IS in cut. After both flip, still in cut. No change.
                # Wait, that's not right. If i and j have different labels, and both flip, they still have different labels.
                # So the edge (i,j) status doesn't change.
                # But the edges from i to other vertices and j to other vertices do change.
                
                # Let's compute:
                # For vertex i, its neighbors k != j:
                #   If label[k] == label[i]: edge (i,k) not in cut. After i flips, label[k] != new_label[i], so edge IS in cut. Gain +1.
                #   If label[k] != label[i]: edge (i,k) in cut. After i flips, label[k] == new_label[i], so edge NOT in cut. Gain -1.
                # Similarly for j.
                # But we need to exclude the edge (i,j) from both counts.
                
                same_i = adj[i][labels == labels[i]].sum() - (1 if labels[j] == labels[i] else 0)
                other_i = adj[i][labels != labels[i]].sum() - (1 if labels[j] != labels[i] else 0)
                gain_i = same_i - other_i
                
                same_j = adj[j][labels == labels[j]].sum() - (1 if labels[i] == labels[j] else 0)
                other_j = adj[j][labels != labels[j]].sum() - (1 if labels[i] != labels[j] else 0)
                gain_j = same_j - other_j
                
                total_gain = gain_i + gain_j
                if total_gain > 0:
                    labels[i] = 1 - labels[i]
                    labels[j] = 1 - labels[j]
                    cut += total_gain
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return labels, cut


def partition(adjacency):
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.int8)
    
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    best_labels = None
    best_cut = -1
    
    # Multi-start with different initializations
    rng = np.random.default_rng(42)
    
    # Start 1: alternating
    labels = np.arange(n) % 2
    labels, cut = _local_search(labels, adj, max_iters=200, deadline=deadline)
    if cut > best_cut:
        best_cut = cut
        best_labels = labels.copy()
    
    # Start 2: random
    for restart in range(5):
        if time.time() > deadline:
            break
        labels = rng.integers(0, 2, size=n).astype(np.int8)
        labels, cut = _local_search(labels, adj, max_iters=200, deadline=deadline)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # Start 3: degree-based initialization
    degrees = adj.sum(axis=1)
    sorted_idx = np.argsort(degrees)
    labels = np.zeros(n, dtype=np.int8)
    labels[sorted_idx::2] = 0
    labels[sorted_idx[1::2]] = 1
    labels, cut = _local_search(labels, adj, max_iters=200, deadline=deadline)
    if cut > best_cut:
        best_cut = cut
        best_labels = labels.copy()
    
    # Apply pairwise local search on the best solution
    if time.time() < deadline:
        best_labels, best_cut = _pairwise_local_search(best_labels, adj, max_iters=100, deadline=deadline)
    
    # Final single-flip polish
    if time.time() < deadline:
        best_labels, best_cut = _local_search(best_labels, adj, max_iters=100, deadline=deadline)
    
    return best_labels
# EVOLVE-BLOCK-END
