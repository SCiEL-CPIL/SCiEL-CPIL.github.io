"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    n = len(labels)
    # Count edges where endpoints have different labels
    # For undirected graph, each crossing edge is counted twice in adjacency
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _local_search(adjacency, labels, max_passes=20):
    """Greedy local improvement: flip vertices that increase cut size."""
    n = len(labels)
    A = adjacency.astype(np.float64)
    labels = labels.copy()
    
    for _ in range(max_passes):
        improved = False
        for v in range(n):
            # Compute gain if we flip vertex v
            # Current: edges from v to opposite set contribute to cut
            # After flip: edges from v to same set become crossing
            # Gain = (edges to same set) - (edges to opposite set)
            same_count = A[v][labels == labels[v]].sum()
            other_count = A[v][labels != labels[v]].sum()
            gain = same_count - other_count
            if gain > 0:
                labels[v] = 1 - labels[v]
                improved = True
        if not improved:
            break
    return labels


def _spectral_partition(adjacency):
    """Partition using Fiedler vector of the graph Laplacian."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    D = A.sum(axis=1)
    L = np.diag(D) - A
    
    # Use numpy's eigensolver for the Laplacian
    # For 100 nodes, full eigendecomposition is fast
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
    fiedler = eigenvectors[:, 1]
    
    # Partition by sign
    labels = (fiedler >= 0).astype(int)
    
    # Ensure both sets are non-empty
    if labels.sum() == 0 or labels.sum() == n:
        # Fallback to random
        labels = (np.random.rand(n) > 0.5).astype(int)
    
    return labels


def _simulated_annealing(adjacency, labels, start_temp=10.0, end_temp=0.1, 
                          n_iterations=5000, deadline=None):
    """Simulated annealing to escape local optima."""
    n = len(labels)
    A = adjacency.astype(np.float64)
    labels = labels.copy()
    
    current_cut = _cut_size(adjacency, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    for i in range(n_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute temperature
        progress = i / max(n_iterations - 1, 1)
        temp = start_temp * (end_temp / start_temp) ** progress
        
        # Pick a random vertex to flip
        v = random.randint(0, n - 1)
        
        # Compute gain of flipping v
        same_count = A[v][labels == labels[v]].sum()
        other_count = A[v][labels != labels[v]].sum()
        gain = same_count - other_count
        
        # Accept if gain > 0 or with probability exp(gain / temp)
        if gain > 0:
            labels[v] = 1 - labels[v]
            current_cut += gain
        else:
            # Accept with probability exp(gain / temp)
            if temp > 0:
                prob = np.exp(gain / temp)
                if random.random() < prob:
                    labels[v] = 1 - labels[v]
                    current_cut += gain
        
        # Update best
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
    
    return best_labels, best_cut


def partition(adjacency):
    """Partition vertices into two sets to maximize cut size."""
    n = len(adjacency)
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Strategy 1: Spectral partition + local search
    labels_spec = _spectral_partition(adjacency)
    labels_spec_ls = _local_search(adjacency, labels_spec)
    cut_spec = _cut_size(adjacency, labels_spec_ls)
    
    if time.time() > deadline:
        return labels_spec_ls
    
    # Strategy 2: Alternating partition + local search
    labels_alt = np.arange(n) % 2
    labels_alt_ls = _local_search(adjacency, labels_alt)
    cut_alt = _cut_size(adjacency, labels_alt_ls)
    
    if time.time() > deadline:
        return labels_alt_ls if cut_alt > cut_spec else labels_spec_ls
    
    # Strategy 3: Random partition + local search
    best_random_labels = None
    best_random_cut = 0
    n_random = 5
    for _ in range(n_random):
        if time.time() > deadline:
            break
        labels_rand = (np.random.rand(n) > 0.5).astype(int)
        labels_rand_ls = _local_search(adjacency, labels_rand)
        cut_rand = _cut_size(adjacency, labels_rand_ls)
        if cut_rand > best_random_cut:
            best_random_cut = cut_rand
            best_random_labels = labels_rand_ls
    
    if best_random_labels is not None and time.time() > deadline:
        # Pick best so far
        candidates = [
            (labels_spec_ls, cut_spec),
            (labels_alt_ls, cut_alt),
            (best_random_labels, best_random_cut),
        ]
        best = max(candidates, key=lambda x: x[1])
        return best[0]
    
    # Pick the best local search result
    candidates = [
        (labels_spec_ls, cut_spec),
        (labels_alt_ls, cut_alt),
    ]
    if best_random_labels is not None:
        candidates.append((best_random_labels, best_random_cut))
    
    best_labels, best_cut = max(candidates, key=lambda x: x[1])
    
    if time.time() > deadline:
        return best_labels
    
    # Simulated annealing on the best result
    sa_labels, sa_cut = _simulated_annealing(adjacency, best_labels, 
                                              start_temp=5.0, end_temp=0.1,
                                              n_iterations=3000, deadline=deadline)
    
    if sa_cut > best_cut:
        best_labels = sa_labels
        best_cut = sa_cut
    
    # Final local search polish
    final_labels = _local_search(adjacency, best_labels, max_passes=10)
    final_cut = _cut_size(adjacency, final_labels)
    
    if final_cut > best_cut:
        best_labels = final_labels
    
    return best_labels
# EVOLVE-BLOCK-END
