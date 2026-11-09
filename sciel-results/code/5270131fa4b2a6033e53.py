"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _local_search(adjacency, labels, max_passes=10):
    """Greedy local improvement: flip vertices that increase cut size."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    
    # Precompute current cut size
    current_cut = _cut_size(adjacency, labels)
    
    for _ in range(max_passes):
        improved = False
        # Randomize the order to break ties
        order = np.random.permutation(n)
        for v in order:
            # Compute gain if we flip vertex v
            # Current edges crossing from v: edges to opposite set
            # After flip: edges to same set become crossing
            # Gain = (edges to same set) - (edges to opposite set)
            same_side = (labels == labels[v]).sum()
            # Number of neighbors on same side vs opposite side
            neighbors_same = A[v, labels == labels[v]].sum()
            neighbors_diff = A[v, labels != labels[v]].sum()
            
            gain = neighbors_same - neighbors_diff
            if gain > 0:
                labels[v] = 1 - labels[v]
                current_cut += gain
                improved = True
        
        if not improved:
            break
    
    return labels, current_cut


def _spectral_partition(adjacency):
    """Spectral partitioning using the Fiedler vector."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    
    # Compute Laplacian L = D - A
    D = np.diag(A.sum(axis=1))
    L = D - A
    
    # Find the second smallest eigenvalue and its eigenvector
    # For small n, we can use dense eigendecomposition
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        fiedler = eigenvectors[:, 1]
    except np.linalg.LinAlgError:
        # Fallback to random
        fiedler = np.random.randn(n)
    
    # Partition based on the sign of the Fiedler vector
    labels = (fiedler > 0).astype(np.int64)
    
    # Ensure both partitions are non-empty
    if labels.sum() == 0 or labels.sum() == n:
        labels = np.random.randint(0, 2, n)
    
    return labels


def _simulated_annealing(adjacency, labels, initial_temp=1.0, cooling_rate=0.995, max_iters=2000):
    """Simulated annealing for Max Cut."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    
    current_cut = _cut_size(adjacency, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    temp = initial_temp
    for it in range(max_iters):
        # Pick a random vertex to flip
        v = np.random.randint(n)
        
        # Compute gain
        neighbors_same = A[v, labels == labels[v]].sum()
        neighbors_diff = A[v, labels != labels[v]].sum()
        gain = neighbors_same - neighbors_diff
        
        new_cut = current_cut + gain
        
        # Accept or reject
        if gain > 0 or np.random.random() < np.exp(gain / temp):
            labels[v] = 1 - labels[v]
            current_cut = new_cut
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        
        temp *= cooling_rate
        
        if temp < 1e-6:
            break
    
    return best_labels, best_cut


def _walksat(adjacency, labels, max_iters=1000, p_flip=0.5):
    """WalkSAT-inspired heuristic for Max Cut."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    
    current_cut = _cut_size(adjacency, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    for it in range(max_iters):
        # Find vertices that would improve the cut if flipped
        gains = np.zeros(n)
        for v in range(n):
            neighbors_same = A[v, labels == labels[v]].sum()
            neighbors_diff = A[v, labels != labels[v]].sum()
            gains[v] = neighbors_same - neighbors_diff
        
        # If there's a positive gain, flip the best one
        max_gain = gains.max()
        if max_gain > 0:
            v = np.argmax(gains)
            labels[v] = 1 - labels[v]
            current_cut += max_gain
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        else:
            # No improving move; do a random walk
            # Weight by "unsatisfied" degree (edges within same set)
            unsat = np.zeros(n)
            for v in range(n):
                unsat[v] = A[v, labels == labels[v]].sum()
            
            # Pick a vertex with probability proportional to its unsat degree
            if unsat.sum() > 0:
                probs = unsat / unsat.sum()
                v = np.random.choice(n, p=probs)
            else:
                v = np.random.randint(n)
            
            # Flip it
            gain = gains[v]
            labels[v] = 1 - labels[v]
            current_cut += gain
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
    
    return best_labels, best_cut


def partition(adjacency):
    """
    Partition the graph into two sets to maximize the number of crossing edges.
    Uses multiple strategies: spectral initialization, local search, simulated annealing,
    and WalkSAT-inspired exploration.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline with safety margin
    
    n = len(adjacency)
    adjacency = np.asarray(adjacency, dtype=bool)
    
    best_labels = None
    best_cut = -1
    
    # Strategy 1: Spectral initialization + local search
    labels = _spectral_partition(adjacency)
    labels, cut = _local_search(adjacency, labels, max_passes=20)
    if cut > best_cut:
        best_cut = cut
        best_labels = labels.copy()
    
    # Check deadline
    if time.time() > deadline:
        return best_labels
    
    # Strategy 2: Multiple random initializations + local search
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        
        labels = np.random.randint(0, 2, n)
        # Ensure both partitions are non-empty
        if labels.sum() == 0 or labels.sum() == n:
            labels[0] = 0
            labels[1] = 1
        
        labels, cut = _local_search(adjacency, labels, max_passes=10)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # Check deadline
    if time.time() > deadline:
        return best_labels
    
    # Strategy 3: Simulated annealing from best solution
    if time.time() < deadline - 20:
        labels = best_labels.copy()
        labels, cut = _simulated_annealing(adjacency, labels, initial_temp=0.5, cooling_rate=0.998, max_iters=5000)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # Check deadline
    if time.time() > deadline:
        return best_labels
    
    # Strategy 4: WalkSAT-inspired exploration
    if time.time() < deadline - 10:
        labels = best_labels.copy()
        labels, cut = _walksat(adjacency, labels, max_iters=2000)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # Final local search polish
    if time.time() < deadline - 5:
        labels = best_labels.copy()
        labels, cut = _local_search(adjacency, labels, max_passes=50)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    return best_labels
# EVOLVE-BLOCK-END
