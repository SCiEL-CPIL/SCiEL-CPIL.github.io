"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    # For each node, count neighbors with opposite label, sum, divide by 2
    opposite = labels[None, :] != labels[:, None]
    return int((adjacency & opposite).sum() // 2)


def _local_search(adjacency, labels, deadline):
    """Greedy single-flip local search until no improvement or deadline."""
    n = len(labels)
    adj = adjacency
    # Precompute row sums for fast gain computation
    # gain of flipping v: if currently 0, flipping to 1 changes cut by
    #   (neighbors with label 1) - (neighbors with label 0)
    # We can maintain this incrementally.
    labels = labels.copy()
    # For each node, compute current "excess" = (#same) - (#other)
    # If excess > 0, flipping improves the cut by excess.
    # Actually: cut contribution of v = #neighbors with opposite label
    # After flip: cut contribution = #neighbors with same label (before)
    # Gain = same_before - other_before
    # same_before = deg - other_before, so gain = deg - 2*other_before
    # other_before = sum(adj[v] & (labels != labels[v]))
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for v in range(n):
            if time.time() > deadline:
                break
            other = int((adj[v] & (labels != labels[v])).sum())
            same = int((adj[v] & (labels == labels[v])).sum())
            if same > other:
                labels[v] = 1 - labels[v]
                improved = True
    return labels


def _simulated_annealing(adjacency, labels, deadline, rng):
    """Simulated annealing with single-flip moves."""
    n = len(labels)
    adj = adjacency
    labels = labels.copy()
    current_cut = _cut_size(adj, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    # Annealing schedule
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    while time.time() < deadline and T > min_T:
        # Make a random flip
        v = rng.integers(n)
        old_label = labels[v]
        labels[v] = 1 - old_label
        new_cut = _cut_size(adj, labels)
        delta = new_cut - current_cut
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_cut = new_cut
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        else:
            labels[v] = old_label
        
        T *= cooling
    
    # Final local search
    best_labels = _local_search(adj, best_labels, deadline)
    return best_labels


def partition(adjacency):
    """Partition vertices into two sets to maximize the cut."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n = len(adjacency)
    adjacency = np.asarray(adjacency, dtype=bool)
    rng = np.random.default_rng(42)
    
    best_labels = None
    best_cut = -1
    
    # Multiple restarts with different initializations
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        
        # Choose initialization
        if restarts == 1:
            # Alternating start
            labels = np.arange(n) % 2
        elif restarts == 2:
            # Random start
            labels = rng.integers(0, 2, n)
        else:
            # Weighted random: bias toward larger side
            p = rng.random()
            labels = (rng.random(n) < p).astype(int)
        
        # Local search first
        labels = _local_search(adjacency, labels, deadline)
        cut = _cut_size(adjacency, labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # If we have time, do simulated annealing from this point
        if time.time() < deadline - 5:
            sa_labels = _simulated_annealing(adjacency, labels, deadline, rng)
            sa_cut = _cut_size(adjacency, sa_labels)
            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels.copy()
        
        # Check if we should stop
        if time.time() > deadline - 10:
            break
        
        # After a few restarts, focus on SA from best
        if restarts >= 5:
            # Do more SA iterations from the best solution
            if time.time() < deadline - 5:
                sa_labels = _simulated_annealing(adjacency, best_labels, deadline, rng)
                sa_cut = _cut_size(adjacency, sa_labels)
                if sa_cut > best_cut:
                    best_cut = sa_cut
                    best_labels = sa_labels.copy()
    
    # Final local search on best
    if time.time() < deadline:
        best_labels = _local_search(adjacency, best_labels, deadline)
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
