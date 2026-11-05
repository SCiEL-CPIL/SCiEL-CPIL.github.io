# EVOLVE-BLOCK-START
import time
import numpy as np

# Safety margin before the 150 s wall-clock kill.
DEADLINE_MARGIN = 5.0


def _cut_size(labels, adj):
    """Compute the cut size (number of crossing edges) for a labeling."""
    diff = labels[None, :] != labels[:, None]
    return int(np.sum(adj & diff))


def _flip_gain(v, labels, adj):
    """
    Gain in cut size if vertex v is flipped.
    """
    same = int(adj[v, labels == labels[v]].sum())
    other = int(adj[v, labels != labels[v]].sum())
    return same - other


def _local_search(labels, adj, deadline):
    """
    Greedy local search: repeatedly flip the vertex with the best positive gain.
    Returns improved labels and the cut size.
    """
    n = len(labels)
    labels = labels.copy()
    best_cut = _cut_size(labels, adj)
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        best_gain = 0
        best_v = -1
        for v in range(n):
            gain = _flip_gain(v, labels, adj)
            if gain > best_gain:
                best_gain = gain
                best_v = v
        if best_v != -1:
            labels[best_v] = 1 - labels[best_v]
            best_cut += best_gain
            improved = True
    
    return labels, best_cut


def _random_restarts(adj, n_restarts, deadline):
    """
    Placeholder to maintain interface. Not used in new logic.
    """
    pass


def partition(adjacency):
    """
    Partition vertices into two sets to maximize the number of crossing edges.
    Uses Iterated Local Search (ILS) with random perturbations.
    """
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=bool)
    
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    # 1. Initial Solution
    # Try alternating first, then random
    best_labels = np.arange(n) % 2
    best_labels, best_cut = _local_search(best_labels, adj, deadline)
    
    # Try one random restart to potentially get a different basin
    if time.time() < deadline:
        rand_labels = np.random.randint(0, 2, size=n)
        rand_labels, rand_cut = _local_search(rand_labels, adj, deadline)
        if rand_cut > best_cut:
            best_labels = rand_labels
            best_cut = rand_cut
    
    # 2. Iterated Local Search (ILS)
    # Perturb the best solution, run local search, keep if better.
    perturb_ratio = 0.05
    min_perturb = 2
    
    while time.time() < deadline:
        # Create a copy of the best solution
        current_labels = best_labels.copy()
        
        # Determine number of vertices to flip
        k = max(min_perturb, int(perturb_ratio * n))
        
        # Randomly select k distinct vertices to flip
        if k < n:
            flip_indices = np.random.choice(n, k, replace=False)
            current_labels[flip_indices] = 1 - current_labels[flip_indices]
        else:
            # If k >= n, flip all (or a large random subset)
            current_labels = 1 - current_labels
            
        # Run local search on the perturbed solution
        new_labels, new_cut = _local_search(current_labels, adj, deadline)
        
        # Update best if improved
        if new_cut > best_cut:
            best_cut = new_cut
            best_labels = new_labels.copy()
            
    return best_labels
# EVOLVE-BLOCK-END
