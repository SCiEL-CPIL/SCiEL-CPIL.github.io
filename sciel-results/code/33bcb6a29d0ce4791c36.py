"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _flip_gain(adjacency, labels, v):
    """Compute the change in cut size if we flip vertex v.
    Positive gain means flipping improves the cut.
    """
    # Count neighbors on same side vs opposite side
    same_mask = (labels == labels[v])
    other_mask = (labels != labels[v])
    same_count = adjacency[v, same_mask].sum()
    other_count = adjacency[v, other_mask].sum()
    # Flipping v: edges to same side become crossing (+same_count), edges to other side stop crossing (-other_count)
    return int(same_count - other_count)


def _local_search(adjacency, labels, max_time=None):
    """Iterated local search via single-flip improvement with vectorized gain computation."""
    n = len(adjacency)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    
    if max_time is None:
        max_time = 145.0
    
    start_time = time.time()
    
    improved = True
    while improved:
        improved = False
        # Compute flip gains for all vertices at once
        # For each vertex v, gain = sum(adjacency[v] & (labels == labels[v])) - sum(adjacency[v] & (labels != labels[v]))
        # = sum(adjacency[v]) - 2 * sum(adjacency[v] & (labels != labels[v]))
        # But let's just compute it directly per vertex for clarity and correctness
        
        order = np.random.permutation(n)
        for v in order:
            if max_time is not None and time.time() - start_time > max_time:
                return labels, current_cut
            
            gain = _flip_gain(adjacency, labels, v)
            
            if gain > 0:
                labels[v] = 1 - labels[v]
                current_cut += gain
                improved = True
    
    return labels, current_cut


def _spectral_partition(adjacency):
    """Use the Fiedler vector (second eigenvector of Laplacian) for initialization."""
    n = len(adjacency)
    # Build Laplacian: L = D - A
    degree = adjacency.sum(axis=1)
    L = np.diag(degree) - adjacency.astype(float)
    
    # Find eigenvalues and eigenvectors
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        fiedler = eigvecs[:, 1]
        # Threshold at zero
        labels = (fiedler >= 0).astype(int)
        return labels
    except Exception:
        return np.random.randint(0, 2, size=n)


def _greedy_partition(adjacency):
    """Greedy construction: add vertices one by one, assigning to the side that maximizes cut."""
    n = len(adjacency)
    labels = np.zeros(n, dtype=int)
    unassigned = set(range(n))
    
    # Start with a random vertex
    start_v = random.choice(list(unassigned))
    labels[start_v] = 0
    unassigned.remove(start_v)
    
    while unassigned:
        # For each unassigned vertex, compute gain if assigned 0 vs 1
        # Pick the vertex-label combo with highest gain
        best_v = None
        best_label = 0
        best_gain = -1
        
        # Sample a subset for efficiency on larger graphs
        candidates = list(unassigned)
        random.shuffle(candidates)
        # Consider up to 20 candidates per step
        for v in candidates[:20]:
            for lbl in [0, 1]:
                # Count edges from v to assigned vertices with opposite label
                assigned_mask = np.zeros(n, dtype=bool)
                assigned_mask[np.array(list(set(range(n)) - unassigned))] = True
                # Actually, let's compute differently
                # For vertex v, if we assign label lbl, the cut contribution is:
                # number of neighbors of v with label (1-lbl) among assigned vertices
                # But we also need to consider future assignments, which greedy doesn't do perfectly
                # Simple approach: count current crossing edges if v is assigned lbl
                neighbor_labels = labels.copy()
                # For unassigned vertices, their label doesn't matter yet for this calculation
                # Just count edges to assigned vertices with opposite label
                assigned_vertices = [i for i in range(n) if i not in unassigned]
                if not assigned_vertices:
                    gain = 0
                else:
                    opp_label = 1 - lbl
                    gain = sum(1 for i in assigned_vertices if adjacency[v, i] and labels[i] == opp_label)
                
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
                    best_label = lbl
        
        if best_v is not None:
            labels[best_v] = best_label
            unassigned.remove(best_v)
        else:
            # Fallback
            v = random.choice(list(unassigned))
            labels[v] = random.randint(0, 1)
            unassigned.remove(v)
    
    return labels


def _random_labels(n):
    """Generate random 0/1 labels."""
    return np.random.randint(0, 2, size=n)


def partition(adjacency):
    n = len(adjacency)
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    best_labels = None
    best_cut = -1
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Spectral partition
    initializations.append(_spectral_partition(adjacency))
    
    # 2. Greedy partition
    initializations.append(_greedy_partition(adjacency))
    
    # 3. Random
    initializations.append(_random_labels(n))
    
    # 4. Alternating
    initializations.append(np.arange(n) % 2)
    
    # Run local search from each initialization
    for init_labels in initializations:
        if time.time() > deadline - 15:
            break
        
        labels, cut = _local_search(adjacency, init_labels, max_time=deadline - 5)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # Iterated local search: perturb best solution and re-optimize
    num_perturbations = 20
    for i in range(num_perturbations):
        if time.time() > deadline - 10:
            break
        
        if best_labels is None:
            current_labels = _random_labels(n)
        else:
            current_labels = best_labels.copy()
            # Flip a random subset
            num_flips = max(1, n // 10)
            flip_indices = np.random.choice(n, size=num_flips, replace=False)
            current_labels[flip_indices] = 1 - current_labels[flip_indices]
        
        labels, cut = _local_search(adjacency, current_labels, max_time=deadline - 5)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    if best_labels is None:
        best_labels = _random_labels(n)
    
    return best_labels
# EVOLVE-BLOCK-END
