# EVOLVE-BLOCK-START
import numpy as np
import time

def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    # labels are 0/1
    # diff is True where labels differ
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)

def _compute_gains(adjacency, labels):
    """
    Compute the gain array for all vertices.
    gain[v] = change in cut size if we flip vertex v.
    gain[v] = 2 * (number of neighbors with same label) - degree(v)
    """
    n = len(labels)
    gains = np.zeros(n, dtype=np.int32)
    # Vectorized approach might be tricky for per-vertex neighbor sums without loops in pure numpy for sparse-ish updates, 
    # but for N=100, a loop over vertices is fine if we vectorize the neighbor check.
    
    # Alternative: 
    # same_mask = (labels[:, None] == labels[None, :])
    # same_counts = (adjacency & same_mask).sum(axis=1)
    # degrees = adjacency.sum(axis=1)
    # gains = 2 * same_counts - degrees
    
    same_mask = (labels[:, None] == labels[None, :])
    same_counts = (adjacency & same_mask).sum(axis=1)
    degrees = adjacency.sum(axis=1)
    gains = 2 * same_counts - degrees
    return gains

def _update_gains_after_flip(adjacency, labels, gains, v):
    """
    Update gains array after flipping vertex v.
    Assumes labels[v] has ALREADY been flipped.
    """
    # The gain of v itself flips sign
    gains[v] = -gains[v]
    
    # For each neighbor u of v, the gain changes by +/- 2
    # If labels[u] == labels[v] (current, after flip), they are same.
    # Before flip, labels[u] != labels[v_old]. So they were different.
    # Edge (u,v) was cut, now it is not.
    # For u: 'same' count increased by 1, 'opp' decreased by 1.
    # gain[u] = 2*same - deg. Change in same is +1 => gain[u] += 2.
    
    # If labels[u] != labels[v] (current), they are different.
    # Before flip, labels[u] == labels[v_old]. So they were same.
    # Edge (u,v) was not cut, now it is.
    # For u: 'same' count decreased by 1, 'opp' increased by 1.
    # gain[u] = 2*same - deg. Change in same is -1 => gain[u] -= 2.
    
    neighbors = np.where(adjacency[v])[0]
    if len(neighbors) > 0:
        # Check current labels of neighbors against current label of v
        # If labels[neighbors] == labels[v], they are same => gain[neighbors] += 2
        # If labels[neighbors] != labels[v], they are diff => gain[neighbors] -= 2
        
        # Create a delta array
        # diff_mask is True if labels differ
        diff_mask = (labels[neighbors] != labels[v])
        
        # For same (not diff), delta is +2
        # For diff, delta is -2
        deltas = np.where(diff_mask, -2, 2)
        
        gains[neighbors] += deltas

def _local_search(adjacency, labels, deadline, use_sa=False, T_start=10.0, T_end=0.1, alpha=0.95):
    """
    Fast local search with optional Simulated Annealing.
    Returns best labels and cut size.
    """
    n = len(labels)
    labels = labels.copy()
    gains = _compute_gains(adjacency, labels)
    current_cut = _cut_size(adjacency, labels)
    
    # For SA, we need temperature
    T = T_start
    num_steps = 0
    max_steps = 10000 # Safety cap
    
    while True:
        if time.time() > deadline:
            break
        
        # Find best move
        # For strict local search, we pick the vertex with max positive gain
        # For SA, we might pick random or best
        if use_sa and T > T_end:
            # Pick a random vertex to flip, or best? 
            # Standard SA picks a random neighbor in state space. Here, flipping any vertex is a move.
            # Let's pick a random vertex to maintain randomness, or pick best to be more directed.
            # Directed SA (pick best, accept if negative with prob) is often better for MaxCut.
            v = np.argmax(gains)
            delta = gains[v]
            
            # Acceptance criterion
            if delta > 0:
                accept = True
            else:
                # Metropolis criterion
                if T > 0:
                    prob = np.exp(delta / T)
                    accept = np.random.random() < prob
                else:
                    accept = False
            
            if accept:
                labels[v] = 1 - labels[v]
                _update_gains_after_flip(adjacency, labels, gains, v)
                current_cut += delta
                num_steps += 1
                T *= alpha
            else:
                # If we rejected the best move, we might be stuck in a local optimum for SA?
                # Actually, if the best move is negative and rejected, all other moves are worse.
                # So we can break or lower T faster.
                # To avoid infinite loop on rejection, we break if best move is negative and rejected.
                break
        else:
            # Strict Local Search
            v = np.argmax(gains)
            delta = gains[v]
            if delta <= 0:
                break
            
            labels[v] = 1 - labels[v]
            _update_gains_after_flip(adjacency, labels, gains, v)
            current_cut += delta
            num_steps += 1
        
        if num_steps > max_steps:
            break

    return labels, current_cut

def _spectral_partition(adjacency, n):
    """Spectral partition using the Fiedler vector."""
    try:
        A = adjacency.astype(np.float64)
        D = np.diag(A.sum(axis=1))
        L = D - A
        
        try:
            vals, vecs = np.linalg.eigh(L)
            idx = np.argsort(vals)
            fiedler_vec = vecs[:, idx[1]]
        except Exception:
            return np.arange(n) % 2
        
        labels = (fiedler_vec > 0).astype(np.int8)
        
        if labels.sum() == 0 or labels.sum() == n:
            median = np.median(fiedler_vec)
            labels = (fiedler_vec > median).astype(np.int8)
            if labels.sum() == 0 or labels.sum() == n:
                labels = np.arange(n) % 2
        
        return labels
    except Exception:
        return np.arange(n) % 2

def _greedy_partition(adjacency, n, deadline):
    """Greedy construction: assign vertices to maximize cut, ordered by degree."""
    labels = np.zeros(n, dtype=np.int8)
    degrees = adjacency.sum(axis=1)
    order = np.argsort(-degrees)
    
    for v in order:
        if time.time() > deadline:
            break
        # Count neighbors in set 1 and set 0
        cut_if_0 = adjacency[v][labels == 1].sum()
        cut_if_1 = adjacency[v][labels == 0].sum()
        labels[v] = 0 if cut_if_0 >= cut_if_1 else 1
    
    return labels

def _random_labels(n, seed):
    """Generate random 0/1 labels."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 2, size=n).astype(np.int8)

def _perturb(labels, n, num_flips, rng):
    """Perturb a solution by flipping a few random vertices."""
    perturbed = labels.copy()
    indices = rng.choice(n, size=min(num_flips, n), replace=False)
    perturbed[indices] = 1 - perturbed[indices]
    return perturbed

def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    adjacency = np.asarray(adjacency, dtype=bool)
    
    start_time = time.time()
    # Safety margin: 150s limit, use 145s
    deadline = start_time + 145
    
    best_labels = None
    best_cut = -1
    
    def _update_best(labels, cut):
        nonlocal best_labels, best_cut
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # Phase 1: Generate diverse initializations
    initializations = []
    
    # 1. Spectral
    initializations.append(_spectral_partition(adjacency, n))
    
    # 2. Greedy
    initializations.append(_greedy_partition(adjacency, n, deadline))
    
    # 3. Alternating
    initializations.append(np.arange(n) % 2)
    
    # 4-10. Random
    for seed in range(7):
        initializations.append(_random_labels(n, 100 + seed))
        
    # Phase 2: Local Search on each initialization
    for init_labels in initializations:
        if time.time() > deadline:
            break
        labels, cut = _local_search(adjacency, init_labels, deadline, use_sa=False)
        _update_best(labels, cut)

    # Phase 3: Iterated Local Search / Simulated Annealing on the best solution
    # We will use ILS: Perturb -> Local Search
    # And occasionally use SA from the current best
    rng = np.random.RandomState(42)
    
    # If we have time, run ILS
    if best_labels is not None:
        current_best = best_labels.copy()
        
        # Determine how many ILS iterations we can afford
        # Each ILS iteration is roughly one local search (fast) + perturbation
        # We'll just loop until deadline
        
        while time.time() < deadline - 2: # Leave 2s for final polish
            # Perturb the best solution
            # Number of flips: random between 1 and 10
            num_flips = rng.randint(1, 11)
            perturbed = _perturb(current_best, n, num_flips, rng)
            
            # Try both strict local search and SA
            # Let's try strict local search first
            labels, cut = _local_search(adjacency, perturbed, deadline, use_sa=False)
            _update_best(labels, cut)
            
            # If we improved, update current_best for next perturbation
            if cut > best_cut:
                current_best = labels.copy()
            elif cut >= best_cut:
                # Keep the one that is equal or better? Usually keep best.
                pass
            
            # Occasionally try SA from the perturbed state
            if rng.random() < 0.2:
                labels_sa, cut_sa = _local_search(adjacency, perturbed, deadline, use_sa=True, T_start=5.0, T_end=0.1, alpha=0.98)
                _update_best(labels_sa, cut_sa)
                if cut_sa > best_cut:
                    current_best = labels_sa.copy()

    # Phase 4: Final Polish
    # Run strict local search on the absolute best found
    if best_labels is not None:
        final_labels, final_cut = _local_search(adjacency, best_labels, deadline, use_sa=False)
        _update_best(final_labels, final_cut)
        
        # One more deep polish: try small perturbations (1-2 flips) and search
        if time.time() < deadline - 0.5:
            for k in range(1, 3):
                if time.time() > deadline:
                    break
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    perturbed = _perturb(best_labels, n, k, rng)
                    labels, cut = _local_search(adjacency, perturbed, deadline, use_sa=False)
                    _update_best(labels, cut)

    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
