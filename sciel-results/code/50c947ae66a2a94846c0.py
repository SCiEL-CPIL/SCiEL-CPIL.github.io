# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _flip_gain(adjacency, labels, v):
    """Compute the change in cut size if we flip vertex v."""
    same_count = adjacency[v][labels == labels[v]].sum()
    opp_count = adjacency[v][labels != labels[v]].sum()
    return int(same_count - opp_count)


def _all_flip_gains(adjacency, labels):
    """Compute flip gains for all vertices at once."""
    n = len(labels)
    # For each vertex v, gain = (neighbors with same label) - (neighbors with opposite label)
    # same_count[v] = sum of adjacency[v, u] where labels[u] == labels[v]
    # opp_count[v] = sum of adjacency[v, u] where labels[u] != labels[v]
    
    # Build label matrix for computation
    same_mask = (labels[:, None] == labels[None, :])
    opp_mask = ~same_mask
    
    same_counts = (adjacency & same_mask).sum(axis=1)
    opp_counts = (adjacency & opp_mask).sum(axis=1)
    
    gains = same_counts - opp_counts
    return gains.astype(np.int32)


def _local_search(adjacency, labels, deadline, max_rounds=50):
    """Iterative local search: flip vertices if it improves cut size."""
    n = len(labels)
    labels = labels.copy()
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        gains = _all_flip_gains(adjacency, labels)
        # Find all vertices with positive gain
        positive = np.where(gains > 0)[0]
        if len(positive) == 0:
            break
        # Flip all vertices with positive gain simultaneously
        labels[positive] = 1 - labels[positive]
    
    return labels


def _spectral_partition(adjacency, n):
    """Use Fiedler vector for initial partition."""
    A = adjacency.astype(float)
    D = np.diag(A.sum(axis=1))
    L = D - A
    
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        fiedler = eigenvectors[:, 1]
        labels = (fiedler >= 0).astype(np.int8)
        return labels
    except:
        return np.random.randint(0, 2, size=n).astype(np.int8)


def _greedy_partition(adjacency, n, deadline):
    """Greedy construction: add vertices one by one, choosing label that maximizes cut."""
    labels = np.zeros(n, dtype=np.int8)
    degrees = adjacency.sum(axis=1)
    order = np.argsort(-degrees)
    
    for v in order:
        if time.time() > deadline:
            break
        cut_if_0 = adjacency[v][labels == 1].sum()
        cut_if_1 = adjacency[v][labels == 0].sum()
        labels[v] = 0 if cut_if_0 >= cut_if_1 else 1
    
    return labels


def _random_labels(n, seed):
    """Generate random 0/1 labels."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 2, size=n).astype(np.int8)


def _simulated_annealing(adjacency, labels, deadline, start_temp=10.0, end_temp=0.01, cooling=0.995):
    """Simulated annealing to escape local optima."""
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    temp = start_temp
    rng = np.random.RandomState(123)
    
    while time.time() < deadline and temp > end_temp:
        # Pick a random vertex to flip
        v = rng.randint(n)
        gain = _flip_gain(adjacency, labels, v)
        
        if gain >= 0:
            # Accept improving or neutral moves
            labels[v] = 1 - labels[v]
            current_cut += gain
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        else:
            # Accept worsening moves with probability exp(gain/temp)
            if rng.random() < np.exp(gain / temp):
                labels[v] = 1 - labels[v]
                current_cut += gain
        
        temp *= cooling
    
    return best_labels, best_cut


def _tabu_search(adjacency, labels, deadline, tabu_length=10, max_iters=2000):
    """Tabu search to escape local optima."""
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    # Tabu list: (vertex, iteration)
    tabu = {}
    rng = np.random.RandomState(456)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        gains = _all_flip_gains(adjacency, labels)
        
        # Find best non-tabu move
        best_v = -1
        best_gain = -np.inf
        for v in range(n):
            if v in tabu and it - tabu[v] < tabu_length:
                continue
            if gains[v] > best_gain:
                best_gain = gains[v]
                best_v = v
        
        if best_v == -1:
            # All moves are tabu, pick best tabu move
            best_v = np.argmax(gains)
            best_gain = gains[best_v]
        
        if best_gain <= 0 and best_cut > current_cut:
            # No improving moves, try random
            best_v = rng.randint(n)
            best_gain = gains[best_v]
        
        # Make the move
        labels[best_v] = 1 - labels[best_v]
        tabu[best_v] = it
        current_cut += best_gain
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
    
    return best_labels, best_cut


def _two_flip_local_search(adjacency, labels, deadline):
    """Local search allowing pairs of simultaneous flips."""
    n = len(labels)
    labels = labels.copy()
    
    for _ in range(20):
        if time.time() > deadline:
            break
        improved = False
        # Try single flips first
        gains = _all_flip_gains(adjacency, labels)
        positive = np.where(gains > 0)[0]
        if len(positive) > 0:
            labels[positive] = 1 - labels[positive]
            improved = True
        
        # Try double flips
        if not improved:
            best_pair_gain = 0
            best_pair = None
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i+1, n):
                    if labels[i] == labels[j]:
                        # Flipping both changes their relationship with each other
                        # and with all other vertices
                        # This is complex; skip for now
                        pass
            if best_pair:
                labels[best_pair[0]] = 1 - labels[best_pair[0]]
                labels[best_pair[1]] = 1 - labels[best_pair[1]]
                improved = True
        
        if not improved:
            break
    
    return labels


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    adjacency = np.asarray(adjacency, dtype=bool)
    
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_labels = None
    best_cut = -1
    
    # Phase 1: Multiple diverse initializations
    initializations = []
    
    # 1. Spectral partition
    initializations.append(_spectral_partition(adjacency, n))
    
    # 2. Greedy partition
    initializations.append(_greedy_partition(adjacency, n, deadline))
    
    # 3. Alternating
    initializations.append((np.arange(n) % 2).astype(np.int8))
    
    # 4-13. Random initializations
    for seed in range(10):
        initializations.append(_random_labels(n, 200 + seed))
    
    # Phase 2: Local search from each initialization
    for init_labels in initializations:
        if time.time() > deadline:
            break
        labels = _local_search(adjacency, init_labels, deadline)
        cut = _cut_size(adjacency, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    if best_labels is None:
        best_labels = (np.arange(n) % 2).astype(np.int8)
        best_cut = _cut_size(adjacency, best_labels)
    
    # Phase 3: Simulated annealing from best solution
    sa_deadline = min(deadline, time.time() + 30)
    sa_labels, sa_cut = _simulated_annealing(adjacency, best_labels, sa_deadline)
    if sa_cut > best_cut:
        best_cut = sa_cut
        best_labels = sa_labels.copy()
    
    # Phase 4: Tabu search from best solution
    tabu_deadline = min(deadline, time.time() + 30)
    tabu_labels, tabu_cut = _tabu_search(adjacency, best_labels, tabu_deadline)
    if tabu_cut > best_cut:
        best_cut = tabu_cut
        best_labels = tabu_labels.copy()
    
    # Phase 5: More local search with perturbations
    rng = np.random.RandomState(789)
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        num_perturbations = min(100, int(remaining_time * 10))
        for i in range(num_perturbations):
            if time.time() > deadline:
                break
            num_flips = 1 + (i % 7)
            perturbed = best_labels.copy()
            indices = rng.choice(n, size=min(num_flips, n), replace=False)
            perturbed[indices] = 1 - perturbed[indices]
            labels = _local_search(adjacency, perturbed, deadline)
            cut = _cut_size(adjacency, labels)
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
    
    # Phase 6: Final local search
    if time.time() < deadline - 2:
        labels = _local_search(adjacency, best_labels, deadline)
        cut = _cut_size(adjacency, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
