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
    same_mask = (labels[:, None] == labels[None, :])
    opp_mask = ~same_mask
    same_counts = (adjacency & same_mask).sum(axis=1)
    opp_counts = (adjacency & opp_mask).sum(axis=1)
    gains = same_counts - opp_counts
    return gains.astype(np.int32)


def _local_search(adjacency, labels, deadline, max_rounds=100):
    """Iterative local search: flip all positive-gain vertices in rounds."""
    labels = labels.copy()
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        gains = _all_flip_gains(adjacency, labels)
        positive = np.where(gains > 0)[0]
        if len(positive) == 0:
            break
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


def _walksat(adjacency, labels, deadline, max_iters=5000, p=0.5, seed=42):
    """WalkSAT-style local search for Max Cut."""
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    rng = np.random.RandomState(seed)
    adj_int = adjacency.astype(np.int32)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        if rng.random() < p:
            same_mask = (labels[:, None] == labels[None, :])
            uncut = adj_int & same_mask
            rows, cols = np.where(uncut > 0)
            if len(rows) == 0:
                break
            idx = rng.randint(len(rows))
            u, v = rows[idx], cols[idx]
            gain_u = _flip_gain(adjacency, labels, u)
            gain_v = _flip_gain(adjacency, labels, v)
            w_u = max(gain_u, 0)
            w_v = max(gain_v, 0)
            total_w = w_u + w_v
            if total_w == 0:
                flip_v = u if rng.random() < 0.5 else v
            else:
                r = rng.random() * total_w
                flip_v = u if r < w_u else v
            labels[flip_v] = 1 - labels[flip_v]
            current_cut = _cut_size(adjacency, labels)
        else:
            v = rng.randint(n)
            labels[v] = 1 - labels[v]
            current_cut = _cut_size(adjacency, labels)
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
    return best_labels, best_cut


def _simulated_annealing(adjacency, labels, deadline, start_temp=5.0, end_temp=0.1, cooling=0.99):
    """Simulated annealing to escape local optima."""
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    temp = start_temp
    rng = np.random.RandomState(123)
    
    while time.time() < deadline and temp > end_temp:
        v = rng.randint(n)
        gain = _flip_gain(adjacency, labels, v)
        if gain >= 0:
            labels[v] = 1 - labels[v]
            current_cut += gain
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        else:
            if rng.random() < np.exp(gain / temp):
                labels[v] = 1 - labels[v]
                current_cut += gain
        temp *= cooling
    return best_labels, best_cut


def _tabu_search(adjacency, labels, deadline, tabu_length=15, max_iters=3000):
    """Tabu search to escape local optima."""
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    tabu = np.full(n, -100, dtype=np.int32)
    rng = np.random.RandomState(456)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        gains = _all_flip_gains(adjacency, labels)
        is_tabu = (it - tabu) < tabu_length
        masked_gains = gains.copy()
        masked_gains[is_tabu] = -np.inf
        best_v = int(np.argmax(masked_gains))
        best_gain = int(gains[best_v])
        if np.isinf(masked_gains[best_v]):
            best_v = int(np.argmax(gains))
            best_gain = int(gains[best_v])
        if best_gain <= 0 and best_cut > current_cut and it > 100:
            best_v = rng.randint(n)
            best_gain = int(gains[best_v])
        labels[best_v] = 1 - labels[best_v]
        tabu[best_v] = it
        current_cut += best_gain
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
    return best_labels, best_cut


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    adjacency = np.asarray(adjacency, dtype=bool)
    
    start_time = time.time()
    deadline = start_time + 140
    
    best_labels = None
    best_cut = -1
    
    # Multiple initializations
    initializations = []
    initializations.append(_spectral_partition(adjacency, n))
    initializations.append(_greedy_partition(adjacency, n, deadline))
    initializations.append((np.arange(n) % 2).astype(np.int8))
    for seed in range(10):
        initializations.append(_random_labels(n, 200 + seed))
    
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
    
    # Iterated local search with various perturbation strengths
    rng = np.random.RandomState(789)
    perturbations_done = 0
    while time.time() < deadline - 5:
        # Vary perturbation size
        base = 2 + (perturbations_done % 5)
        num_flips = min(base + perturbations_done // 10, n // 2)
        perturbed = best_labels.copy()
        indices = rng.choice(n, size=min(num_flips, n), replace=False)
        perturbed[indices] = 1 - perturbed[indices]
        labels = _local_search(adjacency, perturbed, deadline)
        cut = _cut_size(adjacency, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        perturbations_done += 1
    
    # Simulated annealing
    if time.time() < deadline - 3:
        sa_deadline = deadline - 2
        sa_labels, sa_cut = _simulated_annealing(adjacency, best_labels, sa_deadline)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
    
    # Tabu search
    if time.time() < deadline - 3:
        tabu_deadline = deadline - 2
        tabu_labels, tabu_cut = _tabu_search(adjacency, best_labels, tabu_deadline)
        if tabu_cut > best_cut:
            best_cut = tabu_cut
            best_labels = tabu_labels.copy()
    
    # WalkSAT
    if time.time() < deadline - 3:
        ws_deadline = deadline - 2
        ws_labels, ws_cut = _walksat(adjacency, best_labels, ws_deadline, max_iters=5000, p=0.6, seed=99)
        if ws_cut > best_cut:
            best_cut = ws_cut
            best_labels = ws_labels.copy()
    
    # Final local search polish
    if time.time() < deadline - 1:
        labels = _local_search(adjacency, best_labels, deadline)
        cut = _cut_size(adjacency, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    return best_labels.astype(np.int8)
