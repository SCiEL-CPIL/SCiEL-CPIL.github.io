# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _flip_gain(adjacency, labels, v):
    """Compute the change in cut size if we flip vertex v."""
    same_count = adjacency[v][labels == labels[v]].sum()
    opp_count = adjacency[v][labels != labels[v]].sum()
    return int(same_count - opp_count)


def _local_search(adjacency, labels, deadline):
    """Iterative local search with running cut count and shuffled order."""
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)

    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        order = np.random.permutation(n)
        for v in order:
            if time.time() > deadline:
                break
            gain = _flip_gain(adjacency, labels, v)
            if gain > 0:
                labels[v] = 1 - labels[v]
                current_cut += gain
                improved = True

    return labels, current_cut


def _spectral_partition(adjacency, n):
    """Spectral partition using the Fiedler vector."""
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh

        A = adjacency.astype(np.float64)
        D = np.diag(A.sum(axis=1))
        L = D - A

        try:
            vals, vecs = eigsh(L, k=2, which='SA')
            idx = np.argsort(vals)
            fiedler_vec = vecs[:, idx[1]]
        except Exception:
            vals, vecs = np.linalg.eigh(L)
            idx = np.argsort(vals)
            fiedler_vec = vecs[:, idx[1]]

        labels = (fiedler_vec > 0).astype(np.int8)

        if labels.sum() == 0 or labels.sum() == n:
            median = np.median(fiedler_vec)
            labels = (fiedler_vec > median).astype(np.int8)
            if labels.sum() == 0 or labels.sum() == n:
                labels = np.arange(n) % 2

        return labels
    except Exception:
        return np.arange(n) % 2


def _greedy_partition(adjacency, n):
    """Greedy construction: assign vertices to maximize cut."""
    labels = np.zeros(n, dtype=np.int8)
    unassigned = set(range(n))

    first = random.randint(0, n - 1)
    labels[first] = 0
    unassigned.remove(first)

    while unassigned:
        if time.time() > 100:
            break
        best_v = None
        best_gain = -1
        best_label = 0

        for v in list(unassigned):
            cut_if_0 = adjacency[v][labels == 1].sum()
            cut_if_1 = adjacency[v][labels == 0].sum()

            gain_0 = int(cut_if_0)
            gain_1 = int(cut_if_1)

            if gain_0 > best_gain:
                best_gain = gain_0
                best_v = v
                best_label = 0
            elif gain_1 > best_gain:
                best_gain = gain_1
                best_v = v
                best_label = 1

        if best_v is None:
            break

        labels[best_v] = best_label
        unassigned.remove(best_v)

    return labels


def _random_labels(n, seed):
    """Generate random 0/1 labels."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 2, size=n)


def _simulated_annealing(adjacency, labels, deadline, start_temp=10.0, cooling=0.999):
    """Simulated annealing to escape local optima."""
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    temp = start_temp

    while time.time() < deadline:
        temp *= cooling
        if temp < 0.01:
            break

        v = random.randint(0, n - 1)
        gain = _flip_gain(adjacency, labels, v)

        if gain > 0 or random.random() < np.exp(gain / temp):
            labels[v] = 1 - labels[v]
            current_cut += gain

            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()

    return best_labels, best_cut


def _tabu_search(adjacency, labels, deadline, tabu_tenure=10, iterations=500):
    """Tabu search to escape local optima."""
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    best_labels = labels.copy()
    best_cut = current_cut

    tabu = {v: 0 for v in range(n)}
    tabu_counter = 0

    for _ in range(iterations):
        if time.time() > deadline:
            break

        best_v = None
        best_gain = -np.inf

        for v in range(n):
            if tabu.get(v, 0) > tabu_counter - tabu_tenure:
                continue
            gain = _flip_gain(adjacency, labels, v)
            if gain > best_gain:
                best_gain = gain
                best_v = v

        if best_v is None:
            # If all moves are tabu, pick the best tabu move
            for v in range(n):
                gain = _flip_gain(adjacency, labels, v)
                if gain > best_gain:
                    best_gain = gain
                    best_v = v

        if best_v is not None:
            labels[best_v] = 1 - labels[best_v]
            current_cut += best_gain
            tabu[best_v] = tabu_counter
            tabu_counter += 1

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
    deadline = start_time + 115

    best_labels = None
    best_cut = -1

    def _update_best(labels, cut):
        nonlocal best_labels, best_cut
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # Strategy 1: Spectral initialization
    if n >= 2:
        spec_labels = _spectral_partition(adjacency, n)
        spec_labels, spec_cut = _local_search(adjacency, spec_labels, deadline)
        _update_best(spec_labels, spec_cut)

    # Strategy 2: Greedy construction
    greedy_labels = _greedy_partition(adjacency, n)
    greedy_labels, greedy_cut = _local_search(adjacency, greedy_labels, deadline)
    _update_best(greedy_labels, greedy_cut)

    # Strategy 3: Alternating
    alt_labels = np.arange(n) % 2
    alt_labels, alt_cut = _local_search(adjacency, alt_labels, deadline)
    _update_best(alt_labels, alt_cut)

    # Strategy 4: Multiple random restarts with local search
    base_seed = 42
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        seed = base_seed + restart
        labels = _random_labels(n, seed)
        labels, cut = _local_search(adjacency, labels, deadline)
        _update_best(labels, cut)

    # Strategy 5: Simulated annealing from best
    if best_labels is not None and time.time() < deadline - 10:
        sa_labels, sa_cut = _simulated_annealing(adjacency, best_labels, deadline)
        _update_best(sa_labels, sa_cut)

    # Strategy 6: Tabu search from best
    if best_labels is not None and time.time() < deadline - 5:
        tabu_labels, tabu_cut = _tabu_search(adjacency, best_labels, deadline)
        _update_best(tabu_labels, tabu_cut)

    # Strategy 7: More random restarts if time remains
    if time.time() < deadline - 10:
        for restart in range(10):
            if time.time() > deadline:
                break
            seed = base_seed + 100 + restart
            labels = _random_labels(n, seed)
            labels, cut = _local_search(adjacency, labels, deadline)
            _update_best(labels, cut)

    if best_labels is None:
        best_labels = np.arange(n) % 2

    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
