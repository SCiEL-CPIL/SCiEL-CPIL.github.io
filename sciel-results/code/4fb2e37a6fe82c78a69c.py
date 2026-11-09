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


def _two_flip_gain(adjacency, labels, u, v):
    """Compute the change in cut size if we flip both vertices u and v."""
    lu = labels[u]
    lv = labels[v]
    
    # Gain from flipping u alone (before v is flipped)
    gain_u = _flip_gain(adjacency, labels, u)
    # Gain from flipping v alone (before u is flipped)
    gain_v = _flip_gain(adjacency, labels, v)
    
    # Interaction term: if u and v are connected, flipping both changes the edge status twice
    # Actually, let's compute directly
    lu_new = 1 - lu
    lv_new = 1 - lv
    
    # Edges from u: those to same-side become cross, those to opposite become non-cross
    # After both flips:
    # For neighbor w of u:
    #   if w == v: edge u-v was (lu,lv), now (lu_new, lv_new)
    #   else: edge u-w was (lu, labels[w]), now (lu_new, labels[w])
    
    # Direct computation
    old_cut_u = 0
    old_cut_v = 0
    new_cut_u = 0
    new_cut_v = 0
    
    for w in np.where(adjacency[u])[0]:
        if w == v:
            continue
        if labels[w] != lu:
            old_cut_u += 1
        if labels[w] != lu_new:
            new_cut_u += 1
    
    for w in np.where(adjacency[v])[0]:
        if w == u:
            continue
        if labels[w] != lv:
            old_cut_v += 1
        if labels[w] != lv_new:
            new_cut_v += 1
    
    # Edge u-v
    if adjacency[u, v]:
        if lu != lv:
            old_cut_uv = 1
        else:
            old_cut_uv = 0
        if lu_new != lv_new:
            new_cut_uv = 1
        else:
            new_cut_uv = 0
    else:
        old_cut_uv = 0
        new_cut_uv = 0
    
    # Total old cut contribution from u and v (excluding double-counting u-v)
    old_total = old_cut_u + old_cut_v + old_cut_uv
    new_total = new_cut_u + new_cut_v + new_cut_uv
    
    return new_total - old_total


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
            # Count same-side and opposite-side neighbors
            same_count = adjacency[v][labels == labels[v]].sum()
            other_count = adjacency[v][labels != labels[v]].sum()
            
            if same_count > other_count:
                labels[v] = 1 - labels[v]
                current_cut += (same_count - other_count)
                improved = True
    
    # 2-flip local search to escape local optima
    if time.time() < deadline - 2:
        two_improved = True
        while two_improved and time.time() < deadline - 1:
            two_improved = False
            order = np.random.permutation(n)
            for i in range(n):
                if time.time() > deadline - 1:
                    break
                u = order[i]
                for j in range(i + 1, n):
                    v = order[j]
                    if labels[u] == labels[v]:
                        gain = _two_flip_gain(adjacency, labels, u, v)
                        if gain > 0:
                            labels[u] = 1 - labels[u]
                            labels[v] = 1 - labels[v]
                            current_cut += gain
                            two_improved = True
                            break
                if two_improved:
                    break
    
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


def _hyperplane_partition(adjacency, n, num_hyperplanes=50):
    """Goemans-Williamson style: generate random hyperplane cuts from Fiedler vector.
    
    The Fiedler vector gives a 1D embedding where vertices are separated by a hyperplane.
    We rotate the hyperplane slightly around the mean direction to get diverse partitions.
    """
    try:
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
        
        # Normalize the Fiedler vector
        fiedler_vec = fiedler_vec / np.linalg.norm(fiedler_vec)
        
        # Generate random hyperplanes by rotating around the Fiedler vector
        # In 1D, this means adding a small random perturbation
        best_labels = None
        best_cut = -1
        
        for _ in range(num_hyperplanes):
            # Create a perturbed version of the Fiedler vector
            # This simulates rotating the hyperplane
            noise = np.random.normal(0, 0.1, size=n)
            perturbed_vec = fiedler_vec + noise
            
            # Partition based on sign of perturbed vector
            labels = (perturbed_vec > 0).astype(np.int8)
            
            # Ensure both sides have vertices
            if labels.sum() == 0 or labels.sum() == n:
                continue
            
            cut = _cut_size(adjacency, labels)
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
        
        if best_labels is not None:
            return best_labels
        else:
            return np.arange(n) % 2
    except Exception:
        return np.arange(n) % 2


def _simulated_annealing(adjacency, labels, deadline, start_temp=10.0, cool_rate=0.995):
    """Simulated Annealing for maximum cut."""
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    best_cut = current_cut
    best_labels = labels.copy()
    
    temp = start_temp
    rng = np.random.RandomState(42)
    
    while temp > 0.01 and time.time() < deadline:
        # Pick a random vertex to flip
        v = rng.randint(0, n)
        gain = _flip_gain(adjacency, labels, v)
        
        if gain > 0:
            # Always accept improving moves
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
        
        temp *= cool_rate
    
    return best_labels, best_cut


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
    
    # Strategy 2: Hyperplane partitioning (Goemans-Williamson style)
    if n >= 2 and time.time() < deadline - 10:
        hp_labels = _hyperplane_partition(adjacency, n, num_hyperplanes=30)
        hp_labels, hp_cut = _local_search(adjacency, hp_labels, deadline)
        _update_best(hp_labels, hp_cut)
    
    # Strategy 3: Greedy construction
    greedy_labels = _greedy_partition(adjacency, n)
    greedy_labels, greedy_cut = _local_search(adjacency, greedy_labels, deadline)
    _update_best(greedy_labels, greedy_cut)
    
    # Strategy 4: Simulated Annealing
    if time.time() < deadline - 15:
        sa_labels = np.arange(n) % 2
        sa_labels, sa_cut = _simulated_annealing(adjacency, sa_labels, deadline - 5)
        sa_labels, sa_cut2 = _local_search(adjacency, sa_labels, deadline)
        _update_best(sa_labels, sa_cut2)
    
    # Strategy 5: Multiple random restarts with local search
    base_seed = 42
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        seed = base_seed + restart
        labels = _random_labels(n, seed)
        labels, cut = _local_search(adjacency, labels, deadline)
        _update_best(labels, cut)
    
    # Strategy 6: Perturbation-based restarts from best
    if best_labels is not None and time.time() < deadline - 5:
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            perturbed = best_labels.copy()
            num_flips = max(1, n // 15)
            flip_indices = np.random.choice(n, size=num_flips, replace=False)
            perturbed[flip_indices] = 1 - perturbed[flip_indices]
            perturbed, pert_cut = _local_search(adjacency, perturbed, deadline)
            _update_best(perturbed, pert_cut)
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
