# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    diff = np.abs(labels[:, np.newaxis] - labels[np.newaxis, :])
    return np.sum(adj * diff) / 2.0


def _all_flip_gains(adj, labels):
    """Compute the gain from flipping each vertex v.
    gain[v] = (neighbors in same set as v) - (neighbors in opposite set)
    Vectorized: for each v, same = sum(adj[v] * (labels==labels[v])), opp = sum(adj[v] * (labels!=labels[v]))
    """
    n = len(labels)
    # For each vertex v:
    #   same_count[v] = sum over u of adj[v,u] * (labels[u] == labels[v])
    #   opp_count[v] = sum over u of adj[v,u] * (labels[u] != labels[v])
    #   gain[v] = same_count[v] - opp_count[v]
    
    # labels as float: 0 or 1
    lab_f = labels.astype(np.float64)
    # same_mask[v,u] = 1 if labels[u]==labels[v] else 0
    # = 1 - |labels[v] - labels[u]| = 1 - (labels[v] + labels[u] - 2*labels[v]*labels[u]) for binary
    # Actually simpler: (labels[v] == labels[u]) = 1 - abs(labels[v]-labels[u])
    # abs diff matrix:
    absdiff = np.abs(lab_f[:, np.newaxis] - lab_f[np.newaxis, :])
    same = 1.0 - absdiff  # 1 where same, 0 where different
    opp = absdiff  # 1 where different, 0 where same
    
    # same_count[v] = sum_u adj[v,u] * same[v,u]
    same_count = adj @ same.sum(axis=1)  # WRONG: this is not right
    # Let me redo:
    # same_count[v] = sum_u adj[v,u] * (1 - abs(labels[v]-labels[u]))
    # = sum_u adj[v,u] - sum_u adj[v,u] * abs(labels[v]-labels[u])
    # = degree[v] - (adj @ absdiff_col)[v] where absdiff_col[u] = abs(labels[v]-labels[u])... no, it's per-v.
    
    # Better: 
    # gain[v] = sum_u adj[v,u] * (2*(labels[v]==labels[u]) - 1)
    # = sum_u adj[v,u] * (1 - 2*abs(labels[v]-labels[u]))  ... wait
    # if labels[v]==labels[u]: abs=0, term = 1
    # if labels[v]!=labels[u]: abs=1, term = -1
    # So gain[v] = sum_u adj[v,u] * (1 - 2*abs(labels[v]-labels[u]))
    
    # Compute: for each v, sum_u adj[v,u] * (1 - 2*abs(labels[v]-labels[u]))
    # = degree[v] - 2 * sum_u adj[v,u] * abs(labels[v]-labels[u])
    
    # abs(labels[v]-labels[u]) = labels[v]*labels[u]... no.
    # For binary: abs(a-b) = a+b-2ab
    # So sum_u adj[v,u] * abs(labels[v]-labels[u]) = sum_u adj[v,u]*(labels[v]+labels[u]-2*labels[v]*labels[u])
    # = labels[v]*sum_u adj[v,u] + sum_u adj[v,u]*labels[u] - 2*labels[v]*sum_u adj[v,u]*labels[u]
    # = labels[v]*deg[v] + (adj@labels)[v] - 2*labels[v]*(adj@labels)[v]
    
    deg = adj.sum(axis=1)
    adj_labels = adj @ lab_f  # (adj@labels)[v] = sum_u adj[v,u]*labels[u]
    
    # sum_u adj[v,u]*abs(labels[v]-labels[u]) = labels[v]*deg[v] + adj_labels[v] - 2*labels[v]*adj_labels[v]
    abs_sum = lab_f * deg + adj_labels - 2.0 * lab_f * adj_labels
    
    gain = deg - 2.0 * abs_sum
    return gain


def _local_search_1flip(adj, labels, max_time=None, start_time=None):
    """Greedy 1-flip local search using vectorized gain computation."""
    n = len(labels)
    if start_time is None:
        start_time = time.time()
    if max_time is None:
        max_time = start_time + 100
    
    improved = True
    while improved:
        improved = False
        if time.time() > max_time:
            break
        gains = _all_flip_gains(adj, labels)
        # Find all vertices with positive gain, flip the best one
        best_v = int(np.argmax(gains))
        if gains[best_v] > 0:
            labels[best_v] = 1 - labels[best_v]
            improved = True
    return labels


def _spectral_init(adj, n):
    """Initialize labels using the Fiedler vector (eigenvector of Laplacian)."""
    degree = adj.sum(axis=1)
    L = np.diag(degree) - adj
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        idx = np.argsort(eigvals)
        fiedler = eigvecs[:, idx[1]]
        labels = (fiedler > 0).astype(int)
    except:
        labels = np.arange(n) % 2
    return labels


def _go_heuristic(adj, labels, max_time=None):
    """Greedy Ordering Heuristic (GOH) for Max-Cut.
    Process vertices in a given order, assign each to the side that maximizes cut edges to already-assigned neighbors.
    """
    n = len(labels)
    if max_time is None:
        max_time = time.time() + 100
    
    # Work on a copy
    lab = labels.copy()
    assigned = np.zeros(n, dtype=bool)
    
    # Process in order 0..n-1
    for v in range(n):
        if time.time() > max_time:
            break
        # Count edges to already-assigned neighbors in each partition
        # neighbors of v that are already assigned
        neighbors = np.where(adj[v] > 0)[0]
        assigned_neighbors = neighbors[assigned[neighbors]]
        if len(assigned_neighbors) == 0:
            # No assigned neighbors, assign to 0
            lab[v] = 0
        else:
            # Count how many assigned neighbors are in set 0 and set 1
            in_set0 = (lab[assigned_neighbors] == 0).sum()
            in_set1 = (lab[assigned_neighbors] == 1).sum()
            # Assign to the side with fewer assigned neighbors (to maximize cut)
            if in_set0 < in_set1:
                lab[v] = 0
            elif in_set1 < in_set0:
                lab[v] = 1
            else:
                # Tie: random
                lab[v] = random.randint(0, 1)
        assigned[v] = True
    
    return lab


def _simulated_annealing(adj, labels, max_iters=2000, max_time=None, t0=10.0, t_end=0.1, cooling=0.999):
    """Simulated annealing for Max-Cut using 1-flip moves."""
    n = len(labels)
    if max_time is None:
        max_time = time.time() + 100
    
    start_time = time.time()
    cur_labels = labels.copy()
    cur_cut = _cut_size(adj, cur_labels)
    best_labels = cur_labels.copy()
    best_cut = cur_cut
    
    t = t0
    for it in range(max_iters):
        if time.time() > max_time:
            break
        t = max(t_end, t * cooling)
        
        # Pick a random vertex to flip
        v = random.randint(0, n - 1)
        gain = _all_flip_gains(adj, cur_labels)[v]
        
        if gain > 0:
            # Always accept improving moves
            cur_labels[v] = 1 - cur_labels[v]
            cur_cut += gain
        else:
            # Accept worsening moves with probability exp(gain/t)
            if gain < 0 and random.random() < np.exp(gain / t):
                cur_labels[v] = 1 - cur_labels[v]
                cur_cut += gain
        
        if cur_cut > best_cut:
            best_cut = cur_cut
            best_labels = cur_labels.copy()
    
    return best_labels, best_cut


def _tabu_search(adj, labels, tabu_len=7, max_iters=500, max_time=None):
    """Cerny tabu search for maximum cut."""
    n = len(labels)
    start_time = time.time()
    if max_time is None:
        max_time = start_time + 100
    
    best_labels = labels.copy()
    best_cut = _cut_size(adj, labels)
    current_cut = best_cut
    
    # Tabu list: stores the iteration until which each vertex is tabu
    tabu_until = np.zeros(n, dtype=int)
    current_iter = 0
    
    for it in range(max_iters):
        if time.time() > max_time:
            break
        current_iter = it
        
        gains = _all_flip_gains(adj, labels)
        
        # Find the best move (considering tabu restrictions)
        best_gain = -np.inf
        best_v = -1
        
        for v in range(n):
            is_tabu = (it < tabu_until[v])
            
            # Aspiration criterion: accept tabu move if it improves best
            if is_tabu and gains[v] > best_cut:
                is_tabu = False
            
            if is_tabu:
                continue
            
            if gains[v] > best_gain:
                best_gain = gains[v]
                best_v = v
        
        if best_v == -1:
            # All moves are tabu, pick the best one anyway
            for v in range(n):
                if gains[v] > best_gain:
                    best_gain = gains[v]
                    best_v = v
            if best_v == -1:
                break
        
        # Perform the flip
        labels[best_v] = 1 - labels[best_v]
        tabu_until[best_v] = current_iter + tabu_len
        current_cut += best_gain
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
    
    return best_labels, best_cut


def partition(adjacency):
    n = len(adjacency)
    adj = np.array(adjacency, dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit
    
    best_labels = None
    best_cut = -1
    
    np.random.seed(42)
    random.seed(42)
    
    # Strategy: Multiple diverse initializations, each followed by
    # local search -> tabu search -> simulated annealing
    
    inits = []
    
    # Spectral
    spec_labels = _spectral_init(adj, n)
    inits.append(spec_labels)
    
    # Alternating
    alt_labels = np.arange(n) % 2
    inits.append(alt_labels)
    
    # Random inits
    for _ in range(6):
        rand_labels = np.random.randint(0, 2, n).astype(int)
        inits.append(rand_labels)
    
    # GOH with different orderings
    for _ in range(3):
        order = list(range(n))
        random.shuffle(order)
        g_labels = _go_heuristic(adj, np.zeros(n, dtype=int), max_time=deadline)
        # The GOH uses natural order; to get diversity, we permute the graph
        # Actually, let's just use random inits for simplicity
        inits.append(g_labels)
    
    for init_idx, init_labels in enumerate(inits):
        if time.time() > deadline:
            break
        
        labels = init_labels.copy()
        
        # Phase 1: Local search to get to a local optimum
        labels = _local_search_1flip(adj, labels, max_time=deadline)
        cut = _cut_size(adj, labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # Phase 2: Tabu search starting from this local optimum
        if time.time() > deadline:
            break
        
        ts_labels, ts_cut = _tabu_search(adj, labels.copy(), tabu_len=7, max_iters=300, max_time=deadline)
        
        # Phase 3: Final local search on the tabu result
        ts_labels = _local_search_1flip(adj, ts_labels, max_time=deadline)
        ts_cut = _cut_size(adj, ts_labels)
        
        if ts_cut > best_cut:
            best_cut = ts_cut
            best_labels = ts_labels.copy()
        
        # Phase 4: Simulated annealing from the best so far
        if time.time() > deadline:
            break
        
        sa_labels, sa_cut = _simulated_annealing(adj, best_labels.copy(), max_iters=1000, max_time=deadline)
        sa_labels = _local_search_1flip(adj, sa_labels, max_time=deadline)
        sa_cut = _cut_size(adj, sa_labels)
        
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
