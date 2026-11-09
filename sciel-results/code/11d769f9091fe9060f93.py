"""Maximum Cut solver using spectral initialization + WalkSAT-style stochastic local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adj & diff).sum() // 2)


def _spectral_partition(adj):
    """
    Use spectral method (Fiedler vector of Laplacian) to get an initial partition.
    """
    n = len(adj)
    A = adj.astype(np.float64)
    d = A.sum(axis=1)
    # Laplacian L = D - A
    L = np.diag(d) - A
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        L_sparse = csr_matrix(L)
        # Get the second smallest eigenvalue's eigenvector (Fiedler vector)
        # For connected graph, the smallest eigenvalue is 0 with eigenvector [1,1,...,1]
        vals, vecs = eigsh(L_sparse, k=2, which='SM')
        fiedler = vecs[:, 1]
    except Exception:
        # Fallback to dense
        try:
            vals, vecs = np.linalg.eigh(L)
            fiedler = vecs[:, 1]
        except Exception:
            fiedler = np.random.randn(n)
    
    # Partition based on sign of Fiedler vector
    labels = (fiedler >= 0).astype(np.int8)
    return labels


def _fast_local_search(adj, labels, deadline, max_iters=10000):
    """
    Fast greedy local search: repeatedly flip the vertex with maximum positive gain.
    Uses gain array for O(deg) updates per flip.
    """
    n = len(labels)
    labels = labels.copy()
    
    # Precompute degree
    deg = adj.sum(axis=1)
    
    # Initialize gain array
    # gain[v] = (# neighbors in opposite set) - (# neighbors in same set)
    diff_mask = (labels[None, :] != labels[:, None])
    diff_counts = (adj * diff_mask).sum(axis=1)
    gains = 2 * diff_counts - deg
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        # Find the vertex with the maximum positive gain
        max_gain_idx = np.argmax(gains)
        max_gain_val = gains[max_gain_idx]
        
        if max_gain_val <= 0:
            break
        
        v = max_gain_idx
        # Flip v
        labels[v] = 1 - labels[v]
        
        # Update gains for neighbors of v
        neighbors = np.where(adj[v] == 1)[0]
        
        for u in neighbors:
            if labels[u] != labels[v]:
                diff_counts[u] += 1
            else:
                diff_counts[u] -= 1
            gains[u] = 2 * diff_counts[u] - deg[u]
        
        # Update v's own gain
        diff_counts[v] = (adj[v] * (labels != labels[v])).sum()
        gains[v] = 2 * diff_counts[v] - deg[v]
    
    return labels


def _walksat_search(adj, labels, deadline, max_iters=50000):
    """
    WalkSAT-style stochastic local search for Max-Cut.
    Formulates as Max-2-SAT where each edge (u,v) is a clause (x_u != x_v).
    An uncut edge is an unsatisfied clause.
    """
    n = len(labels)
    labels = labels.copy()
    
    # Precompute adjacency lists
    adj_lists = [np.where(adj[i] == 1)[0] for i in range(n)]
    
    # Compute initial unsatisfied edges count and per-vertex unsat counts
    # unsat[v] = number of edges (v,u) where labels[v] == labels[u]
    unsat = np.zeros(n, dtype=np.int32)
    for v in range(n):
        for u in adj_lists[v]:
            if u > v and labels[v] == labels[u]:
                unsat[v] += 1
                unsat[u] += 1
    
    total_unsat = unsat.sum() // 2
    best_labels = labels.copy()
    best_unsat = total_unsat
    
    rng = np.random.RandomState(42)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        if total_unsat == 0:
            break
        
        # Pick a random unsatisfied edge
        # Build list of unsatisfied edges
        unsat_edges = []
        for v in range(n):
            if unsat[v] > 0:
                for u in adj_lists[v]:
                    if u > v and labels[v] == labels[u]:
                        unsat_edges.append((v, u))
        
        if not unsat_edges:
            break
        
        v, u = unsat_edges[rng.randint(len(unsat_edges))]
        
        # Try flipping v or u
        # Compute the gain of flipping each
        # Flipping v: changes unsat count for edges incident to v
        # For each neighbor w of v:
        #   If labels[v] == labels[w], edge (v,w) is unsat. After flip, it becomes sat. unsat decreases by 1.
        #   If labels[v] != labels[w], edge (v,w) is sat. After flip, it becomes unsat. unsat increases by 1.
        # Net change in unsat for flipping v = (# sat neighbors) - (# unsat neighbors)
        # = (deg[v] - unsat[v]) - unsat[v] = deg[v] - 2*unsat[v]
        # But we want to minimize unsat, so we want negative change.
        # Change in total_unsat if we flip v = (unsat[v] - (deg[v] - unsat[v])) = 2*unsat[v] - deg[v]
        # Wait, let me think again.
        # Currently, for neighbor w:
        #   If labels[v] == labels[w]: edge is unsat. After flipping v, labels[v] != labels[w], so edge becomes sat. So unsat[v] decreases by 1, unsat[w] decreases by 1.
        #   If labels[v] != labels[w]: edge is sat. After flipping v, labels[v] == labels[w], so edge becomes unsat. So unsat[v] increases by 1, unsat[w] increases by 1.
        # Number of unsat neighbors of v = unsat[v]
        # Number of sat neighbors of v = deg[v] - unsat[v]
        # After flip:
        #   The unsat neighbors become sat: unsat[v] decreases by unsat[v], unsat[those neighbors] each decrease by 1
        #   The sat neighbors become unsat: unsat[v] increases by (deg[v] - unsat[v]), unsat[those neighbors] each increase by 1
        # Net change in unsat[v] = -unsat[v] + (deg[v] - unsat[v]) = deg[v] - 2*unsat[v]
        # Net change in total unsat edges = (deg[v] - 2*unsat[v]) / 2? No.
        # Each edge is counted once in total_unsat. When we flip v, each incident edge changes state.
        # An unsat edge becomes sat: total_unsat decreases by 1.
        # A sat edge becomes unsat: total_unsat increases by 1.
        # So delta_total = (number of sat neighbors) - (number of unsat neighbors) = (deg[v] - unsat[v]) - unsat[v] = deg[v] - 2*unsat[v]
        
        gain_v = -(adj[v].sum() - 2 * unsat[v])  # negative means unsat decreases
        gain_u = -(adj[u].sum() - 2 * unsat[u])
        
        # With probability p, flip the one with better gain; otherwise flip randomly
        p = 0.8
        if rng.random() < p:
            if gain_v <= gain_u:
                flip_vertex = v
            else:
                flip_vertex = u
        else:
            flip_vertex = v if rng.random() < 0.5 else u
        
        # Perform the flip
        w = flip_vertex
        old_label = labels[w]
        labels[w] = 1 - old_label
        
        # Update unsat counts
        deg_w = adj[w].sum()
        old_unsat_w = unsat[w]
        
        for neighbor in adj_lists[w]:
            if labels[w] != labels[neighbor]:
                # Edge is now sat, was unsat
                unsat[w] -= 1
                unsat[neighbor] -= 1
            else:
                # Edge is now unsat, was sat
                unsat[w] += 1
                unsat[neighbor] += 1
        
        total_unsat = unsat.sum() // 2
        
        if total_unsat < best_unsat:
            best_unsat = total_unsat
            best_labels = labels.copy()
    
    return best_labels


def _random_labels(n, seed=None):
    """Generate random partition labels."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    return rng.randint(0, 2, size=n).astype(np.int8)


def partition(adjacency):
    """Main partition function using spectral init + local search + WalkSAT."""
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.int8)
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_labels = None
    best_cut = -1
    
    # Phase 1: Spectral initialization + local search
    spectral_labels = _spectral_partition(adj)
    refined = _fast_local_search(adj, spectral_labels, deadline)
    cut = _cut_size(adj, refined)
    if cut > best_cut:
        best_cut = cut
        best_labels = refined.copy()
    
    # Phase 2: Multiple random restarts with local search
    max_restarts = 200
    for i in range(max_restarts):
        if time.time() > deadline:
            break
        labels = _random_labels(n)
        improved = _fast_local_search(adj, labels, deadline)
        cut = _cut_size(adj, improved)
        if cut > best_cut:
            best_cut = cut
            best_labels = improved.copy()
    
    # Phase 3: WalkSAT from the best solution found so far
    if best_labels is not None and time.time() < deadline:
        ws_labels = _walksat_search(adj, best_labels, deadline)
        cut = _cut_size(adj, ws_labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = ws_labels.copy()
    
    if best_labels is None:
        return _random_labels(n)
    
    return best_labels
# EVOLVE-BLOCK-END
