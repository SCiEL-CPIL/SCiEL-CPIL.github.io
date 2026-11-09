# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # labels is 0/1. cut = sum over v of (number of neighbors of v with label != labels[v])
    # = sum_v adj[v] @ (1 - labels) ... but this counts each edge twice? No.
    # For an undirected graph, edge (v,u) contributes 1 to cut if labels differ.
    # sum_v (adj[v] @ (1-labels)) counts for each v, the number of neighbors not in same set.
    # If v is in set 0, it counts neighbors in set 1. If v is in set 1, it counts neighbors in set 0.
    # Each crossing edge (v,u) with v in 0, u in 1: counted once in v's sum (u is in 1, 1-labels[u]=0? No.)
    # Let's be careful.
    # adj[v] is a row. (1-labels) is a vector.
    # adj[v] @ (1-labels) = sum_u adj[v,u] * (1-labels[u]).
    # If labels[v]=0, we want to count neighbors with labels[u]=1. (1-labels[u]) is 1 when labels[u]=0. That's wrong.
    # We want neighbors with labels[u] != labels[v].
    # If labels[v]=0, we want labels[u]=1.
    # If labels[v]=1, we want labels[u]=0.
    # So we want adj[v] @ (labels != labels[v]).
    # This is equivalent to:
    # If labels[v]=0: adj[v] @ (labels)  (since labels[u]=1 for opposite)
    # If labels[v]=1: adj[v] @ (1-labels) (since labels[u]=0 for opposite)
    # More simply: cut = 0.5 * sum_v sum_u adj[v,u] * |labels[v] - labels[u]|
    # Since labels are 0/1, |labels[v]-labels[u]| is 1 if different, 0 if same.
    # Vectorized:
    # diff = np.abs(labels[:, np.newaxis] - labels[np.newaxis, :])
    # cut = np.sum(adj * diff) / 2.0
    # But for 100 nodes, this is 10000 operations, very fast.
    # Alternatively:
    # cut = np.sum(adj @ (1 - 2*labels + 1) ... no.
    # Let's use the efficient matrix trick:
    # Let x be the label vector (0/1).
    # cut = x^T (D - A) (1-x) ... no.
    # cut = sum_{v<u} A_{vu} (x_v != x_u)
    # = 1/2 sum_{v,u} A_{vu} (x_v != x_u)
    # = 1/2 sum_{v,u} A_{vu} (1 - (1-2x_v)(1-2x_u)/2 ... )
    # Simpler:
    # Let s = 2x - 1 (so s is -1/1).
    # x_v != x_u iff s_v != s_u iff s_v s_u = -1.
    # x_v == x_u iff s_v s_u = 1.
    # cut = 1/2 sum_{v,u} A_{vu} (1 if s_v != s_u else 0)
    # = 1/2 sum_{v,u} A_{vu} (1 - s_v s_u)/2 ... no.
    # Indicator(s_v != s_u) = (1 - s_v s_u) / 2.
    # cut = 1/2 sum_{v,u} A_{vu} (1 - s_v s_u) / 2
    # = 1/4 (sum_{v,u} A_{vu} - sum_{v,u} A_{vu} s_v s_u)
    # sum_{v,u} A_{vu} = 2 * E (where E is number of edges).
    # sum_{v,u} A_{vu} s_v s_u = s^T A s.
    # cut = 1/4 (2E - s^T A s) = E/2 - (s^T A s)/4.
    # Since s^T A s = (2x-1)^T A (2x-1) = 4 x^T A x - 2 x^T A 1 - 2 1^T A x + 1^T A 1
    # This is getting complex. Let's just use the direct method for n=100.
    # It's O(n^2) which is 10^4, negligible.
    diff = np.abs(labels[:, np.newaxis] - labels[np.newaxis, :])
    return np.sum(adj * diff) / 2.0


def _greedy_flip_improve(adj, labels):
    """Improve partition by greedy flipping until no improvement."""
    n = len(labels)
    improved = True
    while improved:
        improved = False
        for v in range(n):
            # Compute gain from flipping v
            # Current contribution of v to cut: number of neighbors in opposite set
            # New contribution: number of neighbors in same set
            # gain = new - old
            # old = adj[v] @ (1-labels) if labels[v]=0, else adj[v] @ labels
            # Actually, let's compute the change.
            # If we flip v, the edges incident to v change status.
            # Edges to same set become crossing, edges to opposite become non-crossing.
            # Let same_count = number of neighbors in same set
            # Let opp_count = number of neighbors in opposite set
            # Current cut contribution from v's edges: opp_count
            # New cut contribution from v's edges: same_count
            # Gain = same_count - opp_count
            # Note: this gain is counted once for v, but the edge (v,u) is also counted for u.
            # However, when we flip v, only the status of edges (v,u) changes.
            # The total cut size change is exactly same_count - opp_count.
            # Because each edge (v,u) contributes 1 to cut if they differ.
            # Before: contributes 1 if labels[v]!=labels[u].
            # After: contributes 1 if (1-labels[v])!=labels[u] i.e. labels[v]==labels[u].
            # So the change for edge (v,u) is:
            # If labels[v]==labels[u]: 0 -> 1 (gain +1)
            # If labels[v]!=labels[u]: 1 -> 0 (gain -1)
            # Total gain = sum_{u: labels[v]==labels[u]} 1 + sum_{u: labels[v]!=labels[u]} (-1)
            # = same_count - opp_count
            same_mask = (labels == labels[v])
            opp_mask = (labels != labels[v])
            same_count = adj[v][same_mask].sum()
            opp_count = adj[v][opp_mask].sum()
            if same_count > opp_count:
                labels[v] = 1 - labels[v]
                improved = True
    return labels


def _spectral_init(adj, n):
    """Initialize labels using the Fiedler vector (eigenvector of Laplacian)."""
    # Laplacian L = D - A
    degree = adj.sum(axis=1)
    L = np.diag(degree) - adj
    # Find eigenvector corresponding to smallest non-zero eigenvalue
    # Use numpy's eigh for symmetric matrices
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # The Fiedler vector is the eigenvector with the second smallest eigenvalue
        # (smallest is 0 for connected graphs, corresponding to constant vector)
        # Sort by eigenvalue
        idx = np.argsort(eigvals)
        # Take the second smallest (index 1)
        fiedler = eigvecs[:, idx[1]]
        # Partition: positive -> 1, negative -> 0
        labels = (fiedler > 0).astype(int)
    except:
        # Fallback to random or alternating
        labels = np.arange(n) % 2
    return labels


def partition(adjacency):
    n = len(adjacency)
    adj = np.array(adjacency, dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_labels = None
    best_cut = -1
    
    # Strategy: Multiple restarts with different initializations
    # 1. Spectral init
    # 2. Random inits
    # 3. Alternating init
    
    inits = []
    
    # Spectral
    spec_labels = _spectral_init(adj, n)
    inits.append(spec_labels)
    
    # Alternating
    alt_labels = np.arange(n) % 2
    inits.append(alt_labels)
    
    # Random inits
    np.random.seed(42)
    for _ in range(5):
        rand_labels = np.random.randint(0, 2, n)
        inits.append(rand_labels)
    
    for init_labels in inits:
        if time.time() > deadline:
            break
        
        labels = init_labels.copy()
        
        # Greedy improvement
        labels = _greedy_flip_improve(adj, labels)
        
        cut = _cut_size(adj, labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # Simulated annealing for further improvement
        if time.time() > deadline:
            break
        
        sa_labels = labels.copy()
        current_cut = cut
        temperature = 5.0
        cooling = 0.95
        
        sa_improved = True
        while sa_improved and temperature > 0.01:
            sa_improved = False
            for v in range(n):
                if time.time() > deadline:
                    break
                
                # Compute gain from flipping v
                same_mask = (sa_labels == sa_labels[v])
                opp_mask = (sa_labels != sa_labels[v])
                same_count = adj[v][same_mask].sum()
                opp_count = adj[v][opp_mask].sum()
                gain = same_count - opp_count
                
                if gain > 0:
                    sa_labels[v] = 1 - sa_labels[v]
                    current_cut += gain
                    sa_improved = True
                else:
                    # Accept with probability exp(gain / temperature)
                    if np.random.random() < np.exp(gain / temperature):
                        sa_labels[v] = 1 - sa_labels[v]
                        current_cut += gain
                        sa_improved = True
            
            temperature *= cooling
        
        sa_cut = _cut_size(adj, sa_labels)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
    
    if best_labels is None:
        # Fallback
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
