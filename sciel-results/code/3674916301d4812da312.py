"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # labels are 0/1. An edge (i,j) crosses if labels[i] != labels[j].
    # cut = sum_{i<j} adj[i,j] * (labels[i] != labels[j])
    # Using matrix ops: let x = 2*labels - 1 (values in {-1, 1})
    # Then (labels[i] != labels[j]) corresponds to x[i]*x[j] = -1
    # cut = 0.5 * (sum_{i,j} adj[i,j] * (1 - x[i]*x[j]) / 2) ... let me think again
    # Actually: cut = 0.5 * sum_{i,j} adj[i,j] * (1 - x[i]*x[j]) / 2
    # Wait: for i != j, adj[i,j] = adj[j,i]. The sum over all i,j of adj[i,j]*(1-x[i]x[j])
    # = sum_{i!=j} adj[i,j]*(1-x[i]x[j]) since adj[i,i]=0 typically
    # For an edge (i,j) with i!=j: it contributes adj[i,j]*(1-x[i]x[j]) + adj[j,i]*(1-x[j]x[i])
    # = 2*adj[i,j]*(1-x[i]x[j])
    # If labels differ: x[i]x[j]=-1, so 1-(-1)=2, contribution = 2*adj[i,j]*2 = 4*adj[i,j]
    # If labels same: x[i]x[j]=1, so 1-1=0, contribution = 0
    # So total sum = 4 * cut
    # Thus cut = sum_{i,j} adj[i,j]*(1-x[i]x[j]) / 4
    # But we can also just do: for each vertex v, count neighbors with different label
    # cut = 0.5 * sum_v (adj[v] @ (1 - labels)) if labels are 0/1? 
    # No: adj[v] @ (1-labels) counts edges from v to vertices with label 0 (if labels[v]=1) or label 1 (if labels[v]=0)... 
    # Actually simpler: cut = sum_v sum_u adj[v,u] * (1 - labels[v]*labels[u] - (1-labels[v])*(1-labels[u]))
    # That's getting complicated. Let me just use the x formulation.
    x = 2 * labels - 1  # -1 or 1
    # cut = 0.25 * x @ adj @ x ... no
    # Let's just compute directly with numpy
    # For each vertex, the number of edges to opposite side:
    # If label[v] = 0: edges to vertices with label 1
    # If label[v] = 1: edges to vertices with label 0
    # total cut = sum_v (adj[v] @ (1 - labels)) if we consider each edge twice? 
    # adj[v] @ (1-labels) for a vertex v with label 0: counts neighbors with label 1 (correct, these are crossing)
    # adj[v] @ (1-labels) for a vertex v with label 1: counts neighbors with label 0 (correct, these are crossing)
    # So sum_v adj[v] @ (1-labels) counts each crossing edge exactly twice (once from each end)
    # Therefore cut = 0.5 * sum_v adj[v] @ (1-labels)
    # = 0.5 * (adj @ (1-labels)).sum()
    return 0.5 * (adj @ (1 - labels)).sum()


def _greedy_flip_improve(adj, labels, max_passes=100, time_limit=None, start_time=None):
    """
    Fast greedy local search: repeatedly flip vertices that improve the cut.
    Use dot products for O(N) per flip decision.
    """
    n = len(labels)
    labels = labels.copy()
    
    if time_limit is None:
        time_limit = 150.0
    if start_time is None:
        start_time = time.time()
    
    # Precompute degree
    degrees = adj.sum(axis=1)
    
    # For each vertex v, define:
    # same_count[v] = number of neighbors of v with same label
    # other_count[v] = number of neighbors of v with different label
    # Flipping v changes cut by: other_count[v] - same_count[v]
    # (we lose other_count edges that were crossing, gain same_count edges that become crossing)
    # So flip if same_count[v] > other_count[v], i.e., same_count[v] > degree[v]/2
    
    # Efficient update: maintain for each vertex the count of same-label neighbors
    # When we flip vertex u, for each neighbor v of u:
    #   if v had same label as u, now it's different: same_count[v] decreases by 1
    #   if v had different label as u, now it's same: same_count[v] increases by 1
    
    # Let's maintain same_count array
    same_count = np.zeros(n, dtype=np.int32)
    for v in range(n):
        same_count[v] = adj[v] @ labels  # number of neighbors with same label (since labels are 0/1)
    
    # Actually adj[v] @ labels gives sum of adj[v,u]*labels[u] for all u
    # If labels[v]=0: same-label neighbors have label 0, so adj[v]@(1-labels) counts same
    # If labels[v]=1: same-label neighbors have label 1, so adj[v]@labels counts same
    # Let me redo: 
    # same_count[v] = adj[v] @ labels if labels[v]==1, else adj[v] @ (1-labels)
    # More uniformly: let's use x = 2*labels-1
    # Or just compute properly:
    
    same_count = np.zeros(n, dtype=np.int32)
    for v in range(n):
        if labels[v] == 1:
            same_count[v] = adj[v] @ labels
        else:
            same_count[v] = adj[v] @ (1 - labels)
    
    # Greedy loop
    improved = True
    passes = 0
    while improved and passes < max_passes:
        if time.time() - start_time > time_limit - 5:
            break
        improved = False
        passes += 1
        for v in range(n):
            if time.time() - start_time > time_limit - 5:
                break
            deg_v = degrees[v]
            sc = same_count[v]
            oc = deg_v - sc
            if sc > oc:
                # Flip vertex v
                labels[v] = 1 - labels[v]
                improved = True
                # Update same_count for all neighbors of v
                # Neighbors of v: where adj[v] != 0
                neighbors = np.where(adj[v] != 0)[0]
                for u in neighbors:
                    if labels[u] == labels[v]:
                        # Now same label, so same_count[u] increases by 1
                        same_count[u] += 1
                    else:
                        # Now different label, so same_count[u] decreases by 1
                        same_count[u] -= 1
                # Update same_count for v itself
                # v's same count is now: neighbors with same new label
                # Before flip: sc was same with old label. After flip, same with new label = oc (old other)
                # Wait, let me just recompute for v
                if labels[v] == 1:
                    same_count[v] = adj[v] @ labels
                else:
                    same_count[v] = adj[v] @ (1 - labels)
    
    return labels


def _spectral_init(adj):
    """
    Use spectral method to get an initial partition.
    Use the Fiedler vector (eigenvector corresponding to the second smallest eigenvalue
    of the Laplacian) to partition.
    """
    n = len(adj)
    degrees = adj.sum(axis=1)
    # Laplacian L = D - A
    L = np.diag(degrees) - adj
    
    # For efficiency with n=100, we can use scipy
    try:
        from scipy.linalg import eigh
        # Get eigenvalues and eigenvectors
        eigvals, eigvecs = eigh(L)
        # Fiedler vector is the eigenvector with second smallest eigenvalue
        fiedler = eigvecs[:, 1]
        # Partition: positive -> 1, negative -> 0
        labels = (fiedler > 0).astype(np.int32)
        return labels
    except Exception:
        # Fallback to random
        return np.random.randint(0, 2, n).astype(np.int32)


def partition(adjacency):
    """
    Maximum Cut via multiple restarts with spectral initialization and fast greedy local search.
    """
    adj = np.array(adjacency, dtype=np.float64)
    n = len(adj)
    
    start_time = time.time()
    time_limit = 150.0
    
    best_labels = None
    best_cut = -1
    
    # Multiple restarts with different initializations
    num_restarts = 200  # We'll limit by time
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 10:
            break
        
        # Choose initialization strategy
        if restart % 3 == 0:
            # Spectral initialization
            labels = _spectral_init(adj)
        elif restart % 3 == 1:
            # Random initialization
            labels = np.random.randint(0, 2, n).astype(np.int32)
        else:
            # Alternating initialization
            labels = np.arange(n) % 2
        
        # Greedy local search
        labels = _greedy_flip_improve(adj, labels, max_passes=50, time_limit=time_limit, start_time=start_time)
        
        # Evaluate
        cut = _cut_size(adj, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
