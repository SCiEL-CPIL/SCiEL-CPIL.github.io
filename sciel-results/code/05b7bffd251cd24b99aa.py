"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _cut_size(adj, labels):
    """Calculate the number of edges crossing the partition."""
    # labels are 0/1. An edge (i, j) crosses if labels[i] != labels[j].
    # We can compute this as: sum over all edges where labels differ.
    # Using matrix multiplication:
    # Let L be the adjacency matrix.
    # Let x be the label vector (0 or 1).
    # The number of crossing edges is 0.5 * sum_{i,j} L_{ij} * (x_i - x_j)^2
    # Since x_i in {0,1}, (x_i - x_j)^2 is 1 if different, 0 if same.
    # So cut size = 0.5 * x^T (D - A) x? No, simpler:
    # cut = sum_{i<j, L_{ij}=1} (x_i != x_j)
    # Vectorized: 
    # diff = labels[:, None] != labels[None, :]
    # cut = np.sum(adj * diff) // 2 (since undirected, each edge counted twice)
    # Or simply:
    # Let's use the property: 
    # Let y be labels in {-1, 1}.
    # Cut size = (Total Edges - sum_{i,j} A_{ij} y_i y_j) / 4? 
    # Actually, standard formula: 
    # Let x in {0,1}.
    # Crossing edges = sum_{i,j} A_{ij} x_i (1-x_j) for i!=j? No.
    # Easier: 
    # same_partition = np.sum(adj * (labels[:, None] == labels[None, :])) // 2
    # total_edges = np.sum(adj) // 2
    # return total_edges - same_partition
    
    # Most efficient vectorized way for 0/1 labels:
    # A[i][j] is 1 if edge.
    # We want count of (i,j) where A[i][j]=1 and labels[i] != labels[j].
    # Since graph is undirected, we can just sum A * (labels[:, None] != labels[None, :]) and divide by 2.
    # But constructing the full NxN diff matrix is O(N^2) memory. N=100 is fine.
    diff_mask = labels[:, None] != labels[None, :]
    return np.sum(adj * diff_mask) // 2

def _local_search(adj, labels, max_iters=100):
    """
    Vectorized greedy local search.
    Repeatedly find the vertex with the maximum gain and flip it if gain > 0.
    """
    n = len(labels)
    adj = adj.astype(np.int8)
    labels = labels.copy()
    
    # Precompute degrees
    deg = adj.sum(axis=1)
    
    # Current cut size
    current_cut = _cut_size(adj, labels)
    best_cut = current_cut
    best_labels = labels.copy()
    
    for _ in range(max_iters):
        # Calculate gain for flipping each vertex v.
        # If v is in set S (label 1), flipping it to T (label 0) changes cut by:
        # (neighbors in T) - (neighbors in S)
        # Let S_mask = (labels == 1), T_mask = (labels == 0)
        # Neighbors in T for v: adj[v, :][T_mask].sum()
        # Neighbors in S for v: adj[v, :][S_mask].sum()
        # Gain = neighbors_in_opposite - neighbors_in_same
        
        # Vectorized calculation:
        # Let x = labels. 
        # neighbors_in_opposite = adj @ (1 - labels)
        # neighbors_in_same = adj @ labels
        # gain = neighbors_in_opposite - neighbors_in_same
        
        # Note: adj @ labels counts neighbors with label 1.
        # If label[v] is 1:
        #   same = (adj @ labels)[v]
        #   opp = deg[v] - same
        #   gain = opp - same = deg[v] - 2 * same
        # If label[v] is 0:
        #   same = deg[v] - (adj @ labels)[v]  (neighbors with 0)
        #   opp = (adj @ labels)[v]           (neighbors with 1)
        #   gain = opp - same = (adj @ labels)[v] - (deg[v] - (adj @ labels)[v]) = 2*(adj @ labels)[v] - deg[v]
        
        # Unified formula:
        # Let s = adj @ labels
        # If labels[v] == 1: gain = deg[v] - 2*s[v]
        # If labels[v] == 0: gain = 2*s[v] - deg[v]
        # This can be written as: gain = (2*labels[v] - 1) * (2*s[v] - deg[v]) ?
        # Let's check:
        # If label=1: (1) * (2s - d) = 2s - d. But we wanted d - 2s. So sign is flipped.
        # If label=0: (-1) * (2s - d) = d - 2s. But we wanted 2s - d. So sign is flipped.
        # Actually, gain = (2*labels[v] - 1) * (deg[v] - 2*s[v])
        # If label=1: 1 * (d - 2s) = d - 2s. Correct.
        # If label=0: -1 * (d - 2s) = 2s - d. Correct.
        
        s = adj @ labels
        gains = (2 * labels - 1) * (deg - 2 * s)
        
        # Find the vertex with maximum positive gain
        max_gain = np.max(gains)
        if max_gain <= 0:
            break
        
        v = np.argmax(gains)
        labels[v] = 1 - labels[v]
        current_cut += max_gain
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
            
    return best_labels, best_cut

def partition(adjacency):
    # Convert to numpy array of integers for fast arithmetic
    adj = np.array(adjacency, dtype=np.int8)
    n = len(adj)
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_labels = None
    best_cut = -1
    
    # 1. Spectral Initialization
    # Compute Laplacian L = D - A
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Fiedler vector is the eigenvector for the 2nd smallest eigenvalue (index 1)
        fiedler = eigvecs[:, 1]
        # Threshold at 0 to get initial labels
        initial_labels = (fiedler > 0).astype(np.int8)
    except:
        initial_labels = np.random.randint(0, 2, n).astype(np.int8)
    
    # Run local search from spectral start
    labels, cut = _local_search(adj, initial_labels)
    best_labels = labels
    best_cut = cut
    
    # 2. Multi-start with random perturbations
    # We have time budget. Let's try several random restarts or perturbations.
    # Since N=100, local search is very fast (microseconds to milliseconds).
    # We can afford hundreds of restarts.
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Generate a random initial label
        # Mix of random and perturbed spectral
        if rng.random() < 0.5:
            start_labels = rng.integers(0, 2, n).astype(np.int8)
        else:
            # Perturb spectral solution
            start_labels = initial_labels.copy()
            # Flip a random subset of vertices
            mask = rng.random(n) < 0.2
            start_labels[mask] = 1 - start_labels[mask]
            
        labels, cut = _local_search(adj, start_labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels
            
    return best_labels
# EVOLVE-BLOCK-END
