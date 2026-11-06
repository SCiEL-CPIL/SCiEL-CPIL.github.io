"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    # Convert to numpy array for fast operations
    adj = np.array(adjacency, dtype=np.int8)
    n = len(adj)
    
    # Deadline guard: kill before 150s wall-clock limit
    deadline = time.time() + 145.0
    
    # Compute cut size
    def cut_size(labels):
        # Cut edges are those where labels differ
        # For each edge (i,j) where adj[i,j]=1, it's a cut edge if labels[i] != labels[j]
        # Vectorized: sum of adj[i,j] where labels[i] != labels[j]
        # This is equivalent to: (sum of all edges) - (sum of edges within same partition)
        # Or directly: 0.5 * sum over i of sum over j of adj[i,j] * (labels[i] != labels[j])
        # Since adj is symmetric, we can compute:
        # For each vertex i, count neighbors in opposite partition
        # sum_i sum_j adj[i,j] * (labels[i] != labels[j]) / 2
        # But simpler: 
        # total_cut = 0
        # for each i: total_cut += adj[i, labels != labels[i]].sum()
        # Then divide by 2? No, wait.
        # Actually, if we sum over all i of (number of neighbors of i in opposite partition),
        # each cut edge is counted twice (once from each endpoint).
        # So cut_size = 0.5 * sum_i adj[i, labels != labels[i]].sum()
        
        # More efficient vectorized approach:
        # labels is 0/1 array
        # For each i, the number of neighbors with label 1-labels[i]
        # Let's compute: for label 0 vertices, count neighbors with label 1
        #             for label 1 vertices, count neighbors with label 0
        mask_0 = labels == 0
        mask_1 = labels == 1
        # For vertices in partition 0, their cut edges are edges to partition 1
        # For vertices in partition 1, their cut edges are edges to partition 0
        cut = adj[mask_0][:, mask_1].sum() + adj[mask_1][:, mask_0].sum()
        # This counts each cut edge exactly once? No.
        # adj[mask_0][:, mask_1] gives edges from 0-partition to 1-partition
        # adj[mask_1][:, mask_0] gives edges from 1-partition to 0-partition
        # Since adj is symmetric, these are the same set of edges, so this double counts.
        # So cut_size = 0.5 * (that sum)? No wait.
        # Actually adj[mask_0][:, mask_1] already contains all edges between the two partitions.
        # Each such edge appears once in this submatrix.
        # So adj[mask_0][:, mask_1].sum() is already the cut size.
        # Let me verify: if there's an edge between i (in 0) and j (in 1),
        # then adj[i,j]=1 and it appears in adj[mask_0][:, mask_1] at position corresponding to i,j.
        # It does NOT appear in adj[mask_1][:, mask_0] as a separate entry - wait, it does,
        # because adj[j,i]=1 and j is in mask_1, i is in mask_0.
        # So adj[mask_1][:, mask_0] also contains this edge.
        # Therefore the sum is 2 * cut_size.
        # So cut_size = 0.5 * (adj[mask_0][:, mask_1].sum() + adj[mask_1][:, mask_0].sum())
        # But since adj is symmetric, adj[mask_0][:, mask_1].sum() == adj[mask_1][:, mask_0].sum()
        # So cut_size = adj[mask_0][:, mask_1].sum()
        
        # Let me use a cleaner vectorized approach:
        # cut_size = sum over all i,j of adj[i,j] * (labels[i] != labels[j]) / 2
        # Since adj is symmetric, this equals sum over i<j of adj[i,j] * (labels[i] != labels[j])
        
        # Most efficient: 
        # Let L = labels as int array
        # diff = (L[:, None] != L[None, :]) * adj
        # cut = 0.5 * diff.sum()
        # This is O(n^2) but vectorized.
        
        L = labels.astype(np.int8)
        diff = (L[:, None] != L[None, :]).astype(np.int8)
        return 0.5 * (diff * adj).sum()
    
    # Local search: flip vertex if it improves cut
    def local_search(labels):
        best_cut = cut_size(labels)
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for v in range(n):
                if time.time() > deadline:
                    break
                # Try flipping vertex v
                new_labels = labels.copy()
                new_labels[v] = 1 - new_labels[v]
                new_cut = cut_size(new_labels)
                if new_cut > best_cut:
                    labels = new_labels
                    best_cut = new_cut
                    improved = True
        return labels, best_cut
    
    # Spectral initialization using Fiedler vector
    def spectral_init():
        deg = adj.sum(axis=1).astype(np.float64)
        L = np.diag(deg) - adj.astype(np.float64)
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            # Fiedler vector is the eigenvector for the second smallest eigenvalue
            # eigvals are sorted in ascending order
            fiedler = eigvecs[:, 1]
            # Partition based on sign
            labels = (fiedler > 0).astype(np.int8)
            # Ensure both partitions are non-empty
            if labels.sum() == 0 or labels.sum() == n:
                labels = np.arange(n) % 2
            return labels
        except:
            return np.arange(n) % 2
    
    # Random initialization
    def random_init():
        return np.random.randint(0, 2, n)
    
    # Run multiple restarts with different initializations
    best_labels = None
    best_cut = -1
    
    # First, try spectral initialization
    labels_spec = spectral_init()
    labels_ls, cut_ls = local_search(labels_spec)
    if cut_ls > best_cut:
        best_cut = cut_ls
        best_labels = labels_ls.copy()
    
    # Then, try random restarts
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        labels_rand = random_init()
        labels_ls, cut_ls = local_search(labels_rand)
        if cut_ls > best_cut:
            best_cut = cut_ls
            best_labels = labels_ls.copy()
    
    # If we still have time, do more random restarts
    while time.time() < deadline - 5:  # 5s safety margin
        labels_rand = random_init()
        labels_ls, cut_ls = local_search(labels_rand)
        if cut_ls > best_cut:
            best_cut = cut_ls
            best_labels = labels_ls.copy()
    
    return best_labels
# EVOLVE-BLOCK-END
