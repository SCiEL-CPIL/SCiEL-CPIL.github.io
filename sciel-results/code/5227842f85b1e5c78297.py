"""UPGMA phylogenetic topology recovery."""
# EVOLVE-BLOCK-START
import numpy as np

def _jc_distances(sequences, labels):
    """Compute Jukes-Cantor corrected distance matrix."""
    n = len(labels)
    seqs = [sequences[lab] for lab in labels]
    L = len(seqs[0])
    
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arrs = [np.array([char_map[c] for c in s], dtype=np.uint8) for s in seqs]
    stacked = np.stack(arrs, axis=0)
    
    dist_mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diffs = np.sum(stacked[i] != stacked[j])
            p = diffs / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
            dist_mat[i, j] = d
            dist_mat[j, i] = d
            
    return dist_mat

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";" if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"
        
    dist = _jc_distances(sequences, labels)
    
    # UPGMA
    # Clusters: list of (label, size, cluster_tree)
    # Initially, each taxon is a cluster.
    clusters = []
    for lab in labels:
        clusters.append({
            'label': lab,
            'size': 1,
            'tree': lab
        })
        
    # Distance matrix between clusters
    # We'll maintain a list of active cluster indices
    active = list(range(n))
    
    # dist[active[i], active[j]]
    # We'll use a dictionary or just re-index.
    # Simpler: maintain a list of clusters and a distance matrix that we update.
    
    # Let's use a list of cluster objects and a matrix D of size n x n.
    # When we merge, we remove the merged clusters from active and add a new one.
    
    D = dist.copy()
    cluster_trees = [lab for lab in labels]
    cluster_sizes = [1] * n
    active_indices = list(range(n))
    
    while len(active_indices) > 1:
        # Find min distance
        min_d = float('inf')
        min_i, min_j = 0, 1
        for i_idx in range(len(active_indices)):
            for j_idx in range(i_idx + 1, len(active_indices)):
                i = active_indices[i_idx]
                j = active_indices[j_idx]
                if D[i, j] < min_d:
                    min_d = D[i, j]
                    min_i, min_j = i, j
                    
        # Merge min_i and min_j
        # New cluster index: let's just use a new ID or reuse one.
        # We'll remove min_j from active and update min_i.
        
        # New tree
        new_tree = f"({cluster_trees[min_i]},{cluster_trees[min_j]})"
        new_size = cluster_sizes[min_i] + cluster_sizes[min_j]
        
        # Update distances from new cluster to others
        # d(new, k) = (size_i * d(i, k) + size_j * d(j, k)) / (size_i + size_j)
        for k_idx in range(len(active_indices)):
            k = active_indices[k_idx]
            if k == min_i or k == min_j:
                continue
            D[min_i, k] = (cluster_sizes[min_i] * D[min_i, k] + cluster_sizes[min_j] * D[min_j, k]) / new_size
            D[k, min_i] = D[min_i, k]
            
        # Update cluster info
        cluster_trees[min_i] = new_tree
        cluster_sizes[min_i] = new_size
        
        # Remove min_j from active
        active_indices.remove(min_j)
        
    # Final tree
    return cluster_trees[active_indices[0]] + ";"

# EVOLVE-BLOCK-END
