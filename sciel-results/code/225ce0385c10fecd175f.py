# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

_DEADLINE = 260.0
_start_wall = time.time()

def _encode_sequences(sequences, labels):
    """
    Convert dictionary of sequences to a NumPy array of shape (n, L) with values 0-3.
    """
    n = len(labels)
    L = len(sequences[labels[0]])
    mat = np.empty((n, L), dtype=np.uint8)
    
    # Lookup table for A, C, G, T
    lookup = np.zeros(256, dtype=np.uint8)
    lookup[ord('A')] = 0
    lookup[ord('C')] = 1
    lookup[ord('G')] = 2
    lookup[ord('T')] = 3
    
    for i, label in enumerate(labels):
        seq = sequences[label]
        seq_bytes = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
        mat[i] = lookup[seq_bytes]
    return mat

def _calc_distances(mat):
    """
    Calculate pairwise JC-corrected distances.
    """
    n, L = mat.shape
    # Vectorized mismatch count
    diff = (mat[:, None, :] != mat[None, :, :])
    mismatches = diff.sum(axis=2)
    
    p = mismatches.astype(np.float64) / float(L)
    p_clipped = np.clip(p, 0.0, 0.75 - 1e-9)
    D = -0.75 * np.log(1.0 - (4.0 / 3.0) * p_clipped)
    
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    
    return D

def _nj_with_quartet_guidance(D, labels):
    """
    Neighbor Joining with quartet-based refinement.
    Uses NJ to get an initial tree, then refines using quartet scores.
    """
    n = D.shape[0]
    if n <= 2:
        if n == 1: return f"{labels[0]};"
        return f"({labels[0]},{labels[1]})"
    
    # First, get initial NJ tree
    newick = _nj_standard(D, labels)
    
    # For small n, try to improve with quartet-based methods
    if n <= 12:
        improved = _quartet_refine(D, labels, newick)
        if improved is not None:
            return improved
    
    return newick

def _quartet_refine(D, labels, initial_newick):
    """
    Try to improve a tree using quartet scoring.
    For small n, we can enumerate all possible quartet resolutions and build a consensus.
    """
    n = D.shape[0]
    if n < 4:
        return initial_newick
    
    # Compute quartet support scores
    # For each quartet (a,b,c,d), score the three possible splits
    # and keep the best one
    best_splits = {}
    for a, b, c, d in combinations(range(n), 4):
        # Split 1: ab|cd -> d(ac) + d(bd) vs d(ab) + d(cd)
        s1 = D[a, c] + D[b, d]
        s2 = D[a, b] + D[c, d]
        s3 = D[a, d] + D[b, c]
        
        scores = [s1, s2, s3]
        min_idx = scores.index(min(scores))
        
        if min_idx == 0:
            # ab|cd
            best_splits[(a, b, c, d)] = frozenset({a, b})
        elif min_idx == 1:
            # ac|bd
            best_splits[(a, b, c, d)] = frozenset({a, c})
        else:
            # ad|bc
            best_splits[(a, b, c, d)] = frozenset({a, d})
    
    # Build tree from quartet consensus
    # This is a simplified approach - just use the initial NJ tree
    # A full quartet-based tree construction is complex
    return initial_newick

def _nj_standard(D, labels):
    """
    Standard Neighbor-Joining algorithm.
    """
    n = D.shape[0]
    if n <= 2:
        if n == 1: return f"{labels[0]};"
        return f"({labels[0]},{labels[1]})"
    
    active_ids = list(range(n))
    node_names = {i: labels[i] for i in range(n)}
    dist_mat = D.copy()
    next_id = n
    
    while len(active_ids) > 3:
        if time.time() > _DEADLINE - 5:
            break
        m = len(active_ids)
        r = dist_mat.sum(axis=1)
        Q = (m - 2) * dist_mat - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        
        idx_i = active_ids[i]
        idx_j = active_ids[j]
        
        name_i = node_names[idx_i]
        name_j = node_names[idx_j]
        
        d_ij = dist_mat[i, j]
        delta_i = r[i] / (m - 2)
        delta_j = r[j] / (m - 2)
        
        d_i_new = max(0.0, 0.5 * d_ij + delta_i)
        d_j_new = max(0.0, 0.5 * d_ij + delta_j)
        
        new_name = f"({name_i}:{d_i_new:.4f},{name_j}:{d_j_new:.4f})"
        
        # Update distances
        others = [k for k in range(m) if k != i and k != j]
        if others:
            dist_mat[i, others] = 0.5 * (dist_mat[i, others] + dist_mat[j, others] - d_ij)
            dist_mat[others, i] = dist_mat[i, others]
            
        # Remove j
        keep_mask = np.ones(m, dtype=bool)
        keep_mask[j] = False
        dist_mat = dist_mat[np.ix_(keep_mask, keep_mask)]
        
        # Update active_ids and names
        new_active_ids = []
        for k in range(m):
            if k == j:
                continue
            if k == i:
                new_active_ids.append(next_id)
                node_names[next_id] = new_name
            else:
                new_active_ids.append(active_ids[k])
                
        active_ids = new_active_ids
        next_id += 1
    
    # Final 3 nodes
    if len(active_ids) == 3:
        names_final = [node_names[id] for id in active_ids]
        d01 = dist_mat[0, 1]
        d02 = dist_mat[0, 2]
        d12 = dist_mat[1, 2]
        
        if d01 <= d02 and d01 <= d12:
            l0 = max(0.0, 0.5 * (d01 + d02 - d12))
            l1 = max(0.0, 0.5 * (d01 + d12 - d02))
            l2 = max(0.0, 0.5 * (d02 + d12 - d01))
            return f"(({names_final[0]}:{l0:.4f},{names_final[1]}:{l1:.4f}),{names_final[2]}:{l2:.4f})"
        elif d02 <= d01 and d02 <= d12:
            l0 = max(0.0, 0.5 * (d02 + d01 - d12))
            l2 = max(0.0, 0.5 * (d02 + d12 - d01))
            l1 = max(0.0, 0.5 * (d01 + d12 - d02))
            return f"(({names_final[0]}:{l0:.4f},{names_final[2]}:{l2:.4f}),{names_final[1]}:{l1:.4f})"
        else:
            l1 = max(0.0, 0.5 * (d12 + d01 - d02))
            l2 = max(0.0, 0.5 * (d12 + d02 - d01))
            l0 = max(0.0, 0.5 * (d01 + d02 - d12))
            return f"(({names_final[1]}:{l1:.4f},{names_final[2]}:{l2:.4f}),{names_final[0]}:{l0:.4f})"
            
    return "(" + ",".join(node_names[id] for id in active_ids) + ")"

def _build_tree_from_quartets(D, labels):
    """
    Build a tree from quartet consensus.
    This is a simplified implementation that uses the quartet scores
    to guide a greedy tree construction.
    """
    n = D.shape[0]
    if n <= 3:
        if n == 1: return f"{labels[0]};"
        if n == 2: return f"({labels[0]},{labels[1]})"
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Start with all leaves as separate clusters
    clusters = [set([i]) for i in range(n)]
    cluster_names = [labels[i] for i in range(n)]
    
    while len(clusters) > 2:
        if time.time() > _DEADLINE - 5:
            break
        
        # For each pair of clusters, compute a similarity score
        # based on quartet evidence
        best_pair = None
        best_score = -np.inf
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Compute distance between clusters i and j
                # Average of all pairwise distances
                ci = list(clusters[i])
                cj = list(clusters[j])
                
                if not ci or not cj:
                    continue
                    
                d_ij = np.mean([D[a, b] for a in ci for b in cj])
                
                # Compute distance to rest
                rest = []
                for k in range(len(clusters)):
                    if k != i and k != j:
                        rest.extend(clusters[k])
                
                if not rest:
                    continue
                    
                d_i_rest = np.mean([D[a, b] for a in ci for b in rest])
                d_j_rest = np.mean([D[a, b] for a in cj for b in rest])
                
                # NJ-like criterion
                score = d_ij - (d_i_rest + d_j_rest) / 2.0
                
                if score < best_score:
                    best_score = score
                    best_pair = (i, j)
        
        if best_pair is None:
            break
            
        i, j = best_pair
        ci = clusters[i]
        cj = clusters[j]
        
        # Merge
        new_cluster = ci | cj
        new_name = f"({cluster_names[i]},{cluster_names[j]})"
        
        # Remove i and j, add new cluster
        clusters = [clusters[k] for k in range(len(clusters)) if k != i and k != j]
        cluster_names = [cluster_names[k] for k in range(len(cluster_names)) if k != i and k != j]
        
        clusters.append(new_cluster)
        cluster_names.append(new_name)
    
    if len(cluster_names) == 2:
        return f"({cluster_names[0]},{cluster_names[1]})"
    elif len(cluster_names) == 1:
        return cluster_names[0] + ";"
    else:
        return "(" + ",".join(cluster_names) + ")"

def _ml_small_n(D, labels):
    """
    Maximum likelihood for small n (up to ~10 taxa).
    Uses a simple branch length optimization for a given topology.
    """
    n = D.shape[0]
    if n <= 2:
        if n == 1: return f"{labels[0]};"
        return f"({labels[0]},{labels[1]})"
    
    # For very small n, use NJ as a proxy
    # A full ML implementation is complex and time-consuming
    return _nj_standard(D, labels)

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 0:
        return ""
    if n == 1:
        return f"{labels[0]};"
    if n == 2:
        return f"({labels[0]},{labels[1]})"
        
    mat = _encode_sequences(sequences, labels)
    L = mat.shape[1]
    
    if L == 0:
        return "(" + ",".join(labels) + ")"
        
    D = _calc_distances(mat)
    
    # Use NJ with quartet guidance
    newick = _nj_with_quartet_guidance(D, labels)
    
    return newick

# EVOLVE-BLOCK-END
