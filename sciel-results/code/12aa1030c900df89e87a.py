"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
        
    # Compute conflict matrix: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(777)
    
    # Spectral clustering to identify natural communities
    # Build adjacency matrix (1 if NOT conflicting, i.e., compatible)
    compatibility = ~conflicts
    np.fill_diagonal(compatibility, False)
    
    # Use spectral embedding on the compatibility graph
    # The Laplacian of the compatibility graph will reveal clusters
    L = np.diag(neighbor_counts) - (compatibility.astype(np.float64) - np.eye(N4))
    # Actually, let's use the standard graph Laplacian: L = D - A where A is adjacency
    # For compatibility graph, A[i,j] = 1 if compatible
    A_compat = compatibility.astype(np.float64)
    deg_compat = A_compat.sum(axis=1)
    L = np.diag(deg_compat) - A_compat
    
    # Get smallest eigenvectors (excluding trivial)
    try:
        if N4 > 10:
            k_eig = min(8, N4 - 1)
            eigenvalues, eigenvectors = eigsh(L, k=k_eig, which='SM')
            # Use eigenvectors corresponding to smallest eigenvalues
            embedding = eigenvectors[:, :min(k_eig, 6)]
        else:
            embedding = None
    except Exception:
        embedding = None
    
    # Perform hierarchical clustering on the embedding
    cluster_labels = None
    if embedding is not None and embedding.shape[1] >= 2:
        try:
            dist_matrix = squareform(1 - np.corrcoef(embedding.T, rowvar=False))
            dist_matrix = np.maximum(dist_matrix, 0)
            np.fill_diagonal(dist_matrix, 0)
            Z = linkage(squareform(dist_matrix, checks=False), method='ward')
            # Try different numbers of clusters
            for n_clusters in [2, 3, 4, 5]:
                labels = fcluster(Z, t=n_clusters, criterion='maxclust')
                # Check if clusters are reasonably balanced
                counts = np.bincount(labels, minlength=n_clusters)
                if np.min(counts) > N4 * 0.05 and np.max(counts) < N4 * 0.6:
                    cluster_labels = labels
                    break
        except Exception:
            cluster_labels = None
    
    # If spectral clustering didn't work well, use a simpler approach
    if cluster_labels is None:
        # Use greedy coloring-like approach to partition
        cluster_labels = np.zeros(N4, dtype=int)
        assigned = np.zeros(N4, dtype=bool)
        cluster_id = 0
        while not np.all(assigned):
            # Find a maximal independent set in remaining vertices
            remaining = np.where(~assigned)[0]
            if len(remaining) == 0:
                break
            in_cluster = np.zeros(N4, dtype=bool)
            alive = np.ones(N4, dtype=bool)
            alive[~assigned] = True
            while np.any(alive):
                alive_idx = np.where(alive)[0]
                if len(alive_idx) == 0:
                    break
                # Pick vertex with minimum degree among alive
                degs = np.array([len(neighbors[v]) for v in alive_idx])
                min_deg = np.min(degs)
                cands = alive_idx[degs == min_deg]
                chosen = cands[0]
                in_cluster[chosen] = True
                to_remove = np.unique(np.concatenate(([chosen], neighbors[chosen])))
                alive[to_remove] = False
            cluster_labels[in_cluster] = cluster_id
            assigned[in_cluster] = True
            cluster_id += 1
            if cluster_id >= 8:
                # Assign remaining to last cluster
                cluster_labels[~assigned] = cluster_id
                break
    
    # Solve maximum independent set within each cluster
    def solve_cluster_mis(cluster_indices, time_budget=2.0):
        """Solve MIS within a subset of vertices using SA."""
        if len(cluster_indices) == 0:
            return np.zeros(N4, dtype=bool), 0
        
        local_conflicts = conflicts[np.ix_(cluster_indices, cluster_indices)]
        local_neighbors = [np.where(local_conflicts[i])[0] for i in range(len(cluster_indices))]
        local_n = len(cluster_indices)
        
        if local_n <= 1:
            result = np.zeros(N4, dtype=bool)
            result[cluster_indices[0]] = True
            return result, 1
        
        # Greedy initialization
        in_set = np.zeros(local_n, dtype=bool)
        alive = np.ones(local_n, dtype=bool)
        degrees = np.array([len(n) for n in local_neighbors], dtype=np.int32)
        
        while np.any(alive):
            alive_idx = np.where(alive)[0]
            if len(alive_idx) == 0:
                break
            min_deg = np.min(degrees[alive_idx])
            cands = alive_idx[degrees[alive_idx] == min_deg]
            chosen = cands[0]
            in_set[chosen] = True
            to_remove = np.unique(np.concatenate(([chosen], local_neighbors[chosen])))
            alive[to_remove] = False
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= local_conflicts[rem][:, to_remove].sum(axis=1)
        
        current_size = int(in_set.sum())
        
        # SA refinement
        T = 1.0
        cooling = 0.9995
        min_T = 0.05
        sa_deadline = time.time() + time_budget
        
        count_conflicts = np.zeros(local_n, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[local_neighbors[i]] += 1
        
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        
        while time.time() < sa_deadline:
            not_in = np.where(~in_set)[0]
            in_idx = np.where(in_set)[0]
            
            if len(not_in) == 0:
                break
            
            move_type = rng.random()
            
            if move_type < 0.6 and len(not_in) > 0:
                counts = count_conflicts[not_in]
                min_count = np.min(counts)
                min_cands = not_in[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                k = int(min_count)
                delta = 1 - k
                
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.3
                else:
                    accept = rng.random() < np.exp(delta / T)
                
                if accept:
                    to_remove = np.where(local_conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[local_neighbors[u]] -= 1
                    in_set[v] = True
                    current_size += delta
                    count_conflicts[local_neighbors[v]] += 1
            elif move_type < 0.8 and len(in_idx) > 0:
                v = in_idx[rng.integers(len(in_idx))]
                delta = -1
                accept = rng.random() < np.exp(delta / T)
                if accept:
                    in_set[v] = False
                    current_size += delta
                    count_conflicts[local_neighbors[v]] -= 1
            else:
                if len(in_idx) > 0 and len(not_in) > 0:
                    v_rm = in_idx[rng.integers(len(in_idx))]
                    v_ad = not_in[rng.integers(len(not_in))]
                    conflicts_with = np.where(local_conflicts[v_ad] & in_set)[0]
                    other_conflicts = conflicts_with[conflicts_with != v_rm]
                    if len(other_conflicts) == 0:
                        in_set[v_rm] = False
                        in_set[v_ad] = True
                        count_conflicts[local_neighbors[v_rm]] -= 1
                        count_conflicts[local_neighbors[v_ad]] += 1
                    else:
                        to_remove = np.unique(np.concatenate(([v_rm], other_conflicts)))
                        delta = 1 - len(to_remove)
                        if delta > 0 or rng.random() < np.exp(delta / T):
                            in_set[to_remove] = False
                            for u in to_remove:
                                count_conflicts[local_neighbors[u]] -= 1
                            in_set[v_ad] = True
                            current_size += delta
                            count_conflicts[local_neighbors[v_ad]] += 1
            
            if current_size > local_best_size:
                local_best_size = current_size
                local_best_in_set = in_set.copy()
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        # Map back to global indices
        global_result = np.zeros(N4, dtype=bool)
        for i in range(local_n):
            if local_best_in_set[i]:
                global_result[cluster_indices[i]] = True
        
        return global_result, local_best_size
    
    # Solve each cluster and merge
    best_in_set = np.zeros(N4, dtype=bool)
    total_size = 0
    
    unique_clusters = np.unique(cluster_labels)
    time_per_cluster = max(0.5, (deadline - start_time - 10) / max(1, len(unique_clusters)))
    
    for cluster_id in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        
        # Check if we can add all non-conflicting vertices from this cluster
        # to the current global solution
        result, size = solve_cluster_mis(cluster_indices, time_budget=time_per_cluster)
        
        # Try to merge: check if adding this cluster's solution conflicts with existing
        if not np.any(best_in_set):
            best_in_set = result
            total_size = size
        else:
            # Check compatibility
            merged = best_in_set.copy()
            for idx in np.where(result)[0]:
                if not np.any(conflicts[idx] & best_in_set):
                    merged[idx] = True
            
            if int(merged.sum()) > total_size:
                best_in_set = merged
                total_size = int(merged.sum())
    
    # Global SA refinement
    def sa_global(in_set_init, max_time=15.0):
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        T = 2.0
        cooling = 0.9998
        min_T = 0.05
        
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        while time.time() < sa_deadline:
            in_set_indices = np.where(in_set)[0]
            not_in_set = np.where(~in_set)[0]
            
            if len(not_in_set) == 0:
                break
            
            move_type = rng.random()
            
            if move_type < 0.55 and len(not_in_set) > 0:
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                k = int(min_count)
                delta = 1 - k
                
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.3
                else:
                    accept = rng.random() < np.exp(delta / T)
                
                if accept:
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                    in_set[v] = True
                    current_size += delta
                    count_conflicts[neighbors[v]] += 1
                else:
                    stuck_counter += 1
            elif move_type < 0.70 and len(in_set_indices) > 0:
                v = in_set_indices[rng.integers(len(in_set_indices))]
                delta = -1
                accept = rng.random() < np.exp(delta / T)
                if accept:
                    in_set[v] = False
                    current_size += delta
                    count_conflicts[neighbors[v]] -= 1
                else:
                    stuck_counter += 1
            elif move_type < 0.90 and len(in_set_indices) > 0 and len(not_in_set) > 0:
                v_rm = in_set_indices[rng.integers(len(in_set_indices))]
                n_sample = min(20, len(not_in_set))
                sample_idx = rng.choice(len(not_in_set), size=n_sample, replace=False)
                sample_cands = not_in_set[sample_idx]
                
                best_swap = None
                best_delta = -100
                for v_ad in sample_cands:
                    conflicts_with = np.where(conflicts[v_ad] & in_set)[0]
                    other_conflicts = conflicts_with[conflicts_with != v_rm]
                    d = 1 - 1 - len(other_conflicts)
                    if d > best_delta:
                        best_delta = d
                        best_swap = (v_rm, v_ad, other_conflicts)
                
                if best_swap is not None and best_delta >= -3:
                    v_r, v_a, others = best_swap
                    if best_delta >= 0 or rng.random() < np.exp(best_delta / T):
                        to_remove = np.unique(np.concatenate(([v_r], others))) if len(others) > 0 else np.array([v_r])
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                        in_set[v_a] = True
                        current_size += best_delta
                        count_conflicts[neighbors[v_a]] += 1
                else:
                    stuck_counter += 1
            else:
                if len(in_set_indices) >= 2 and len(not_in_set) >= 2:
                    v1 = in_set_indices[rng.integers(len(in_set_indices))]
                    v2 = in_set_indices[rng.integers(len(in_set_indices))]
                    if v1 != v2:
                        n_sample = min(10, len(not_in_set))
                        sample_idx = rng.choice(len(not_in_set), size=n_sample, replace=False)
                        sample_cands = not_in_set[sample_idx]
                        
                        found = []
                        for v_ad in sample_cands:
                            conflicts_with = np.where(conflicts[v_ad] & in_set)[0]
                            other_conflicts = conflicts_with[~np.isin(conflicts_with, [v1, v2])]
                            if len(other_conflicts) == 0:
                                found.append(v_ad)
                                if len(found) == 2:
                                    break
                        
                        if len(found) == 2:
                            in_set[v1] = False
                            in_set[v2] = False
                            count_conflicts[neighbors[v1]] -= 1
                            count_conflicts[neighbors[v2]] -= 1
                            for v_ad in found:
                                in_set[v_ad] = True
                                count_conflicts[neighbors[v_ad]] += 1
                        else:
                            stuck_counter += 1
                else:
                    stuck_counter += 1
            
            if current_size > local_best_size:
                local_best_size = current_size
                local_best_in_set = in_set.copy()
                stuck_counter = 0
            
            if stuck_counter > 800:
                T = max(T, 2.5)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        return local_best_in_set, local_best_size
    
    # Multiple SA restarts with different initializations
    best_size = total_size
    best_final_in_set = best_in_set.copy()
    
    # Greedy polish
    def greedy_polish(in_set_init):
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            not_in = np.where(~in_set)[0]
            if len(not_in) == 0:
                break
            
            conflict_counts = np.array([
                int((conflicts[v] & in_set).sum()) for v in not_in
            ])
            addable = not_in[conflict_counts == 0]
            
            if len(addable) > 0:
                remaining = addable.copy()
                while len(remaining) > 0:
                    v = remaining[0]
                    in_set[v] = True
                    current_size += 1
                    changed = True
                    nbrs_set = set(neighbors[v])
                    nbrs_set.add(v)
                    remaining = np.array([x for x in remaining if x not in nbrs_set])
        
        return in_set, current_size
    
    polish_set, polish_size = greedy_polish(best_final_in_set)
    if polish_size > best_size:
        best_size = polish_size
        best_final_in_set = polish_set
    
    # SA restarts
    n_restarts = 0
    remaining_time = deadline - 5 - time.time()
    
    while time.time() < deadline - 3 and n_restarts < 30:
        n_restarts += 1
        
        # Different initialization strategies
        if n_restarts % 3 == 0:
            # Use spectral cluster solution
            init_set = best_final_in_set.copy()
        elif n_restarts % 3 == 1:
            # Greedy min-degree
            init_set = np.zeros(N4, dtype=bool)
            alive = np.ones(N4, dtype=bool)
            degrees = neighbor_counts.copy()
            while np.any(alive) and time.time() < deadline - 4:
                alive_idx = np.where(alive)[0]
                if len(alive_idx) == 0:
                    break
                min_deg = np.min(degrees[alive_idx])
                cands = alive_idx[degrees[alive_idx] == min_deg]
                chosen = cands[0]
                init_set[chosen] = True
                to_remove = np.unique(np.concatenate(([chosen], neighbors[chosen])))
                alive[to_remove] = False
                rem = np.where(alive)[0]
                if len(rem) > 0:
                    degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
        else:
            # Random sampling + polish
            init_set = np.zeros(N4, dtype=bool)
            sample_size = min(N4, 200)
            sample_idx = rng.choice(N4, size=sample_size, replace=False)
            for v in sample_idx:
                if not np.any(conflicts[v] & init_set):
                    init_set[v] = True
            init_set, _ = greedy_polish(init_set)
        
        current_size = int(init_set.sum())
        if current_size > best_size:
            best_size = current_size
            best_final_in_set = init_set.copy()
        
        # SA local search
        sa_time = min(remaining_time * 0.4, 8.0)
        sa_time = max(sa_time, 1.0)
        sa_result, sa_size = sa_global(init_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_final_in_set = sa_result.copy()
        
        # Polish
        if time.time() < deadline - 2:
            polish_set, polish_size = greedy_polish(sa_result if sa_size >= current_size else init_set)
            if polish_size > best_size:
                best_size = polish_size
                best_final_in_set = polish_set
    
    # Final polish
    if time.time() < deadline - 1:
        final_polish, final_size = greedy_polish(best_final_in_set)
        if final_size > best_size:
            best_size = final_size
            best_final_in_set = final_polish
    
    # Final result
    final_v4_indices = np.where(best_final_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
