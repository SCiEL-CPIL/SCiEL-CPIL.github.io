import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.sparse.linalg import eigsh

def priority(adjacency):
    """
    Assigns a priority score to each vertex for a greedy decoder that picks
    vertices in descending score order, skipping neighbours of selected vertices.
    
    Uses a structural approach based on:
    1. Local density (degree and neighbor degree)
    2. Spectral partitioning (Fiedler vector) to identify the 'sparsest cut'
    3. Territory scoring: prioritizing vertices in the larger partition or those
       with more 'room' relative to the cut
    4. Iterative refinement of scores using simulated greedy outcomes
    5. Local search swaps
    """
    start_time = time.time()
    deadline = start_time + 135.0  # 15s margin for safety
    
    A = np.asarray(adjacency)
    n = A.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Precompute basic features
    degree = A.sum(axis=1).astype(float)
    neighbor_deg_sum = A @ degree
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_neighbor_deg = np.where(degree > 0, neighbor_deg_sum / degree, 0.0)
    
    # Helper: Fast Greedy Decode for evaluation
    def greedy_decode_size(scores):
        """
        Simulates the greedy decoder.
        Returns the size of the independent set.
        """
        order = np.argsort(-scores)
        
        selected = np.zeros(n, dtype=bool)
        count = 0
        
        for v in order:
            if not np.dot(selected, A[v]):
                selected[v] = True
                count += 1
        
        return count
    
    # Candidate Strategies
    candidates = []
    
    # 1. Pure Degree (baseline)
    candidates.append(-degree)
    
    # 2. Degree + Mean Neighbor Degree
    candidates.append(-(degree + 0.5 * mean_neighbor_deg))
    
    # 3. Degree - Mean Neighbor Degree
    candidates.append(-(degree - 0.3 * mean_neighbor_deg))
    
    # 4. Spectral features
    try:
        L = np.diag(degree) - A
        if n > 10:
            vals, vecs = eigsh(L, k=3, which='SM')
            idx = np.argsort(vals)
            # The second smallest eigenvalue corresponds to the Fiedler vector
            # idx[0] is 0, idx[1] is the Fiedler vector
            fiedler = vecs[:, idx[1]]
            # Ensure the Fiedler vector is oriented such that the sum is 0 (centered)
            fiedler = fiedler - np.mean(fiedler)
            
            # If we have a third vector, use it for refinement
            if len(idx) > 2:
                second = vecs[:, idx[2]]
                second = second - np.mean(second)
            else:
                second = np.zeros(n)
        else:
            eigvals, eigvecs = np.linalg.eigh(L)
            idx = np.argsort(eigvals)
            fiedler = eigvecs[:, idx[1]]
            fiedler = fiedler - np.mean(fiedler)
            second = eigvecs[:, idx[2]] if len(idx) > 2 else np.zeros(n)
            second = second - np.mean(second)
        
        f_min = fiedler.min()
        f_max = fiedler.max()
        if f_max - f_min > 1e-9:
            fiedler_norm = (fiedler - f_min) / (f_max - f_min)
        else:
            fiedler_norm = np.zeros(n)
        
        s_min = second.min()
        s_max = second.max()
        if s_max - s_min > 1e-9:
            second_norm = (second - s_min) / (s_max - s_min)
        else:
            second_norm = np.zeros(n)
        
        # Add spectral candidates
        candidates.append(-degree + 0.5 * fiedler_norm)
        candidates.append(-degree - 0.5 * fiedler_norm)
        candidates.append(-degree + 0.3 * fiedler_norm + 0.2 * second_norm)
        candidates.append(-degree - 0.3 * fiedler_norm + 0.2 * second_norm)
        
        # 5. Territory Score based on Fiedler Vector
        # The Fiedler vector partitions the graph into two sets.
        # Vertices with high absolute value of the Fiedler vector are "deep" in their partition.
        # Vertices with low absolute value are near the cut.
        # We want to prioritize vertices that are "deep" in the larger partition or have more "room".
        
        # Determine the two partitions based on the sign of the Fiedler vector
        # Since the Fiedler vector is centered, we can split by sign
        pos_mask = fiedler > 0
        neg_mask = fiedler <= 0
        
        # Count the size of each partition
        pos_count = np.sum(pos_mask)
        neg_count = np.sum(neg_mask)
        
        # The "territory" of a vertex is how much "space" it has in its partition.
        # We can approximate this by the number of vertices in the same partition with similar Fiedler values.
        # Or simply, we can use the magnitude of the Fiedler vector as a proxy for "distance from the cut".
        
        # A better heuristic: 
        # If a vertex is in the larger partition, it might be safer to pick.
        # If a vertex is far from the cut (high |fiedler|), it is less likely to conflict with the other side.
        
        # Let's create a score that combines:
        # - Degree (lower is better)
        # - Fiedler magnitude (higher is better, means far from cut)
        # - Partition size bonus (bonus if in the larger partition)
        
        # Normalize fiedler to [0, 1] for the magnitude
        fiedler_mag = np.abs(fiedler)
        fiedler_mag_norm = fiedler_mag / (np.max(fiedler_mag) + 1e-9)
        
        # Partition bonus: 1 if in larger partition, 0.5 otherwise
        larger_partition_mask = pos_mask if pos_count >= neg_count else neg_mask
        partition_bonus = np.where(larger_partition_mask, 1.0, 0.5)
        
        # Combine
        territory_score = -degree + 0.5 * fiedler_mag_norm + 0.2 * partition_bonus
        
        candidates.append(territory_score)
        
        # Also try a version that penalizes vertices near the cut more heavily
        # Vertices near the cut (low |fiedler|) are bottlenecks
        bottleneck_penalty = 1.0 / (fiedler_mag_norm + 0.1)
        candidates.append(-degree - 0.3 * bottleneck_penalty)
        
    except Exception:
        pass
    
    # 6. Eigenvector centrality (structural separation)
    try:
        A_csr = csr_matrix(A)
        
        # Power iteration for eigenvector centrality
        x = np.ones(n) / n
        for _ in range(20):
            x_new = A_csr @ x
            x_new = x_new / (np.sum(x_new) + 1e-10)
            x = x_new
        
        # Normalize
        x_min = x.min()
        x_max = x.max()
        if x_max - x_min > 1e-9:
            eig_centrality = (x - x_min) / (x_max - x_min)
        else:
            eig_centrality = np.zeros(n)
        
        candidates.append(-degree - 0.5 * eig_centrality)
        candidates.append(-degree + 0.3 * eig_centrality)
        
    except Exception:
        pass
    
    # Find a good initial candidate
    best_size = -1
    best_scores = candidates[0]
    
    for s in candidates:
        if time.time() > deadline - 30.0:
            break
        if np.any(~np.isfinite(s)):
            s = np.where(np.isfinite(s), s, 0.0)
        size = greedy_decode_size(s)
        if size > best_size:
            best_size = size
            best_scores = s
    
    # Iterative refinement: adjust scores to reduce conflicts
    refined_scores = best_scores.copy()
    
    for iteration in range(10):
        if time.time() > deadline - 15.0:
            break
        
        # Simulate greedy decode and track which vertices cause conflicts
        order = np.argsort(-refined_scores)
        selected = np.zeros(n, dtype=bool)
        conflict_count = np.zeros(n)  # How many times each vertex blocks others
        
        for v in order:
            if np.dot(selected, A[v]):
                # This vertex is blocked - find which selected vertex blocked it
                blockers = np.where(selected & A[v])[0]
                if len(blockers) > 0:
                    # Distribute conflict to the blocker(s)
                    for b in blockers:
                        conflict_count[b] += 1
            else:
                selected[v] = True
                # Mark neighbors as potentially conflicted
                neighbors = np.where(A[v])[0]
                for u in neighbors:
                    if not selected[u]:
                        conflict_count[v] += 0.5  # v might block u
        
        # Adjust scores: penalize vertices that cause many conflicts
        # But also reward vertices that are isolated from high-score neighbors
        conflict_norm = conflict_count / (np.max(conflict_count) + 1e-10)
        
        # New score: original score minus conflict penalty
        new_scores = refined_scores - 0.3 * conflict_norm
        
        # Check if this improves
        new_size = greedy_decode_size(new_scores)
        if new_size >= best_size:
            refined_scores = new_scores
            best_size = new_size
            best_scores = new_scores
        else:
            # Try a smaller adjustment
            new_scores = refined_scores - 0.1 * conflict_norm
            new_size = greedy_decode_size(new_scores)
            if new_size >= best_size:
                refined_scores = new_scores
                best_size = new_size
                best_scores = new_scores
    
    # 7. Local search: try swapping low-priority selected vertices with high-priority unselected
    if time.time() < deadline - 10.0:
        order = np.argsort(-best_scores)
        selected = np.zeros(n, dtype=bool)
        selected_set = []
        
        for v in order:
            if not np.dot(selected, A[v]):
                selected[v] = True
                selected_set.append(v)
        
        # Try to improve by swapping
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i, v in enumerate(selected_set):
                # Try removing v and adding a better vertex
                selected[v] = False
                # Find best vertex that can now be added
                best_new = -1
                best_new_score = -np.inf
                for u in order:
                    if not selected[u] and not np.dot(selected, A[u]):
                        # Check if u is better than v
                        if best_scores[u] > best_scores[v] or (best_scores[u] == best_scores[v] and u < v):
                            # Check if adding u doesn't reduce set size
                            # (it won't since we're replacing one with one)
                            if best_scores[u] > best_new_score:
                                best_new = u
                                best_new_score = best_scores[u]
                
                if best_new != -1 and best_scores[best_new] > best_scores[v]:
                    selected_set[i] = best_new
                    selected[best_new] = True
                    improved = True
                
                # Restore v if not swapped
                if selected_set[i] != v:
                    selected[v] = True
    
    # Final evaluation
    final_size = greedy_decode_size(best_scores)
    
    # If refinement didn't help, fall back to best candidate
    if final_size < best_size:
        # Re-evaluate all candidates to find the true best
        for s in candidates:
            if np.any(~np.isfinite(s)):
                s = np.where(np.isfinite(s), s, 0.0)
            size = greedy_decode_size(s)
            if size > final_size:
                best_scores = s
                final_size = size
    
    return best_scores
