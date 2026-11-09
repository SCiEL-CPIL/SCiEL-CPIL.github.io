import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.sparse.linalg import eigsh

def priority(adjacency):
    """
    Assigns a priority score to each vertex for a greedy decoder that picks
    vertices in descending score order, skipping neighbors of already picked vertices.
    
    Uses a structural approach based on:
    1. Local density (degree and neighbor degree)
    2. Betweenness-like centrality (how much a vertex separates the graph)
    3. Iterative refinement of scores using simulated greedy outcomes
    4. Community structure via spectral clustering
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
    
    # Helper: Compute local independence potential
    def local_independence_score(v, scores):
        """
        Estimate how well vertex v fits into an independent set given current scores.
        Higher score = better fit (less conflict with high-score neighbors)
        """
        neighbors = np.where(A[v])[0]
        if len(neighbors) == 0:
            return scores[v]
        # Penalize if neighbors have high scores
        neighbor_score_max = np.max(scores[neighbors]) if len(neighbors) > 0 else 0
        # The lower the neighbor scores, the better
        return scores[v] - 0.5 * neighbor_score_max
    
    # Candidate Strategies
    candidates = []
    
    # 1. Pure Degree (baseline)
    candidates.append(-degree)
    
    # 2. Degree + Mean Neighbor Degree
    candidates.append(-(degree + 0.5 * mean_neighbor_deg))
    
    # 3. Degree - Mean Neighbor Degree (prefer low-degree with low-neighbor-degree)
    candidates.append(-(degree - 0.3 * mean_neighbor_deg))
    
    # 4. Spectral features
    try:
        L = np.diag(degree) - A
        if n > 10:
            vals, vecs = eigsh(L, k=3, which='SM')
            idx = np.argsort(vals)
            fiedler = vecs[:, idx[1]]
            second = vecs[:, idx[2]] if len(idx) > 2 else np.zeros(n)
        else:
            eigvals, eigvecs = np.linalg.eigh(L)
            idx = np.argsort(eigvals)
            fiedler = eigvecs[:, idx[1]]
            second = eigvecs[:, idx[2]] if len(idx) > 2 else np.zeros(n)
        
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
        
    except Exception:
        pass
    
    # 5. Betweenness centrality approximation (structural separation)
    # Vertices that are bridges or articulation points should be deprioritized
    # because selecting them blocks many other vertices
    try:
        # Approximate betweenness by counting how many shortest paths pass through each vertex
        # Use a simplified version: for each vertex, count its "centrality" in the graph
        # A vertex with high betweenness is a bottleneck - deprioritize it
        
        # Compute degree-based betweenness proxy
        # For each edge (u,v), the number of shortest paths through u is roughly
        # related to how many vertices are "behind" u from v's perspective
        
        # Simpler approach: use eigenvector centrality as a proxy
        # High eigenvector centrality = connected to many high-degree vertices = bad for IS
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
        candidates.append(-degree + 0.3 * eig_centrality)  # Sometimes high centrality helps
        
    except Exception:
        pass
    
    # 6. Iterative refinement: Start with a good initial score, then refine
    # by simulating greedy decode and adjusting scores based on conflicts
    
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
            
            selected[v] = True  # Restore if not swapped
    
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

# EVOLVE-BLOCK-END
