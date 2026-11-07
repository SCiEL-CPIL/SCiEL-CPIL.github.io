import numpy as np
import time

def _fast_aco_simulate(distance_matrix, heuristic, n_ants=20, n_iterations=30, rho=0.9, seed=42):
    """
    Fast ACO simulation to evaluate heuristic quality.
    Returns the minimum tour length found.
    """
    n = distance_matrix.shape[0]
    tau = np.ones((n, n))
    rng = np.random.default_rng(seed)
    
    best_tour_len = float('inf')
    
    for it in range(n_iterations):
        tour_lengths = []
        for ant in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            tour_len = 0.0
            tour_path = [0]
            
            for _ in range(n - 1):
                # Vectorized probability calculation
                probs = tau[current, :] * heuristic[current, :]
                probs[visited] = 0.0
                
                total_prob = probs.sum()
                if total_prob < 1e-10:
                    # Fallback: pick random unvisited
                    unvisited_indices = np.where(~visited)[0]
                    next_node = unvisited_indices[rng.integers(len(unvisited_indices))]
                else:
                    probs = probs / total_prob
                    next_node = rng.choice(n, p=probs)
                
                tour_len += distance_matrix[current, next_node]
                visited[next_node] = True
                tour_path.append(next_node)
                current = next_node
            
            tour_len += distance_matrix[current, 0]
            tour_lengths.append(tour_len)
            
            deposit = 1.0 / tour_len
            # Vectorized pheromone update
            for i in range(n):
                u = tour_path[i]
                v = tour_path[(i + 1) % n]
                tau[u, v] += deposit
                # Note: The contract says "in travel direction only", so tau[v,u] is NOT updated
                # unless the ant travels v->u later, which it won't in a simple cycle unless it's a 2-cycle (impossible for n>2)
                # Wait, the contract says: "each ant adds 1/length to pheromone[i, j] for each step i to j of its closed tour, in travel direction only."
                # My loop does exactly that.
        
        tau *= rho
        
        current_best = min(tour_lengths)
        if current_best < best_tour_len:
            best_tour_len = current_best
            
    return best_tour_len

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 280.0
    
    n = distance_matrix.shape[0]
    dm = distance_matrix.copy()
    
    # 1. Compute the Minimum Spanning Tree (MST)
    # We can use Prim's algorithm or Kruskal's. 
    # Since n=50, a simple O(n^2) Prim's is efficient and doesn't require sorting all edges.
    
    # Initialize MST structure
    in_mst = np.zeros(n, dtype=bool)
    in_mst[0] = True
    min_dist = dm[0].copy()
    min_dist[0] = np.inf
    
    mst_edges = set()
    
    for _ in range(n - 1):
        # Find the vertex with the minimum distance to the current MST
        # Exclude vertices already in MST
        temp_min_dist = min_dist.copy()
        temp_min_dist[in_mst] = np.inf
        
        if np.all(temp_min_dist == np.inf):
            break # Disconnected graph? Should not happen for TSP in unit square
            
        next_node = np.argmin(temp_min_dist)
        dist_val = temp_min_dist[next_node]
        
        # Find the predecessor in MST
        # The edge is (prev, next_node) where prev is in MST and dist[prev, next_node] == dist_val
        # Actually, min_dist stores the edge weight from the MST to the node.
        # We need to find which node in MST provided this min_dist.
        # This O(n) search is fine for n=50.
        
        # However, standard Prim's keeps track of the parent. Let's refactor slightly for clarity and correctness.
        # Let's restart Prim's with parent tracking.
        
    # Refactored Prim's Algorithm with Parent Tracking
    in_mst = np.zeros(n, dtype=bool)
    in_mst[0] = True
    min_dist = dm[0].copy()
    parent = np.full(n, -1, dtype=int)
    
    for _ in range(n - 1):
        temp_min_dist = min_dist.copy()
        temp_min_dist[in_mst] = np.inf
        
        if np.all(temp_min_dist == np.inf):
            break
            
        next_node = np.argmin(temp_min_dist)
        
        # Find parent
        # The parent is the node in MST such that dm[parent, next_node] == min_dist[next_node]
        # Since we updated min_dist from the MST, the parent is the one that set this value.
        # But we didn't track which node updated it.
        # We can find it by checking all nodes in MST.
        for i in range(n):
            if in_mst[i] and dm[i, next_node] == min_dist[next_node]:
                parent[next_node] = i
                break
        
        in_mst[next_node] = True
        mst_edges.add((min(next_node, parent[next_node]), max(next_node, parent[next_node])))
        
        # Update min_dist for neighbors
        for j in range(n):
            if not in_mst[j] and dm[next_node, j] < min_dist[j]:
                min_dist[j] = dm[next_node, j]
                # We don't strictly need to update parent here if we do the O(n) search later,
                # but for efficiency in the search above, we could store it.
                # For n=50, the O(n) search is negligible.

    # 2. Construct the Heuristic Matrix
    # High value for MST edges, low value for others.
    # Let high = 10.0 and low = 1e-9 (as per contract, non-positive becomes 1e-9, but we want explicit low)
    # Actually, contract says "Heuristic entries that are not positive become 1e-9".
    # So we can set non-MST edges to 0.0, and they will be handled, or just set them to a small positive number.
    # Let's set MST edges to 1.0 and others to 1e-9.
    # Wait, if I set others to 1e-9, the ants will almost never traverse them unless pheromone is huge.
    # Pheromone starts at 1. 
    # Prob ~ tau * H.
    # If H=1e-9, Prob is tiny.
    # If H=1.0, Prob is normal.
    # This forces the ACO to stick to the MST.
    # The MST is a tree, so it's not a tour. The ants must close the loop.
    # The last step (return to 0) might not be in the MST.
    # If the edge (last, 0) is not in MST, H[last, 0] = 1e-9.
    # The ant *must* go back to 0.
    # If H is 1e-9, the probability is very low, but it's the only unvisited node (or one of few).
    # Actually, at the last step, there is only 1 unvisited node (node 0).
    # So probs[0] = tau[last,0] * H[last,0].
    # probs[other] = 0.
    # So the ant WILL go to 0, regardless of H[last,0], as long as it's > 0.
    # So the return edge is always taken.
    # The constraint is on the path construction.
    
    H = np.full((n, n), 1e-9, dtype=float)
    
    for u, v in mst_edges:
        H[u, v] = 1.0
        H[v, u] = 1.0 # MST is undirected, but ACO is directed. 
        # Should we allow both directions?
        # The tour is a cycle. The MST is a tree.
        # If we restrict to MST edges, the ant can only move along tree edges.
        # Since the tree is connected, the ant can reach any node.
        # However, the "return to 0" edge might not be in the MST.
        # If the ant is at node X and 0 is the only unvisited node, it must go X->0.
        # If (X,0) is not in MST, H[X,0] is 1e-9.
        # This is fine because it's the only choice.
        
    # Ensure diagonal is 0 or 1e-9? 
    # Contract: "Heuristic entries that are not positive become 1e-9".
    # Usually diagonal is ignored because visited[0] is True initially, and we don't revisit.
    # But to be safe, set diagonal to 1e-9.
    np.fill_diagonal(H, 1e-9)
    
    # 3. Evaluation (Optional but good for sanity check within budget)
    # We have plenty of time (280s deadline).
    # Let's run a quick simulation to ensure it doesn't crash and gives a finite score.
    # We won't use this to refine, as the strategy is fixed (MST backbone).
    # But we can check if the MST edges are sufficient.
    
    # Check time
    if time.time() > deadline - 10.0:
        return H
        
    # Quick simulation to verify validity
    try:
        sim_score = _fast_aco_simulate(dm, H, n_ants=5, n_iterations=5, seed=123)
        if not np.isfinite(sim_score):
            # Fallback to inverse distance if MST heuristic fails (shouldn't happen)
            safe_dist = dm.copy()
            np.fill_diagonal(safe_dist, 1.0)
            safe_dist[safe_dist < 1e-9] = 1e-9
            H = 1.0 / safe_dist
            H[H <= 0] = 1e-9
            H[~np.isfinite(H)] = 1e-9
    except Exception:
        # Fallback
        safe_dist = dm.copy()
        np.fill_diagonal(safe_dist, 1.0)
        safe_dist[safe_dist < 1e-9] = 1e-9
        H = 1.0 / safe_dist
        H[H <= 0] = 1e-9
        H[~np.isfinite(H)] = 1e-9

    # Final sanity checks
    H[H <= 0] = 1e-9
    H[~np.isfinite(H)] = 1e-9
    
    return H
