import numpy as np
import time
import random

def _construct_randomized_nn(dist, n, seed=0):
    """
    Construct a tour using randomized nearest neighbor.
    At each step, pick the next node from the top-k nearest unvisited nodes with some randomness.
    """
    rng = random.Random(seed)
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    current = 0
    tour = [0]
    
    for _ in range(n - 1):
        unvisited = np.where(~visited)[0]
        if len(unvisited) == 0:
            break
        
        # Get distances to unvisited
        dists = dist[current, unvisited]
        
        # Sort by distance
        sorted_indices = np.argsort(dists)
        sorted_unvisited = unvisited[sorted_indices]
        sorted_dists = dists[sorted_indices]
        
        # Pick from top 3 with probability weighting (favor closer)
        k = min(3, len(sorted_unvisited))
        candidates = sorted_unvisited[:k]
        
        # Use softmax-like selection based on inverse distance or just random from top k
        # Let's use inverse distance weights
        weights = 1.0 / (sorted_dists[:k] + 1e-9)
        total_w = np.sum(weights)
        if total_w == 0:
            next_node = candidates[0]
        else:
            probs = weights / total_w
            next_node = rng.choices(candidates, weights=probs, k=1)[0]
        
        visited[next_node] = True
        tour.append(next_node)
        current = next_node
        
    tour.append(0)
    return np.array(tour)

def _cheapest_insertion(dist, n, seed=0):
    """
    Construct a tour using cheapest insertion heuristic.
    Start with a random edge, then insert remaining nodes at the position with minimum cost increase.
    """
    rng = random.Random(seed)
    
    # Start with a random pair
    start_nodes = rng.sample(range(n), 2)
    a, b = start_nodes[0], start_nodes[1]
    
    # Initial path: a -> b
    path = [a, b]
    visited = set([a, b])
    
    remaining = list(set(range(n)) - visited)
    rng.shuffle(remaining)
    
    while remaining:
        # Pick the next node to insert (first in remaining for determinism in insertion logic, 
        # but we shuffled remaining so the order of consideration is random)
        node = remaining.pop(0)
        
        # Find best position to insert node in current path
        min_cost = float('inf')
        best_pos = -1
        
        m = len(path)
        for i in range(m):
            u = path[i]
            v = path[(i + 1) % m]
            # Cost increase if we insert node between u and v
            # New edges: u->node, node->v
            # Removed edge: u->v
            cost_inc = dist[u, node] + dist[node, v] - dist[u, v]
            if cost_inc < min_cost:
                min_cost = cost_inc
                best_pos = i
        
        # Insert node after path[best_pos]
        if best_pos == m - 1:
            path.append(node)
        else:
            path.insert(best_pos + 1, node)
            
    # Close the tour
    path.append(path[0])
    return np.array(path)

def _two_opt_improve(dist, tour, n, max_iters=50):
    """Improve tour using 2-opt. tour is a closed tour (first==last)."""
    path = list(tour[:-1])
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(n - 2):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = path[i], path[i + 1]
                c, d = path[j], path[(j + 1) % n]
                old_cost = dist[a, b] + dist[c, d]
                new_cost = dist[a, c] + dist[b, d]
                if new_cost < old_cost - 1e-10:
                    path[i + 1:j + 1] = path[i + 1:j + 1][::-1]
                    improved = True
    return np.array(path + [path[0]])

def _or_opt_improve(dist, tour, n, max_iters=20):
    """
    Or-Opt: move a segment of 1, 2, or 3 nodes to another position.
    """
    path = list(tour[:-1])
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        # Try segment sizes 1, 2, 3
        for seg_len in [1, 2, 3]:
            if seg_len >= n:
                continue
            for i in range(n):
                # Extract segment starting at i
                segment = []
                for k in range(seg_len):
                    segment.append(path[(i + k) % n])
                
                # Identify edges removed
                # Edges: prev->first, last->next
                prev_node = path[(i - 1) % n]
                next_node = path[(i + seg_len) % n]
                
                # Try inserting segment at every other position j
                for j in range(n):
                    if j == i or j == (i + seg_len) % n:
                        continue
                    # Check if j is inside the segment being moved
                    # Simple check: if j is in [i, i+seg_len-1], skip
                    if j >= i and j < i + seg_len:
                        continue
                    # Also handle wrap-around for simplicity, though n=50 is small
                    
                    # Edges at insertion point j:
                    # Insert before path[j]
                    # Current edges: path[j-1] -> path[j]
                    # New edges: path[j-1] -> segment[0], segment[-1] -> path[j]
                    
                    u = path[(j - 1) % n]
                    v = path[j]
                    
                    # Cost change:
                    # Remove: prev->seg[0], seg[-1]->next, u->v
                    # Add: prev->u, seg[-1]->v, seg[0]->v ? No, wait.
                    
                    # Let's trace carefully.
                    # Original path: ... prev -> s0 -> s1 -> ... -> sk-1 -> next ...
                    # We remove s0...sk-1.
                    # New path part 1: ... prev -> next ...
                    # We insert s0...sk-1 between u and v.
                    # New path part 2: ... u -> s0 -> ... -> sk-1 -> v ...
                    
                    # Removed edges:
                    # 1. prev -> s0
                    # 2. sk-1 -> next
                    # 3. u -> v
                    # Added edges:
                    # 1. prev -> next
                    # 2. u -> s0
                    # 3. sk-1 -> v
                    
                    cost_removed = dist[prev_node, segment[0]] + dist[segment[-1], next_node] + dist[u, v]
                    cost_added = dist[prev_node, next_node] + dist[u, segment[0]] + dist[segment[-1], v]
                    
                    if cost_added < cost_removed - 1e-10:
                        # Perform move
                        # 1. Remove segment from i
                        # We need to be careful with indices if i < j
                        # Let's just rebuild the list to be safe, n=50 is small
                        
                        new_path = []
                        # Copy path, skipping segment at i
                        for k in range(n):
                            if k >= i and k < i + seg_len:
                                continue
                            new_path.append(path[k])
                        
                        # Now insert segment before index j in new_path
                        # Note: indices in new_path might have shifted
                        # Find the index of v in new_path to insert before it
                        # Since v = path[j] (original), and we didn't move v, v is still in new_path
                        # Find v's index in new_path
                        v_idx = new_path.index(v)
                        
                        # Insert segment
                        new_path[v_idx:v_idx] = segment
                        
                        # Check if valid (no duplicates, length n)
                        if len(new_path) == n and len(set(new_path)) == n:
                            path = new_path
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    return np.array(path + [path[0]])

def _local_search(dist, tour, n, max_2opt=50, max_oropt=10):
    """Combine 2-opt and Or-Opt."""
    current_tour = tour
    for _ in range(3): # Alternate 2-opt and Or-Opt
        current_tour = _two_opt_improve(dist, current_tour, n, max_iters=max_2opt)
        current_tour = _or_opt_improve(dist, current_tour, n, max_iters=max_oropt)
    return current_tour

def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for ACO TSP.
    Strategy: Multi-start diverse construction + Local Search -> Consensus Heuristic.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    n = distance_matrix.shape[0]
    
    # Baseline: Inverse distance
    heuristic = 1.0 / (distance_matrix + 1e-9)
    
    # Edge frequency counter
    edge_counts = np.zeros((n, n), dtype=np.int32)
    
    # Number of diverse tours to generate
    num_tours = 20
    
    for i in range(num_tours):
        if time.time() > deadline - 10: # Leave 10s for final processing
            break
            
        # Alternate construction methods
        if i % 2 == 0:
            # Randomized Nearest Neighbor
            seed = i * 12345
            tour = _construct_randomized_nn(distance_matrix, n, seed=seed)
        else:
            # Cheapest Insertion
            seed = i * 98765
            tour = _cheapest_insertion(distance_matrix, n, seed=seed)
        
        # Local Search
        tour = _local_search(distance_matrix, tour, n, max_2opt=30, max_oropt=5)
        
        # Count edges
        for k in range(n):
            u = tour[k]
            v = tour[k + 1]
            edge_counts[u, v] += 1
            edge_counts[v, u] += 1
            
    # Calculate max count to normalize
    max_count = np.max(edge_counts)
    
    if max_count > 0:
        # Consensus heuristic:
        # If an edge is in many good tours, it's likely part of the optimal tour.
        # We want to boost these edges significantly.
        # Base: inverse distance.
        # Boost: proportional to frequency.
        
        # Create a mask for edges that appeared in at least one tour
        # We want to strongly favor high-frequency edges.
        
        # Let's use a multiplicative factor based on frequency.
        # freq_factor = 1 + (edge_counts / max_count) * boost_strength
        # But we want a steeper curve. Maybe exp or power?
        
        # Let's try:
        # heuristic[i,j] = base[i,j] * (1 + (edge_counts[i,j] / max_count) * 20)
        # This gives a 21x boost for the most common edges.
        
        # However, we must be careful not to make non-tour edges too small if they are short.
        # The ACO probability is proportional to pheromone * heuristic.
        # Pheromone starts at 1.
        # If heuristic is huge for tour edges, ants will likely pick them.
        
        # Let's refine:
        # For edges in the consensus:
        #   value = base * (1 + 20 * (count / max_count))
        # For edges not in any consensus:
        #   value = base
        
        boost_strength = 20.0
        normalized_counts = edge_counts / max_count
        
        # Apply boost
        boost_factor = 1.0 + boost_strength * normalized_counts
        heuristic = heuristic * boost_factor
        
        # Ensure diagonal is small
        np.fill_diagonal(heuristic, 1e-9)
        
    else:
        # Fallback if no tours were generated
        np.fill_diagonal(heuristic, 1e-9)
        
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
