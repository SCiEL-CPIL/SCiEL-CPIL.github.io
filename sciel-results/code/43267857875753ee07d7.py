"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_TIME_LIMIT = 115.0  # Safety margin from 120s limit

def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _TIME_LIMIT

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    _ensure_start_time()
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # If time is running out or the problem is large enough that deep lookahead is risky,
    # fall back to nearest neighbor to ensure we finish in time.
    # However, for 50 cities, even a shallow lookahead is very fast.
    # Let's use a lookahead depth of 3 or 4.
    
    # Optimization: Pre-compute distances from current_node to all unvisited
    dists_to_unvisited = distance_matrix[current_node, unvisited_nodes]
    
    # If time is critical, just use nearest neighbor
    if _time_exceeded():
        return int(unvisited_nodes[np.argmin(dists_to_unvisited)])
    
    best_candidate_idx = 0
    best_score = float('inf')
    
    # Lookahead depth: How many steps ahead to simulate
    # For 50 nodes, simulating 3-4 steps for each of ~50 candidates is 150-200 operations.
    # This is trivial for numpy/python in 120s.
    # We can even look ahead to the end if we are fast, but let's stick to a moderate depth
    # to balance quality and risk. Depth 3 is usually a good sweet spot.
    lookahead_depth = 3
    
    for i, candidate_node in enumerate(unvisited_nodes):
        # Cost to get to candidate
        cost = dists_to_unvisited[i]
        
        # Simulate next `lookahead_depth` steps from candidate_node
        # We need to know which nodes are visited in the simulation.
        # Since we are only simulating a few steps, we can just track the set of visited nodes locally.
        # But wait, `unvisited_nodes` is the list of ALL unvisited nodes.
        # In the simulation, we visit `candidate_node`, so it is no longer available for the subsequent steps.
        
        # To make this efficient, we can just pick the nearest neighbor from the remaining unvisited nodes
        # excluding the one we just picked.
        
        # For simplicity and speed, let's just do a quick greedy simulation.
        # We need to be careful not to modify the original unvisited_nodes array.
        
        # A cheaper approximation:
        # Score = dist(current, candidate) + dist(candidate, nearest_unvisited_excluding_candidate) + ...
        
        # Let's try a slightly more robust lookahead:
        # 1. Pick candidate.
        # 2. From candidate, pick nearest neighbor in (unvisited - {candidate}).
        # 3. From that neighbor, pick nearest neighbor in (unvisited - {candidate, neighbor}).
        
        # To do this efficiently without creating new arrays every time:
        # We can just check indices.
        
        # Actually, for 50 nodes, creating a small list is fine.
        # But let's try to be smart.
        
        # Let's just do a simple 1-step lookahead first (greedy with 1-step look ahead)
        # Score = dist(current, cand) + dist(cand, nearest(remaining))
        
        # If we have time, we can do deeper.
        
        # Let's implement a variable depth lookahead based on remaining time?
        # No, keep it simple. Depth 2 or 3 is usually better than 1.
        
        # Let's do Depth 2:
        # Cost = d(c, cand) + min_{u in unvisited \ {cand}} d(cand, u)
        
        # We can precompute the two nearest neighbors for each node in the unvisited set?
        # No, that's complex.
        
        # Simple loop:
        if i < n_unvisited - 1:
            # Find the nearest neighbor to candidate_node among the other unvisited nodes
            # We can use the distance matrix directly.
            # dists_from_candidate = distance_matrix[candidate_node, unvisited_nodes]
            # We need to exclude the candidate itself (index i in unvisited_nodes)
            
            # To avoid creating a copy of the array, we can just find the min and check if it's the candidate.
            # But argmin gives the index in the subarray.
            
            # Let's just compute the min distance to any OTHER node.
            # This is O(N) per candidate, so O(N^2) total. 50^2 = 2500 ops. Very fast.
            
            # Get distances from candidate to all unvisited
            # We already have distance_matrix.
            # Let's just do:
            # min_dist_next = np.min(distance_matrix[candidate_node, unvisited_nodes])
            # But this might pick the candidate itself if distance is 0 (which it is).
            # So we need to exclude index i.
            
            # Efficient way:
            # Create a mask or just check the min.
            # Since N is small, let's just do a quick check.
            
            # Actually, we can precompute the two smallest distances from each node to the unvisited set?
            # No, let's just do the inner loop for the lookahead.
            
            # Lookahead 1 step:
            # Find nearest neighbor to candidate in unvisited_nodes \ {candidate_node}
            
            # We can use np.argmin on a modified view?
            # Let's just use a simple Python loop for the inner step since it's small,
            # or use numpy with a mask.
            
            # Mask approach:
            # mask = np.ones(n_unvisited, dtype=bool)
            # mask[i] = False
            # min_dist = np.min(distance_matrix[candidate_node, unvisited_nodes][mask])
            
            # This creates a boolean array and an indexed array. 50 * 50 = 2500 operations.
            # It's fine.
            
            # Let's try to optimize by just checking the global min and second min?
            # No, let's stick to the mask for clarity and correctness, it's fast enough.
            
            # But wait, we can do better.
            # Let's just use the fact that we want to minimize d(c, cand) + d(cand, next)
            
            # Let's implement a small recursive lookahead or iterative.
            # Iterative is safer.
            
            # State: current_sim_node = candidate_node
            # Visited in sim: {candidate_node}
            # Steps remaining: lookahead_depth
            
            # For simplicity, let's just do 1-step lookahead first.
            # If that's not enough, we can increase depth.
            
            # Let's do 2-step lookahead.
            
            # Step 1: From candidate, go to nearest unvisited (excluding candidate)
            # Step 2: From that node, go to nearest unvisited (excluding candidate and step1 node)
            
            # To do this efficiently:
            # We can pre-compute the nearest neighbor for each node in the unvisited set, 
            # but we have to exclude already visited nodes.
            
            # Given the small size, let's just do the simulation.
            
            # Simulation:
            sim_current = candidate_node
            sim_visited = set([i]) # Index in unvisited_nodes
            
            sim_cost = 0.0
            for _ in range(lookahead_depth):
                # Find nearest neighbor to sim_current in unvisited_nodes not in sim_visited
                # We need to map node ID to index in unvisited_nodes?
                # No, unvisited_nodes contains node IDs.
                # distance_matrix is indexed by node ID.
                
                # Get distances from sim_current to all unvisited nodes
                dists = distance_matrix[sim_current, unvisited_nodes]
                
                # Mask out visited nodes
                # We need to know the index of each node in unvisited_nodes to mask it.
                # This is tricky because unvisited_nodes is an array of node IDs.
                
                # Alternative:
                # Just iterate through unvisited_nodes and check if it's in sim_visited?
                # sim_visited contains indices in unvisited_nodes.
                
                # Let's create a mask based on indices.
                # mask = np.ones(n_unvisited, dtype=bool)
                # for idx in sim_visited:
                #     mask[idx] = False
                
                # This is O(K) where K is size of sim_visited (small).
                # Then find argmin of dists[mask].
                
                # If no unvisited nodes left, break.
                if len(sim_visited) == n_unvisited:
                    break
                
                mask = np.ones(n_unvisited, dtype=bool)
                for idx in sim_visited:
                    mask[idx] = False
                
                if not np.any(mask):
                    break
                    
                # Find nearest
                # We need the index in the masked array, then map back to original index
                # Actually, np.argmin on the masked array gives the index in the masked array?
                # No, if we use dists[mask], the length is reduced.
                # We need the index in the original unvisited_nodes array.
                
                # Let's use np.where or similar.
                # Or just set visited distances to inf.
                
                # Faster:
                # dists_masked = dists.copy()
                # for idx in sim_visited:
                #     dists_masked[idx] = np.inf
                # next_idx = np.argmin(dists_masked)
                
                # This is O(N) per step.
                
                dists_masked = dists.copy()
                for idx in sim_visited:
                    dists_masked[idx] = np.inf
                
                next_idx = int(np.argmin(dists_masked))
                
                if next_idx == -1 or dists_masked[next_idx] == np.inf:
                    break
                
                sim_cost += dists_masked[next_idx]
                sim_visited.add(next_idx)
                sim_current = unvisited_nodes[next_idx]
            
            total_score = dists_to_unvisited[i] + sim_cost
            
            if total_score < best_score:
                best_score = total_score
                best_candidate_idx = i
        else:
            # Only one unvisited node left (or this is the only option)
            # Actually, if i is the last index, and n_unvisited > 1, there are others.
            # The logic above handles it.
            # Wait, the `if i < n_unvisited - 1` was to avoid edge cases?
            # No, the simulation handles it.
            # Let's remove the if/else and just run the simulation for all.
            pass
            
    # The above loop had a flaw: I put the logic in an if block.
    # Let's restructure the main loop properly.
    
    best_candidate_idx = 0
    best_score = float('inf')
    
    for i, candidate_node in enumerate(unvisited_nodes):
        cost = dists_to_unvisited[i]
        
        # Lookahead simulation
        sim_current = candidate_node
        sim_visited = set([i])
        sim_cost = 0.0
        
        for _ in range(lookahead_depth):
            if len(sim_visited) == n_unvisited:
                break
            
            dists = distance_matrix[sim_current, unvisited_nodes]
            dists_masked = dists.copy()
            for idx in sim_visited:
                dists_masked[idx] = np.inf
            
            next_idx = int(np.argmin(dists_masked))
            
            if dists_masked[next_idx] == np.inf:
                break
            
            sim_cost += dists_masked[next_idx]
            sim_visited.add(next_idx)
            sim_current = unvisited_nodes[next_idx]
            
        total_score = cost + sim_cost
        
        if total_score < best_score:
            best_score = total_score
            best_candidate_idx = i
            
    return int(unvisited_nodes[best_candidate_idx])
# EVOLVE-BLOCK-END
