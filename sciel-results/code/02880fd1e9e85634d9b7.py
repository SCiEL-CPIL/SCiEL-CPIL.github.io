"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    adj_bool = (adj > 0.5).astype(np.int8)
    degrees = adj_bool.sum(axis=1).astype(np.int32)
    
    # Precompute neighbor lists
    neighbor_indices = [np.nonzero(adj_bool[i])[0] for i in range(n)]
    
    def greedy_size_from_scores(scores):
        """Compute the size of the greedy independent set given a score vector."""
        order = np.argsort(-scores, kind='stable')
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                count += 1
                selected[v] = True
                if len(neighbor_indices[v]) > 0:
                    selected[neighbor_indices[v]] = True
        return count
    
    def order_to_scores(order, n):
        """Convert a peeling order to scores such that argsort(-scores) matches the order."""
        scores = np.zeros(n, dtype=np.float64)
        for i, v in enumerate(order):
            scores[v] = float(n - i)  # First selected gets highest score
        # Vertices not in order (if peeling terminated early) get low scores
        for v in range(n):
            if scores[v] == 0.0:
                scores[v] = 0.5
        return scores
    
    # --- Core Strategy: Maximal Independent Set Construction + Local Search (Swap Improvement) ---
    # Instead of just ordering vertices for greedy, we:
    # 1. Build a large independent set via randomized greedy (multiple restarts)
    # 2. Improve it via local search: try swapping out one vertex for two others
    # 3. Assign high scores to the final set members so the decoder picks them first
    # 4. For remaining vertices, use min-degree ordering
    
    def randomized_greedy_mis(adj_bool, neighbor_indices, n, rng):
        """
        Build a maximal independent set using randomized greedy.
        Returns the set as a boolean array.
        """
        active = set(range(n))
        selected = np.zeros(n, dtype=bool)
        
        while active:
            if time.time() > deadline - 20.0:
                break
            
            # Pick a random vertex from active
            v = rng.choice(list(active))
            selected[v] = True
            active.discard(v)
            
            # Remove neighbors
            for u in neighbor_indices[v]:
                if u in active:
                    active.discard(u)
        
        return selected
    
    def local_search_swap_improvement(selected, adj_bool, neighbor_indices, n, max_iter=200):
        """
        Try to improve the independent set by swapping out one vertex for two others.
        Returns the improved set as a boolean array.
        """
        selected_set = set(np.nonzero(selected)[0])
        not_selected_set = set(range(n)) - selected_set
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            if time.time() > deadline - 20.0:
                break
            improved = False
            iter_count += 1
            
            # Try swapping out each selected vertex
            for v in list(selected_set):
                # Find vertices not in the set that are not neighbors of v
                # and are neighbors of at least two other non-selected vertices
                # such that adding them and removing v increases the set size
                
                # For each non-selected vertex u that is NOT a neighbor of v,
                # check if we can add u and another vertex w (also not neighbor of v)
                # while removing v
                
                # Simpler: try to remove v and add any two vertices that were blocked by v
                # but are not neighbors of each other
                candidates = []
                for u in not_selected_set:
                    if u == v:
                        continue
                    # u is not in the set, so it's blocked by some selected vertex
                    # Check if v is one of the blockers of u
                    if adj_bool[v, u] == 1:
                        candidates.append(u)
                
                if len(candidates) < 2:
                    continue
                
                # Try pairs of candidates
                for i in range(min(len(candidates), 10)):
                    for j in range(i+1, min(len(candidates), 10)):
                        u, w = candidates[i], candidates[j]
                        # Check if u and w are not neighbors
                        if adj_bool[u, w] == 0:
                            # Check that removing v doesn't block u or w from other selected vertices
                            # Actually, u and w were blocked by v (and possibly others)
                            # We need to ensure that after removing v, u and w can both be added
                            # This means no other selected vertex is adjacent to u or w
                            
                            can_add_u = True
                            can_add_w = True
                            for s in selected_set:
                                if s == v:
                                    continue
                                if adj_bool[s, u] == 1:
                                    can_add_u = False
                                    break
                                if adj_bool[s, w] == 1:
                                    can_add_w = False
                                    break
                            
                            if can_add_u and can_add_w:
                                # Perform the swap: remove v, add u and w
                                selected_set.discard(v)
                                selected_set.add(u)
                                selected_set.add(w)
                                not_selected_set.discard(u)
                                not_selected_set.discard(w)
                                not_selected_set.add(v)
                                selected[v] = False
                                selected[u] = True
                                selected[w] = True
                                improved = True
                                break
                    if improved:
                        break
            
            if not improved:
                # Try single-vertex additions (in case the set wasn't maximal)
                for u in not_selected_set:
                    can_add = True
                    for s in selected_set:
                        if adj_bool[s, u] == 1:
                            can_add = False
                            break
                    if can_add:
                        selected_set.add(u)
                        not_selected_set.discard(u)
                        selected[u] = True
                        improved = True
                        break
        
        return selected
    
    def min_degree_order_for_remaining(adj_bool, neighbor_indices, n, blocked):
        """
        For vertices not in the independent set, create a min-degree ordering
        among the remaining graph.
        """
        active = set(range(n)) - set(np.nonzero(blocked)[0])
        deg = degrees.astype(np.float64).copy()
        
        # Update degrees for the remaining graph
        for v in active:
            deg[v] = sum(1 for u in neighbor_indices[v] if u in active)
        
        order = []
        while active:
            if time.time() > deadline - 20.0:
                break
            
            # Find vertex with minimum degree
            min_deg = float('inf')
            best_v = -1
            for v in active:
                if deg[v] < min_deg:
                    min_deg = deg[v]
                    best_v = v
            
            if best_v == -1:
                break
            
            order.append(best_v)
            active.discard(best_v)
            
            for u in neighbor_indices[best_v]:
                if u in active:
                    active.discard(u)
                    for w in neighbor_indices[u]:
                        if w in active:
                            deg[w] -= 1
        
        return order
    
    # --- Main: Multi-restart randomized greedy + local search ---
    
    best_scores = None
    best_size = -1
    
    num_restarts = 30
    
    for r in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
        
        rng = np.random.RandomState(12345 + r)
        
        # Build a maximal independent set
        mis = randomized_greedy_mis(adj_bool, neighbor_indices, n, rng)
        current_size = mis.sum()
        
        # Local search to improve
        mis = local_search_swap_improvement(mis.copy(), adj_bool, neighbor_indices, n, max_iter=100)
        current_size = mis.sum()
        
        if current_size > best_size:
            best_size = current_size
            
            # Create scores: high for MIS members, then min-degree for rest
            scores = np.zeros(n, dtype=np.float64)
            mis_members = np.nonzero(mis)[0]
            
            # Assign high scores to MIS members (in reverse order of their "importance")
            # Just assign them all the same high score; ties broken by index
            for v in mis_members:
                scores[v] = n + 1.0  # High score
            
            # For non-MIS vertices, use min-degree ordering
            remaining_order = min_degree_order_for_remaining(adj_bool, neighbor_indices, n, mis)
            for i, v in enumerate(remaining_order):
                if not mis[v]:
                    scores[v] = float(len(remaining_order) - i)
            
            best_scores = scores
    
    # Fallback: if no valid scores, use simple degree-based
    if best_scores is None:
        best_scores = -degrees.astype(np.float64)
    
    return best_scores.astype(float)

# EVOLVE-BLOCK-END
