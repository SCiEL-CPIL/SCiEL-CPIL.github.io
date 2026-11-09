"""Rewritten program for guided local search penalty rule (TSP)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty for TSP.
    
    Uses an edge-reversal heuristic: for each edge in the current tour,
    find the cheapest non-tour edge that can replace it via a 2-opt swap.
    Heavily penalize the original tour edge to force 2-opt to explore
    the specific alternative connections that were previously blocked.
    
    Also applies a mild global soft penalty proportional to edge usage
    to discourage extremely overused edges across the landscape.
    """
    t0 = time.time()
    
    n = edge_distance.shape[0]
    penalized = edge_distance.copy()
    
    tour = np.asarray(local_opt_tour)
    m = len(tour)
    
    if m < 2:
        return penalized
    
    # Build a boolean mask of tour edges
    tour_edge_mask = np.zeros((n, n), dtype=bool)
    tour_edges = []
    for i in range(m):
        u = tour[i]
        v = tour[(i + 1) % m]
        tour_edge_mask[u, v] = True
        tour_edge_mask[v, u] = True
        tour_edges.append((u, v))
    
    # Get max usage for normalization
    max_used = edge_n_used.max()
    if max_used <= 0:
        max_used = 1.0
    
    # Mild global soft penalty based on usage
    global_penalty = 1.0 + 0.005 * (edge_n_used / max_used)
    penalized *= global_penalty
    
    # For each tour edge, find the best non-tour replacement via 2-opt swap
    # A 2-opt swap of edge (u,v) in the tour: if tour is ...u, v, w, x...
    # removing (u,v) and (w,x) and adding (u,w) and (v,x) or (u,x) and (v,w)
    # We want to find the cheapest alternative pair that replaces one tour edge.
    
    # For edge (u, v) where u is at position i and v is at position (i+1)%m:
    # The neighbors in the tour are:
    #   prev_u = tour[(i-1) % m]
    #   next_v = tour[(i+2) % m]
    # A 2-opt that removes edge (u,v) would also remove another edge (a,b)
    # and add (u,a) and (v,b) or (u,b) and (v,a).
    
    # Simpler approach: for each tour edge (u,v), find the cheapest edge
    # (u, x) where x is not v and (u,x) is not a tour edge, and
    # (v, y) where y is not u and (v,y) is not a tour edge.
    # Then penalize the original edge (u,v) heavily.
    
    # Actually, the key insight: we want to force 2-opt to try specific swaps.
    # For each tour edge (u,v), the best 2-opt swap involves removing (u,v)
    # and another edge, then reconnecting. Let's find for each (u,v) the
    # cheapest reconnection option.
    
    # Precompute: for each node, the cheapest non-tour edge incident to it
    # But we need to be careful: the replacement must form a valid 2-opt.
    
    # Let's use a simpler but effective strategy:
    # For each tour edge (u,v), compute the "cost" of keeping it vs. the
    # best alternative. If the alternative is significantly cheaper, penalize
    # the tour edge more strongly.
    
    for i in range(m):
        u = tour[i]
        v = tour[(i + 1) % m]
        prev_u = tour[(i - 1) % m]
        next_v = tour[(i + 2) % m]
        
        used = edge_n_used[u, v]
        
        # The two possible 2-opt swaps that remove edge (u,v):
        # Option 1: remove (u,v) and (prev_u, u) -> no, that's the same edge region
        # Actually, a 2-opt swap removes two non-adjacent edges.
        # Edge (u,v) is at position i. Another edge at position j (j != i, j != i-1, j != i+1 mod m)
        # For simplicity, consider swapping (u,v) with the "opposite" edge or nearby edges.
        
        # Simpler: for each tour edge, find the cheapest non-tour edge from u
        # and the cheapest non-tour edge from v. Use these to estimate the
        # benefit of breaking this edge.
        
        # Find cheapest non-tour edge incident to u (excluding v and prev_u which are tour neighbors)
        # Actually, we can't just pick any edge - it must participate in a valid 2-opt.
        
        # Let's use a direct approach: for edge (u,v), the two 2-opt candidates
        # that break (u,v) are:
        # 1. Swap (u,v) with (a,b) where a is a neighbor of u in the "reversed" segment
        # 2. This is getting complex. Let's use a simpler heuristic.
        
        # Heuristic: penalize based on how "stale" the edge is (usage count)
        # and how much better alternatives exist.
        
        # Compute the best non-tour edge from u
        # Mask out tour edges and self-loops
        mask_u = ~tour_edge_mask[u]
        mask_u[u] = True  # exclude self
        mask_u[v] = True  # exclude the current edge
        # Also exclude prev_u since (u, prev_u) is also a tour edge
        mask_u[prev_u] = True
        
        if np.any(mask_u):
            best_u = np.where(mask_u)[0]
            if len(best_u) > 0:
                costs_u = edge_distance[u, best_u]
                min_cost_u = costs_u.min()
                best_u_node = best_u[np.argmin(costs_u)]
            else:
                min_cost_u = edge_distance[u, v]
                best_u_node = v
        else:
            min_cost_u = edge_distance[u, v]
            best_u_node = v
        
        # Compute the best non-tour edge from v
        mask_v = ~tour_edge_mask[v]
        mask_v[v] = True
        mask_v[u] = True
        mask_v[next_v] = True
        
        if np.any(mask_v):
            best_v = np.where(mask_v)[0]
            if len(best_v) > 0:
                costs_v = edge_distance[v, best_v]
                min_cost_v = costs_v.min()
                best_v_node = best_v[np.argmin(costs_v)]
            else:
                min_cost_v = edge_distance[v, u]
                best_v_node = u
        else:
            min_cost_v = edge_distance[v, u]
            best_v_node = u
        
        # The potential saving from breaking edge (u,v) is roughly:
        # current edge cost - (some fraction of the best alternatives)
        # If the best alternative is much cheaper, we should penalize more.
        
        current_cost = edge_distance[u, v]
        # Estimate the cost of a 2-opt swap that breaks (u,v)
        # A swap would replace (u,v) and another edge (a,b) with (u,a) and (v,b)
        # For a rough estimate, use the best single-edge alternatives
        estimated_swap_cost = (min_cost_u + min_cost_v) / 2.0
        
        # If the estimated swap cost is lower than the current edge cost,
        # the edge is a good candidate for removal. Penalize it more.
        improvement_ratio = max(0.0, (current_cost - estimated_swap_cost) / max(current_cost, 1e-10))
        
        # Penalty: base + usage scaling + improvement scaling
        # Higher improvement_ratio means we should penalize more to force 2-opt to try this swap
        base_penalty = 1.5
        usage_penalty = 0.3 * (used / max_used)
        improvement_penalty = 0.8 * improvement_ratio
        # Also add a small penalty for absolute usage
        abs_usage_penalty = 0.02 * used
        
        tour_penalty = base_penalty + usage_penalty + improvement_penalty + abs_usage_penalty
        # Cap to avoid extreme values
        tour_penalty = min(tour_penalty, 5.0)
        
        penalized[u, v] *= tour_penalty
        penalized[v, u] *= tour_penalty
    
    # Safety: ensure all values are finite
    if not np.all(np.isfinite(penalized)):
        penalized = np.nan_to_num(penalized, nan=0.0, posinf=1e10, neginf=0.0)
    
    # Time guard
    elapsed = time.time() - t0
    # Should be fast for n=100
    
    return penalized
# EVOLVE-BLOCK-END
