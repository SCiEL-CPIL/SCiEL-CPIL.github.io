import time
import random
import math

def _build_adjacency(n, edges):
    adj = [[0] * n for _ in range(n)]
    deg = [0] * n
    for u, v in edges:
        adj[u][v] = 1
        adj[v][u] = 1
        deg[u] += 1
        deg[v] += 1
    return adj, deg

def _compute_scores(n, adj, deg):
    """
    Computes max triangle degree sum and sum of squared triangle degree sums.
    """
    max_sum = 0
    sum_sq = 0
    
    # Enumerate all triangles
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                for w in range(v + 1, n):
                    if adj[u][w] and adj[v][w]:
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
                        sum_sq += s * s
    return max_sum, sum_sq

def construct():
    n = 16
    m = 65
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Initial construction: K_{8,8} + 1 edge
    # Vertices 0-7 and 8-15
    initial_edges = set()
    for u in range(8):
        for v in range(8, 16):
            initial_edges.add((u, v))
    # Add one edge within part A to make 65 edges and create triangles
    initial_edges.add((0, 1))
    
    # Convert to list for easy manipulation, keep set for O(1) lookup
    cur_edges = set(initial_edges)
    adj, deg = _build_adjacency(n, list(cur_edges))
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    all_edges_set = set(all_edges)
    
    # Initial scores
    cur_max, cur_sq = _compute_scores(n, adj, deg)
    best_max = cur_max
    best_sq = cur_sq
    best_edges = set(cur_edges)
    
    # Simulated Annealing Parameters
    T_start = 10.0
    T_end = 0.01
    # We don't know exact iterations, so we loop until deadline
    # But we need a cooling schedule. Let's estimate iterations or just use time-based cooling.
    # Better: fixed number of iterations per "restart" or just continuous cooling.
    # Let's do continuous cooling based on time elapsed.
    
    rng = random.Random(42)
    
    # To speed up random selection of non-edges, we can maintain a list of non-edges,
    # but updating it is O(m). n=16 is small, so picking a random edge and checking if it's in set is O(1) average.
    # Total edges = 120. Current = 65. Non-edges = 55.
    # Probability of picking a non-edge randomly from all 120 is 55/120 ~ 45%.
    # So on average 2-3 tries to find a non-edge. This is fine.
    
    while time.time() < deadline:
        # Calculate temperature based on time
        elapsed = time.time() - start_time
        total_time = deadline - start_time
        progress = elapsed / total_time
        T = T_start * (T_end / T_start) ** progress
        
        # Pick edge to remove
        e_out = rng.choice(list(cur_edges))
        u1, v1 = e_out
        
        # Pick edge to add
        # Retry a few times if we pick an existing edge
        e_in = None
        for _ in range(10):
            candidate = rng.choice(all_edges)
            if candidate not in cur_edges:
                e_in = candidate
                break
        
        if e_in is None:
            continue
            
        u2, v2 = e_in
        
        # Perform swap
        cur_edges.discard(e_out)
        cur_edges.add(e_in)
        
        adj[u1][v1] = 0
        adj[v1][u1] = 0
        deg[u1] -= 1
        deg[v1] -= 1
        
        adj[u2][v2] = 1
        adj[v2][u2] = 1
        deg[u2] += 1
        deg[v2] += 1
        
        # Compute new scores
        new_max, new_sq = _compute_scores(n, adj, deg)
        
        # Cost function: Maximize penalty for high max sum, and penalize variance (sum of squares)
        # We want to minimize MaxSum.
        # Surrogate: SumSq helps balance.
        # Let's use Cost = new_max * 100 + new_sq
        # Or simply compare new_max first, then new_sq.
        # SA usually works best with a single scalar cost.
        # Let's use Cost = new_max ** 2 + 0.1 * new_sq
        old_cost = (cur_max ** 2) + 0.1 * cur_sq
        new_cost = (new_max ** 2) + 0.1 * new_sq
        
        delta = new_cost - old_cost
        
        # Acceptance criterion
        accepted = False
        if delta <= 0:
            accepted = True
        else:
            # Boltzmann distribution
            if T > 0:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    accepted = True
        
        if accepted:
            cur_max = new_max
            cur_sq = new_sq
            # Update best
            if cur_max < best_max or (cur_max == best_max and cur_sq < best_sq):
                best_max = cur_max
                best_sq = cur_sq
                best_edges = set(cur_edges)
        else:
            # Revert
            cur_edges.discard(e_in)
            cur_edges.add(e_out)
            
            adj[u1][v1] = 1
            adj[v1][u1] = 1
            deg[u1] += 1
            deg[v1] += 1
            
            adj[u2][v2] = 0
            adj[v2][u2] = 0
            deg[u2] -= 1
            deg[v2] -= 1
            
            cur_max, cur_sq = _compute_scores(n, adj, deg) # Recompute to be safe, or keep old values
            # Actually, we know the old values were cur_max, cur_sq before the swap
            # But we overwrote cur_max/cur_sq? No, we stored them in old_cost calculation implicitly?
            # No, cur_max/cur_sq are still the old values because we only updated them if accepted.
            # Wait, in the accepted branch, we set cur_max = new_max.
            # In the rejected branch, we revert the graph. The scores should be back to the pre-swap values.
            # Since we didn't update cur_max/cur_sq in the rejected branch, they are still the pre-swap values.
            # So no need to recompute.

    # Return best found
    edges_list = [[u, v] for u, v in best_edges if u < v]
    return {'n': 16, 'edges': edges_list}
