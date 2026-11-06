import time
import random
import numpy as np

def construct():
    n = 31
    target_edges = 91
    start_time = time.time()
    deadline = start_time + 145.0

    # 1. Construct Polarity Graph of PG(2,5) - This is the unique (31,90) C4-free graph.
    q = 5
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    key = (1, y * pow(x, q-2, q) % q, z * pow(x, q-2, q) % q)
                elif y != 0:
                    key = (0, 1, z * pow(y, q-2, q) % q)
                else:
                    key = (0, 0, 1)
                if key not in seen:
                    seen.add(key)
                    points.append(key)

    assert len(points) == 31, f"Expected 31 points, got {len(points)}"

    def is_on_polar(p, qpt):
        x, y, z = p
        X, Y, Z = qpt
        return (x * Y + y * X + z * Z) % q == 0

    # Build base graph (90 edges)
    base_edges_set = set()
    adj_base = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if is_on_polar(points[i], points[j]):
                base_edges_set.add((i, j))
                adj_base[i].add(j)
                adj_base[j].add(i)
    
    assert len(base_edges_set) == 90, f"Expected 90 edges, got {len(base_edges_set)}"
    
    # 2. Identify the "best" single edge to add to the base graph.
    # We want to minimize the number of C4s created.
    # A C4 is formed by adding (u,v) if there are 2 common neighbors.
    # The number of new C4s is C(k, 2) where k is the number of common neighbors in the base graph.
    # We want to minimize C(k, 2), which means minimizing k.
    
    best_add = None
    min_common = float('inf')
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for i, j in all_pairs:
        if (i, j) not in base_edges_set:
            common = len(adj_base[i] & adj_base[j])
            if common < min_common:
                min_common = common
                best_add = (i, j)
                
    # The base graph has 0 C4s.
    # Adding 1 edge creates C(min_common, 2) C4s.
    # In PG(2,5) polarity graph, non-adjacent points can have 0, 1, or 2 common neighbors?
    # Actually, for q=5, the max degree is 6.
    # Let's just pick the one with min_common.
    
    initial_edges = base_edges_set | {best_add}
    initial_adj = [set(s) for s in adj_base]
    u0, v0 = best_add
    initial_adj[u0].add(v0)
    initial_adj[v0].add(u0)
    
    def binom2(x):
        return x * (x - 1) // 2

    def count_c4(edges_set, adj_list):
        total = 0
        for u, v in edges_set:
            common = len(adj_list[u] & adj_list[v])
            total += binom2(common)
        return total // 4

    current_edges = set(initial_edges)
    current_adj = [set(s) for s in initial_adj]
    current_c4 = count_c4(current_edges, current_adj)
    
    best_c4 = current_c4
    best_edges = set(current_edges)
    best_adj = [set(s) for s in current_adj]
    
    # 3. Local Search: 1-swap (remove one edge, add another)
    # We can remove the "expensive" edge (best_add) and try to find a cheaper configuration?
    # Or remove any edge and add any other.
    # Since we must have 91 edges, and the base is 90 C4-free, any 91-edge graph must have at least 1 C4?
    # Wait, ex(31; C4) = 90. So any 91-edge graph has at least 1 C4.
    # The minimum is likely small.
    
    # Strategy:
    # Start with the best 91-edge graph found by adding 1 edge to the polarity graph.
    # Then perform swaps: remove (u1, v1) and add (u2, v2).
    # We want to minimize C4 count.
    
    # To make search faster, we can precompute common neighbors for all pairs?
    # Or just use sets. With n=31, sets are fast enough.
    
    random.seed(42)
    
    # Precompute list of all non-edges for faster random selection?
    # Or just pick random pairs and check if they are edges.
    
    # We will do a simulated annealing or simple hill climbing with restarts.
    # Given the small search space of "deviations" from the polarity graph,
    # maybe we can enumerate all single-edge additions, pick the best, then swap.
    
    # Let's try to improve the initial best_add choice by looking at 2-edge deviations?
    # No, 1-swap is enough if we start from the best 1-addition.
    
    # Actually, let's try all possible single additions to the base graph, and keep the best one.
    # Then do local search on that.
    
    # Re-evaluate: Is it possible that removing a base edge and adding two non-base edges yields fewer C4s?
    # Yes. So we need to search the space of 91-edge graphs.
    # The polarity graph is a strong attractor.
    
    # Let's use a more aggressive local search.
    
    # We'll maintain a list of candidate edges to add (non-edges) and edges to remove.
    
    # For speed, let's use numpy for adjacency matrix?
    # n=31 is small. Sets are fine.
    
    # Let's try to optimize the count_c4 calculation.
    # We can track the C4 count incrementally.
    
    def get_c4_delta(remove_u, remove_v, add_u, add_v, adj, edges_set):
        # Returns (delta_c4, new_c4)
        # Nodes involved: remove_u, remove_v, add_u, add_v
        # Edges affected:
        # 1. (remove_u, remove_v) is removed.
        # 2. (add_u, add_v) is added.
        # 3. All edges incident to remove_u, remove_v, add_u, add_v.
        
        involved = {remove_u, remove_v, add_u, add_v}
        
        # Find all edges in current graph that are incident to involved nodes
        affected_edges = set()
        for u in involved:
            for v in adj[u]:
                if u < v:
                    affected_edges.add((u, v))
        
        # Calculate old contribution of affected edges
        old_contrib = 0
        for u, v in affected_edges:
            if (u, v) == (remove_u, remove_v) or (u, v) == (remove_v, remove_u):
                continue # This edge is being removed, its contribution disappears
            common = len(adj[u] & adj[v])
            old_contrib += binom2(common)
            
        # Note: (remove_u, remove_v) was in affected_edges, but we skip it in old_contrib.
        # Wait, old_contrib should be the sum of contributions of ALL affected edges that REMAIN.
        # The edge (remove_u, remove_v) is removed, so its contribution is gone.
        # The other affected edges remain, but their common neighbor counts might change.
        
        # So we need to calculate:
        # 1. Sum of binom2(common) for all affected edges that are NOT (remove_u, remove_v) in the OLD graph.
        # 2. Sum of binom2(common) for all affected edges in the NEW graph.
        #    Affected edges in NEW graph:
        #    - All old affected edges except (remove_u, remove_v)
        #    - The new edge (add_u, add_v)
        
        # Let's refine:
        # Old Affected Edges Set A_old = { (u,v) in E | u in involved or v in involved }
        # New Affected Edges Set A_new = (A_old \ {(remove_u, remove_v)}) U {(add_u, add_v)}
        
        # We need to compute:
        # Delta = Sum_{e in A_new} binom2(common_new(e)) - Sum_{e in A_old} binom2(common_old(e))
        
        # Note that for e in A_old \ {(remove_u, remove_v)}, common_new(e) might differ from common_old(e).
        
        # Let's compute common_new for each edge in A_new.
        
        # Helper to get new adjacency for a node x
        def get_new_adj(x):
            n_x = set(adj[x])
            if x == remove_u:
                n_x.discard(remove_v)
            elif x == remove_v:
                n_x.discard(remove_u)
            if x == add_u:
                n_x.add(add_v)
            elif x == add_v:
                n_x.add(add_u)
            return n_x
        
        # Precompute new adj for involved nodes to avoid repeated set copying
        new_adj_cache = {}
        for x in involved:
            new_adj_cache[x] = get_new_adj(x)
            
        # Calculate old sum
        old_sum = 0
        for u, v in affected_edges:
            if (u, v) == (remove_u, remove_v) or (u, v) == (remove_v, remove_u):
                continue
            common = len(adj[u] & adj[v])
            old_sum += binom2(common)
            
        # Wait, A_old includes (remove_u, remove_v). We subtracted it.
        # But A_old also includes other edges incident to remove_u/remove_v.
        # Their common counts in the OLD graph are based on OLD adj.
        
        # Calculate new sum
        new_sum = 0
        # Iterate over A_new
        # A_new = (A_old \ {(remove_u, remove_v)}) U {(add_u, add_v)}
        
        # We can iterate over A_old, skip the removed edge, and add the new edge.
        
        # But we need to be careful: if (add_u, add_v) was already in A_old?
        # (add_u, add_v) is a non-edge in the current graph (we only add non-edges).
        # So (add_u, add_v) is NOT in A_old.
        # Unless add_u or add_v is involved and the edge exists? No, it's a non-edge.
        
        # So A_new is disjoint from (remove_u, remove_v) and contains (add_u, add_v).
        
        for u, v in affected_edges:
            if (u, v) == (remove_u, remove_v) or (u, v) == (remove_v, remove_u):
                continue
            # This edge is in A_new
            # Get common neighbors in NEW graph
            # n_u_new = new_adj_cache[u]
            # n_v_new = new_adj_cache[v]
            # But wait, u and v might not be in involved?
            # No, affected_edges are defined by having an endpoint in involved.
            # So u or v is in involved.
            # If u is in involved, new_adj_cache has u.
            # If v is in involved, new_adj_cache has v.
            # If u is NOT in involved, then v IS in involved.
            # We need the full new adjacency for u and v.
            # If u is not in involved, its adjacency doesn't change.
            # So we can use adj[u] if u not in new_adj_cache, else new_adj_cache[u].
            
            n_u = new_adj_cache.get(u, adj[u])
            n_v = new_adj_cache.get(v, adj[v])
            common = len(n_u & n_v)
            new_sum += binom2(common)
            
        # Add the new edge (add_u, add_v)
        u_new, v_new = add_u, add_v
        n_u_new = new_adj_cache[u_new]
        n_v_new = new_adj_cache[v_new]
        common_new_edge = len(n_u_new & n_v_new)
        new_sum += binom2(common_new_edge)
        
        delta = new_sum - old_sum
        return delta // 4 # Since C4 = sum / 4, and sum is divisible by 4?
        # Actually, the total sum is always divisible by 4.
        # The delta might not be divisible by 4?
        # Let's check.
        # The change in total sum is Delta_Sum.
        # New C4 = (Old_Sum - old_sum + new_sum) / 4
        # Delta C4 = (new_sum - old_sum) / 4
        # Is new_sum - old_sum always divisible by 4?
        # Not necessarily. But the total C4 count is an integer.
        # So we should calculate the new total C4 and compare, or ensure delta is handled correctly.
        # Actually, it's safer to return the new total C4 or the exact integer delta.
        # Since we are minimizing, we can just compare new_c4 < best_c4.
        # Let's return the new_c4 directly to avoid division issues.
        
    # Let's refactor get_c4_delta to return new_c4
    def calc_new_c4(remove_u, remove_v, add_u, add_v, adj, edges_set, current_c4):
        involved = {remove_u, remove_v, add_u, add_v}
        
        affected_edges = set()
        for u in involved:
            for v in adj[u]:
                if u < v:
                    affected_edges.add((u, v))
        
        # Old sum of affected edges (excluding the one being removed)
        old_sum = 0
        for u, v in affected_edges:
            if (u, v) == (remove_u, remove_v) or (u, v) == (remove_v, remove_u):
                continue
            common = len(adj[u] & adj[v])
            old_sum += binom2(common)
            
        # New sum
        new_adj_cache = {}
        for x in involved:
            n_x = set(adj[x])
            if x == remove_u:
                n_x.discard(remove_v)
            elif x == remove_v:
                n_x.discard(remove_u)
            if x == add_u:
                n_x.add(add_v)
            elif x == add_v:
                n_x.add(add_u)
            new_adj_cache[x] = n_x
            
        new_sum = 0
        for u, v in affected_edges:
            if (u, v) == (remove_u, remove_v) or (u, v) == (remove_v, remove_u):
                continue
            n_u = new_adj_cache.get(u, adj[u])
            n_v = new_adj_cache.get(v, adj[v])
            common = len(n_u & n_v)
            new_sum += binom2(common)
            
        # Add new edge contribution
        u_new, v_new = add_u, add_v
        n_u_new = new_adj_cache[u_new]
        n_v_new = new_adj_cache[v_new]
        common_new_edge = len(n_u_new & n_v_new)
        new_sum += binom2(common_new_edge)
        
        # Total old sum = current_total_sum
        # We don't have current_total_sum stored, only current_c4.
        # current_total_sum = 4 * current_c4
        # new_total_sum = 4 * current_c4 - old_sum + new_sum
        # new_c4 = new_total_sum // 4
        
        new_total_sum = 4 * current_c4 - old_sum + new_sum
        return new_total_sum // 4

    # Local Search Loop
    iterations = 0
    max_iterations = 500000
    
    while time.time() < deadline - 5 and iterations < max_iterations:
        iterations += 1
        
        # Pick a random edge to remove
        u1, v1 = random.choice(list(current_edges))
        
        # Pick a random non-edge to add
        # To speed up, we can pick random pairs and check if they are edges.
        # Or maintain a list of non-edges.
        # With 91 edges, there are 465 - 91 = 374 non-edges.
        # Picking random pairs and checking is fast enough.
        
        improved = False
        for _ in range(10): # Try a few candidates per removal
            u2, v2 = random.choice(all_pairs)
            if (u2, v2) in current_edges or (v2, u2) in current_edges:
                continue
            if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                continue
            
            new_c4 = calc_new_c4(u1, v1, u2, v2, current_adj, current_edges, current_c4)
            
            if new_c4 < best_c4:
                # Accept
                # Update state
                current_adj[u1].discard(v1)
                current_adj[v1].discard(u1)
                current_adj[u2].add(v2)
                current_adj[v2].add(u2)
                
                current_edges.discard((u1, v1))
                current_edges.add((u2, v2))
                
                current_c4 = new_c4
                best_c4 = new_c4
                best_edges = set(current_edges)
                best_adj = [set(s) for s in current_adj]
                improved = True
                break
        
        if not improved and iterations % 1000 == 0:
            # Optional: Restart or perturb?
            # For now, just continue.
            pass

    return list(best_edges)
