"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_3_colorable_mrv(adj_masks, n):
    """Fast 3-colorability check using iterative DP with bitmasks and memoization."""
    full_mask = (1 << n) - 1
    
    # Memoization: key = (uncolored_mask, cc0, cc1)
    # cc2 is implicitly full_mask ^ uncolored_mask ^ cc0 ^ cc1
    memo = {}
    
    def dp(uncolored_mask, cc0, cc1):
        if uncolored_mask == 0:
            return True
        
        key = (uncolored_mask, cc0, cc1)
        if key in memo:
            return memo[key]
        
        cc2 = full_mask ^ uncolored_mask ^ cc0 ^ cc1
        
        # Select vertex with fewest available colors (MRV heuristic)
        best_v = -1
        best_avail = 4
        m = uncolored_mask
        while m:
            v = (m & (-m)).bit_length() - 1
            m &= m - 1
            nb = adj_masks[v]
            avail = 3
            if nb & cc0:
                avail -= 1
            if nb & cc1:
                avail -= 1
            if nb & cc2:
                avail -= 1
            if avail == 0:
                memo[key] = False
                return False
            if avail < best_avail:
                best_avail = avail
                best_v = v
                if avail == 1:
                    break
        
        v = best_v
        nb = adj_masks[v]
        
        # Try each available color
        for c in range(3):
            if c == 0:
                if nb & cc0:
                    continue
                new_cc0 = cc0 | (1 << v)
                new_cc1 = cc1
            elif c == 1:
                if nb & cc1:
                    continue
                new_cc0 = cc0
                new_cc1 = cc1 | (1 << v)
            else:
                if nb & cc2:
                    continue
                new_cc0 = cc0
                new_cc1 = cc1
            
            if dp(uncolored_mask & ~(1 << v), new_cc0, new_cc1):
                memo[key] = True
                return True
        
        memo[key] = False
        return False
    
    return dp(full_mask, 0, 0)


def _triangle_free(adj_masks, n):
    """Check if graph is triangle-free using bitmasks."""
    for u in range(n):
        for v in range(u + 1, n):
            if (adj_masks[u] >> v) & 1:
                common = adj_masks[u] & adj_masks[v]
                common &= ~(1 << u)
                common &= ~(1 << v)
                if common:
                    return False
    return True


def _groetzsch_edges():
    """Return edges of the Grötzsch graph (Mycielski of C5) on vertices 0..10."""
    # Cycle C5: 0-1-2-3-4-0
    # Mycielski: add vertices 5-9 (copies of 0-4) and vertex 10 (universal to 5-9)
    edges = [[0,1],[1,2],[2,3],[3,4],[4,0]]
    # Copies: vertex 5+i is copy of vertex i
    # 5+0=5 connected to neighbors of 0: 1,4
    # 5+1=6 connected to neighbors of 1: 0,2
    # 5+2=7 connected to neighbors of 2: 1,3
    # 5+3=8 connected to neighbors of 3: 2,4
    # 5+4=9 connected to neighbors of 4: 3,0
    edges += [[5,1],[5,4],[6,0],[6,2],[7,1],[7,3],[8,2],[8,4],[9,3],[9,0]]
    # Vertex 10 connected to all copies 5-9
    edges += [[10,5],[10,6],[10,7],[10,8],[10,9]]
    return edges


def _build_initial_graph(n=16):
    """Build a good initial 4-chromatic triangle-free graph on n vertices."""
    edges = _groetzsch_edges()
    adj = _edges_to_adj(edges, n)
    
    # Vertices 11-15 are isolated. Try to connect them to independent sets.
    # For each isolated vertex, find the largest independent set it can connect to
    # without creating triangles or making the graph 3-colorable.
    
    isolated = [11, 12, 13, 14, 15]
    
    for v in isolated:
        # Find all vertices that v can potentially connect to
        # v can connect to u if:
        # 1. u is not already adjacent to v
        # 2. No common neighbor of v and u (triangle-free check)
        # 3. The resulting graph is still not 3-colorable
        
        # Since v is isolated, it has no neighbors, so condition 2 is trivially satisfied
        # for any u. But we need to ensure adding (v,u) keeps the graph non-3-colorable.
        
        # Strategy: try to add multiple edges from v to an independent set
        # First, find independent sets in the existing graph
        
        # Simple approach: greedily add edges from v to vertices
        # Try to add as many edges as possible while maintaining 4-chromaticity
        
        added = 0
        # Sort potential neighbors by degree (highest first) for better pruning
        candidates = sorted(range(11), key=lambda x: -bin(adj[x]).count('1'))
        
        for u in candidates:
            if (adj[v] >> u) & 1:
                continue
            # Check triangle-free: v and u should have no common neighbor
            common = adj[v] & adj[u]
            common &= ~(1 << v)
            common &= ~(1 << u)
            if common:
                continue
            
            new_adj = adj[:]
            new_adj[v] |= (1 << u)
            new_adj[u] |= (1 << v)
            
            if not _is_3_colorable_mrv(new_adj, n):
                adj = new_adj
                added += 1
        
        # After adding some edges, try again to add more (the independent set might have changed)
        for _ in range(2):
            more_added = 0
            for u in range(11):
                if (adj[v] >> u) & 1:
                    continue
                common = adj[v] & adj[u]
                common &= ~(1 << v)
                common &= ~(1 << u)
                if common:
                    continue
                new_adj = adj[:]
                new_adj[v] |= (1 << u)
                new_adj[u] |= (1 << v)
                if not _is_3_colorable_mrv(new_adj, n):
                    adj = new_adj
                    more_added += 1
            if more_added == 0:
                break
    
    return _adj_to_edges(adj, n), adj


def _edges_to_adj(edges, n):
    """Convert edge list to adjacency bitmasks."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj


def _adj_to_edges(adj, n):
    """Convert adjacency bitmasks to edge list."""
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                edges.append([u, v])
    return edges


def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 16
    
    # Build a good initial graph
    best_edges, adj = _build_initial_graph(n)
    best_count = len(best_edges)
    
    def try_add_edge(adj, u, v, n):
        """Try to add edge (u,v). Return new adj if valid, else None."""
        common = adj[u] & adj[v]
        common &= ~(1 << u)
        common &= ~(1 << v)
        if common:
            return None
        
        new_adj = adj[:]
        new_adj[u] |= (1 << v)
        new_adj[v] |= (1 << u)
        
        if _is_3_colorable_mrv(new_adj, n):
            return None
        
        return new_adj
    
    def get_non_edges_ordered(adj, n):
        non_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    continue
                deg_u = bin(adj[u]).count('1')
                deg_v = bin(adj[v]).count('1')
                # Prefer edges between high-degree vertices (more likely to maintain non-3-colorability)
                non_edges.append((-(deg_u + deg_v), u, v))
        non_edges.sort()
        return [(u, v) for _, u, v in non_edges]
    
    # Aggressive systematic greedy: repeatedly scan all non-edges, add if possible
    # Do multiple passes to ensure we've found a local optimum
    for _ in range(3):
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            non_edges = get_non_edges_ordered(adj, n)
            for u, v in non_edges:
                if time.time() > deadline - 10:
                    break
                new_adj = try_add_edge(adj, u, v, n)
                if new_adj is not None:
                    adj = new_adj
                    changed = True
                    best_edges = _adj_to_edges(adj, n)
                    best_count = len(best_edges)
    
    # Local search with simulated annealing
    rng = random.Random(42)
    current_adj = adj[:]
    current_edges_count = best_count
    
    # Adaptive iteration count based on remaining time
    remaining = deadline - 10 - time.time()
    max_iterations = int(remaining * 150)  # ~150 iterations per second with faster color check
    max_iterations = max(1000, min(max_iterations, 10000))
    
    for iteration in range(max_iterations):
        if time.time() > deadline - 10:
            break
        
        # Pick a move type - bias towards edge addition
        move_type = rng.random()
        
        if move_type < 0.6:
            # Try adding a random non-edge
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if (current_adj[u] >> v) & 1:
                continue
            new_adj = try_add_edge(current_adj, u, v, n)
            if new_adj is not None:
                current_adj = new_adj
                current_edges_count += 1
                if current_edges_count > best_count:
                    best_count = current_edges_count
                    best_edges = _adj_to_edges(current_adj, n)
                    adj = current_adj[:]
        elif move_type < 0.85:
            # Remove one edge, then greedily try to add 1-3 edges
            edges = _adj_to_edges(current_adj, n)
            if not edges:
                continue
            ru, rv = rng.choice(edges)
            
            new_adj = current_adj[:]
            new_adj[ru] &= ~(1 << rv)
            new_adj[rv] &= ~(1 << ru)
            
            # Try to add edges back, prioritizing high-degree pairs
            added = 0
            for _ in range(40):
                u = rng.randint(0, n - 1)
                v = rng.randint(0, n - 1)
                if u == v or u > v:
                    continue
                if (new_adj[u] >> v) & 1:
                    continue
                common = new_adj[u] & new_adj[v]
                common &= ~(1 << u)
                common &= ~(1 << v)
                if common:
                    continue
                test_adj = new_adj[:]
                test_adj[u] |= (1 << v)
                test_adj[v] |= (1 << u)
                if not _is_3_colorable_mrv(test_adj, n):
                    new_adj = test_adj
                    added += 1
                    if added >= 3:
                        break
            
            new_count = len(_adj_to_edges(new_adj, n))
            # Accept if better or with some probability (annealing)
            if new_count >= current_edges_count:
                current_adj = new_adj
                current_edges_count = new_count
                if current_edges_count > best_count:
                    best_count = current_edges_count
                    best_edges = _adj_to_edges(current_adj, n)
                    adj = current_adj[:]
            else:
                # Accept worse moves with decreasing probability
                prob = 0.15 * (1.0 - iteration / max_iterations)
                if rng.random() < prob:
                    current_adj = new_adj
                    current_edges_count = new_count
        else:
            # 2-opt style: remove 2 edges, add 2-4 different edges
            edges = _adj_to_edges(current_adj, n)
            if len(edges) < 2:
                continue
            e1, e2 = rng.sample(edges, 2)
            
            new_adj = current_adj[:]
            new_adj[e1[0]] &= ~(1 << e1[1])
            new_adj[e1[1]] &= ~(1 << e1[0])
            new_adj[e2[0]] &= ~(1 << e2[1])
            new_adj[e2[1]] &= ~(1 << e2[0])
            
            added = 0
            for _ in range(50):
                u = rng.randint(0, n - 1)
                v = rng.randint(0, n - 1)
                if u == v or u > v:
                    continue
                if (new_adj[u] >> v) & 1:
                    continue
                common = new_adj[u] & new_adj[v]
                common &= ~(1 << u)
                common &= ~(1 << v)
                if common:
                    continue
                test_adj = new_adj[:]
                test_adj[u] |= (1 << v)
                test_adj[v] |= (1 << u)
                if not _is_3_colorable_mrv(test_adj, n):
                    new_adj = test_adj
                    added += 1
                    if added >= 4:
                        break
            
            new_count = len(_adj_to_edges(new_adj, n))
            if new_count >= current_edges_count:
                current_adj = new_adj
                current_edges_count = new_count
                if current_edges_count > best_count:
                    best_count = current_edges_count
                    best_edges = _adj_to_edges(current_adj, n)
                    adj = current_adj[:]
            else:
                prob = 0.08 * (1.0 - iteration / max_iterations)
                if rng.random() < prob:
                    current_adj = new_adj
                    current_edges_count = new_count
    
    # Final polish: systematic greedy pass on the best found
    if time.time() < deadline - 1:
        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            non_edges = get_non_edges_ordered(adj, n)
            for u, v in non_edges:
                if time.time() > deadline - 1:
                    break
                new_adj = try_add_edge(adj, u, v, n)
                if new_adj is not None:
                    adj = new_adj
                    changed = True
                    best_edges = _adj_to_edges(adj, n)
                    best_count = len(best_edges)
    
    return best_edges

# EVOLVE-BLOCK-END
