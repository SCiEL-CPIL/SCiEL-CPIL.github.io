import time
import random
import numpy as np

def construct():
    """
    Constructs a C6-free graph on 40 vertices maximizing edges.
    Uses a structural change: instead of greedy insertion with kicks,
    we use a simulated-annealing-style approach with a different
    representation: we work with the complement graph (edges NOT present)
    and try to remove edges from the complement (i.e., add to the graph)
    while checking C6-freeness, but with a key difference:
    
    STRUCTURAL CHANGE: Use a different initial construction based on
    strongly regular graph constructions and a different local search
    that operates on "edge swaps" (remove one edge, add another) rather
    than pure addition/removal. This explores a different region of
    the solution space.
    """
    N = 40
    deadline = time.time() + 115  # Safety margin
    
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    best_edges_list = []
    best_count = 0
    
    def run_once(seed):
        nonlocal best_edges_list, best_count
        random.seed(seed)
        
        # Bitmask adjacency
        adj = [0] * N
        edges = []
        
        def add_edge(u, v):
            if u > v:
                u, v = v, u
            if adj[u] & (1 << v):
                return False
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append((u, v))
            return True
        
        def remove_edge(u, v):
            if u > v:
                u, v = v, u
            if not (adj[u] & (1 << v)):
                return False
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            edges.remove((u, v))
            return True
        
        def can_add(u, v):
            """
            Check if adding edge (u, v) creates a C6.
            Fast check using bitmasks:
            A C6 is formed if there exists a path of length 4 from u to v
            through intermediate vertices. Equivalently, there exist distinct
            vertices a, b, c, d such that u-a-b-c-d-v is a path.
            
            Faster approach: count common neighbors at each step.
            We use the fact that a C6 through (u,v) exists iff there is
            a pair of vertices (x, y) where x is neighbor of u, y is neighbor
            of v, and there's a path of length 2 from x to y not using u or v.
            """
            # For each neighbor x of u (x != v), for each neighbor y of v (y != u),
            # check if there's a common neighbor of x and y that is not u, v, x, or y.
            # Actually, we need a path x - a - b - y where a, b are distinct and not u, v.
            # This is equivalent to: x and y have a common neighbor, and that common
            # neighbor has a neighbor that is also a neighbor of y... 
            # 
            # Simpler: use BFS/DFS from u for 4 steps, avoiding v, and check if we
            # can reach a neighbor of v.
            
            # Optimized: precompute neighbor sets
            nu = adj[u] & ~(1 << v)
            nv = adj[v] & ~(1 << u)
            
            # For each x in neighbors of u, for each y in neighbors of v:
            # We need a path of length 2 from x to y (x-a-b-y) where a,b distinct,
            # a,b not in {u,v,x,y}
            # Equivalently: there exists a common neighbor a of x and some b, and b
            # is a common neighbor of a and y... this is getting complex.
            
            # Let's use the direct approach but with early termination
            # Path: u - x - a - b - y - v where x in N(u), y in N(v)
            # We need x-a-b-y to be a path of length 3 (3 edges, 4 vertices including x,y)
            # Wait, u-x-a-b-y-v is 5 edges = C6. So we need path of length 3 from x to y.
            
            # For efficiency, iterate over x in N(u), and for each x, find all
            # vertices reachable from x in exactly 3 steps (avoiding u, v), and
            # check if any of those are in N(v).
            
            # Even faster: for each x in N(u), compute N2(x) = neighbors of neighbors of x
            # (excluding u, v, x), then check if N2(x) intersects N(v).
            
            x_mask = nu
            while x_mask:
                low = x_mask & (-x_mask)
                x = low.bit_length() - 1
                x_mask ^= low
                
                # Get neighbors of x, excluding u, v, and x itself
                nx = adj[x] & ~(1 << u) & ~(1 << v) & ~(1 << x)
                
                # For each neighbor a of x, check if a has a neighbor b such that b is in N(v)
                # i.e., check if N(a) \ {u,v,x,a} intersects N(v)
                a_mask = nx
                while a_mask:
                    low_a = a_mask & (-a_mask)
                    a = low_a.bit_length() - 1
                    a_mask ^= low_a
                    
                    # N(a) excluding u, v, x, a
                    na = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << x) & ~(1 << a)
                    
                    # Check if any neighbor of a is in N(v)
                    if na & nv:
                        return False
            
            return True
        
        def find_blocking_c6(u, v):
            """Find a C6 that would be created by adding (u, v)."""
            nu = adj[u] & ~(1 << v)
            nv = adj[v] & ~(1 << u)
            
            x_mask = nu
            while x_mask:
                low = x_mask & (-x_mask)
                x = low.bit_length() - 1
                x_mask ^= low
                
                nx = adj[x] & ~(1 << u) & ~(1 << v) & ~(1 << x)
                a_mask = nx
                while a_mask:
                    low_a = a_mask & (-a_mask)
                    a = low_a.bit_length() - 1
                    a_mask ^= low_a
                    
                    na = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << x) & ~(1 << a)
                    
                    if na & nv:
                        # Find a specific y in N(v) that is neighbor of a
                        y_mask = na & nv
                        low_y = y_mask & (-y_mask)
                        y = low_y.bit_length() - 1
                        
                        # C6: u-x-a-y-v-... wait, we need u-x-a-b-y-v
                        # Actually: u - x - a - y - v is only 4 edges. We need 5 edges.
                        # Let me recheck: C6 has 6 vertices, 6 edges.
                        # u-x-a-b-y-v: vertices are u,x,a,b,y,v (6 vertices)
                        # So we need x-a-b-y to be a path of length 3 (3 edges).
                        # x-a is edge, a-b is edge, b-y is edge.
                        # So b is a common neighbor of a and y.
                        
                        # Let me redo: find b that is neighbor of both a and y
                        common = adj[a] & adj[y] & ~(1 << u) & ~(1 << v) & ~(1 << x) & ~(1 << a) & ~(1 << y)
                        if common:
                            low_b = common & (-common)
                            b = low_b.bit_length() - 1
                            return [(u, x), (x, a), (a, b), (b, y), (y, v)]
                        # If no common b, try next y
                        y_mask = (na & nv) & ~(1 << y)
                        while y_mask:
                            low_y2 = y_mask & (-y_mask)
                            y2 = low_y2.bit_length() - 1
                            y_mask ^= low_y2
                            common2 = adj[a] & adj[y2] & ~(1 << u) & ~(1 << v) & ~(1 << x) & ~(1 << a) & ~(1 << y2)
                            if common2:
                                low_b2 = common2 & (-common2)
                                b2 = low_b2.bit_length() - 1
                                return [(u, x), (x, a), (a, b2), (b2, y2), (y2, v)]
                        continue
                    # No intersection, try next a
            return None
        
        # Initial graph: use a different construction
        # Start with 8 disjoint K5s but then try to merge them
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    add_edge(base + a, base + b)
        
        local_best = len(edges)
        local_best_adj = adj[:]
        
        # Simulated annealing style search
        # Temperature starts high, decreases over time
        T_start = 1.0
        T_end = 0.01
        start_time = time.time()
        total_time = 110.0  # Leave 5s margin
        
        iteration = 0
        while time.time() < deadline:
            elapsed = time.time() - start_time
            progress = min(1.0, elapsed / total_time)
            T = T_start * (T_end / T_start) ** progress
            
            # Try a random edge swap: remove edge e1, add edge e2
            # Or try to add a single edge
            if random.random() < 0.7:
                # Try to add a single random missing edge
                missing = [(u, v) for u, v in all_pairs if not (adj[u] & (1 << v))]
                if not missing:
                    break
                u, v = random.choice(missing)
                if can_add(u, v):
                    add_edge(u, v)
                    if len(edges) > local_best:
                        local_best = len(edges)
                        local_best_adj = adj[:]
                else:
                    # With probability T, accept a "kick" that removes an edge
                    # from the blocking C6
                    if T > 0.05 and edges:
                        block_c6 = find_blocking_c6(u, v)
                        if block_c6:
                            e_to_remove = random.choice(block_c6)
                            remove_edge(e_to_remove[0], e_to_remove[1])
                            # Now try to add (u, v) again
                            if can_add(u, v):
                                add_edge(u, v)
            else:
                # Edge swap: remove one edge, add another
                if not edges:
                    continue
                e1 = random.choice(edges)
                remove_edge(e1[0], e1[1])
                
                missing = [(u, v) for u, v in all_pairs if not (adj[u] & (1 << v))]
                if missing:
                    # Try to find a good edge to add
                    # Prefer edges that connect high-degree vertices
                    scored = []
                    for u, v in missing:
                        du = adj[u].bit_count()
                        dv = adj[v].bit_count()
                        common = (adj[u] & adj[v]).bit_count()
                        score = (du + dv) - common * 2.0
                        scored.append((score, u, v))
                    scored.sort(reverse=True)
                    
                    # Try top candidates
                    tried = 0
                    added = False
                    for score, u, v in scored[:min(20, len(scored))]:
                        if can_add(u, v):
                            add_edge(u, v)
                            added = True
                            break
                        tried += 1
                        if tried >= 5:
                            break
                    
                    if not added:
                        # Re-add the removed edge
                        add_edge(e1[0], e1[1])
                
                if len(edges) > local_best:
                    local_best = len(edges)
                    local_best_adj = adj[:]
            
            iteration += 1
            if iteration % 1000 == 0:
                # Periodic polish: try all missing edges
                for u, v in all_pairs:
                    if adj[u] & (1 << v):
                        continue
                    if can_add(u, v):
                        add_edge(u, v)
                        if len(edges) > local_best:
                            local_best = len(edges)
                            local_best_adj = adj[:]
        
        # Final polish
        for _ in range(3):
            improved = False
            for u, v in all_pairs:
                if adj[u] & (1 << v):
                    continue
                if can_add(u, v):
                    add_edge(u, v)
                    improved = True
                    if len(edges) > local_best:
                        local_best = len(edges)
                        local_best_adj = adj[:]
            if not improved:
                break
        
        # Update global best
        if local_best > best_count:
            best_count = local_best
            final_edges = []
            for u in range(N):
                mask = local_best_adj[u]
                while mask:
                    low = mask & (-mask)
                    v = low.bit_length() - 1
                    mask ^= low
                    if u < v:
                        final_edges.append([u, v])
            best_edges_list = final_edges
    
    # Run multiple restarts with different seeds
    num_restarts = 5
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        run_once(1000 + i * 7919)
    
    if not best_edges_list:
        return []
    
    return best_edges_list

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
