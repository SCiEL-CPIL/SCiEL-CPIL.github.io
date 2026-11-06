import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    def check_c4_safe(adj, n, u, v):
        """Check if adding edge (u,v) creates a C4."""
        nu = adj[u]
        nv = adj[v]
        # A C4 is u-x-y-v-u. x in N(u), y in N(v), x!=y, x~y.
        # x and y are neighbors of u and v respectively.
        # If x==y, it's a triangle u-x-v-u, not a C4.
        # We need to check if there exists x in N(u) and y in N(v) such that x!=y and x~y.
        
        # Optimize: iterate over smaller neighbor set
        if len(nu) > len(nv):
            nu, nv = nv, nu
            
        for x in nu:
            if x == v: continue # x is neighbor of u, if x==v, v is in N(u), but we are adding (u,v) so v not in N(u) yet. 
                                # Wait, adj[u] does not contain v yet.
            # x is a neighbor of u.
            # We need y in N(v) such that x~y.
            # y cannot be x (if x in N(v), then x~v and x~u, triangle).
            # y cannot be u (u not in N(v) yet).
            
            # Check if any neighbor of x is in N(v)
            # N(x) intersects N(v) ?
            # If yes, say z in N(x) and z in N(v).
            # Then u-x-z-v-u is a cycle.
            # Length: u-x (1), x-z (1), z-v (1), v-u (1, the new edge).
            # So u-x-z-v-u is a C4 if x!=z and x!=v and z!=u and x!=v.
            # x is in N(u). z is in N(v).
            # If x==z, then u-x-v-u is a triangle. Not a C4.
            # So we need x != z.
            
            # So: Does N(x) intersect (N(v) \ {x})?
            # Note: x might not be in N(v).
            
            for z in adj[x]:
                if z in nv:
                    if z != x:
                        return True
        return False

    def try_build(n, min_deg_req, time_limit, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        
        def add_edge(u, v):
            adj[u].add(v)
            adj[v].add(u)
            
        def remove_edge(u, v):
            adj[u].discard(v)
            adj[v].discard(u)
            
        def get_min_deg():
            return min(len(adj[i]) for i in range(n))
            
        # Initialization: Random greedy fill
        # Try to add random edges that don't create C4s
        vertices = list(range(n))
        rng.shuffle(vertices)
        
        # Phase 1: Greedy addition
        for u in vertices:
            candidates = [v for v in range(n) if v > u and v not in adj[u]]
            rng.shuffle(candidates)
            for v in candidates:
                if not check_c4_safe(adj, n, u, v):
                    add_edge(u, v)
                    
        # Phase 2: Local Search to maximize min degree
        # We want min_deg >= min_deg_req
        # Use a simple hill climbing with restarts if stuck
        t0 = time.time()
        stuck_counter = 0
        last_best = get_min_deg()
        
        while time.time() - t0 < time_limit:
            if get_min_deg() >= min_deg_req:
                return adj
                
            # Find vertex with min degree
            target = min(range(n), key=lambda i: len(adj[i]))
            
            # Try to add an edge to target
            candidates = [v for v in range(n) if v != target and v not in adj[target]]
            rng.shuffle(candidates)
            
            added = False
            for v in candidates:
                if not check_c4_safe(adj, n, target, v):
                    add_edge(target, v)
                    added = True
                    stuck_counter = 0
                    break
            
            if not added:
                stuck_counter += 1
                if stuck_counter > 50:
                    # Try to remove an edge to unblock
                    # Prefer edges not incident to target
                    edges_to_remove = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                if u != target and v != target:
                                    edges_to_remove.append((u,v))
                    if edges_to_remove:
                        u, v = rng.choice(edges_to_remove)
                        remove_edge(u, v)
                        stuck_counter = 0
                    else:
                        # Try removing an edge incident to target if no others
                        # Or restart
                        break
                        
        return None

    best_result = None
    
    # Try n from 47 down
    for n in range(47, 39, -1):
        if time.time() - start_time > deadline:
            break
            
        min_deg_req = max(1, n - 39)
        
        # Allocate time based on n
        # Higher n is harder, give it more time
        time_budget = 20.0
        if n <= 45:
            time_budget = 10.0
            
        attempts = 5
        success = False
        
        for i in range(attempts):
            if time.time() - start_time > deadline:
                break
            seed = 1000 + n * 100 + i
            # Adjust time limit per attempt
            t_limit = time_budget / attempts
            res = try_build(n, min_deg_req, t_limit, seed)
            if res is not None:
                # Verify
                # Check min degree
                if min(len(res[i]) for i in range(n)) >= min_deg_req:
                    # Check C4 free (should be guaranteed by construction, but verify for safety)
                    # For n=47, full C4 check is O(n^4) ~ 5M ops, acceptable once.
                    c4_found = False
                    for u in range(n):
                        for v in res[u]:
                            if u < v:
                                # Check common neighbors
                                common = res[u].intersection(res[v])
                                if len(common) >= 2:
                                    c4_found = True
                                    break
                        if c4_found: break
                    if not c4_found:
                        edges = []
                        for u in range(n):
                            for v in res[u]:
                                if u < v:
                                    edges.append([u, v])
                        best_result = {'n': n, 'edges': edges}
                        success = True
                        break
        if success:
            return best_result
            
    # Fallback
    if best_result is None:
        # Return a small valid graph
        # n=3, K3 is C4-free, min deg 2 >= 3-39 (1)
        return {'n': 3, 'edges': [[0,1],[1,2],[0,2]]}
        
    return best_result
