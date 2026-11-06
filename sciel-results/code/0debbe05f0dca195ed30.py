import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def _creates_c4(adj, u, v):
        """
        Check if adding edge (u, v) creates a C4.
        adj is a list of sets.
        A C4 is created if there exists a path u-x-y-v where x in N(u), y in N(v), and (x,y) is an edge.
        """
        nu = adj[u]
        nv = adj[v]
        # Check if any neighbor of u is adjacent to any neighbor of v.
        # To optimize, iterate over the smaller neighbor set.
        if len(nu) > len(nv):
            nu, nv = nv, nu
            
        for x in nu:
            if adj[x] & nv:
                return True
        return False

    def _try_construct(n, min_deg, time_limit):
        """
        Attempt to construct a C4-free graph on n vertices with min degree >= min_deg.
        Returns a dict {'n': n, 'edges': [...]} on success, None on failure.
        """
        start_time_local = time.time()
        
        # Multiple restarts
        # We want to be fast. Each attempt should be short if it fails.
        max_attempts = 5000
        
        for attempt in range(max_attempts):
            if time.time() - start_time_local > time_limit:
                break
                
            adj = [set() for _ in range(n)]
            
            # Strategy: Sparse Start + Add
            # Start with empty graph (or very sparse) and add edges.
            # This avoids the "dense start" local minima where C4s are hard to break.
            
            success = False
            # We need to ensure min degree.
            # Keep track of degrees implicitly via len(adj[i]).
            
            # Max steps to prevent infinite loops in a single attempt
            # n=46, edges ~ 46*7/2 = 161. 
            # We might need a few extra steps if we get stuck and restart within? No, we restart the whole graph.
            # So we just need enough steps to fill the graph.
            # 500 steps should be plenty for 161 edges.
            max_steps = 1000 
            
            for step in range(max_steps):
                if time.time() - start_time_local > time_limit:
                    break
                    
                # Check if done
                min_deg_curr = min(len(adj[i]) for i in range(n))
                if min_deg_curr >= min_deg:
                    success = True
                    break
                    
                # Find vertex with min degree
                # Optimization: maintain a list of vertices sorted by degree? 
                # For n=46, finding min is O(n). 1000 steps * 46 = 46k ops. Trivial.
                u = min(range(n), key=lambda i: len(adj[i]))
                
                # Find valid neighbors for u
                # We want to pick a v such that adding (u,v) doesn't create C4.
                # To be greedy, maybe pick the v that is also low degree? 
                # Or just random. Random is often better for diversity.
                # Let's shuffle candidates.
                candidates = [v for v in range(n) if v != u and v not in adj[u]]
                
                if not candidates:
                    # u is connected to everyone. If min_deg_curr < min_deg, this is a problem.
                    # But if u is connected to everyone, deg(u) = n-1.
                    # If min_deg <= n-1, this shouldn't happen unless n is small.
                    # For n=46, min_deg=7. u can't be stuck with max degree < 7 easily?
                    # If candidates is empty, u is connected to all.
                    # So deg(u) = 45. This is >= 7.
                    # So if u is the min degree vertex and it has no candidates, 
                    # it means all vertices have no candidates? 
                    # No, u is the min. If u has no candidates, deg(u)=45.
                    # Then min_deg_curr = 45. Which is >= 7.
                    # So we would have broken out earlier.
                    break
                    
                random.shuffle(candidates)
                
                added = False
                for v in candidates:
                    if not _creates_c4(adj, u, v):
                        adj[u].add(v)
                        adj[v].add(u)
                        added = True
                        break
                        
                if not added:
                    # Stuck. u cannot connect to anyone without creating C4.
                    # This implies u is in a "saturated" state relative to C4 constraints.
                    # Restart is the standard remedy for this greedy approach.
                    break
                    
            if success:
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
                
        return None

    # Target n=46, min_deg=7
    # Time budget: 100 seconds
    res = _try_construct(46, 7, 100.0)
    if res:
        return res
        
    # Fallback: n=45, min_deg=6
    remaining = deadline - time.time()
    if remaining > 10.0:
        res = _try_construct(45, 6, remaining - 10.0)
        if res:
            return res

    # Fallback: n=44, min_deg=5
    remaining = deadline - time.time()
    if remaining > 5.0:
        res = _try_construct(44, 5, remaining - 5.0)
        if res:
            return res

    # Absolute fallback: empty graph on 1 vertex (valid, n=1)
    # Or maybe a small valid graph?
    # The task says maximize n. If we fail, we should return the best we have.
    # But we didn't store intermediate results.
    # Let's return a trivial valid graph.
    return {'n': 1, 'edges': []}

# Keep the function name exactly as requested
# The seed program had:
# def construct():
#     ...
#     return best_result
#
# My construct() matches this.

if __name__ == "__main__":
    # Simple test
    res = construct()
    print(res['n'])
    print(len(res['edges']))
