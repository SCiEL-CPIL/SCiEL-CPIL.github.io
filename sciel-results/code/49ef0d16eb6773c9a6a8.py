import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin for 120s limit
    
    # Seed result fallback
    seed_edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        seed_edges += [[a, b], [a, c], [b, c]]
    best_result = {'n': 30, 'edges': seed_edges}
    best_n = 30
    
    def check_c4_free(adj, n):
        # adj is list of sets
        for u in range(n):
            for v in range(u + 1, n):
                if len(adj[u] & adj[v]) > 1:
                    return False
        return True

    def has_independent_set_11(adj, n):
        # Backtracking to find independent set of size 11
        # Simple DFS with pruning
        if n < 11:
            return False
        
        # Order vertices by degree (heuristic)
        # For speed, just use standard backtracking
        def backtrack(start, size, current):
            if len(current) == 11:
                return True
            if start >= n:
                return False
            # Pruning: if remaining vertices + current < 11
            if (n - start) + len(current) < 11:
                return False
            
            # Option 1: Exclude vertex start
            if backtrack(start + 1, size, current):
                return True
            
            # Option 2: Include vertex start if independent
            is_independent = True
            for v in current:
                if v in adj[start]:
                    is_independent = False
                    break
            
            if is_independent:
                if backtrack(start + 1, size + 1, current + [start]):
                    return True
            
            return False
        
        return backtrack(0, 0, [])

    def try_construct(n):
        # Greedy random edge addition
        adj = [set() for _ in range(n)]
        edges = []
        
        # Generate all possible edges
        possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            # Check if adding (u, v) creates a C4
            # A C4 is created if u and v have a common neighbor w, and w is connected to u and v?
            # No, C4 is u-w-v-x-u.
            # Adding (u, v) creates a C4 if there exists w such that w is neighbor of u and v?
            # If w is neighbor of u and v, then u-w-v is a path of length 2.
            # If we add (u, v), we close the triangle u-v-w-u. That's a C3, not C4.
            # Wait.
            # C4 involves 4 vertices.
            # If we add edge (u, v), a C4 is formed if there exists a vertex w such that
            # w is connected to u and v? No, that forms a triangle.
            # A C4 is formed if there exists a vertex w such that
            # w is connected to u, and there is a path of length 2 from w to v via some x?
            # No.
            # Adding (u, v) creates a C4 if there exists a vertex w such that
            # w is a common neighbor of u and v? No, that's a triangle.
            
            # Let's re-evaluate.
            # A graph has a C4 if there are two vertices with 2 common neighbors.
            # If we add (u, v), do we create a pair with 2 common neighbors?
            # The new common neighbors of u and v are the vertices connected to both.
            # Before adding (u, v), let $C = N(u) \cap N(v)$.
            # If $|C| \ge 2$, then u and v already had 2 common neighbors.
            # But if u and v were not connected, they could have common neighbors.
            # If they had 2 common neighbors, the graph already had a C4 (u-c1-v-c2-u).
            # So we assume the graph was C4-free.
            # Thus $|N(u) \cap N(v)| \le 1$.
            
            # Does adding (u, v) create a NEW C4?
            # It creates a C4 if there is a path of length 3 between u and v in the OLD graph.
            # Path: u - w - x - v.
            # Then u-w-x-v-u is a C4.
            # So we must check if there is a path of length 3 between u and v.
            # This is equivalent to checking if $N(u) \cap N(N(v))$ contains any vertex?
            # Specifically, if there exists $w \in N(u)$ such that $w$ has a neighbor $x \in N(v)$ and $x \neq u$?
            # Since graph is simple, $x \neq u$ is implied if $u \notin N(v)$ (which is true before adding).
            
            # So condition: 
            # For all $w \in N(u)$, for all $x \in N(v)$, if $w \sim x$, then C4 formed.
            # We need to ensure NO such pair (w, x) exists.
            
            creates_c4 = False
            for w in adj[u]:
                for x in adj[v]:
                    if w == x:
                        # Common neighbor, forms triangle, not C4.
                        # But if there are 2 common neighbors, C4 existed before.
                        # We assume C4-free, so at most 1 common neighbor.
                        pass
                    elif w in adj[x]:
                        # Path u-w-x-v exists. Adding u-v closes C4.
                        creates_c4 = True
                        break
                if creates_c4:
                    break
            
            if not creates_c4:
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
        
        # Check if valid
        if not check_c4_free(adj, n):
            return None
        
        if has_independent_set_11(adj, n):
            return None
        
        return {'n': n, 'edges': edges}

    # Try to construct larger graphs
    for n in range(39, 30, -1):
        if time.time() > deadline:
            break
        
        # Try a few random shuffles
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            result = try_construct(n)
            if result:
                best_result = result
                best_n = n
                break
        
        if best_n == n:
            # Found a valid graph for this n, try larger?
            # We are iterating downwards from 39.
            # If we found 39, we can stop or try 40?
            # The loop is range(39, 30, -1), so it starts at 39.
            # If we find 39, we break the inner loop.
            # Should we try 40?
            # The prompt says best known is 39.
            # Let's try 43 down to 39.
            pass

    # If we didn't find 39, try 40-43?
    # My loop starts at 39.
    # Let's restart loop from 43 down to 39.
    
    best_result = {'n': 30, 'edges': seed_edges}
    best_n = 30
    
    for n in range(43, 38, -1):
        if time.time() > deadline:
            break
        
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            result = try_construct(n)
            if result:
                best_result = result
                best_n = n
                break
        
        if best_n == n:
            # Found n, can we go higher?
            # The loop continues to n-1.
            # We want the MAX n.
            # So if we find 43, we are done.
            # If we find 42, we try 43? No, we started at 43.
            # So if we find 43, we break.
            if n == 43:
                break

    return best_result
