import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    # This ensures we always return a valid graph.
    # We reuse the logic from the seed program for the fallback.
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    best_n, best_edges = luw_construct(k, g)
    
    # Heuristic search for smaller graph
    # Try N from Moore bound upwards
    # Moore bound for k=3, g=16 is 510.
    # Let's start from 512.
    
    def generate_random_3_reg(n):
        # Create stubs
        stubs = [i for i in range(n) for _ in range(3)]
        random.shuffle(stubs)
        # Pair them
        adj = [[] for _ in range(n)]
        # This naive pairing can create loops and multi-edges.
        # We need to resolve them.
        # Simple approach: keep swapping until valid.
        # But this can be slow.
        # Alternative: use a configuration model with rejection.
        
        # Better: Start with a valid 3-reg graph?
        # Or just generate and fix.
        
        # Let's use a simple fixer.
        # 1. Pair stubs arbitrarily.
        pairs = [(stubs[i], stubs[i+1]) for i in range(0, len(stubs), 2)]
        
        # Build adjacency
        for u, v in pairs:
            if u == v:
                # Loop, need to fix
                pass
            else:
                if v not in adj[u]:
                    adj[u].append(v)
                    adj[v].append(u)
                else:
                    # Multi-edge, need to fix
                    pass
        
        # This is getting complicated.
        # Let's use a different method: 
        # Start with a cycle of n vertices (2-reg).
        # Then add n/2 chords to make it 3-reg.
        # For n even:
        # 1. Cycle: (0,1), (1,2), ..., (n-1,0).
        # 2. Chords: connect i to i + n/2.
        # This creates a 3-reg graph.
        # Is it simple? Yes, if n > 4.
        # Girth? The cycle has length n. The chords create 4-cycles?
        # (i, i+1, i+1+n/2, i+n/2, i). Length 4.
        # So girth is 4.
        # We can start with this and improve.
        
        adj = [[] for _ in range(n)]
        for i in range(n):
            adj[i].append((i + 1) % n)
            adj[i].append((i - 1) % n)
        for i in range(n // 2):
            u = i
            v = i + n // 2
            adj[u].append(v)
            adj[v].append(u)
        
        # Verify 3-regular
        for i in range(n):
            if len(adj[i]) != 3:
                return None # Should not happen
        
        # Convert to edge list
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        return adj, edges

    def get_girth(adj, n, max_girth):
        # BFS from each vertex
        # Returns min girth found, or max_girth if none shorter
        min_girth = max_girth
        for start in range(n):
            # BFS
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            found_cycle = False
            while queue:
                u = queue.popleft()
                du = dist[u]
                if du >= max_girth // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Check if this creates a cycle
                        # If v is visited, and v is not parent of u
                        if v != parent[u]:
                            cycle_len = du + dist[v] + 1
                            if cycle_len < min_girth:
                                min_girth = cycle_len
                                if min_girth == 3:
                                    return 3
            if min_girth < max_girth:
                # We can early terminate if we found a very small girth?
                # But we want the exact min.
                pass
        return min_girth

    def swap_edge(adj, n, edges):
        # Pick two random edges
        # Find indices
        # This is slow if we use list of edges.
        # Better to pick random u, v from adj.
        
        # Pick u
        u = random.randint(0, n - 1)
        # Pick a neighbor v of u
        v = random.choice(adj[u])
        
        # Pick x
        x = random.randint(0, n - 1)
        # Pick a neighbor y of x
        # Ensure y != x and (x,y) is not (u,v) or (v,u)
        # Also ensure u != x, u != y, v != x, v != y to avoid loops/multi-edges easily
        # Actually, we can just pick and check.
        
        # To be efficient, pick x, y from a random edge not incident to u or v?
        # Let's just pick random x, y.
        
        # We need to remove (u,v) and (x,y) and add (u,x) and (v,y) (or u,y, v,x)
        
        # Let's pick x, y such that (x,y) is an edge.
        # We can precompute a list of edges for random selection.
        # But we update edges.
        
        # Let's just pick random u, v, x, y and check if they form valid swap.
        
        # This is getting complex.
        # Let's use a simpler heuristic:
        # Pick a random edge (u,v).
        # Pick a random edge (x,y).
        # If they are disjoint, swap.
        
        # To get random edge, we can maintain a set of edges.
        
        pass # Placeholder

    # Given the complexity and time limit, and the fact that LUW is a valid fallback,
    # and the difficulty of implementing a robust 3-regular graph generator + girth optimizer in < 120s,
    # I will stick to the LUW construction but try to optimize it or use a different known construction if possible.
    
    # However, the LUW graph is very large.
    # Is there a smaller explicit construction?
    # For (3,16), the cage is 936.
    # If I cannot find it, I return LUW.
    # The score will be low (large n), but valid.
    
    # Let's check if the LUW construction is the best I can do.
    # The prompt says "Score: n, minimized".
    # If I return n=354294, and the seed is n=354294, score is 1.0.
    # If I can return n=1000, score is 354.
    
    # I will implement a basic random search for N=1000.
    
    def try_search(n_target, time_limit):
        # Generate random 3-reg graph
        # Use the cycle + chords method
        adj = [[] for _ in range(n_target)]
        for i in range(n_target):
            adj[i].append((i + 1) % n_target)
            adj[i].append((i - 1) % n_target)
        for i in range(n_target // 2):
            u = i
            v = i + n_target // 2
            adj[u].append(v)
            adj[v].append(u)
            
        # Edge set for quick lookup
        edge_set = set()
        for u in range(n_target):
            for v in adj[u]:
                if u < v:
                    edge_set.add((u, v))
        
        current_girth = get_girth(adj, n_target, g)
        
        # Local search
        no_improve_count = 0
        while time.time() < time_limit:
            if current_girth >= g:
                # Found valid graph
                edges = []
                for u in range(n_target):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n_target, edges
            
            # Pick random swap
            # To make it efficient, pick random u, v from adj
            u = random.randint(0, n_target - 1)
            v = random.choice(adj[u])
            
            # Pick random x, y
            x = random.randint(0, n_target - 1)
            y = random.choice(adj[x])
            
            # Check disjointness and validity
            if u == x or u == y or v == x or v == y:
                continue
            if (u, v) == (x, y) or (u, v) == (y, x):
                continue
            
            # Swap (u,v) and (x,y) with (u,x) and (v,y)
            # Check if (u,x) and (v,y) are already edges
            if (min(u,x), max(u,x)) in edge_set or (min(v,y), max(v,y)) in edge_set:
                continue
            
            # Perform swap
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            edge_set.remove((min(u,v), max(u,v)))
            edge_set.remove((min(x,y), max(x,y)))
            
            adj[u].append(x)
            adj[x].append(u)
            adj[v].append(y)
            adj[y].append(v)
            
            edge_set.add((min(u,x), max(u,x)))
            edge_set.add((min(v,y), max(v,y)))
            
            new_girth = get_girth(adj, n_target, g)
            
            if new_girth > current_girth:
                current_girth = new_girth
                no_improve_count = 0
            else:
                no_improve_count += 1
                # Revert if worse? Or keep if same?
                # To avoid getting stuck, revert if worse.
                if new_girth < current_girth:
                    # Revert
                    adj[u].remove(x)
                    adj[x].remove(u)
                    adj[v].remove(y)
                    adj[y].remove(v)
                    
                    edge_set.remove((min(u,x), max(u,x)))
                    edge_set.remove((min(v,y), max(v,y)))
                    
                    adj[u].append(v)
                    adj[v].append(u)
                    adj[x].append(y)
                    adj[y].append(x)
                    
                    edge_set.add((min(u,v), max(u,v)))
                    edge_set.add((min(x,y), max(x,y)))
                
                if no_improve_count > 1000:
                    # Restart
                    break
        
        return None

    # Try a few sizes
    for n_try in [600, 700, 800, 900, 1000]:
        if time.time() > deadline - 10:
            break
        result = try_search(n_try, deadline - 5)
        if result:
            n, edges = result
            if n < best_n:
                best_n, best_edges = n, edges
                # If we found a small one, we might stop?
                # But we want the smallest.
                # If we found 600, we can stop?
                # 600 is close to Moore bound 512.
                # If we found 600, it's likely optimal or close.
                break

    return best_n, best_edges
