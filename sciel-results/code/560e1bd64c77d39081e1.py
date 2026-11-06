import time
import random

def construct():
    """
    Constructs a simple graph on 20 vertices with edge colours 0..19.
    Maximizes the number of edges while ensuring every 4-cycle is rainbow.
    
    Strategy:
    The current approach finds a C4-free graph, limiting edges to ~60.
    This approach attempts to find a dense graph (potentially near-complete) 
    and assigns colours based on a deterministic function of vertex indices 
    that ensures any 4-cycle (u, v, w, x) has distinct edge colours.
    
    We test several algebraic colouring rules. If a rule works for the 
    complete graph K_20, we return all 190 edges. If not, we fall back 
    to a dense random graph with greedy colour repair or a known dense 
    structure like a complete bipartite graph (which has many C4s but 
    can be coloured carefully) or simply a very dense random graph 
    with a robust local search colouring algorithm.
    
    Given the tight time limit and the complexity of verifying all C4s 
    in a dense graph, we prioritize:
    1. Testing if a simple algebraic colouring works on K_20.
    2. If not, constructing a dense graph and using a local search 
       to fix violations, starting from a high edge count.
    """
    n = 20
    deadline = time.time() + 100  # Safety margin

    # List of all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Helper: Check if a specific graph and colouring is valid
    def is_valid(edges, colours):
        """
        edges: list of (u, v)
        colours: dict {(u, v): c}
        Returns True if all 4-cycles are rainbow.
        """
        # Build adjacency list with colours
        adj = {i: {} for i in range(n)}
        for (u, v) in edges:
            c = colours[(u, v)]
            adj[u][v] = c
            adj[v][u] = c
            
        # Find all 4-cycles. A 4-cycle is u-v-w-x-u with u<v<w<x or similar ordering?
        # To avoid duplicates and check all, we can iterate over pairs of disjoint edges?
        # Or iterate over u < v < w < x and check if u-v, v-w, w-x, x-u exist?
        # That only finds "convex" 4-cycles in that order. 
        # A 4-cycle in a simple graph can be identified by 4 distinct vertices.
        # There are 3 ways to pair 4 vertices into a cycle: (u,v,w,x), (u,v,x,w), (u,x,v,w).
        # Actually, a 4-cycle is just a set of 4 vertices {u,v,w,x} where the induced subgraph 
        # contains a cycle of length 4. 
        # Efficient check: For every pair of vertices (u, v), look at their common neighbors.
        # If u and v share two neighbors w and x, then u-w-v-x-u is a 4-cycle.
        # The edges are (u,w), (w,v), (v,x), (x,u).
        # We must check that colours of these 4 edges are distinct.
        
        for u in range(n):
            for v in range(u + 1, n):
                # Find common neighbors of u and v
                common = []
                for w in range(n):
                    if w != u and w != v:
                        if w in adj[u] and w in adj[v]:
                            common.append(w)
                            
                # If there are at least 2 common neighbors, we have C4s
                if len(common) >= 2:
                    # Check all pairs of common neighbors (w, x)
                    for i in range(len(common)):
                        for j in range(i + 1, len(common)):
                            w = common[i]
                            x = common[j]
                            # Cycle: u - w - v - x - u
                            # Edges: (u,w), (w,v), (v,x), (x,u)
                            try:
                                c1 = adj[u][w]
                                c2 = adj[w][v]
                                c3 = adj[v][x]
                                c4 = adj[x][u]
                                if len({c1, c2, c3, c4}) < 4:
                                    return False
                            except KeyError:
                                # Should not happen if w, x are common neighbors
                                continue
        return True

    # Strategy 1: Try algebraic colourings on the complete graph K_20
    # If any works, we return 190 edges, which is the maximum possible.
    # This is the "jump" we hope for.
    
    # Common algebraic patterns:
    # 1. c(u, v) = (u * v) % 20
    # 2. c(u, v) = (u + v) % 20
    # 3. c(u, v) = (u * 7 + v * 3) % 20
    # 4. c(u, v) = (u ^ v) % 20 (XOR)
    
    candidates = [
        lambda u, v: (u * v) % 20,
        lambda u, v: (u + v) % 20,
        lambda u, v: (u * 7 + v * 3) % 20,
        lambda u, v: (u ^ v) % 20,
        lambda u, v: ((u + 1) * (v + 1)) % 20,
        lambda u, v: (u * 11 + v * 5) % 20,
    ]
    
    # Pre-compute edges for K_20
    k20_edges = all_edges
    
    for func in candidates:
        colours = {}
        for u, v in k20_edges:
            colours[(u, v)] = func(u, v)
        
        # Check validity. This might be slow for K_20 if not optimized, 
        # but n=20 is small.
        # Optimization: The check above is O(n^2 * deg^2). For K_20, deg=19.
        # Pairs (u,v) ~ 190. Common neighbors ~ 18. Pairs of common ~ 153.
        # Total checks ~ 190 * 153 ~ 30k. Very fast.
        
        if is_valid(k20_edges, colours):
            # Success! Return full K_20
            return [[u, v, colours[(u, v)]] for u, v in k20_edges]

    # Strategy 2: If no simple algebraic rule works for K_20, 
    # try to find a dense subgraph that works with a randomised algebraic rule 
    # or use a local search on a dense random graph.
    
    # Let's try to construct a dense graph and colour it.
    # We will start with a random graph with M edges and try to assign colours 
    # to satisfy the rainbow property.
    # Since finding a valid colouring for a fixed dense graph is hard, 
    # we will instead search for a graph+colouring pair simultaneously.
    
    # However, a simpler robust fallback: 
    # The C4-free strategy yielded ~60 edges (score 1.47).
    # Can we do better than 60?
    # The maximum number of edges in a C4-free graph on 20 vertices is t_3(20).
    # t_3(20) = floor(20^2/4) = 100? No, that's for K_4-free?
    # Zarankiewicz number z(n, 2, 2) is max edges in C4-free graph.
    # For n=20, z(20,2,2) is known to be around 60-70.
    # So C4-free is limited.
    
    # If we allow C4s, we can have more edges, but must colour them correctly.
    # Let's try a local search approach on a dense graph.
    # Start with a random graph with M=100 edges.
    # Randomly assign colours.
    # While violations exist, pick a violating C4 and change one of its edge colours.
    
    def local_search(max_edges_target, time_limit):
        # Start with a random graph
        m = max_edges_target
        if m > len(all_edges):
            m = len(all_edges)
        
        # Randomly select m edges
        random.shuffle(all_edges)
        edges = all_edges[:m]
        
        # Initial random colouring
        colours = {}
        for u, v in edges:
            colours[(u, v)] = random.randint(0, 19)
        
        start_time = time.time()
        no_improve_count = 0
        
        while time.time() < start_time + time_limit:
            if is_valid(edges, colours):
                return [[u, v, colours[(u, v)]] for u, v in edges]
            
            # Find a violation
            # We need a function to find ONE violating C4
            def find_violation():
                for u in range(n):
                    for v in range(u + 1, n):
                        common = []
                        for w in range(n):
                            if w != u and w != v:
                                if w in adj[u] and w in adj[v]:
                                    common.append(w)
                        if len(common) >= 2:
                            for i in range(len(common)):
                                for j in range(i + 1, len(common)):
                                    w = common[i]
                                    x = common[j]
                                    try:
                                        c1 = adj[u][w]
                                        c2 = adj[w][v]
                                        c3 = adj[v][x]
                                        c4 = adj[x][u]
                                        if len({c1, c2, c3, c4}) < 4:
                                            return [(u, w), (w, v), (v, x), (x, u)]
                                    except KeyError:
                                        continue
                return None
            
            # Rebuild adj for speed in find_violation
            adj = {i: {} for i in range(n)}
            for (u, v) in edges:
                c = colours[(u, v)]
                adj[u][v] = c
                adj[v][u] = c
                
            viol = find_violation()
            if viol is None:
                # Should have returned in is_valid, but just in case
                return [[u, v, colours[(u, v)]] for u, v in edges]
                
            # Pick a random edge from the violation and change its colour
            edge_to_change = random.choice(viol)
            old_colour = colours[edge_to_change]
            new_colour = random.randint(0, 19)
            if new_colour == old_colour:
                new_colour = (new_colour + 1) % 20
            colours[edge_to_change] = new_colour
            
            # Check if this change fixed the specific violation and didn't break others?
            # Full is_valid check is expensive if done every step.
            # But n=20 is small. Let's just do a full check every 10 steps or so?
            # Or just rely on the loop condition.
            
            # To avoid infinite loops on bad configurations, we can add a restart.
            # But let's just run for the time limit.
            
            # Optimization: If we get stuck, restart with a new random graph?
            # For now, keep it simple.
            
            # Small optimization: if we change a colour and it's still invalid, 
            # we might be oscillating. 
            # Let's add a simple restart mechanism if no progress in X seconds.
            # But tracking "progress" is hard. 
            # Let's just run until deadline.
            
            # If we are close to deadline, break
            if time.time() > deadline:
                break
                
        # If we didn't find a valid one, return None
        return None

    # Try local search with a dense graph
    # Start with 100 edges
    result = local_search(100, 50)
    if result:
        return result
    
    # If that fails, try 80 edges
    result = local_search(80, 30)
    if result:
        return result
    
    # Fallback to C4-free strategy (known to work)
    # Re-implement the C4-free search briefly
    def generate_random_graph(m):
        all_edges_copy = all_edges[:]
        random.shuffle(all_edges_copy)
        return all_edges_copy[:m]

    def is_c4_free(edges):
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        for u in range(n):
            for w in range(u + 1, n):
                common = adj[u].intersection(adj[w])
                if len(common) > 1:
                    return False
        return True

    best_edges = None
    for m in range(70, 19, -1):
        trials = 200
        for _ in range(trials):
            if time.time() > deadline:
                break
            edges = generate_random_graph(m)
            if is_c4_free(edges):
                # Colour all 0
                return [[u, v, 0] for u, v in edges]
        
    # Final fallback: Star
    return [[0, v, 0] for v in range(1, 20)]

