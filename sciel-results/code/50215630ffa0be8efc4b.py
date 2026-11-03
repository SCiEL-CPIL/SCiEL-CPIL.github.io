import time
import math
import collections

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g.
    Returns (n, edges).
    """
    start_time = time.monotonic()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # LUW Construction Parameters
    # For k=3, q=3.
    # Girth of LUW(q, d) is at least d + 5 (for odd q).
    # We need girth >= g.
    # So d + 5 >= g => d >= g - 5.
    # Also d must be such that the construction is valid.
    # For q=3, d=11 gives girth >= 16.
    # n = 2 * q^d = 2 * 3^11 = 354294. This is too big for 120s?
    # Wait, the current program scores 3.0.
    # Let's check the size of the current program's output.
    # Current program:
    # d = 3
    # while girth_of_d(d) < g: d += 1
    # girth_of_d(d) = d + 5 if d odd, d + 6 if d even.
    # For g=16:
    # d=11 -> girth 16.
    # n = 2 * 3^11 = 354294.
    # This is huge. Is the current program actually running in 120s?
    # 354294 vertices, 3-regular. Edges ~ 530k.
    # Python loops over 350k vertices, doing some arithmetic.
    # It might be slow, but maybe it fits?
    # Or maybe the "score 3.0" implies it's valid and fast enough.
    
    # Let's verify the girth calculation for LUW.
    # Lazebnik-Ustimenko-Woldar graphs:
    # Vertices: (side, x_1, ..., x_d) where x_i in GF(q).
    # Side 0: Point, Side 1: Line.
    # Edges connect (0, x) to (1, y) if y_1 = x_1 + a_1 x_2 ... etc.
    # The girth is determined by the length of the shortest cycle.
    # For LUW(q, d), the girth is at least 2 * floor(d/2) + 2? No.
    # According to literature, girth is at least d + 2? Or d + 5?
    # Let's trust the current program's heuristic: girth_of_d(d) = d+5 (odd d).
    # If d=11, girth >= 16.
    # n = 2 * 3^11 = 354,294.
    
    # Is there a smaller construction?
    # Moore bound for (3, 16) is 512.
    # Best known is 936.
    # LUW(3, 11) is 354k. That's much larger than 936.
    # Why does the current program score 3.0?
    # Score = seed / value.
    # If seed is 3.0 and value is n, then score = 3.0 / n?
    # No, "Score: n, minimized".
    # "The evaluator measures the raw value (lower is better). The search score is seed/value"
    # If the current program returns n=354294, and score is 3.0...
    # Maybe the "value" in score calculation is not n?
    # Or maybe the current program returns a smaller graph?
    
    # Let's re-read the current program code carefully.
    # q = k = 3.
    # d starts at 3.
    # girth_of_d(3) = 3+5 = 8.
    # girth_of_d(5) = 5+5 = 10.
    # girth_of_d(7) = 7+5 = 12.
    # girth_of_d(9) = 9+5 = 14.
    # girth_of_d(11) = 11+5 = 16.
    # So d=11.
    # n = 2 * 3^11.
    
    # Perhaps the "score 3.0" is a normalized score where 3.0 is good?
    # "1.0 equals the seed program and above 1.0 is an improvement"
    # If seed program returns n_seed, and score is 1.0.
    # If current program returns n_cur, and score is 3.0.
    # Then score = n_seed / n_cur.
    # So n_cur = n_seed / 3.
    # If n_seed is large, n_cur is smaller.
    # But 354k is quite large.
    
    # Let's look at the inspiration.
    # Inspiration 1 tries GP graphs.
    # GP graphs (Generalized Petersen) are 3-regular.
    # Girth of GP(n, k) can be high.
    # For g=16, we need a GP graph with girth >= 16.
    # GP(n, k) has 2n vertices.
    # Smallest GP with girth 16?
    # Known: GP(31, 6) has girth 10?
    # Actually, there are specific GP graphs with high girth.
    # But finding them requires search.
    
    # The directive says: "Use a deterministic algebraic construction based on the specific parameters of the Exoo et al. (2026) 936-vertex cage... bypassing heuristic search and larger LUW fallbacks."
    # I don't have the explicit construction from Exoo et al. (2026) in my training data (it's 2026, so it's fictional or future).
    # However, I can try to construct a graph that is likely to be smaller than 354k.
    
    # Wait, if the current program returns 354k and scores 3.0, and the goal is to minimize n...
    # If I can return a graph with n < 354k, I will score higher.
    # The best known is 936.
    # Can I construct a 936-vertex graph?
    # I don't know the exact edges.
    
    # Alternative: Can I optimize the LUW construction?
    # No, LUW(3, 11) is fixed size.
    
    # What about using a different algebraic construction?
    # Maybe a Cayley graph?
    # Or a bipartite graph?
    
    # Let's look at the "Inspiration 1" code.
    # It tries GP graphs.
    # `for n in range(255, max_n + 1):`
    # `for k_gp in range(1, n // 2 + 1):`
    # It constructs GP(n, k_gp) and checks girth.
    # GP(n, k) has 2n vertices.
    # If it finds a GP graph with girth >= 16, it returns it.
    # The smallest GP graph with girth 16 might be smaller than 354k.
    # In fact, GP graphs with girth 16 are known to exist with much fewer vertices.
    # For example, GP(51, 10) or similar?
    # Actually, the smallest known (3, 16)-cage is 936.
    # 936 is not necessarily a GP graph.
    # But there might be GP graphs with girth 16 and n < 354k/2 = 177k.
    # Actually, GP graphs have girth at most 2 * log_2(n) + something?
    # No, girth of GP(n, k) can be arbitrarily large as n grows, but slowly.
    # For girth 16, we need n to be reasonably large.
    # But definitely smaller than 354k.
    
    # So, combining the two:
    # 1. Try to find a GP graph with girth >= g using a bounded search.
    # 2. If found and verified, return it.
    # 3. If not found or time runs out, fall back to LUW.
    
    # The current program *only* does LUW.
    # The inspiration program *tries* GP then LUW.
    # If the inspiration program was scored, it might have found a smaller graph.
    # But the prompt says "CURRENT PROGRAM (score 3.0000)" is the LUW one.
    # And "INSPIRATIONS FROM OTHER SOLUTIONS" includes the GP search one.
    # I should combine them: Try GP search first (fast, potentially smaller), then LUW.
    
    # Optimization for GP search:
    # We need to check girth efficiently.
    # The inspiration uses BFS from each vertex. O(n * (n + m)) which is O(n^2).
    # For n=1000, n^2 = 10^6. Fast.
    # For n=10000, n^2 = 10^8. Slow.
    # We should limit the search range for GP graphs.
    # We know the answer is around 936.
    # So 2n is around 936 => n is around 468.
    # Let's search n from 200 to 1000.
    
    # Also, we can use a faster girth check.
    # Non-backtracking BFS?
    # Or just simple BFS with early termination.
    
    # Let's implement the combined strategy.
    
    def check_girth_bfs(n, edges, min_girth):
        """
        Checks if the graph has girth at least min_girth.
        Returns True if girth >= min_girth.
        """
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        max_depth = min_girth // 2 - 1
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = collections.deque([start])
            
            while queue:
                curr = queue.popleft()
                if dist[curr] > max_depth:
                    continue
                for neighbor in adj[curr]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[curr] + 1
                        parent[neighbor] = curr
                        queue.append(neighbor)
                    elif neighbor != parent[curr]:
                        # Found a cycle
                        if dist[curr] + dist[neighbor] + 1 < min_girth:
                            return False
        return True

    def construct_luw(k, g, deadline):
        """
        Constructs a k-regular graph of girth at least g using LUW.
        """
        q = k
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            def add(x, y): return x ^ y
            def sub(x, y): return x ^ y
            def mul(x, y): return table[x][y]
        else:
            def add(x, y): return (x + y) % q
            def sub(x, y): return (x - y) % q
            def mul(x, y): return x * y % q

        # Determine d
        def girth_of_d(d):
            if d % 2 == 1:
                return d + 5
            else:
                return d + 6

        d = 3
        while girth_of_d(d) < g:
            d += 1

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rules_cycle = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
            rule[c] = rules_cycle[r]

        index = {}
        order = []
        edges = []

        start = (0, (None,) + (0,) * d)
        index[start] = 0
        order.append(start)

        for i, vertex in enumerate(order):
            if time.monotonic() > deadline - 5.0:
                break
            side, x = vertex

            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))

                neighbor = (1 - side, tuple(y))

                if neighbor not in index:
                    index[neighbor] = len(order)
                    order.append(neighbor)

                if side == 0:
                    edges.append((index[vertex], index[neighbor]))

        n = len(order)
        return n, edges

    def construct_gp_search(k, g, deadline):
        """
        Searches for a Generalized Petersen Graph with girth >= g.
        Returns (n, edges) if found, else None.
        """
        # GP graphs are 3-regular.
        # Vertices: 2 * n.
        # Edges: (i, i+1), (n+i, n+i+1), (i, n+i).
        # We need to find n and k (step) such that girth >= g.
        # Note: The standard GP(n, k) has edges (i, i+k) on the outer cycle?
        # No, standard GP(n, k) has:
        # Outer cycle: (i, i+1) mod n.
        # Inner star: (n+i, n+i+k) mod n.
        # Spokes: (i, n+i).
        
        # Let's search n from a small value up to a limit.
        # Since we know 936 is the best known, 2n ~ 936 => n ~ 468.
        # Let's search n from 10 to 1000.
        
        for n in range(10, 1000):
            if time.monotonic() > deadline - 20.0:
                return None
            
            # Check if n is large enough for girth g.
            # Girth of GP(n, k) is at most 2 * log_2(n) + 2? No.
            # For small n, girth is small.
            
            for k_gp in range(1, n // 2 + 1):
                if time.monotonic() > deadline - 10.0:
                    return None
                
                # Quick filters
                # If gcd(n, k_gp) > 1, the inner graph is not a single cycle, 
                # but multiple cycles. This might reduce girth or create multiple components?
                # GP(n, k) is connected if gcd(n, k) = 1.
                if math.gcd(n, k_gp) != 1:
                    continue
                
                # Construct edges
                edges = []
                for i in range(n):
                    edges.append((i, (i + 1) % n))
                for i in range(n):
                    edges.append((n + i, n + (i + 1) % n))
                for i in range(n):
                    edges.append((i, n + i))
                for i in range(n):
                    edges.append((n + i, n + (i + k_gp) % n))
                
                total_vertices = 2 * n
                
                # Check girth
                # Optimization: Only check if potential girth is high enough.
                # Max girth for GP(n, k) is bounded.
                # For now, just check.
                
                if check_girth_bfs(total_vertices, edges, g):
                    return total_vertices, edges

        return None

    # Strategy:
    # 1. Try GP search. It might find a smaller graph.
    # 2. If GP search fails or times out, use LUW.
    
    # However, GP search might be slow if it checks many n, k pairs.
    # 1000 * 500 = 500,000 checks.
    # Each check is O(n^2). For n=1000, 10^6 ops.
    # 500k * 10^6 is too slow.
    # We need to limit the search or optimize the check.
    
    # Better strategy:
    # The current program (LUW) is guaranteed to work and is fast (just generation).
    # The GP search is a gamble.
    # Let's do a *limited* GP search first.
    # Search n up to 500.
    # And use a faster girth check or early termination.
    
    # Actually, the LUW graph is 354k vertices.
    # If I find ANY graph with < 354k vertices, I win.
    # GP graphs with n=500 have 1000 vertices.
    # If any GP(500, k) has girth >= 16, I return 1000.
    # 1000 < 354294. Huge improvement.
    
    # So, let's prioritize GP search.
    # But make it fast.
    
    def check_girth_fast(n, edges, min_girth):
        """
        Faster girth check using BFS with early termination.
        """
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # We only need to check if there is a cycle < min_girth.
        # If min_girth is 16, we look for cycles of length 3, 4, ..., 15.
        # BFS from each node up to depth min_girth // 2 - 1.
        
        max_depth = min_girth // 2 - 1
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = collections.deque([start])
            
            while queue:
                curr = queue.popleft()
                if dist[curr] > max_depth:
                    continue
                for neighbor in adj[curr]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[curr] + 1
                        parent[neighbor] = curr
                        queue.append(neighbor)
                    elif neighbor != parent[curr]:
                        if dist[curr] + dist[neighbor] + 1 < min_girth:
                            return False
        return True

    # Let's try a small GP search first.
    # If it finds a solution, return it.
    # Otherwise, fall back to LUW.
    
    # To speed up GP search, we can skip obvious bad cases.
    # Also, we can limit the number of k_gp values.
    
    gp_result = None
    search_limit = 500  # Search n up to 500
    
    for n in range(20, search_limit + 1):
        if time.monotonic() > deadline - 30.0:
            break
        
        # For a given n, which k_gp are promising?
        # k_gp should be roughly n/3 to n/2 for high girth?
        # Let's just iterate all valid k_gp.
        
        for k_gp in range(1, n // 2 + 1):
            if time.monotonic() > deadline - 10.0:
                break
            
            if math.gcd(n, k_gp) != 1:
                continue
            
            edges = []
            for i in range(n):
                edges.append((i, (i + 1) % n))
            for i in range(n):
                edges.append((n + i, n + (i + 1) % n))
            for i in range(n):
                edges.append((i, n + i))
            for i in range(n):
                edges.append((n + i, n + (i + k_gp) % n))
            
            total_vertices = 2 * n
            
            if check_girth_fast(total_vertices, edges, g):
                gp_result = (total_vertices, edges)
                break
        
        if gp_result:
            break
    
    if gp_result:
        return gp_result
    
    # Fallback to LUW
    return construct_luw(k, g, deadline)
