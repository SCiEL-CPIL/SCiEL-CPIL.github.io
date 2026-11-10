import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth >= g on as few vertices as possible.
    Uses a randomized local search for small k, g (specifically k=4, g=10)
    to find a graph much smaller than the L-U-W construction.
    Falls back to L-U-W if the search fails or for other parameters.
    """
    t0 = time.time()
    DEADLINE = 110.0  # Safety margin
    
    def elapsed():
        return time.time() - t0
    
    # Specific optimization for k=4, g=10
    if k == 4 and g == 10:
        # Try to find a 4-regular graph of girth 10 on n vertices
        # Start with known best n=320, then try larger if needed
        # We restrict to bipartite graphs as the cage is bipartite
        
        def try_bipartite_local_search(n, time_limit):
            """
            Attempts to construct a 4-regular bipartite graph on n vertices
            (n/2 in each part) with girth >= 10.
            Returns (n, edges) if successful, else None.
            """
            if n % 2 != 0:
                return None
            
            half_n = n // 2
            # Vertices 0..half_n-1 are in Left, half_n..n-1 are in Right
            
            def generate_random_4_reg_bipartite():
                """
                Generate a random 4-regular bipartite graph using configuration model
                with conflict resolution.
                """
                # Use a more robust method: start with a random perfect matching
                # and compose multiple matchings
                
                adj = [[] for _ in range(n)]
                
                # Create 4 random perfect matchings
                for m in range(4):
                    # Create a random permutation of left vertices
                    left_order = list(range(half_n))
                    random.shuffle(left_order)
                    
                    # Assign each left vertex to a unique right vertex
                    for i in range(half_n):
                        left_v = left_order[i]
                        right_v = half_n + i
                        adj[left_v].append(right_v)
                        adj[right_v].append(left_v)
                
                return adj

            def count_short_cycles(adj, max_len=8):
                """
                Count cycles of length <= max_len (even lengths only for bipartite).
                Returns a dict mapping cycle length to count.
                """
                cycle_counts = {}
                n_vertices = len(adj)
                
                for start in range(n_vertices):
                    # BFS with depth limit
                    dist = [-1] * n_vertices
                    parent = [-1] * n_vertices
                    dist[start] = 0
                    queue = [start]
                    head = 0
                    while head < len(queue):
                        u = queue[head]
                        head += 1
                        if dist[u] >= max_len // 2:
                            continue
                        for v in adj[u]:
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                parent[v] = u
                                queue.append(v)
                            elif parent[u] != v and parent[v] != u:
                                # Found a cycle
                                cycle_len = dist[u] + dist[v] + 1
                                if cycle_len <= max_len:
                                    cycle_counts[cycle_len] = cycle_counts.get(cycle_len, 0) + 1
                                break  # Only count first cycle from this start
                
                return cycle_counts

            def get_girth(adj):
                """
                Returns the length of the shortest cycle in the graph.
                Optimized for bipartite graphs.
                """
                n_vertices = len(adj)
                
                for start in range(n_vertices):
                    # BFS
                    dist = [-1] * n_vertices
                    parent = [-1] * n_vertices
                    dist[start] = 0
                    queue = [start]
                    head = 0
                    while head < len(queue):
                        u = queue[head]
                        head += 1
                        for v in adj[u]:
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                parent[v] = u
                                queue.append(v)
                            elif parent[u] != v and parent[v] != u:
                                # Found a cycle
                                cycle_len = dist[u] + dist[v] + 1
                                if cycle_len < 10:
                                    return cycle_len
                
                return 10  # Assume no short cycle found (or >= 10)

            def try_swap(adj, cycle_info=None):
                """
                Attempts a random 2-opt swap to improve girth.
                If cycle_info is provided, bias toward breaking short cycles.
                Returns True if a swap was made.
                """
                # Pick a random edge (u, v)
                # u in Left, v in Right
                # Pick another edge (u', v')
                # Swap to (u, v') and (u', v)
                
                # To be efficient, pick a random Left vertex u, and a neighbor v.
                u = random.randint(0, half_n - 1)
                if not adj[u]:
                    return False
                v = random.choice(adj[u])
                
                # Pick another Left vertex u'
                u_prime = random.randint(0, half_n - 1)
                if u_prime == u:
                    return False
                if not adj[u_prime]:
                    return False
                v_prime = random.choice(adj[u_prime])
                
                # Ensure we don't create multi-edges
                # New edges: (u, v_prime) and (u_prime, v)
                # Check if (u, v_prime) already exists
                if v_prime in adj[u]:
                    return False
                if v in adj[u_prime]:
                    return False
                
                # Perform swap
                adj[u].remove(v)
                adj[u].append(v_prime)
                adj[v].remove(u)
                adj[v].append(u_prime)
                
                adj[u_prime].remove(v_prime)
                adj[u_prime].append(v)
                adj[v_prime].remove(u_prime)
                adj[v_prime].append(u)
                
                return True

            def count_edges_in_cycles(adj, max_len=8):
                """
                For each edge, count how many short cycles (length <= max_len) it participates in.
                Returns a dict mapping (u, v) to count.
                """
                edge_cycle_count = {}
                n_vertices = len(adj)
                
                for start in range(n_vertices):
                    dist = [-1] * n_vertices
                    parent = [-1] * n_vertices
                    dist[start] = 0
                    queue = [start]
                    head = 0
                    while head < len(queue):
                        u = queue[head]
                        head += 1
                        if dist[u] >= max_len // 2:
                            continue
                        for v in adj[u]:
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                parent[v] = u
                                queue.append(v)
                            elif parent[u] != v and parent[v] != u:
                                cycle_len = dist[u] + dist[v] + 1
                                if cycle_len <= max_len:
                                    # This edge (u, v) is in a short cycle
                                    edge_key = (min(u, v), max(u, v))
                                    edge_cycle_count[edge_key] = edge_cycle_count.get(edge_key, 0) + 1
                                break
                
                return edge_cycle_count

            # Main search loop
            best_girth = 0
            best_adj = None
            
            while time.time() - t0 < time_limit:
                adj = generate_random_4_reg_bipartite()
                
                # Initial girth check
                girth = get_girth(adj)
                
                if girth >= 10:
                    # Success
                    edges = []
                    for u in range(half_n):
                        for v in adj[u]:
                            if u < v: # Only add once
                                edges.append((u, v))
                    return n, edges
                
                # Local search with adaptive strategy
                current_girth = girth
                no_improve_count = 0
                max_no_improve = 200
                
                while time.time() - t0 < time_limit:
                    # Every 50 iterations, compute cycle edge counts to guide search
                    if no_improve_count % 50 == 0 and no_improve_count > 0:
                        edge_cycle_count = count_edges_in_cycles(adj, max_len=8)
                        # Bias toward edges in many short cycles
                        if edge_cycle_count:
                            # Pick an edge with high cycle count
                            sorted_edges = sorted(edge_cycle_count.items(), key=lambda x: -x[1])
                            # Take top 20%
                            top_count = max(1, len(sorted_edges) // 5)
                            top_edges = [e for e, _ in sorted_edges[:top_count]]
                            u, v = random.choice(top_edges)
                            # Ensure u is in left
                            if u >= half_n:
                                u, v = v, u
                            # Now pick u' and v'
                            u_prime = random.randint(0, half_n - 1)
                            if u_prime == u:
                                return False
                            v_prime = random.choice(adj[u_prime])
                            if v_prime in adj[u] or v in adj[u_prime]:
                                pass  # Skip this swap
                            else:
                                # Perform swap
                                adj[u].remove(v)
                                adj[u].append(v_prime)
                                adj[v].remove(u)
                                adj[v].append(u_prime)
                                adj[u_prime].remove(v_prime)
                                adj[u_prime].append(v)
                                adj[v_prime].remove(u_prime)
                                adj[v_prime].append(u)
                                new_girth = get_girth(adj)
                                if new_girth > current_girth:
                                    current_girth = new_girth
                                    no_improve_count = 0
                                    if current_girth >= 10:
                                        edges = []
                                        for u2 in range(half_n):
                                            for v2 in adj[u2]:
                                                if u2 < v2:
                                                    edges.append((u2, v2))
                                        return n, edges
                                else:
                                    no_improve_count += 1
                                    if no_improve_count > max_no_improve:
                                        break
                                continue
                    
                    # Normal random swap
                    if try_swap(adj):
                        new_girth = get_girth(adj)
                        if new_girth > current_girth:
                            current_girth = new_girth
                            no_improve_count = 0
                            if current_girth >= 10:
                                edges = []
                                for u2 in range(half_n):
                                    for v2 in adj[u2]:
                                        if u2 < v2:
                                            edges.append((u2, v2))
                                return n, edges
                        else:
                            no_improve_count += 1
                            if no_improve_count > max_no_improve:
                                break
                    else:
                        break
                        
            # If we exit the inner loop, we restart with a new random graph
            
            return None

        # Try a few n values
        # 320 is the known cage. Try it first.
        # If it fails, try slightly larger n where it might be easier?
        # Or just keep trying 320?
        # Let's try 320 for 60s.
        # If fail, try 324 for 20s.
        # If fail, try 328 for 20s.
        # If fail, fallback.
        
        n_candidates = [320, 324, 328, 332, 336, 340, 350, 400]
        
        for n in n_candidates:
            if time.time() - t0 > DEADLINE - 10:
                break
            
            # Allocate time based on remaining time
            # Simple: 10 seconds per candidate, max 10 candidates
            # But 320 is most important. Give it more time.
            time_limit = min(30, DEADLINE - (time.time() - t0) - 5)
            if time_limit < 1:
                continue
                
            result = try_bipartite_local_search(n, time_limit)
            if result:
                return result

    # Fallback: L-U-W Construction
    # (Same as before)
    
    q = k
    if g % 2 == 1:
        d = max(3, g - 5)
        if d % 2 == 0:
            d += 1
    else:
        d = max(4, g - 4)
        if d % 2 == 1:
            d += 1
    
    if q not in [2, 3, 4, 5, 7, 11, 13]:
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True
        q = 4
        while not (is_prime(q) or q == 4):
            q += 1
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y):
            return x ^ y
        def sub(x, y):
            return x ^ y
        def mul(x, y):
            return table[x][y]
    else:
        def add(x, y):
            return (x + y) % q
        def sub(x, y):
            return (x - y) % q
        def mul(x, y):
            return (x * y) % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side), tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges
# EVOLVE-BLOCK-END
