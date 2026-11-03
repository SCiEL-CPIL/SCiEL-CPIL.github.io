import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    if k != 4 or g != 10:
        # Fallback for other parameters: return LUW or simple construction
        # The task is specifically for (4, 10)
        # For now, we just handle (4, 10) optimally and fallback to LUW logic for others
        pass

    # Moore bound for (4,10) is 243.
    # Smallest known is 320.
    # We will try to find a graph on 320 vertices.
    
    def generate_random_4reg(n):
        """Generate a random 4-regular simple graph on n vertices."""
        if n < 4:
            return None
        stubs = []
        for i in range(n):
            for _ in range(4):
                stubs.append(i)
        
        random.shuffle(stubs)
        
        edges = []
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                return None
            edges.append((u, v))
        
        # Check for multi-edges
        edge_set = set()
        for u, v in edges:
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return None
            edge_set.add(key)
            
        return edges

    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def count_short_cycles(n, adj, max_girth):
        """
        Count the number of distinct cycles of length < max_girth.
        Returns the count.
        """
        # We use a BFS-like approach to find cycles.
        # To avoid double counting, we only count cycles where the start node is the smallest vertex in the cycle.
        # And we only traverse in increasing vertex order? No, that's hard.
        
        # Simpler: Count non-backtracking walks of length L that return to start.
        # A cycle of length L corresponds to L such walks (one for each start node).
        # So total_walks / L gives the number of cycles of length L.
        
        total_defects = 0
        
        # We only need to check up to depth (max_girth - 1) // 2 ?
        # No, a cycle of length 9 requires depth 4 from a node.
        # Let's just do BFS up to depth 4 from each node and count "back edges" to visited nodes at appropriate depths.
        
        # Actually, for local search, we just need a scalar that decreases as we improve.
        # Let's count the number of "bad" non-backtracking walks of length 3, 4, 5, 6, 7, 8, 9.
        
        # This is expensive to do exactly for all nodes every step.
        # Approximation: Pick a random subset of nodes and check cycles starting there.
        # Or just check if girth >= 10. If not, we need a cost.
        
        # Let's use the trace of B^L.
        # Constructing B is O(N*4).
        # B^L is expensive.
        
        # Let's stick to: If girth < 10, return a random high number? No.
        
        # Better: 
        # 1. Check if girth >= 10. If yes, return 0.
        # 2. If no, count the number of cycles of length 3, 4, 5.
        #    Cycles of length 3 and 4 are easy to count.
        #    Cycles of length 5-9 are harder.
        
        # For the purpose of hill climbing, minimizing triangles and 4-cycles is a good first step.
        # Let's count triangles and 4-cycles.
        
        # Triangles:
        # For each edge (u,v), check common neighbors.
        tri_count = 0
        edge_set = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edge_set.add((u, v))
        
        # Count triangles
        for u in range(n):
            neighbors = adj[u]
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    if (neighbors[i], neighbors[j]) in edge_set or (neighbors[j], neighbors[i]) in edge_set:
                        tri_count += 1
                        # Each triangle is counted 3 times (once for each vertex)
        
        # Count 4-cycles
        # For each edge (u,v), count common neighbors of u and v? No, that's triangles.
        # 4-cycles: u-v-w-x-u.
        # Number of 4-cycles = (1/4) * sum_{u,v adjacent} (common_neighbors(u,v) choose 2)? No.
        # For each pair of nodes (u,v) not necessarily adjacent, count common neighbors.
        # If c is number of common neighbors, then C(c, 2) is number of 4-cycles through u and v.
        # Each 4-cycle is counted twice (once for each diagonal pair).
        
        four_count = 0
        # This is O(N^2 * deg^2) which is 320^2 * 16 ~ 1.6M ops. Acceptable.
        # But we can do better.
        
        # Let's just return a penalty based on triangles and 4-cycles.
        # Penalty = 100 * tri_count + 10 * four_count
        
        return tri_count * 100 + four_count # Placeholder, will refine

    def check_girth_fast(n, adj, g):
        """
        Check if girth is at least g.
        Returns True if girth >= g, False otherwise.
        """
        # BFS from each node up to depth (g-1)//2
        max_depth = (g - 1) // 2
        
        for start in range(n):
            depth = [-1] * n
            parent = [-1] * n
            depth[start] = 0
            parent[start] = -2
            queue = [start]
            head = 0
            
            while head < len(queue):
                u = queue[head]
                head += 1
                
                if depth[u] >= max_depth:
                    continue
                
                for v in adj[u]:
                    if depth[v] == -1:
                        depth[v] = depth[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if parent[u] == v:
                            continue
                        # Found a cycle
                        cycle_len = depth[u] + depth[v] + 1
                        if cycle_len < g:
                            return False
                            
        return True

    def get_cost(n, adj, g):
        """
        Calculate a cost function to minimize.
        If girth >= g, cost is 0.
        Otherwise, cost is number of short cycles.
        """
        if check_girth_fast(n, adj, g):
            return 0
        
        # If not girth >= g, estimate cost.
        # We can just return a large number, but for hill climbing we need gradient.
        # Let's count triangles and 4-cycles quickly.
        
        # Triangles
        tri = 0
        # Using adjacency sets for O(1) lookup
        adj_sets = [set(a) for a in adj]
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    # Common neighbors
                    common = adj_sets[u].intersection(adj_sets[v])
                    tri += len(common)
        tri //= 3 # Each triangle counted 3 times
        
        # 4-cycles
        four = 0
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    common = adj_sets[u].intersection(adj_sets[v])
                    four += len(common) * (len(common) - 1) // 2
        four //= 2 # Each 4-cycle counted twice (once for each diagonal)
        
        # Higher cycles are harder.
        # Let's assume minimizing tri and four helps.
        # Weight: Triangles are worse.
        return tri * 1000 + four * 10

    def local_search(n, edges, g, time_limit):
        """
        Perform local search to improve girth.
        """
        adj = build_adj(n, edges)
        current_cost = get_cost(n, adj, g)
        
        if current_cost == 0:
            return edges
        
        # Precompute edge set for fast lookup
        edge_set = set((min(u, v), max(u, v)) for u, v in edges)
        
        # List of edges for random selection
        edge_list = list(edge_set)
        
        while time.time() < time_limit:
            # Pick two random edges
            if len(edge_list) < 2:
                break
                
            e1_idx = random.randint(0, len(edge_list) - 1)
            e2_idx = random.randint(0, len(edge_list) - 1)
            
            if e1_idx == e2_idx:
                continue
                
            u1, v1 = edge_list[e1_idx]
            u2, v2 = edge_list[e2_idx]
            
            # Check if we can swap
            # New edges: (u1, u2) and (v1, v2)
            # Or (u1, v2) and (v1, u2)
            
            # Ensure no self-loops
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                continue
                
            # Ensure no existing edges
            ne1 = (min(u1, u2), max(u1, u2))
            ne2 = (min(v1, v2), max(v1, v2))
            
            if ne1 in edge_set or ne2 in edge_set:
                continue
                
            # Check if the new edges create immediate conflicts (already checked)
            
            # Perform swap
            edge_set.remove((u1, v1))
            edge_set.remove((u2, v2))
            edge_set.add(ne1)
            edge_set.add(ne2)
            
            # Update edge_list (simple: rebuild or swap indices)
            # For simplicity, just rebuild adj and check cost. 
            # Rebuilding adj is O(N).
            adj = build_adj(n, list(edge_set))
            
            new_cost = get_cost(n, adj, g)
            
            if new_cost <= current_cost:
                current_cost = new_cost
                if current_cost == 0:
                    return list(edge_set)
            else:
                # Revert
                edge_set.remove(ne1)
                edge_set.remove(ne2)
                edge_set.add((u1, v1))
                edge_set.add((u2, v2))
                
    # Try to find a 320-vertex graph
    target_n = 320
    found_edges = None
    
    # Try a few random restarts
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        edges = generate_random_4reg(target_n)
        if not edges:
            continue
            
        # Quick check: is it already girth 10?
        adj = build_adj(target_n, edges)
        if check_girth_fast(target_n, adj, g):
            found_edges = edges
            break
            
        # If not, try to improve with local search
        # Limit local search time per attempt
        local_deadline = time.time() + 2.0
        if time.time() > local_deadline:
             local_deadline = deadline - 1.0
             
        improved_edges = local_search(target_n, edges, g, local_deadline)
        
        if improved_edges:
            # Verify
            adj = build_adj(target_n, improved_edges)
            if check_girth_fast(target_n, adj, g):
                found_edges = improved_edges
                break
                
    if found_edges:
        return target_n, found_edges
        
    # If 320 failed, try 322, 324...
    for n in range(322, 400, 2):
        if time.time() > deadline:
            break
            
        # Try one random graph for each n
        edges = generate_random_4reg(n)
        if not edges:
            continue
            
        adj = build_adj(n, edges)
        if check_girth_fast(n, adj, g):
            return n, edges
            
        # Try short local search
        local_deadline = time.time() + 1.0
        improved_edges = local_search(n, edges, g, local_deadline)
        if improved_edges:
            adj = build_adj(n, improved_edges)
            if check_girth_fast(n, adj, g):
                return n, improved_edges
                
    # Fallback: LUW graph
    # Re-implement seed logic for fallback
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

