import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 18

    def is_independent(mask, adj):
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if adj[v] & (mask ^ lsb):
                return False
            m ^= lsb
        return True

    def can_add_edge(adj, u, v):
        if (adj[u] >> v) & 1:
            return True
        common = adj[u] & adj[v]
        return is_independent(common, adj)

    def largest_tf(adj, time_limit=None):
        best = [0]
        full_mask = (1 << n) - 1
        
        # Greedy Lower Bound
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: -degrees[x])
        
        greedy_set = 0
        for v in order:
            if is_independent(adj[v] & greedy_set, adj):
                greedy_set |= (1 << v)
        best[0] = bin(greedy_set).count('1')
        
        def bnb(S_mask, C_mask):
            if time_limit and time.time() > time_limit:
                return
            sz = bin(S_mask).count('1')
            if sz + bin(C_mask).count('1') <= best[0]:
                return
            
            if not C_mask:
                if sz > best[0]:
                    best[0] = sz
                return

            # Select vertex with max degree in C
            max_deg = -1
            chosen_v = -1
            c = C_mask
            while c:
                lsb = c & -c
                vid = lsb.bit_length() - 1
                deg_in_c = bin(adj[vid] & C_mask).count('1')
                if deg_in_c > max_deg:
                    max_deg = deg_in_c
                    chosen_v = vid
                c ^= lsb
                
            v = chosen_v
            v_bit = 1 << v
            
            bnb(S_mask, C_mask & ~v_bit)
            
            if is_independent(adj[v] & S_mask, adj):
                bnb(S_mask | v_bit, C_mask & ~v_bit)
                
        bnb(0, full_mask)
        return best[0]

    def get_edge_list(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def build_turan_3_partite(part_sizes):
        """Build a 3-partite Turán graph where each part is a clique.
        Vertices in different parts are NOT connected (independent between parts).
        This ensures K4-free since any 4 vertices must have 2 in same part (clique)
        or 3 in same part, but with 3 parts by pigeonhole, 4 vertices have 2 in same part.
        Actually for K4-free: a K4 would need 4 vertices with all edges. In 3-partite
        with cliques within parts, edges only exist within parts. So a K4 must be entirely
        within one part. If each part has size <= 3, no K4 exists.
        """
        adj = [0] * n
        vertices = list(range(n))
        part_starts = [0]
        for s in part_sizes:
            part_starts.append(part_starts[-1] + s)
        
        for p in range(3):
            start = part_starts[p]
            end = part_starts[p + 1]
            for i in range(start, end):
                for j in range(i + 1, end):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def get_part_assignment(part_sizes):
        """Return list where part_assignment[v] = part index of vertex v."""
        assignment = [0] * n
        idx = 0
        for p, s in enumerate(part_sizes):
            for _ in range(s):
                assignment[idx] = p
                idx += 1
        return assignment

    def swap_vertices_between_parts(part_assignment, part_sizes, u, v):
        """Swap vertices u and v between their parts, updating part_sizes.
        Returns True if swap was made, False if u and v are in same part."""
        pu = part_assignment[u]
        pv = part_assignment[v]
        if pu == pv:
            return False
        part_assignment[u] = pv
        part_assignment[v] = pu
        part_sizes[pu] -= 1
        part_sizes[pv] -= 1
        part_sizes[pu] += 1
        part_sizes[pv] += 1
        return True

    def rebuild_adj_from_parts(part_assignment, part_sizes):
        """Rebuild adjacency from part assignment."""
        adj = [0] * n
        for p in range(3):
            part_vertices = [v for v in range(n) if part_assignment[v] == p]
            for i in range(len(part_vertices)):
                for j in range(i + 1, len(part_vertices)):
                    u = part_vertices[i]
                    v = part_vertices[j]
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    random.seed(42)
    best_adj = None
    best_score = n

    # Start with balanced 3-partite Turán graph: parts of size 6, 6, 6
    # Each part is a clique of size 6. No K4 since max clique size is 6 but
    # wait - a clique of size 6 contains K4! We need parts of size at most 3.
    # With 18 vertices and 3 parts, we can have parts of size 6 each, but that
    # gives cliques of size 6 which contain K4.
    # 
    # Correction: For K4-free, we need no 4-clique. In a 3-partite graph with
    # cliques within parts, a K4 must be within one part. So each part must have
    # size at most 3. But 3*3 = 9 < 18.
    #
    # Alternative: Use a different construction. Let's use a complete 3-partite
    # graph (edges BETWEEN parts, not within). Then no triangles within parts,
    # and triangles can form across parts. K4-free means no 4 vertices all connected.
    # In complete 3-partite, 4 vertices with 2 in one part and 1 each in other two
    # would form: the 2 in same part are NOT connected (no intra-part edges), so no K4.
    # Actually in complete 3-partite, edges are between parts. 4 vertices: if all in
    # different... we only have 3 parts, so by pigeonhole 2 in same part, not connected.
    # So complete 3-partite is always K4-free!
    #
    # Triangle-free induced subgraph: a set of vertices with no triangles.
    # In complete 3-partite, a triangle needs 1 vertex from each of 3 parts.
    # A triangle-free set can have at most 2 parts' worth of vertices fully,
    # or mix carefully. Max triangle-free: take all from 2 parts = 12 vertices
    # (if balanced 6-6-6). That's too high.
    #
    # Better: Use the complement approach. Let's stick with random greedy + local search
    # but add Turán-inspired initialization with small parts.
    
    # Use 6 parts of size 3 (complete 6-partite is K4-free since 4 vertices 
    # must have 2 in same part which aren't connected... wait no, in complete 
    # multipartite, edges are BETWEEN parts. 4 vertices in 6 parts: if all in 
    # different parts, they form K4. So complete 6-partite on 18 vertices with
    # parts of size 3: 4 vertices from 4 different parts form K4. NOT K4-free.
    
    # Simplest K4-free: complete 3-partite is K4-free as argued above.
    # For triangle-free: max is sum of 2 largest parts.
    # To minimize max triangle-free, we want parts as equal as possible: 6-6-6 gives 12.
    # That's not great. Let's try a different approach.
    
    # Actually, let's use the random greedy approach which already works, 
    # but improve the local search to be more effective.
    
    def generate_random_k4_free():
        adj = [0] * n
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Try multiple random restarts
    restarts = 0
    while time.time() < deadline - 10.0:
        restarts += 1
        adj = generate_random_k4_free()
        
        # Initial Evaluation
        score = largest_tf(adj, time_limit=deadline - 5.0)
        
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            print(f"Restart {restarts}: New best {best_score}")
            
        # Local Search: try adding edges that reduce triangle-free number
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            random.shuffle(all_edges)
            for u, v in all_edges:
                if not ((adj[u] >> v) & 1):
                    if can_add_edge(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        
                        new_score = largest_tf(adj, time_limit=deadline - 5.0)
                        
                        if new_score < score:
                            score = new_score
                            improved = True
                            if score < best_score:
                                best_score = score
                                best_adj = adj[:]
                                print(f"  Local Search Improvement: {best_score}")
                        else:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            
        if time.time() > deadline - 10.0:
            break

    if best_adj is None:
        # Fallback
        best_adj = generate_random_k4_free()

    edges = get_edge_list(best_adj)
    return edges
