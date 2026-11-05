import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 118  # 2s safety margin for 120s limit
    
    n = 18
    total_pairs = n * (n - 1) // 2  # 153
    
    # Precompute all pairs as (i, j) with i < j
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_index = {}
    for idx, (i, j) in enumerate(pairs):
        pair_index[(i, j)] = idx
    
    # Precompute for each pair, the set of pairs that share a vertex with it
    # For K4 check: if we add edge (u,v), a K4 is formed if there exist w,x
    # such that (u,w),(v,w),(u,x),(v,x),(w,x) all exist.
    # Equivalently, the common neighbors of u and v must contain an edge.
    
    def count_k4_violations(adj):
        """Count K4s in the graph. adj is list of int bitmasks."""
        count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    continue
                # Common neighbors of u and v
                common = adj[u] & adj[v]
                common &= ~((1 << u) | (1 << v))
                # Check if any edge exists within common neighbors
                # For each w in common, check if any x > w in common has edge (w,x)
                w_list = [w for w in range(n) if (common >> w) & 1]
                for a in range(len(w_list)):
                    w = w_list[a]
                    for b in range(a + 1, len(w_list)):
                        x = w_list[b]
                        if (adj[w] >> x) & 1:
                            count += 1
        return count
    
    def is_k4_free(adj):
        return count_k4_violations(adj) == 0
    
    def max_triangle_free_size(adj):
        """
        Find the maximum size of an induced triangle-free subgraph.
        
        A set S is triangle-free iff for every edge (u,v) in S, there is no w in S
        with edges (u,w) and (v,w).
        
        Equivalently: in the subgraph induced by S, no triangle exists.
        
        We can think of it as: we want the largest S such that the induced
        subgraph on S has no K3.
        
        Branch and bound: try to build S greedily with backtracking.
        Alternatively, note that a triangle-free graph has at most n^2/4 edges (Mantel).
        
        Better approach: The complement of a triangle-free graph is... not helpful directly.
        
        Let's use: a set S is triangle-free iff for every triple (u,v,w) in S,
        not all three edges exist.
        
        We can model this as: find the maximum S such that no K3 is fully contained in S.
        This is equivalent to: the maximum independent set in the 3-uniform hypergraph
        where hyperedges are triangles of G.
        
        For n=18, the number of triangles is at most C(18,3)=816.
        
        Branch and bound on the hypergraph:
        - State: current set S, remaining candidates R
        - Upper bound: |S| + |R|
        - If upper bound <= best, prune
        - Pick a vertex v from R: either include v (check no triangle in S∪{v} is completed)
          or exclude v.
        
        Actually, a cleaner B&B:
        - We process vertices one by one.
        - At each step, we can add or not add the next vertex.
        - When adding vertex v, we must ensure that no triangle is formed with existing vertices in S.
          i.e., for any two existing vertices u,w in S, if (u,v) and (w,v) are both edges,
          then (u,w) must NOT be an edge (otherwise u,v,w form a triangle).
          Wait no: if u,w,v are all in S and all three edges exist, that's a triangle.
          So when adding v to S, we need: there do NOT exist u,w in S such that
          (u,v), (w,v), (u,w) are all edges.
        
        Let me implement a recursive B&B.
        """
        # Precompute adjacency as set of frozensets for fast triangle check
        # Actually, use bitmasks.
        
        best = [0]
        
        def rec(S_mask, candidate_mask, current_size):
            # Upper bound
            if current_size + bin(candidate_mask).count('1') <= best[0]:
                return
            
            if candidate_mask == 0:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Pick a candidate vertex
            # Heuristic: pick the vertex with most edges to S (hardest to add)
            # or just pick the lowest bit
            # For speed, pick lowest set bit
            v = (candidate_mask & (-candidate_mask)).bit_length() - 1
            new_candidate = candidate_mask & ~(1 << v)
            
            # Option 1: exclude v
            rec(S_mask, new_candidate, current_size)
            
            # Option 2: include v (if it doesn't create a triangle with S)
            # Check: for all pairs (u,w) in S, not (adj[u]&adj[w] has v and adj[u] has v and adj[w] has v)
            # i.e., no u,w in S with edges (u,v),(w,v),(u,w)
            # This is: among neighbors of v in S, there is no edge.
            neigh_in_S = adj[v] & S_mask
            # Check if any edge exists within neigh_in_S
            has_edge = False
            ni = [i for i in range(n) if (neigh_in_S >> i) & 1]
            for a in range(len(ni)):
                u = ni[a]
                for b in range(a + 1, len(ni)):
                    w = ni[b]
                    if (adj[u] >> w) & 1:
                        has_edge = True
                        break
                if has_edge:
                    break
            
            if not has_edge:
                rec(S_mask | (1 << v), new_candidate, current_size + 1)
        
        rec(0, (1 << n) - 1, 0)
        return best[0]
    
    def adj_to_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges
    
    def edges_to_adj(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    # Generate initial Clebsch-based graph
    def generate_initial():
        adj = [0] * n
        # Clebsch on 0..15: adjacent if Hamming distance = 2
        for u in range(16):
            for v in range(u + 1, 16):
                diff = u ^ v
                if bin(diff).count('1') == 2:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        # Connect 16 and 17 to break up large TF sets
        # Random connections
        for i in range(n):
            if i != 16 and random.random() < 0.4:
                adj[16] |= (1 << i)
                adj[i] |= (1 << 16)
            if i != 17 and random.random() < 0.4:
                adj[17] |= (1 << i)
                adj[i] |= (1 << 17)
        if random.random() < 0.5:
            adj[16] |= (1 << 17)
            adj[17] |= (1 << 16)
        return adj
    
    # Simulated annealing
    adj = generate_initial()
    
    # Remove K4s initially
    for _ in range(100):
        if is_k4_free(adj):
            break
        # Find a K4 and remove a random edge from it
        found = False
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    continue
                common = adj[u] & adj[v]
                common &= ~((1 << u) | (1 << v))
                w_list = [w for w in range(n) if (common >> w) & 1]
                for a in range(len(w_list)):
                    w = w_list[a]
                    for b in range(a + 1, len(w_list)):
                        x = w_list[b]
                        if (adj[w] >> x) & 1:
                            # K4: u,v,w,x
                            quad_edges = [(u,v),(u,w),(u,x),(v,w),(v,x),(w,x)]
                            e = random.choice(quad_edges)
                            adj[e[0]] &= ~(1 << e[1])
                            adj[e[1]] &= ~(1 << e[0])
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
    
    current_score = max_triangle_free_size(adj)
    best_score = current_score
    best_adj = adj[:]
    
    # Simulated annealing
    T = 2.0
    T_min = 0.01
    cooling = 0.995
    
    while time.time() < deadline - 2:
        T *= cooling
        if T < T_min:
            T = T_min
        
        # Pick a random edge to flip
        u, v = random.sample(range(n), 2)
        if u > v:
            u, v = v, u
        
        # Try flipping edge (u,v)
        old_val = (adj[u] >> v) & 1
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)
        
        # Check K4-free
        if not is_k4_free(adj):
            # Revert
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            continue
        
        new_score = max_triangle_free_size(adj)
        
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_adj = adj[:]
        else:
            # Accept with probability exp(-(new_score - current_score) / T)
            delta = new_score - current_score
            if delta > 0:
                import math
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_score = new_score
                else:
                    # Revert
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
            # If delta == 0, accept (no cost)
        
        # Occasional restart
        if random.random() < 0.001 and time.time() < deadline - 10:
            adj = generate_initial()
            for _ in range(100):
                if is_k4_free(adj):
                    break
                found = False
                for u2 in range(n):
                    for v2 in range(u2 + 1, n):
                        if not (adj[u2] >> v2) & 1:
                            continue
                        common = adj[u2] & adj[v2]
                        common &= ~((1 << u2) | (1 << v2))
                        w_list = [w for w in range(n) if (common >> w) & 1]
                        for a in range(len(w_list)):
                            w = w_list[a]
                            for b in range(a + 1, len(w_list)):
                                x = w_list[b]
                                if (adj[w] >> x) & 1:
                                    quad_edges = [(u2,v2),(u2,w),(u2,x),(v2,w),(v2,x),(w,x)]
                                    e = random.choice(quad_edges)
                                    adj[e[0]] &= ~(1 << e[1])
                                    adj[e[1]] &= ~(1 << e[0])
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
            current_score = max_triangle_free_size(adj)
    
    return adj_to_edges(best_adj)
