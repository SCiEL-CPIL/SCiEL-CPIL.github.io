import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    def get_edges(adj, n):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges
    
    def has_k4(adj, n):
        """Check if graph has a K4"""
        for u in range(n):
            for v in range(u + 1, n):
                if not ((adj[u] >> v) & 1):
                    continue
                common = adj[u] & adj[v]
                common &= ~(1 << u)
                common &= ~(1 << v)
                c = common
                while c:
                    lsb = c & -c
                    x = lsb.bit_length() - 1
                    c ^= lsb
                    if adj[x] & (common & ~(1 << x)):
                        return True
        return False
    
    def has_indep_6(adj, n):
        """Check if graph has an independent set of size 6"""
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    continue
                for w in range(v + 1, n):
                    if (adj[u] >> w) & 1 or (adj[v] >> w) & 1:
                        continue
                    for x in range(w + 1, n):
                        if (adj[u] >> x) & 1 or (adj[v] >> x) & 1 or (adj[w] >> x) & 1:
                            continue
                        for y in range(x + 1, n):
                            if (adj[u] >> y) & 1 or (adj[v] >> y) & 1 or (adj[w] >> y) & 1 or (adj[x] >> y) & 1:
                                continue
                            for z in range(y + 1, n):
                                if (adj[u] >> z) & 1 or (adj[v] >> z) & 1 or (adj[w] >> z) & 1 or (adj[x] >> z) & 1 or (adj[y] >> z) & 1:
                                    continue
                                return True
        return False
    
    def is_valid(n, adj):
        return not has_k4(adj, n) and not has_indep_6(adj, n)
    
    def count_triangles_in_mask(mask, adj):
        """Count triangles in induced subgraph on vertices in mask"""
        vertices = []
        m = mask
        while m:
            lsb = m & -m
            vertices.append(lsb.bit_length() - 1)
            m ^= lsb
        
        count = 0
        n_v = len(vertices)
        if n_v < 3:
            return 0
        
        for i in range(n_v):
            for j in range(i + 1, n_v):
                if not ((adj[vertices[i]] >> vertices[j]) & 1):
                    continue
                common = adj[vertices[i]] & adj[vertices[j]]
                common &= ~(1 << vertices[i])
                common &= ~(1 << vertices[j])
                common &= mask
                count += common.bit_count()
        return count // 3
    
    def has_indep_k_in_mask(mask, adj, k):
        """Check if induced subgraph on mask has independent set of size k"""
        vertices = []
        m = mask
        while m:
            lsb = m & -m
            vertices.append(lsb.bit_length() - 1)
            m ^= lsb
        
        n_v = len(vertices)
        if n_v < k:
            return False
        
        def backtrack(idx, current_set):
            if len(current_set) == k:
                return True
            if idx >= n_v:
                return False
            # Pruning: not enough vertices left
            if len(current_set) + (n_v - idx) < k:
                return False
            
            v = vertices[idx]
            # Option 1: include v
            if all(not ((adj[v] >> vertices[ci]) & 1) for ci in current_set):
                if backtrack(idx + 1, current_set + [ci for ci in current_set]):
                    return True
            # Option 2: skip v
            if backtrack(idx + 1, current_set):
                return True
            return False
        
        # Better approach: direct search
        def search(start, chosen):
            if len(chosen) == k:
                return True
            remaining = n_v - start
            if len(chosen) + remaining < k:
                return False
            for i in range(start, n_v):
                v = vertices[i]
                # Check v is independent from all in chosen
                ok = True
                for cv in chosen:
                    if (adj[v] >> cv) & 1:
                        ok = False
                        break
                if not ok:
                    continue
                if search(i + 1, chosen + [v]):
                    return True
            return False
        
        return search(0, [])
    
    def valid_adjacency_mask_for_new_vertex(adj, current_n, new_vertex_idx):
        """
        For a new vertex to be added at position new_vertex_idx (connecting to 0..current_n-1),
        return a bitmask of which subsets of {0..current_n-1} are valid neighbor sets.
        Actually, we'll check validity of a specific neighbor mask.
        """
        pass
    
    def check_new_vertex_valid(adj, current_n, new_mask):
        """
        Check if adding a vertex connected to exactly the vertices in new_mask
        (among 0..current_n-1) maintains K4-free and no independent 6-set.
        
        adj: list of bitmasks for vertices 0..current_n-1
        new_mask: bitmask of which existing vertices the new vertex connects to
        """
        # K4 check: any triangle among neighbors of new vertex would form K4
        # Triangle among neighbors: for any three neighbors a,b,c all in new_mask,
        # if all three edges exist, that's a K4 with the new vertex
        neighbors = new_mask
        # Check if neighbors contain a triangle
        n = neighbors
        while n:
            lsb = n & -n
            u = lsb.bit_length() - 1
            n ^= lsb
            # For each pair of other neighbors
            rest = neighbors & ~(1 << u)
            r = rest
            while r:
                lsb2 = r & -r
                v = lsb2.bit_length() - 1
                r ^= lsb2
                if (adj[u] >> v) & 1:
                    # u and v are adjacent, check if they have a common neighbor in neighbors
                    common = adj[u] & adj[v] & neighbors
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    if common:
                        return False  # K4 found
        
        # Independent 6-set check: any independent 5-set among non-neighbors
        # would form independent 6-set with new vertex
        non_neighbors = ((1 << current_n) - 1) & ~new_mask
        if has_indep_k_in_mask(non_neighbors, adj, 5):
            return False
        
        return True
    
    def backtracking_search(target_n, deadline, randomize_order=False):
        """
        Build a graph on target_n vertices by adding vertices one by one.
        adj[i] is the bitmask of neighbors of vertex i among 0..i-1 (and we store full adjacency).
        """
        adj = [0] * target_n
        
        def backtrack(vertex_idx):
            if time.time() > deadline:
                return False
            
            if vertex_idx == target_n:
                return True
            
            # Determine possible neighbor masks for this vertex
            # For vertex vertex_idx, it can connect to any subset of 0..vertex_idx-1
            # But we need to check validity
            
            # Generate candidate neighbor masks
            # For efficiency, we can use the fact that:
            # 1. Neighbors must be triangle-free (no triangle in neighbor set)
            # 2. Non-neighbors must have no independent 5-set
            
            prev_mask = (1 << vertex_idx) - 1
            candidates = []
            
            # We need to enumerate subsets of prev_mask that are valid
            # This is exponential, so we need pruning
            
            # Strategy: try to build the neighbor set incrementally
            # Or use a smarter approach: for each subset, check if valid
            
            # For small vertex_idx, enumerate all 2^vertex_idx subsets
            # For larger, we need to be smarter
            
            if vertex_idx <= 12:
                # Enumerate all subsets
                for mask in range(1 << vertex_idx):
                    if check_new_vertex_valid(adj, vertex_idx, mask):
                        candidates.append(mask)
            else:
                # Use a heuristic: try masks based on degree constraints
                # Expected degree should be around target_n/2
                # Try masks with various densities
                
                # First, try the "natural" extension: connect to vertices
                # based on some pattern
                # Try several candidate masks
                
                # Candidate 1: connect to first half
                half_mask = (1 << (vertex_idx // 2)) - 1
                candidates.append(half_mask)
                
                # Candidate 2: connect to second half  
                second_half = ((1 << vertex_idx) - 1) ^ ((1 << (vertex_idx // 2)) - 1)
                candidates.append(second_half)
                
                # Candidate 3: random masks with various densities
                for density in [0.3, 0.4, 0.5, 0.6, 0.7]:
                    mask = 0
                    for v in range(vertex_idx):
                        if random.random() < density:
                            mask |= (1 << v)
                    candidates.append(mask)
                
                # Filter valid candidates
                valid_candidates = []
                for mask in candidates:
                    if check_new_vertex_valid(adj, vertex_idx, mask):
                        valid_candidates.append(mask)
                
                if valid_candidates:
                    candidates = valid_candidates
                else:
                    # No valid candidate found with heuristics, try more
                    # This is a problem - we need to backtrack
                    return False
            
            if not candidates:
                return False
            
            # Shuffle for randomization
            if randomize_order:
                random.shuffle(candidates)
            
            for mask in candidates:
                # Set adjacency
                for v in range(vertex_idx):
                    if (mask >> v) & 1:
                        adj[vertex_idx] |= (1 << v)
                        adj[v] |= (1 << vertex_idx)
                    else:
                        adj[vertex_idx] &= ~(1 << v)
                        adj[v] &= ~(1 << vertex_idx)
                
                if backtrack(vertex_idx + 1):
                    return True
                
                # Undo
                for v in range(vertex_idx):
                    if (mask >> v) & 1:
                        adj[vertex_idx] &= ~(1 << v)
                        adj[v] &= ~(1 << vertex_idx)
                    else:
                        adj[vertex_idx] |= (1 << v)
                        adj[v] |= (1 << vertex_idx)
            
            return False
        
        return backtrack(0)
    
    def make_paley_graph(p):
        """Create a Paley graph of prime order p (p ≡ 1 mod 4)"""
        adj = [0] * p
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj
    
    def local_search(adj, n, deadline, max_rounds=100):
        """Perform local search to minimize violations"""
        def count_k4(adj, n):
            count = 0
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj[u] >> v) & 1):
                        continue
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    c = common
                    while c:
                        lsb = c & -c
                        x = lsb.bit_length() - 1
                        c ^= lsb
                        if adj[x] & (common & ~(1 << x)):
                            count += 1
            return count
        
        def count_indep_6(adj, n):
            count = 0
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        continue
                    for w in range(v + 1, n):
                        if (adj[u] >> w) & 1 or (adj[v] >> w) & 1:
                            continue
                        for x in range(w + 1, n):
                            if (adj[u] >> x) & 1 or (adj[v] >> x) & 1 or (adj[w] >> x) & 1:
                                continue
                            for y in range(x + 1, n):
                                if (adj[u] >> y) & 1 or (adj[v] >> y) & 1 or (adj[w] >> y) & 1 or (adj[x] >> y) & 1:
                                    continue
                                for z in range(y + 1, n):
                                    if (adj[u] >> z) & 1 or (adj[v] >> z) & 1 or (adj[w] >> z) & 1 or (adj[x] >> z) & 1 or (adj[y] >> z) & 1:
                                        continue
                                    count += 1
            return count
        
        for _round in range(max_rounds):
            if time.time() > deadline:
                break
            
            k4_count = count_k4(adj, n)
            i6_count = count_indep_6(adj, n)
            
            if k4_count == 0 and i6_count == 0:
                break
            
            # Find best edge flip
            best_delta = 0
            best_u = -1
            best_v = -1
            
            for u in range(n):
                for v in range(u + 1, n):
                    is_edge = bool((adj[u] >> v) & 1)
                    
                    # Compute delta
                    # If we add edge (u,v): check if it creates K4
                    # If we remove edge (u,v): check if it creates independent 6-set
                    
                    if is_edge:
                        # Removing edge: check if this creates an independent 6-set
                        # The non-neighborhood of u and v expands
                        # Simple approximation: count how many independent 5-sets
                        # in the non-neighborhood would become independent 6-sets
                        non_common = ((1 << n) - 1) & ~(adj[u] | adj[v]) & ~(1 << u) & ~(1 << v)
                        # This is expensive, use a simpler heuristic
                        delta = -1  # Removing an edge might help with K4 but hurt with I6
                    else:
                        # Adding edge: check if it creates K4
                        common = adj[u] & adj[v]
                        common &= ~(1 << u)
                        common &= ~(1 << v)
                        # Count triangles in common
                        tri_count = 0
                        c = common
                        while c:
                            lsb = c & -c
                            x = lsb.bit_length() - 1
                            c ^= lsb
                            if adj[x] & (common & ~(1 << x)):
                                tri_count += 1
                        delta = tri_count  # Adding edge creates this many K4s
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_u = u
                        best_v = v
            
            if best_u == -1:
                break
            
            if (adj[best_u] >> best_v) & 1:
                adj[best_u] &= ~(1 << best_v)
                adj[best_v] &= ~(1 << best_u)
            else:
                adj[best_u] |= (1 << best_v)
                adj[best_v] |= (1 << best_u)
        
        return adj
    
    best_graph = None
    best_n = 0
    
    # Strategy 1: Backtracking search for n=36
    if time.time() < deadline - 10:
        adj = [0] * 36
        if backtracking_search(36, deadline - 5, randomize_order=True):
            if is_valid(36, adj):
                best_graph = adj
                best_n = 36
    
    # Strategy 2: Backtracking search for n=35
    if best_n < 35 and time.time() < deadline - 10:
        adj = [0] * 35
        if backtracking_search(35, deadline - 5, randomize_order=True):
            if is_valid(35, adj):
                best_graph = adj
                best_n = 35
    
    # Strategy 3: Paley-based construction + local search
    if best_n < 35 and time.time() < deadline - 15:
        for target_n in [35, 34, 33]:
            if time.time() > deadline - 10:
                break
            
            found = False
            for r in range(8):
                if time.time() > deadline - 10:
                    break
                
                if r == 0 and target_n >= 29:
                    base_p = 29
                    adj = make_paley_graph(base_p)
                    adj = adj + [0] * (target_n - base_p)
                    for u in range(base_p, target_n):
                        for v in range(target_n):
                            if v != u and random.random() < 0.45:
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                else:
                    adj = [0] * target_n
                    for u in range(target_n):
                        for v in range(u + 1, target_n):
                            if random.random() < 0.5:
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                
                adj = local_search(adj, target_n, deadline - 5)
                
                if is_valid(target_n, adj):
                    best_graph = adj
                    best_n = target_n
                    found = True
                    break
            
            if found:
                break
    
    # Strategy 4: Smaller n with more robust search
    if best_n < 30 and time.time() < deadline - 10:
        for target_n in range(29, 16, -1):
            if time.time() > deadline - 5:
                break
            
            found = False
            for r in range(5):
                if time.time() > deadline - 5:
                    break
                
                adj = [0] * target_n
                for u in range(target_n):
                    for v in range(u + 1, target_n):
                        if random.random() < 0.5:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                
                adj = local_search(adj, target_n, deadline - 2)
                
                if is_valid(target_n, adj):
                    best_graph = adj
                    best_n = target_n
                    found = True
                    break
            
            if found:
                break
    
    # Fallback
    if best_graph is None:
        adj = make_paley_graph(17)
        return {'n': 17, 'edges': get_edges(adj, 17)}
    else:
        return {'n': best_n, 'edges': get_edges(best_graph, best_n)}
