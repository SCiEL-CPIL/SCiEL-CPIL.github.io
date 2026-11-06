import time
import random
import sys
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin

    def compute_alpha_fast(adj_mask, n):
        """Compute maximum independent set size using improved Branch and Bound with coloring bound."""
        best = [0]
        
        def greedy_color_bound(candidates):
            """Greedy coloring of the induced subgraph on candidates.
            Returns an upper bound on the independence number of that subgraph."""
            colors = {}
            color_count = 0
            remaining = candidates
            while remaining:
                color_count += 1
                # Greedily pick vertices for this color class (maximal independent set)
                color_class = 0
                c = remaining
                while c:
                    lsb = c & (-c)
                    v = lsb.bit_length() - 1
                    c ^= lsb
                    # Check if v can be added to current color class
                    # v is compatible with all in color_class if no edge to any
                    compatible = True
                    cc = color_class
                    while cc:
                        lsb2 = cc & (-cc)
                        w = lsb2.bit_length() - 1
                        cc ^= lsb2
                        if (adj_mask[v] >> w) & 1:
                            compatible = False
                            break
                    if compatible:
                        color_class |= lsb
                        remaining &= ~lsb
                # Remove all vertices in this color class from remaining
                remaining &= ~color_class
            return color_count

        def dfs(candidates, size):
            if size > best[0]:
                best[0] = size
            if candidates == 0:
                return
            
            # Upper bound: current size + number of remaining candidates
            remaining = candidates.bit_count()
            if size + remaining <= best[0]:
                return
            
            # Tighter bound: greedy coloring of candidate subgraph
            color_bound = greedy_color_bound(candidates)
            if size + color_bound <= best[0]:
                return
            
            # Pick a vertex to branch on
            # Heuristic: Pick vertex with max degree in the candidate subgraph
            c = candidates
            max_deg = -1
            chosen = -1
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = (adj_mask[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    chosen = v
            
            if chosen == -1:
                return
                
            # Branch 1: Include 'chosen'
            new_candidates = candidates & ~(adj_mask[chosen] | (1 << chosen))
            dfs(new_candidates, size + 1)
            
            # Branch 2: Exclude 'chosen'
            new_candidates = candidates & ~(1 << chosen)
            dfs(new_candidates, size)

        all_candidates = (1 << n) - 1
        dfs(all_candidates, 0)
        return best[0]

    def check_c4_free(adj_mask, n):
        """Check if graph is C4-free using bitmasks - O(n^2) with bit operations."""
        for u in range(n):
            for v in range(u + 1, n):
                # Count common neighbors
                common = adj_mask[u] & adj_mask[v]
                if common.bit_count() >= 2:
                    return False
        return True

    def try_find_graph(n, time_limit):
        """
        Try to find a C4-free graph on n vertices with alpha < 11.
        Uses a smarter greedy construction with local search.
        """
        if time.time() > deadline:
            return None
            
        adj_mask = [0] * n
        
        # Strategy: Build graph vertex by vertex, trying to maximize degree
        # while maintaining C4-free property
        for i in range(n):
            if time.time() > deadline:
                return None
                
            # Candidates: previous vertices j < i
            # Try to connect i to as many as possible
            candidates = list(range(i))
            random.shuffle(candidates)
            
            for j in candidates:
                if time.time() > deadline:
                    return None
                
                # Check if we can add (i, j) without creating C4
                can_add = True
                v_mask = adj_mask[j]
                while v_mask:
                    lsb = v_mask & (-v_mask)
                    v = lsb.bit_length() - 1
                    v_mask ^= lsb
                    
                    # If N(i) and N(v) share a vertex, adding (i,j) creates C4
                    if adj_mask[i] & adj_mask[v]:
                        can_add = False
                        break
                
                if can_add:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        
        # After construction, verify C4-free
        if not check_c4_free(adj_mask, n):
            return None
            
        # Check alpha
        if time.time() > deadline:
            return None
            
        alpha = compute_alpha_fast(adj_mask, n)
        if alpha < 11:
            return adj_mask
        else:
            return None

    def can_add_edge(adj_mask, i, j):
        """Check if edge (i,j) can be added without creating C4."""
        # Adding (i,j) creates C4 iff N(i) and N(j) share a neighbor
        # (other than each other, but since i,j not yet connected, just check intersection)
        return (adj_mask[i] & adj_mask[j]) == 0

    def try_find_graph_improved(n, time_limit):
        """
        Improved construction with multiple strategies and local search.
        """
        if time.time() > deadline:
            return None
            
        adj_mask = [0] * n
        
        # Strategy: Build graph vertex by vertex
        # For each new vertex, determine which existing vertices it can connect to
        # (maintaining C4-free), then choose a subset to maximize connectivity
        
        # Precompute for efficiency: for each pair (i,j) with i<j, 
        # check if edge can be added based on current adjacencies
        
        # Use a smarter approach: process vertices in an order that helps
        # Try multiple orderings
        
        for attempt_order in range(4):
            if time.time() > deadline:
                return None
                
            adj_mask = [0] * n
            
            if attempt_order == 0:
                order = list(range(n))
            elif attempt_order == 1:
                order = list(range(n))
                random.shuffle(order)
            elif attempt_order == 2:
                # Reverse order
                order = list(range(n - 1, -1, -1))
            else:
                # Random
                order = list(range(n))
                random.shuffle(order)
            
            # Process vertices in given order
            # Map: position in order -> vertex id
            vertex_at_pos = order  # vertex_at_pos[k] = vertex added at step k
            
            for pos in range(n):
                if time.time() > deadline:
                    return None
                    
                v = vertex_at_pos[pos]
                
                # Find all previous vertices that v can connect to
                # (i.e., no C4 would be created)
                compatible = []
                for prev_pos in range(pos):
                    u = vertex_at_pos[prev_pos]
                    if can_add_edge(adj_mask, v, u):
                        compatible.append(u)
                
                # Choose which compatible vertices to connect to
                # Strategy: connect to as many as possible, but be selective
                # to leave room for future vertices
                
                if not compatible:
                    continue
                
                # Randomly select a subset to connect to
                # Bias: connect to vertices that have fewer current neighbors
                # (they have more "room")
                random.shuffle(compatible)
                
                # Try connecting to all compatible vertices
                # But do it in an order that's safe
                # Actually, adding multiple edges from v at once: we need to check
                # that adding (v,u1) doesn't prevent adding (v,u2), etc.
                # Since all are compatible with current state, and v is new,
                # adding (v,u1) changes N(v), which could affect compatibility with u2.
                # Wait: adding (v,u1) means N(v) gains u1. For (v,u2) to still be valid,
                # we need N(v) ∩ N(u2) = ∅ after adding u1 to N(v).
                # N(v) was empty before (v is new), so after adding u1, N(v)={u1}.
                # We need {u1} ∩ N(u2) = ∅, i.e., u1 not in N(u2).
                # So we need u1 and u2 to not be adjacent.
                
                # This is getting complex. Let's just add edges one at a time.
                for u in compatible:
                    if time.time() > deadline:
                        break
                    if can_add_edge(adj_mask, v, u):
                        adj_mask[v] |= (1 << u)
                        adj_mask[u] |= (1 << v)
        
        # Verify C4-free
        if not check_c4_free(adj_mask, n):
            return None
            
        if time.time() > deadline:
            return None
            
        alpha = compute_alpha_fast(adj_mask, n)
        if alpha < 11:
            return adj_mask
        else:
            return None

    # Fallback: 10 disjoint triangles (n=30, alpha=10)
    fallback_edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        fallback_edges += [[a, b], [a, c], [b, c]]
    
    best_n = 30
    best_edges = fallback_edges
    
    # Try to find a larger graph
    # Focus on n values where we're most likely to succeed
    # Best known is 39, so try 43 down to 31
    
    for n in range(43, 30, -1):
        if time.time() > deadline - 15:
            break
            
        # Try multiple random constructions with different seeds
        # More attempts for higher n values
        num_attempts = 100 if n >= 38 else 60
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 15:
                break
                
            random.seed(attempt * 7919 + n * 104729)
            adj_mask = try_find_graph_improved(n, 10)
            if adj_mask:
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if (adj_mask[u] >> v) & 1:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
                
    return {'n': best_n, 'edges': best_edges}
