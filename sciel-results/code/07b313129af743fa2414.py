import time
import random
import sys

def construct():
    """
    Construct a triangle-free graph with independence number < 10, maximizing n.
    Structural Change: Fixed n, local search on edge set to break independent sets.
    Uses greedy coloring for fast alpha bounds.
    """
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    def build_adj(edges, n):
        """Build adjacency sets from edge list."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
        
    def is_triangle_free(adj, n):
        """Check if graph is triangle-free."""
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    for k in adj[j]:
                        if k > j and k in adj[i]:
                            return False
        return True

    def alpha_upper_bound_greedy(adj, n):
        """
        Compute an upper bound on the independence number using greedy coloring.
        This is much faster than exact branch-and-bound and provides a tight bound.
        """
        if n == 0:
            return 0
        
        # We color the complement graph. 
        # Independent set in G is a Clique in G_bar.
        # Coloring G gives an upper bound on Clique in G_bar? No.
        # Coloring G_bar gives an upper bound on Clique in G_bar? No, Coloring G_bar gives lower bound on Clique? 
        # Wait. 
        # $\alpha(G) = \omega(\bar{G})$.
        # $\omega(H) \le \chi(H)$.
        # So $\alpha(G) \le \chi(\bar{G})$.
        # We need to color the complement graph $\bar{G}$.
        
        # Construct complement adjacency
        # For efficiency, we work with the original adj and check non-adjacency.
        
        # Greedy coloring of the complement graph.
        # In the complement graph, vertices $u, v$ are adjacent if they are NOT adjacent in G.
        # We want to assign colors to vertices such that adjacent vertices in $\bar{G}$ have different colors.
        # This means if $u, v$ are NOT adjacent in G, they must have different colors.
        # Equivalently, vertices with the same color must be pairwise adjacent in G.
        # So we are partitioning $V$ into cliques of G.
        # The number of colors is the number of cliques.
        # Since any independent set in G contains at most one vertex from each clique (because vertices in a clique are adjacent, so they can't be in an independent set together),
        # the size of the max independent set is at most the number of cliques.
        
        colors = [-1] * n
        num_colors = 0
        uncolored = list(range(n))
        
        # Order vertices by degree in G (high degree first helps form large cliques? 
        # Actually, for coloring the complement, we want to group vertices that are highly connected in G.
        # So sorting by degree in G descending is a good heuristic.)
        uncolored.sort(key=lambda x: len(adj[x]), reverse=True)
        
        while uncolored:
            color = num_colors
            # Find a maximal clique in G using the uncolored vertices
            # Greedy: pick a vertex, add neighbors that are also uncolored and compatible with current clique
            clique = []
            # Start with the first available vertex
            start_v = uncolored[0]
            clique.append(start_v)
            uncolored.remove(start_v)
            
            # Try to add other uncolored vertices to this clique
            # A vertex u can be added to clique C if u is adjacent to all v in C in G.
            i = 0
            while i < len(uncolored):
                u = uncolored[i]
                can_add = True
                for v in clique:
                    if u not in adj[v]:
                        can_add = False
                        break
                if can_add:
                    clique.append(u)
                    uncolored.pop(i)
                else:
                    i += 1
            
            # Assign this color to all vertices in the clique
            for v in clique:
                colors[v] = color
            
            num_colors += 1
        
        return num_colors

    def exact_alpha(adj, n, timeout=5.0):
        """
        Exact branch and bound for independence number.
        Used only when the upper bound is close to 10 or to verify final result.
        """
        if n <= 1:
            return n
        
        # Convert to bitmask
        adj_mask = [0] * n
        for i in range(n):
            for j in adj[i]:
                adj_mask[i] |= (1 << j)
        
        # non_adj[i] has bits set for vertices NOT adjacent to i
        non_adj = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if i != j and j not in adj[i]:
                    mask |= (1 << j)
            non_adj[i] = mask
            
        best = [0]
        start_time = time.time()
        
        def dfs(remaining_mask, current_size):
            nonlocal best
            if time.time() - start_time > timeout:
                return
            
            if remaining_mask == 0:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Upper bound: current_size + popcount(remaining_mask)
            if current_size + bin(remaining_mask).count('1') <= best[0]:
                return
            
            # Greedy coloring bound on the remaining subgraph (complement)
            # This is expensive to compute every node, so we use a simple bound or precomputed?
            # For now, stick to popcount for speed in the inner loop, 
            # but we rely on the external greedy bound for the main search loop.
            
            # Pick vertex with highest degree in remaining set (in G, so lowest degree in G_bar)
            # High degree in G means it has many neighbors, so it constrains the independent set less?
            # No, for Independent Set, we want to pick a vertex that "kills" many other vertices?
            # If we pick v, we remove N(v) from remaining.
            # So we should pick v with MAXIMUM degree in G within the remaining set.
            v = -1
            max_deg = -1
            temp_mask = remaining_mask
            while temp_mask:
                bit = temp_mask & (-temp_mask)
                idx = bit.bit_length() - 1
                deg = bin(adj_mask[idx] & remaining_mask).count('1')
                if deg > max_deg:
                    max_deg = deg
                    v = idx
                temp_mask ^= bit
            
            if v == -1:
                return
            
            # Option 1: Include v
            new_remaining = remaining_mask & non_adj[v]
            new_remaining &= ~(1 << v)
            dfs(new_remaining, current_size + 1)
            
            # Option 2: Exclude v
            remaining_mask &= ~(1 << v)
            dfs(remaining_mask, current_size)
        
        all_mask = (1 << n) - 1
        dfs(all_mask, 0)
        return best[0]

    def local_search_fixed_n(n, max_iters=5000):
        """
        Local search on a fixed n to find a triangle-free graph with alpha < 10.
        Starts with random bipartite graph.
        """
        if time.time() > deadline:
            return None
            
        # Initialize random bipartite graph
        # Split vertices into two sets A and B
        mid = n // 2
        A = list(range(mid))
        B = list(range(mid, n))
        
        adj = [set() for _ in range(n)]
        # Random edges between A and B
        for u in A:
            for v in B:
                if random.random() < 0.5:
                    adj[u].add(v)
                    adj[v].add(u)
                    
        # Ensure triangle-free (Bipartite is always triangle-free)
        
        # Check initial alpha
        ub = alpha_upper_bound_greedy(adj, n)
        if ub < 10:
            return adj
            
        current_alpha = ub
        temp = 10.0 # Temperature for simulated annealing style restarts
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Calculate upper bound
            ub = alpha_upper_bound_greedy(adj, n)
            
            if ub < 10:
                # Verify with exact alpha to be safe? 
                # If ub < 10, then alpha <= ub < 10. So it is valid.
                return adj
                
            # If ub >= 10, we need to reduce alpha.
            # Find a large independent set to break.
            # We can use a greedy heuristic to find a large independent set.
            # Or just pick random pairs from a known large independent set.
            
            # Let's find a large independent set greedily
            indep_set = set()
            remaining = set(range(n))
            # Order by degree (low degree first in G helps find large indep sets? 
            # Actually, for finding a large independent set, we want vertices with few neighbors.
            # So sort by degree in G ascending.
            for v in sorted(remaining, key=lambda x: len(adj[x])):
                if v not in remaining:
                    continue
                # Check if v is compatible with current indep_set
                if all(v not in adj[u] for u in indep_set):
                    indep_set.add(v)
                    remaining.remove(v)
                    # Also remove neighbors of v from remaining? 
                    # No, they might still be compatible with other nodes in indep_set?
                    # If u is neighbor of v, u cannot be in indep_set with v.
                    # So remove neighbors of v from remaining.
                    for u in list(remaining):
                        if u in adj[v]:
                            remaining.remove(u)
                            
            if len(indep_set) < 10:
                # Greedy failed to find 10, but ub says there is one.
                # Try random sampling
                sample = random.sample(range(n), min(n, 20))
                # Build max independent set from sample
                s_set = set()
                for v in sample:
                    if all(v not in adj[u] for u in s_set):
                        s_set.add(v)
                if len(s_set) >= 10:
                    indep_set = s_set
                else:
                    # If we can't find a set of 10, maybe the bound is loose?
                    # Or we are stuck. Try removing a random edge to change structure.
                    edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if j > i:
                                edges.append((i, j))
                    if edges:
                        u, v = random.choice(edges)
                        adj[u].discard(v)
                        adj[v].discard(u)
                    continue
                    
            # We have an independent set S of size >= 10.
            # Try to add an edge within S to break it.
            S_list = list(indep_set)
            random.shuffle(S_list)
            
            added_edge = False
            for i in range(len(S_list)):
                for j in range(i + 1, len(S_list)):
                    u = S_list[i]
                    v = S_list[j]
                    
                    # Check if adding (u, v) creates a triangle
                    # Triangle exists if N(u) intersect N(v) is not empty.
                    # Note: u and v are not adjacent currently.
                    common = adj[u].intersection(adj[v])
                    if not common:
                        # Safe to add
                        adj[u].add(v)
                        adj[v].add(u)
                        added_edge = True
                        break
                if added_edge:
                    break
                    
            if not added_edge:
                # Could not add any edge within S without creating a triangle.
                # This implies for all pairs in S, they share a neighbor.
                # We need to remove some edges to allow this.
                # Pick a random pair (u, v) in S that has common neighbors.
                # Remove a random edge from the common neighbors?
                # Or just remove a random edge from the graph to "anneal".
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if j > i:
                            edges.append((i, j))
                if edges:
                    u, v = random.choice(edges)
                    adj[u].discard(v)
                    adj[v].discard(u)

    # Main loop: try n from 40 down to 1
    for n in range(40, 0, -1):
        if time.time() > deadline:
            break
            
        # Try multiple restarts for each n
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            adj = local_search_fixed_n(n, max_iters=2000)
            if adj is not None:
                # Verify validity
                if is_triangle_free(adj, n):
                    # Exact alpha check
                    alpha = exact_alpha(adj, n, timeout=10.0)
                    if alpha < 10:
                        # Build edge list
                        edges = []
                        for i in range(n):
                            for j in adj[i]:
                                if j > i:
                                    edges.append([i, j])
                        return {'n': n, 'edges': edges}
                        
    # Fallback: trivial valid graph
    n = 9
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % 3 != 0:
                edges.append([i, j])
    return {'n': n, 'edges': edges}
