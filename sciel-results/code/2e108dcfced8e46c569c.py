"""Improved construction using structured C4-free graphs and local search."""
import random
import time
import numpy as np

def construct():
    start = time.time()
    deadline = 145  # safety margin before 150s kill
    
    def is_c4_free_fast(n, adj):
        """Check C4-freeness using bitmasks for speed."""
        adj_masks = []
        for i in range(n):
            mask = 0
            for j in adj[i]:
                mask |= (1 << j)
            adj_masks.append(mask)
        
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    common = adj_masks[u] & adj_masks[v]
                    if bin(common).count('1') > 1:
                        return False
        return True
    
    def min_degree(adj):
        return min(len(adj[i]) for i in range(len(adj)))
    
    def valid(n, adj):
        return min_degree(adj) >= n - 39 and is_c4_free_fast(n, adj)
    
    def to_edges(adj, n):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) preserves C4-freeness."""
        # Need |adj[u] ∩ adj[v]| <= 1 (since u and v are not adjacent yet)
        common = adj[u] & adj[v]
        return len(common) <= 1
    
    def try_complete_adj(adj, n, min_deg):
        """Try to add edges to reach minimum degree while preserving C4-freeness."""
        n_local = len(adj)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u in range(n_local):
                if len(adj[u]) >= min_deg:
                    continue
                candidates = [v for v in range(n_local) if v != u and v not in adj[u]]
                random.shuffle(candidates)
                for v in candidates:
                    if time.time() > deadline:
                        break
                    if can_add_edge(adj, u, v):
                        adj[u].add(v)
                        adj[v].add(u)
                        improved = True
                        if len(adj[u]) >= min_deg:
                            break
                if time.time() > deadline:
                    break
        return valid(n_local, adj)
    
    def try_remove_edges(adj, n):
        """Try removing edges from high-degree vertices to fix C4 violations."""
        n_local = len(adj)
        # Find all C4s and try to break them
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Find a C4
            for u in range(n_local):
                if time.time() > deadline:
                    break
                for v in adj[u]:
                    if v > u:
                        common = adj[u] & adj[v]
                        if len(common) > 1:
                            # Found C4: u-w-v-x-u where w,x in common
                            common_list = list(common)
                            # Remove one edge from the C4
                            w = common_list[0]
                            # Try removing edge u-w or v-w
                            if random.random() < 0.5:
                                adj[u].discard(w)
                                adj[w].discard(u)
                            else:
                                adj[v].discard(w)
                                adj[w].discard(v)
                            changed = True
                            break
                if time.time() > deadline:
                    break
        return is_c4_free_fast(n_local, adj)
    
    def random_c4_free_graph(n, min_deg):
        """Generate a random C4-free graph with min degree >= min_deg."""
        adj = [set() for _ in range(n)]
        
        # Start with a sparse random graph
        p = min_deg / (n - 1) if n > 1 else 0
        p = max(0.3, min(0.95, p))
        
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p:
                    if can_add_edge(adj, u, v):
                        adj[u].add(v)
                        adj[v].add(u)
        
        # Try to complete
        success = try_complete_adj(adj, n, min_deg)
        if not success:
            # Try to fix C4 violations by removing edges
            if try_remove_edges(adj, n):
                if min_degree(adj) >= min_deg:
                    success = True
        
        return adj if success else None
    
    def structured_polarity_graph(n):
        """
        Construct a C4-free graph using polarity graph construction.
        For a finite projective plane of order q, the polarity graph is C4-free.
        We use a simpler structured approach: complement of a union of cliques.
        """
        # Use a block design approach
        # Divide vertices into blocks, connect across blocks carefully
        
        # Try: complement of a C4-free graph with max degree <= 39
        # Equivalently: graph where each vertex has at most 39 non-neighbors
        
        # Simple approach: use a random regular-like graph
        adj = [set() for _ in range(n)]
        
        # Start with complete graph
        for u in range(n):
            for v in range(u + 1, n):
                adj[u].add(v)
                adj[v].add(u)
        
        # Now remove edges to make it C4-free while keeping min degree >= n-39
        # Each vertex can lose at most 39 edges
        removed = [0] * n
        max_removals = 39
        
        # Find C4s and remove edges
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for u in range(n):
                if time.time() > deadline:
                    break
                for v in adj[u]:
                    if v > u:
                        common = adj[u] & adj[v]
                        if len(common) > 1:
                            # Found C4
                            common_list = list(common)
                            w = common_list[0]
                            # Remove an edge incident to a vertex with capacity
                            if removed[u] < max_removals:
                                adj[u].discard(v)
                                adj[v].discard(u)
                                removed[u] += 1
                                removed[v] += 1
                            elif removed[v] < max_removals:
                                adj[u].discard(w)
                                adj[w].discard(u)
                                removed[u] += 1
                                removed[w] += 1
                            elif removed[w] < max_removals:
                                adj[v].discard(w)
                                adj[w].discard(v)
                                removed[v] += 1
                                removed[w] += 1
                            changed = True
                            break
                if time.time() > deadline:
                    break
        
        if valid(n, adj):
            return adj
        return None
    
    best = {'n': 39, 'edges': []}
    
    # Try descending n from 47 to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        min_deg = n - 39
        found = False
        
        # Strategy 1: Structured construction
        if time.time() < deadline:
            adj = structured_polarity_graph(n)
            if adj is not None:
                best = {'n': n, 'edges': to_edges(adj, n)}
                found = True
        
        # Strategy 2: Random + repair
        if not found and time.time() < deadline:
            max_attempts = 2000
            for attempt in range(max_attempts):
                if time.time() > deadline:
                    break
                
                adj = random_c4_free_graph(n, min_deg)
                if adj is not None:
                    best = {'n': n, 'edges': to_edges(adj, n)}
                    found = True
                    break
        
        # Strategy 3: Start from complete, remove C4s
        if not found and time.time() < deadline:
            adj = [set() for _ in range(n)]
            for u in range(n):
                for v in range(u + 1, n):
                    adj[u].add(v)
                    adj[v].add(u)
            
            removed = [0] * n
            max_removals = 39
            
            # Iteratively find and break C4s
            for iteration in range(5000):
                if time.time() > deadline:
                    break
                
                c4_found = False
                for u in range(n):
                    if time.time() > deadline:
                        break
                    for v in list(adj[u]):
                        if v > u:
                            common = adj[u] & adj[v]
                            if len(common) > 1:
                                common_list = list(common)
                                w = common_list[0]
                                # Choose which edge to remove based on removal counts
                                options = []
                                if removed[u] < max_removals:
                                    options.append(('uv', u, v, removed[u]))
                                if removed[v] < max_removals:
                                    options.append(('uv', v, u, removed[v]))
                                if removed[w] < max_removals:
                                    options.append(('uw', u, w, removed[u]))
                                    options.append(('vw', v, w, removed[v]))
                                
                                if options:
                                    # Pick option with lowest removal count
                                    options.sort(key=lambda x: x[3])
                                    choice = options[0]
                                    if choice[0] == 'uv':
                                        adj[u].discard(v)
                                        adj[v].discard(u)
                                        removed[u] += 1
                                        removed[v] += 1
                                    else:
                                        adj[u].discard(w)
                                        adj[w].discard(u)
                                        removed[u] += 1
                                        removed[w] += 1
                                    c4_found = True
                                    break
                    if time.time() > deadline:
                        break
                    if c4_found:
                        break
                
                if not c4_found:
                    if valid(n, adj):
                        best = {'n': n, 'edges': to_edges(adj, n)}
                        found = True
                    break
        
        if found:
            break
    
    return best

# EVOLVE-BLOCK-END
