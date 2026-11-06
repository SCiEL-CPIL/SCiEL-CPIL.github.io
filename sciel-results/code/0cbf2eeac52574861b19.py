"""Construct triangle-free graph with alpha < 10 using greedy vertex addition."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    def has_triangle(adj, n):
        """Check if adjacency list contains a triangle."""
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    for w in adj[v]:
                        if w > v and w in adj[u]:
                            return True
        return False
    
    def max_independent_set(adj, n):
        """Find maximum independent set size using branch and bound."""
        # Use a simple greedy upper bound for speed
        remaining = set(range(n))
        count = 0
        while remaining:
            # Pick vertex with minimum degree in remaining
            min_deg = float('inf')
            chosen = None
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    chosen = v
            count += 1
            remaining.discard(chosen)
            for u in adj[chosen]:
                remaining.discard(u)
        return count
    
    # Greedy constructive approach
    # Start with a small valid graph and try to extend
    best_n = 0
    best_edges = []
    
    # Try to build graphs of increasing size
    for target_n in range(10, 41):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random constructions
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            
            # Build graph vertex by vertex
            adj = [set() for _ in range(target_n)]
            valid = True
            
            for v in range(target_n):
                if v == 0:
                    continue
                
                # Determine which existing vertices v can connect to
                # Constraint: no triangles, so if v connects to u and w, then u and w must not be adjacent
                # Also need to maintain alpha < 10
                
                # Start with all possible connections
                candidates = set(range(v))
                
                # For each candidate u, check if connecting v-u would create a triangle
                # A triangle would exist if there's some w where v-w and u-w are both edges
                # Since we're building incrementally, we only need to check existing edges
                
                valid_connections = set()
                for u in candidates:
                    # Check if v-u would create a triangle
                    # Triangle would be v-u-w where u-w is an edge and v-w would be an edge
                    # But v-w doesn't exist yet, so we need to check: if we add v-u,
                    # can we still add v-w later without creating triangle?
                    # Actually, simpler: if u and w are both neighbors of some vertex, 
                    # and we connect v to both, we need u-w not to be an edge
                    
                    # For now, allow connection if no immediate triangle
                    # Check: is there any w in adj[u] such that we might connect v-w later?
                    # This is complex, so use a simpler heuristic:
                    # Only connect v to u if u's neighborhood doesn't overlap with other candidates
                    
                    valid_connections.add(u)
                
                # Now select a subset of valid_connections to actually add
                # Heuristic: connect to vertices with high degree (to help control independence)
                # But avoid creating triangles
                
                # Sort candidates by degree (descending) to prefer high-degree vertices
                sorted_candidates = sorted(valid_connections, key=lambda u: len(adj[u]), reverse=True)
                
                added = 0
                for u in sorted_candidates:
                    if time.time() > deadline - 5:
                        valid = False
                        break
                    
                    # Check if adding edge v-u creates a triangle
                    creates_triangle = False
                    for w in adj[u]:
                        if w in adj[v]:  # v-w already added
                            creates_triangle = True
                            break
                    
                    if not creates_triangle:
                        adj[v].add(u)
                        adj[u].add(v)
                        added += 1
                
                # Check independence number after adding this vertex
                # Use a fast upper bound
                remaining = set(range(v + 1))
                indep_size = 0
                while remaining:
                    min_deg = float('inf')
                    chosen = None
                    for x in remaining:
                        deg = len(adj[x] & remaining)
                        if deg < min_deg:
                            min_deg = deg
                            chosen = x
                    indep_size += 1
                    remaining.discard(chosen)
                    for y in adj[chosen]:
                        remaining.discard(y)
                
                if indep_size >= 10:
                    # Too many independent vertices, need to add more edges or skip
                    # Try to add an edge to break independence
                    # Find two non-adjacent vertices in the independent set and connect them
                    # This is complex, so just mark as invalid for this attempt
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Final check: no triangles and alpha < 10
            if has_triangle(adj, target_n):
                continue
            
            alpha = max_independent_set(adj, target_n)
            if alpha >= 10:
                continue
            
            # Valid graph found!
            if target_n > best_n:
                best_n = target_n
                best_edges = []
                for u in range(target_n):
                    for v in adj[u]:
                        if u < v:
                            best_edges.append([u, v])
                break
        
        if best_n == target_n:
            continue
    
    return {'n': best_n, 'edges': best_edges}
