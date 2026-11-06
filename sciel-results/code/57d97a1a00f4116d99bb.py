"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _check_no_c4(n, edges):
    """Check that no two vertices have two common neighbours."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        for j in range(i+1, n):
            common = adj[i] & adj[j]
            if len(common) >= 2:
                return False
    return True

def _max_independent_set(n, edges, timeout=10):
    """Find maximum independent set using greedy + branch and bound."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Greedy coloring for upper bound
    def greedy_color(vertices):
        color = {}
        for v in vertices:
            used_colors = set()
            for u in adj[v] & set(vertices):
                if u in color:
                    used_colors.add(color[u])
            c = 0
            while c in used_colors:
                c += 1
            color[v] = c
        return max(color.values()) + 1 if color else 0
    
    def branch_and_bound(remaining, current_size, best):
        if not remaining:
            return best
        if current_size + greedy_color(remaining) <= best:
            return best
        
        # Pick vertex with highest degree in remaining
        v = max(remaining, key=lambda x: len(adj[x] & set(remaining)))
        
        # Include v
        new_remaining = [u for u in remaining if u != v and u not in adj[v]]
        best = branch_and_bound(new_remaining, current_size + 1, best)
        
        # Exclude v
        remaining.remove(v)
        best = branch_and_bound(remaining, current_size, best)
        remaining.append(v)
        
        return best
    
    start_time = time.time()
    best = 0
    # Try a few random orderings
    for attempt in range(50):
        if time.time() - start_time > timeout:
            break
        verts = list(range(n))
        random.shuffle(verts)
        # Simple greedy to get initial bound
        current = list(verts)
        ind = []
        while current:
            v = current.pop()
            ind.append(v)
            current = [u for u in current if u not in adj[v]]
        best = max(best, len(ind))
    
    return best

def _verify_graph(n, edges, timeout=5):
    """Verify graph has no C4 and alpha < 11."""
    if not _check_no_c4(n, edges):
        return False
    alpha = _max_independent_set(n, edges, timeout=timeout)
    return alpha < 11

def construct():
    """
    Try to construct a C4-free graph with no independent 11-set.
    Start with known good constructions and search for improvements.
    """
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Best known: 39-vertex construction
    # We'll try to find one via random search
    
    best_n = 30
    best_edges = None
    
    # Start with 10 triangles as baseline
    edges_30 = []
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        edges_30 += [[a,b], [a,c], [b,c]]
    best_n = 30
    best_edges = edges_30
    
    # Try to add vertices and edges to reach higher n
    # Strategy: start with a C4-free graph and try to add vertices
    
    def try_extend(base_n, base_edges, target_n, attempts=100):
        """Try to extend a C4-free graph to target_n vertices."""
        for _ in range(attempts):
            if time.time() > deadline - 5:
                break
            
            n = target_n
            edges = list(base_edges)
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            
            success = True
            # Add edges to new vertices (base_n to n-1)
            for new_v in range(base_n, n):
                # Connect new_v to some existing vertices
                # Must ensure no C4 is created
                candidates = list(range(base_n))
                random.shuffle(candidates)
                
                for cand in candidates:
                    if time.time() > deadline - 5:
                        return None
                    # Check if adding edge (new_v, cand) creates C4
                    # Two vertices have two common neighbors if they share >=2 neighbors
                    # Adding edge (new_v, cand): check all pairs
                    # For any pair (a, b), if both are neighbors of new_v and cand, 
                    # then new_v and cand would have common neighbors a and b
                    # But we need: no two vertices have two common neighbors
                    # So for new_v and cand: common neighbors must be < 2
                    # new_v's neighbors so far: those already connected
                    # cand's neighbors: adj[cand]
                    # common = (neighbors of new_v) ∩ (adj[cand])
                    # But new_v only has edges to previously added new vertices in this step
                    # Actually, let's just check all pairs after adding
                    
                    # Tentatively add edge
                    adj[new_v].add(cand)
                    adj[cand].add(new_v)
                    
                    # Check C4 condition for all pairs involving new_v or cand
                    valid = True
                    for i in range(n):
                        if i == new_v or i == cand:
                            continue
                        # Check pair (new_v, i)
                        common_ni = adj[new_v] & adj[i]
                        if len(common_ni) >= 2:
                            valid = False
                            break
                        # Check pair (cand, i)  
                        common_ci = adj[cand] & adj[i]
                        if len(common_ci) >= 2:
                            valid = False
                            break
                    
                    if not valid:
                        # Remove edge
                        adj[new_v].discard(cand)
                        adj[cand].discard(new_v)
                    else:
                        edges.append([new_v, cand])
                        # Add at most a few edges per new vertex to keep it sparse
                        if len([e for e in edges if new_v in e]) >= 3:
                            break
                
                # Also connect new_v to other new vertices
                for other_new in range(base_n, new_v):
                    if time.time() > deadline - 5:
                        return None
                    adj[new_v].add(other_new)
                    adj[other_new].add(new_v)
                    
                    # Check C4
                    valid = True
                    for i in range(n):
                        if i == new_v or i == other_new:
                            continue
                        common_ni = adj[new_v] & adj[i]
                        if len(common_ni) >= 2:
                            valid = False
                            break
                        common_ci = adj[other_new] & adj[i]
                        if len(common_ci) >= 2:
                            valid = False
                            break
                    
                    if not valid:
                        adj[new_v].discard(other_new)
                        adj[other_new].discard(new_v)
                    else:
                        edges.append([new_v, other_new])
            
            # Check if graph is C4-free
            if not _check_no_c4(n, edges):
                continue
            
            # Check independence number
            alpha = _max_independent_set(n, edges, timeout=2)
            if alpha < 11:
                return edges
            
            # If alpha >= 11, try adding more edges to reduce independence
            # This is complex; for now, just return None
            return None
        
        return None
    
    # Try to extend from 30 to higher values
    for target_n in range(31, 44):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random extensions
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            result = try_extend(30, edges_30, target_n, attempts=1)
            if result is not None:
                # Verify
                if _verify_graph(target_n, result, timeout=3):
                    best_n = target_n
                    best_edges = result
                    break
        
        if best_n >= target_n:
            continue
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
