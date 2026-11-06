import time
import sys
from itertools import combinations

def construct():
    """
    Finds a dense triangle-free graph on 16 vertices that is not 3-colorable.
    Strategy: 
    1. Start with the Grötzsch graph (11 vertices, 20 edges, 4-chromatic, triangle-free).
    2. Extend to 16 vertices by adding 5 new vertices.
    3. The neighbors of any new vertex in the Grötzsch core must form an independent set in the core.
    4. Enumerate valid independent sets for the new vertices' connections to the core.
    5. For each configuration, greedily add edges between the new vertices to maximize density,
       ensuring the graph remains triangle-free and 4-chromatic.
    6. Use exact backtracking for 3-colorability checks.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 16
    
    # Grötzsch graph edges (0-10)
    # C5: 0-1-2-3-4-0
    # Copies 5-9: 5 connected to 1,4; 6 to 2,0; 7 to 3,1; 8 to 4,2; 9 to 0,3
    # Root 10 connected to 5,6,7,8,9
    groetzh_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (0, 5), (1, 5),
        (1, 6), (2, 6),
        (2, 7), (3, 7),
        (3, 8), (4, 8),
        (4, 9), (0, 9),
        (10, 5), (10, 6), (10, 7), (10, 8), (10, 9)
    ]
    
    # Build adjacency for core
    core_adj = [set() for _ in range(11)]
    for u, v in groetzh_edges:
        core_adj[u].add(v)
        core_adj[v].add(u)
        
    # Find all independent sets in the Grötzsch graph
    # An independent set is a set of vertices with no edges between them.
    independent_sets = []
    for size in range(1, 5): # Max independent set size in Grotzsch is 4
        for combo in combinations(range(11), size):
            is_ind = True
            for i in range(len(combo)):
                for j in range(i+1, len(combo)):
                    if combo[j] in core_adj[combo[i]]:
                        is_ind = False
                        break
                if not is_ind:
                    break
            if is_ind:
                independent_sets.append(frozenset(combo))
                
    # Sort independent sets by size descending to prioritize dense connections
    independent_sets.sort(key=lambda x: len(x), reverse=True)
    
    # Helper: Check if graph is 3-colorable
    def is_3_colorable(adj):
        colors = [-1] * n
        # Order vertices by degree descending for better pruning
        # We need to map indices 0..15
        order = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
        
        def dfs(idx):
            if idx == n:
                return True
            
            v = order[idx]
            # Get available colors
            used = set()
            for u in adj[v]:
                if colors[u] != -1:
                    used.add(colors[u])
            
            avail = [c for c in range(3) if c not in used]
            
            for c in avail:
                colors[v] = c
                if dfs(idx + 1):
                    return True
                colors[v] = -1
            
            return False
        
        return dfs(0)

    # Helper: Check if adding edge (u,v) creates a triangle
    def creates_triangle(adj, u, v):
        if v in adj[u]:
            return False # Already an edge
        common = adj[u].intersection(adj[v])
        return len(common) > 0

    best_edges = []
    best_score = -1
    
    # We have 5 new vertices: 11, 12, 13, 14, 15
    new_vertices = [11, 12, 13, 14, 15]
    
    # To reduce search space, we can fix the connections of the first few new vertices
    # to the most "dense" independent sets.
    # However, 5 vertices * ~50 independent sets is too large for full enumeration (50^5).
    # We will use a greedy/heuristic approach with local improvement.
    
    # Initialize with a random valid assignment from independent sets
    # Then try to improve by swapping assignments or adding edges between new vertices.
    
    import random
    
    # Start with a specific known good configuration if possible, or random
    # Let's try a few random starts
    for _ in range(20): # 20 restarts
        if time.time() > deadline:
            break
            
        # Assign an independent set to each new vertex
        assignments = []
        for v in new_vertices:
            # Pick a random independent set, preferably large
            # Bias towards larger sets
            if random.random() < 0.5 and independent_sets:
                # Pick from top 10 largest
                top_sets = independent_sets[:10]
                s = random.choice(top_sets)
            else:
                s = random.choice(independent_sets)
            assignments.append(s)
        
        # Build initial adjacency
        adj = [set() for _ in range(n)]
        for u, v in groetzh_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        for i, v in enumerate(new_vertices):
            for u in assignments[i]:
                adj[v].add(u)
                adj[u].add(v)
                
        # Check if this base is 4-chromatic
        if is_3_colorable(adj):
            continue # Reject if 3-colorable, we need 4-chromatic
        
        # Now, try to add edges between new vertices
        # Potential edges: (11,12), (11,13), ... (14,15)
        potential_new_edges = [(i, j) for i in range(5) for j in range(i+1, 5)]
        
        # Shuffle to randomize order
        random.shuffle(potential_new_edges)
        
        current_edges_count = len(groetzh_edges) + sum(len(s) for s in assignments)
        
        for u_idx, v_idx in potential_new_edges:
            if time.time() > deadline:
                break
                
            u = new_vertices[u_idx]
            v = new_vertices[v_idx]
            
            # Check if adding (u,v) creates a triangle
            # Triangle would be (u, v, w) where w is common neighbor.
            # Common neighbors are in core or other new vertices.
            # Since u and v are new, their neighbors are in core (independent sets) and other new vertices.
            # If we add (u,v), we must ensure no common neighbor.
            
            if creates_triangle(adj, u, v):
                continue
                
            # Tentatively add edge
            adj[u].add(v)
            adj[v].add(u)
            
            # Check 3-colorability
            if not is_3_colorable(adj):
                current_edges_count += 1
            else:
                # Remove edge
                adj[u].remove(v)
                adj[v].remove(u)
                
        # If we improved, update best
        if current_edges_count > best_score:
            best_score = current_edges_count
            # Reconstruct edge list
            best_edges = list(groetzh_edges)
            for i, v in enumerate(new_vertices):
                for u in assignments[i]:
                    best_edges.append((u, v))
            # Add the edges between new vertices that were kept
            # We need to re-derive which ones were kept. 
            # Easier: just run the greedy addition again deterministically? 
            # Or store the final adj.
            # Let's just store the final adj and convert at the end.
            # For now, just track score. We will regenerate the best at the end if needed, 
            # or better, store the best_adj.
            
            # To avoid complex state tracking, let's just keep the best_adj
            if not hasattr(construct, '_best_adj'):
                construct._best_adj = None
            construct._best_adj = [set(s) for s in adj]
            construct._best_score = current_edges_count

    # If no valid 4-chromatic graph was found (unlikely), return empty or a known one
    if not hasattr(construct, '_best_adj') or construct._best_adj is None:
        # Fallback: Return Grötzsch + isolated vertices (valid but low score)
        # Actually, we should ensure we return a valid graph.
        # The loop above should find one.
        return []

    best_adj = construct._best_adj
    result = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                result.append([u, v])
                
    return result

# Quick self-test
if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
