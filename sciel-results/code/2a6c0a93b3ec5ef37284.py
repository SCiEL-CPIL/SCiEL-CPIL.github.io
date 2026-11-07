"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    n = 15
    
    def is_valid(edges):
        """Check if the graph satisfies the constraint."""
        # Build adjacency
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        
        # For each edge (u,v), check all s in adj[u] and t in adj[v] where s != t
        # Actually, we need to check all paths s-u-v-t where s-u, u-v, v-t are edges and s != t
        for u, v in edges:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True
    
    def violation_count(edges):
        """Count the number of violations."""
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        
        count = 0
        for u, v in edges:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        count += 1
        return count
    
    def edge_set(edges):
        return set(frozenset(e) for e in edges)
    
    # Start with a star (always valid since only one center, endpoints are all leaves with degree 1)
    # Star: center 0 connected to all others. Leaves have degree 1, center has degree 14.
    # Path s-u-v-t: if u=0, v=leaf, then s must be another leaf, t must be... adj[leaf] = {0}, so t=0.
    # But t=0 and s=leaf, deg(leaf)=1, deg(0)=14, so valid.
    # If u=leaf, v=0, similar.
    # So star is valid with 14 edges.
    
    best_edges = [[0, v] for v in range(1, n)]
    best_count = len(best_edges)
    
    # Try to add edges greedily
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Sort by some heuristic - maybe prioritize edges that connect low-degree vertices
    # But we need to be careful about the constraint
    
    # Let's try a greedy approach: try adding each possible edge, keep if valid
    current_edges = list(best_edges)
    current_set = edge_set(current_edges)
    
    # Build adjacency for current graph
    def build_adj(edges):
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg
    
    # Try adding edges in a smart order
    # First, let's try to make degrees more distinct
    # Target: try to achieve degrees like 1,2,3,...,14 or similar distinct pattern
    
    # Let's try a different approach: start from empty and add edges
    # Or start from star and try to add edges while maintaining validity
    
    # Greedy: try to add each non-edge, keep if valid
    for u, v in all_possible:
        if (u, v) in current_set or (v, u) in current_set:
            continue
        if frozenset((u, v)) in current_set:
            continue
        
        new_edges = current_edges + [[u, v]]
        if is_valid(new_edges):
            current_edges = new_edges
            current_set.add(frozenset((u, v)))
    
    if len(current_edges) > best_count:
        best_edges = current_edges
        best_count = len(current_edges)
    
    # Try local search: swap edges or add/remove
    # First, let's try a more systematic approach
    
    # Let's try to construct a graph with specific degree sequence
    # The constraint is: for any s-u-v-t path, deg(s) != deg(t)
    
    # If we can make all degrees distinct, then the constraint is automatically satisfied!
    # Because if all degrees are distinct, then deg(s) != deg(t) for all s != t.
    
    # So the problem reduces to: find the densest graph on 15 vertices with all distinct degrees.
    # Maximum degree sequence: 14, 13, 12, ..., 0 (all distinct)
    # But is this graphical? The sum would be 14+13+...+0 = 105, which is odd, so not graphical.
    # Next: try 14, 13, 12, ..., 1, 0 -> sum = 105, still odd.
    # We need an even sum. So we need to adjust.
    # 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 -> sum = 105 (odd)
    # Change one: e.g., 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 2, 0 -> sum = 104 (even)
    # But now we have two 2's, so not all distinct.
    
    # Alternatively: 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1 -> sum = 104 (even), but two 1's.
    
    # So we can't have all 15 degrees distinct with an even sum. We need at least one pair of equal degrees.
    
    # Let's try: degrees 14,13,12,11,10,9,8,7,6,5,4,3,2,1,0 is not graphical (odd sum).
    # Closest graphical with mostly distinct degrees:
    
    # Let's use a heuristic: try to build a graph with degrees as distinct as possible.
    
    # Try: start with the star (degrees: 14, 1,1,1,1,1,1,1,1,1,1,1,1,1,1)
    # Then add edges between leaves to increase their degrees.
    
    # Let's try a constructive approach with a target degree sequence
    # Target: 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 2, 1 (sum = 104, even)
    # Or: 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 3, 2, 1 (sum = 104, even)
    
    # Let's try to use Havel-Hakimi to check if a degree sequence is graphical,
    # then try to construct it.
    
    def is_graphical(deg_seq):
        """Havel-Hakimi algorithm."""
        d = sorted(deg_seq, reverse=True)
        if sum(d) % 2 != 0:
            return False
        if any(x < 0 for x in d):
            return False
        while d and d[0] > 0:
            k = d[0]
            d = d[1:]
            if len(d) < k:
                return False
            for i in range(k):
                d[i] -= 1
            d.sort(reverse=True)
            if any(x < 0 for x in d):
                return False
        return True
    
    # Try to find a dense graphical degree sequence with minimal repeated degrees
    # and then verify the path constraint.
    
    # Let's try a few candidate degree sequences
    candidates = []
    
    # Base: 14,13,12,11,10,9,8,7,6,5,4,3,2,1,0 -> sum=105 (odd), not graphical
    # Adjust: reduce one by 1 -> 14,13,12,11,10,9,8,7,6,5,4,3,2,1,1 -> sum=104
    candidates.append([14,13,12,11,10,9,8,7,6,5,4,3,2,1,1])
    candidates.append([14,13,12,11,10,9,8,7,6,5,4,3,3,2,1])
    candidates.append([14,13,12,11,10,9,8,7,6,5,5,4,3,2,1])
    candidates.append([13,12,11,10,9,8,7,6,5,4,3,2,1,1,1])  # sum=86
    candidates.append([14,13,12,11,10,9,8,7,6,5,4,3,2,2,2])  # sum=104
    
    # Try to construct a graph for each candidate using a greedy/Havel-Hakimi construction
    def construct_from_degrees(deg_seq, seed=42):
        """Try to construct a simple graph with the given degree sequence."""
        random.seed(seed)
        n = len(deg_seq)
        if not is_graphical(deg_seq):
            return None
        
        # Havel-Hakimi constructive algorithm
        # Create list of (degree, vertex)
        vertices = list(range(n))
        degs = list(deg_seq)
        
        # We'll build the graph iteratively
        edges = []
        remaining_deg = list(degs)
        remaining_vertices = list(range(n))
        
        # Sort by degree descending
        order = sorted(range(n), key=lambda i: -deg_seq[i])
        
        # For each vertex in order, connect to highest-degree available neighbors
        # This is a heuristic, not guaranteed to work for all graphical sequences
        # Let's use a proper Havel-Hakimi construction
        
        # Actually, let's use a different approach: start with empty graph and add edges
        # to match the degree sequence
        
        # Greedy: for each vertex, add edges to satisfy its degree
        current_deg = [0] * n
        edge_set = set()
        
        # Process vertices in order of decreasing desired degree
        for v in order:
            need = deg_seq[v] - current_deg[v]
            if need <= 0:
                continue
            # Find candidates: vertices with remaining capacity
            candidates_v = [u for u in range(n) if u != v and (v, u) not in edge_set and (u, v) not in edge_set and current_deg[u] < deg_seq[u]]
            # Sort by remaining capacity descending
            candidates_v.sort(key=lambda u: -(deg_seq[u] - current_deg[u]))
            
            for u in candidates_v[:need]:
                if current_deg[u] >= deg_seq[u]:
                    continue
                edges.append((v, u))
                edge_set.add((v, u))
                edge_set.add((u, v))
                current_deg[v] += 1
                current_deg[u] += 1
                if current_deg[v] == deg_seq[v]:
                    break
        
        # Check if we achieved the target degrees
        if all(current_deg[i] == deg_seq[i] for i in range(n)):
            return edges
        return None
    
    # Try each candidate
    for deg_seq in candidates:
        if time.time() > deadline - 5:
            break
        edges = construct_from_degrees(deg_seq)
        if edges is not None:
            if is_valid(edges):
                if len(edges) > best_count:
                    best_edges = [[u, v] for u, v in edges]
                    best_count = len(edges)
    
    # Now try to improve by adding more edges if possible
    current_edges = list(best_edges)
    current_set = set(frozenset(e) for e in current_edges)
    
    for u, v in all_possible:
        if time.time() > deadline - 5:
            break
        if frozenset((u, v)) in current_set:
            continue
        new_edges = current_edges + [[u, v]]
        if is_valid(new_edges):
            current_edges = new_edges
            current_set.add(frozenset((u, v)))
    
    if len(current_edges) > best_count:
        best_edges = current_edges
        best_count = len(current_edges)
    
    # Convert to proper list of lists with Python ints
    result = []
    for e in best_edges:
        result.append([int(e[0]), int(e[1])])
    
    return result
# EVOLVE-BLOCK-END
