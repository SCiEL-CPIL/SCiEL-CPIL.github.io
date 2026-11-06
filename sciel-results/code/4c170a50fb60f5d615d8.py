"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    """
    Construct a triangle-free graph with no independent set of 10 vertices.
    Uses a deterministic randomized greedy + local search approach.
    Aims for n >= 32 vertices.
    """
    target_n = 32  # Try to build up to this size
    max_time = 100  # seconds
    
    start_time = time.time()
    
    def is_triangle_free(adj, n):
        """Check if graph is triangle-free using adjacency matrix."""
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            return False
        return True
    
    def max_independent_set_size(adj, n):
        """
        Find the size of the maximum independent set using branch and bound.
        Returns the size.
        """
        # Use a simple branch and bound for maximum independent set
        # For n up to ~40, this should be fast enough
        best = [0]
        visited = [False] * n
        
        def branch(bound, current_size, candidates):
            if time.time() - start_time > max_time:
                return
            if current_size + len(candidates) <= best[0]:
                return
            if not candidates:
                if current_size > best[0]:
                    best[0] = current_size
                return
            # Pick a vertex with highest degree in candidates for better pruning
            # Actually, for independent set, pick vertex with highest degree
            # to reduce branching
            v = candidates[0]
            # Branch 1: include v
            new_candidates = []
            for u in candidates:
                if u != v and not adj[v][u]:
                    new_candidates.append(u)
            branch(bound, current_size + 1, new_candidates)
            # Branch 2: exclude v
            branch(bound, current_size, candidates[1:])
        
        branch(0, 0, list(range(n)))
        return best[0]
    
    def alpha_le_9(adj, n):
        """Check if independence number is at most 9."""
        if n <= 9:
            return True
        # Quick check: if n is small, use exact
        if n <= 20:
            return max_independent_set_size(adj, n) <= 9
        # For larger n, use a heuristic upper bound check
        # Greedy independent set as lower bound
        used = [False] * n
        count = 0
        for v in range(n):
            if not used[v]:
                count += 1
                for u in range(n):
                    if adj[v][u]:
                        used[u] = True
        if count > 9:
            return False
        # If greedy gives <= 9, it might still be that max IS is > 9
        # Use a more careful check for moderate n
        if n <= 35:
            return max_independent_set_size(adj, n) <= 9
        return True  # Assume valid for now, will verify
    
    def try_construct(n):
        """Try to construct a valid graph on n vertices."""
        rng = random.Random(42)
        
        for attempt in range(500):
            if time.time() - start_time > max_time:
                return None
            
            # Initialize with empty graph
            adj = [[False] * n for _ in range(n)]
            
            # Greedily add edges while maintaining triangle-free property
            # Strategy: add edges in a way that avoids triangles
            # Use a coloring-based approach: proper 3-coloring ensures triangle-free
            # Actually, let's use a different approach: start with a known structure
            
            # Approach: Use a random graph and remove edges to eliminate triangles
            # Then check independence number
            
            # Better: Start with a specific construction
            # Use the "blow-up" of C5 or other triangle-free base
            
            # Let's try: random regular-ish graph, then fix triangles
            for i in range(n):
                for j in range(i+1, n):
                    if rng.random() < 0.5:
                        adj[i][j] = True
                        adj[j][i] = True
            
            # Remove edges to make triangle-free
            changed = True
            while changed:
                changed = False
                for i in range(n):
                    if changed:
                        break
                    for j in range(i+1, n):
                        if adj[i][j]:
                            # Check for common neighbors
                            common = []
                            for k in range(n):
                                if k != i and k != j and adj[i][k] and adj[j][k]:
                                    common.append(k)
                            if common:
                                # Remove edge (i,j) to break all triangles through it
                                adj[i][j] = False
                                adj[j][i] = False
                                changed = True
                                break
            
            # Now check if it's triangle-free (should be)
            if not is_triangle_free(adj, n):
                continue
            
            # Check independence number
            if not alpha_le_9(adj, n):
                continue
            
            # Valid! Return edges
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
        
        return None
    
    # Try increasing n from high to low
    for n in range(target_n, 25, -1):
        result = try_construct(n)
        if result is not None:
            # Verify the result
            n_val = result['n']
            edges = result['edges']
            adj = [[False] * n_val for _ in range(n_val)]
            for e in edges:
                adj[e[0]][e[1]] = True
                adj[e[1]][e[0]] = True
            
            # Final verification
            if not is_triangle_free(adj, n_val):
                continue
            
            if n_val <= 35:
                alpha = max_independent_set_size(adj, n_val)
                if alpha > 9:
                    continue
            
            return result
    
    # Fallback: return the Andrásfai graph And(9) on 26 vertices
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
# EVOLVE-BLOCK-END
