"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

# EVOLVE-BLOCK-START
def _has_triangle(adj, n):
    """Check if graph with adjacency list adj has a triangle."""
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            # Check common neighbors
            for k in adj[i]:
                if k > j and k in adj[j]:
                    return True
    return False

def _count_triangles(adj, n):
    """Count triangles in graph."""
    count = 0
    for i in range(n):
        for j in adj[i]:
            if j > i:
                common = adj[i] & adj[j]
                count += len(common)
    return count // 3

def _max_independent_set_size(adj, n, limit=10):
    """Branch and bound to check if independent set of size 'limit' exists.
    Returns True if independent set of size >= limit exists."""
    adj_mask = [0] * n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        adj_mask[i] = mask
    
    def dfs(available_mask, current_size):
        if current_size >= limit:
            return True
        if available_mask == 0:
            return False
        # Upper bound: current_size + popcount(available_mask) < limit
        if current_size + bin(available_mask).count('1') < limit:
            return False
        
        # Pick vertex with highest degree in available subgraph
        best_v = -1
        best_deg = -1
        m = available_mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            deg = bin(adj_mask[v] & available_mask).count('1')
            if deg > best_deg:
                best_deg = deg
                best_v = v
            m ^= lsb
        
        if best_v == -1:
            return False
        
        # Option 1: Include best_v
        new_mask = available_mask & ~(adj_mask[best_v] | (1 << best_v))
        if dfs(new_mask, current_size + 1):
            return True
        
        # Option 2: Exclude best_v
        new_mask = available_mask & ~(1 << best_v)
        if dfs(new_mask, current_size):
            return True
        
        return False
    
    initial_mask = (1 << n) - 1
    return dfs(initial_mask, 0)

def _remove_triangles(adj, n):
    """Iteratively remove edges to eliminate all triangles."""
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in list(adj[i]):
                if j <= i:
                    continue
                # Check if (i, j) is in a triangle
                if adj[i] & adj[j]:
                    # Remove edge (i, j)
                    adj[i].remove(j)
                    adj[j].remove(i)
                    changed = True
                    break
            if changed:
                break

def construct():
    # Use a Paley-type construction on 41 vertices, then reduce to 40
    # Paley graph P(41)
    p = 41
    n_full = p
    adj_full = [set() for _ in range(n_full)]
    
    # Quadratic residues mod 41
    residues = set()
    for x in range(1, p):
        residues.add((x * x) % p)
    
    for i in range(n_full):
        for j in range(i + 1, n_full):
            diff = (j - i) % p
            if diff in residues:
                adj_full[i].add(j)
                adj_full[j].add(i)
    
    # Remove vertex 40 to get 40 vertices
    n = 40
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in adj_full[i]:
            if j < n:
                adj[i].add(j)
    
    # The Paley graph P(41) has triangles. We must remove them.
    _remove_triangles(adj, n)
    
    # Now check independence number
    # If alpha >= 10, we try to reduce it by adding edges (if possible without creating triangles)
    # Using Simulated Annealing
    
    if _max_independent_set_size(adj, n, 10):
        # Try to improve
        temperature = 1.0
        cooling = 0.999
        iterations = 20000
        
        # We need to add edges to reduce alpha.
        # Candidate edges are non-edges that do not create triangles.
        
        for _ in range(iterations):
            # Pick a random pair (i, j) that is not an edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j or j in adj[i]:
                continue
            
            # Check if adding (i, j) creates a triangle
            if adj[i] & adj[j]:
                continue
            
            # Tentatively add edge
            adj[i].add(j)
            adj[j].add(i)
            
            # Check if alpha is still < 10
            if not _max_independent_set_size(adj, n, 10):
                # Good, keep it
                temperature *= cooling
            else:
                # Bad, remove it
                adj[i].remove(j)
                adj[j].remove(i)
                temperature *= cooling
    
    # Final check
    if not _max_independent_set_size(adj, n, 10) and not _has_triangle(adj, n):
        edges = []
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
    
    # Fallback: Return the Andrasfai graph on 26 vertices (known valid)
    n = 26
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n % 3 == 1:
                adj[i].add(j)
                adj[j].add(i)
    
    edges = []
    for i in range(n):
        for j in adj[i]:
            if j > i:
                edges.append([i, j])
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
