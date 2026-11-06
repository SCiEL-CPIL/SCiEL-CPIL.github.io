"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

# EVOLVE-BLOCK-START
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
        if current_size + bin(available_mask).count('1') < limit:
            return False
        
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
        
        new_mask = available_mask & ~(adj_mask[best_v] | (1 << best_v))
        if dfs(new_mask, current_size + 1):
            return True
        
        new_mask = available_mask & ~(1 << best_v)
        if dfs(new_mask, current_size):
            return True
        
        return False
    
    initial_mask = (1 << n) - 1
    return dfs(initial_mask, 0)

def _count_triangles(adj, n):
    count = 0
    for i in range(n):
        for j in adj[i]:
            if j > i:
                count += len(adj[i] & adj[j])
    return count // 3

def construct():
    random.seed(42)
    
    # Try to construct a graph on 40 vertices
    n_target = 40
    
    # Use a spectral/Paley-type construction as initial guess
    # Then use simulated annealing to fix triangles while keeping alpha < 10
    
    def _make_paley_like(n):
        """Create a Paley-type graph: edge (i,j) if (j-i) is a quadratic residue mod n
        or if n is not prime, use a simple heuristic."""
        adj = [set() for _ in range(n)]
        # For simplicity, use a random triangle-free initialization
        # then improve with SA
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.25:
                    adj[i].add(j)
                    adj[j].add(i)
        return adj
    
    def _remove_all_triangles(adj, n):
        """Iteratively remove edges that are part of triangles."""
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in list(adj[i]):
                    if j <= i:
                        continue
                    if adj[i] & adj[j]:
                        adj[i].remove(j)
                        adj[j].remove(i)
                        changed = True
        return adj
    
    best_graph = None
    
    for attempt in range(5):
        n = n_target
        adj = _make_paley_like(n)
        adj = _remove_all_triangles(adj, n)
        
        # Now try to add edges to reduce alpha while maintaining triangle-free
        # Use simulated annealing
        temperature = 1.0
        cooling = 0.9995
        iterations = 3000
        
        for it in range(iterations):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j or j in adj[i]:
                continue
            
            # Check if adding (i,j) creates a triangle
            if adj[i] & adj[j]:
                continue
            
            # Tentatively add edge
            adj[i].add(j)
            adj[j].add(i)
            
            if not _max_independent_set_size(adj, n, 10):
                # Keep the edge
                temperature *= cooling
            else:
                # Revert
                adj[i].remove(j)
                adj[j].remove(i)
                temperature *= cooling
        
        if not _max_independent_set_size(adj, n, 10):
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if j > i:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
    
    # Fallback to 39 vertices
    n = 39
    adj = _make_paley_like(n)
    adj = _remove_all_triangles(adj, n)
    
    temperature = 1.0
    cooling = 0.9995
    for it in range(3000):
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j or j in adj[i]:
            continue
        if adj[i] & adj[j]:
            continue
        adj[i].add(j)
        adj[j].add(i)
        if not _max_independent_set_size(adj, n, 10):
            temperature *= cooling
        else:
            adj[i].remove(j)
            adj[j].remove(i)
            temperature *= cooling
    
    if not _max_independent_set_size(adj, n, 10):
        edges = []
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
    
    # Fallback: Andrasfai graph on 26 vertices
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
