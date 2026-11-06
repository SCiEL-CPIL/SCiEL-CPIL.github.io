import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Find a 4-coloring of K_n with no monochromatic C6.
    Strategy:
    1. Try to find a coloring for n=17 (known to exist) or n=18 (new).
    2. Use local search: start with random coloring, iteratively fix monochromatic C6s.
    3. Incremental check: maintain a list of "bad" 6-subsets (those containing a mono-C6).
       When an edge is flipped, only re-check subsets containing that edge.
    """
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    # Precompute all 6-subsets for n up to 18
    # For n=18, C(18,6) = 18564. That's manageable.
    
    def get_all_6_subsets(n):
        """Yield all 6-subsets of {0, ..., n-1}."""
        import itertools
        return list(itertools.combinations(range(n), 6))
    
    # Cache 6-subsets per n
    subsets_cache = {}
    def get_subsets(n):
        if n not in subsets_cache:
            subsets_cache[n] = get_all_6_subsets(n)
        return subsets_cache[n]
    
    def edge_index(i, j, n):
        """Get the index of edge (i,j) in the flat color array (i<j)."""
        if i > j:
            i, j = j, i
        # Number of edges before row i: sum_{k=0}^{i-1} (n-1-k)
        # = i*(n-1) - i*(i-1)//2
        # Then add position j in row i: j - i - 1
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    def check_subset_mono_c6(subset, colors, n):
        """
        Check if the 6 vertices in subset contain a monochromatic C6.
        colors is a numpy array of shape (n*(n-1)//2,) with colors 0..3.
        Returns True if a monochromatic C6 exists, False otherwise.
        """
        verts = list(subset)
        # Build adjacency for each color
        for c in range(4):
            # Build adjacency matrix for these 6 vertices
            adj = np.zeros((6, 6), dtype=np.uint8)
            for a in range(6):
                for b in range(a + 1, 6):
                    u, v = verts[a], verts[b]
                    idx = edge_index(u, v, n)
                    if colors[idx] == c:
                        adj[a, b] = 1
                        adj[b, a] = 1
            # Check for Hamiltonian cycle in this 6-node graph
            if has_hamiltonian_cycle(adj):
                return True
        return False
    
    def has_hamiltonian_cycle(adj):
        """Check if a 6x6 adjacency matrix has a Hamiltonian cycle using bitmask DP."""
        n_nodes = adj.shape[0]
        if n_nodes < 3:
            return False
        
        # dp[mask][v] = True if there's a path visiting exactly vertices in mask, ending at v
        # Start from vertex 0
        full_mask = (1 << n_nodes) - 1
        dp = np.zeros((1 << n_nodes, n_nodes), dtype=bool)
        
        for v in range(n_nodes):
            if adj[0, v]:
                dp[1 | (1 << v), v] = True
        
        for mask in range(1, 1 << n_nodes):
            if not (mask & 1):  # Must include vertex 0
                continue
            for v in range(n_nodes):
                if not dp[mask, v]:
                    continue
                for w in range(n_nodes):
                    if adj[v, w] and not (mask & (1 << w)):
                        dp[mask | (1 << w), w] = True
        
        # Check if we can return to 0
        for v in range(n_nodes):
            if dp[full_mask, v] and adj[v, 0]:
                return True
        return False
    
    def initial_bad_subsets(n, colors):
        """Find all 6-subsets that contain a monochromatic C6."""
        subsets = get_subsets(n)
        bad = set()
        for subset in subsets:
            if check_subset_mono_c6(subset, colors, n):
                bad.add(subset)
        return bad
    
    def local_search(n, max_time=10.0):
        """
        Try to find a valid 4-coloring of K_n using local search.
        Returns colors array if found, None otherwise.
        """
        start = time.time()
        num_edges = n * (n - 1) // 2
        
        # Multiple restarts
        for restart in range(50):
            if time.time() - start > max_time:
                break
            
            # Random initialization
            colors = np.random.randint(0, 4, size=num_edges, dtype=np.int32)
            
            # Find initial bad subsets
            bad = initial_bad_subsets(n, colors)
            
            if len(bad) == 0:
                return colors
            
            # Local search loop
            for step in range(10000):
                if time.time() - start > max_time:
                    return None
                if time.time() > start_time + time_limit - 5:
                    return None
                
                if len(bad) == 0:
                    return colors
                
                # Pick a random bad subset
                subset = random.choice(list(bad))
                verts = list(subset)
                
                # Find the actual monochromatic C6 in this subset
                # We need to find which color has the C6 and which edges form it
                cycle_edges = None
                for c in range(4):
                    adj = np.zeros((6, 6), dtype=np.uint8)
                    for a in range(6):
                        for b in range(a + 1, 6):
                            u, v = verts[a], verts[b]
                            idx = edge_index(u, v, n)
                            if colors[idx] == c:
                                adj[a, b] = 1
                                adj[b, a] = 1
                    # Find Hamiltonian cycle edges
                    cycle = find_hamiltonian_cycle_edges(adj)
                    if cycle is not None:
                        cycle_edges = [(verts[a], verts[b]) for a, b in cycle]
                        break
                
                if cycle_edges is None:
                    # Stale bad subset, remove it
                    bad.discard(subset)
                    continue
                
                # Pick a random edge from the cycle to flip
                u, v = random.choice(cycle_edges)
                idx = edge_index(u, v, n)
                old_color = colors[idx]
                new_color = random.randint(0, 3)
                while new_color == old_color:
                    new_color = random.randint(0, 3)
                colors[idx] = new_color
                
                # Update bad sets
                # Remove the subset we fixed
                bad.discard(subset)
                
                # Re-check all subsets containing edge (u,v)
                # Precompute: for each edge, which 6-subsets contain it
                # This is expensive to compute on the fly, so let's do it smarter
                # Actually, let's just re-check all subsets that contain u and v
                subsets = get_subsets(n)
                for s in subsets:
                    if u in s and v in s:
                        if s in bad:
                            # Already known bad, re-check
                            pass  # We'll re-check below
                        if check_subset_mono_c6(s, colors, n):
                            bad.add(s)
                        else:
                            bad.discard(s)
        
        return None
    
    # Try n=18 first (new result), then n=17 (known), then fall back to smaller
    for target_n in [18, 17, 16, 15, 14, 13, 12, 11, 10, 9]:
        if time.time() > start_time + time_limit - 10:
            break
        remaining = time_limit - (time.time() - start_time) - 5
        if remaining < 1.0:
            break
        
        colors = local_search(target_n, max_time=remaining)
        if colors is not None:
            # Verify
            if verify_coloring(target_n, colors):
                flat_colors = colors.tolist()
                return {'n': target_n, 'colors': flat_colors}
    
    # Fallback: return the known n=9 coloring
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def verify_coloring(n, colors):
    """Verify that the coloring has no monochromatic C6."""
    num_edges = n * (n - 1) // 2
    if len(colors) != num_edges:
        return False
    
    for c in colors:
        if c < 0 or c > 3:
            return False
    
    # Check all 6-subsets
    import itertools
    for subset in itertools.combinations(range(n), 6):
        verts = list(subset)
        for color in range(4):
            adj = np.zeros((6, 6), dtype=np.uint8)
            for a in range(6):
                for b in range(a + 1, 6):
                    u, v = verts[a], verts[b]
                    if u > v:
                        u, v = v, u
                    idx = u * (n - 1) - u * (u - 1) // 2 + (v - u - 1)
                    if colors[idx] == color:
                        adj[a, b] = 1
                        adj[b, a] = 1
            if _has_hamiltonian_cycle(adj):
                return False
    return True

def _has_hamiltonian_cycle(adj):
    n_nodes = adj.shape[0]
    if n_nodes < 3:
        return False
    full_mask = (1 << n_nodes) - 1
    dp = np.zeros((1 << n_nodes, n_nodes), dtype=bool)
    for v in range(n_nodes):
        if adj[0, v]:
            dp[1 | (1 << v), v] = True
    for mask in range(1, 1 << n_nodes):
        if not (mask & 1):
            continue
        for v in range(n_nodes):
            if not dp[mask, v]:
                continue
            for w in range(n_nodes):
                if adj[v, w] and not (mask & (1 << w)):
                    dp[mask | (1 << w), w] = True
    for v in range(n_nodes):
        if dp[full_mask, v] and adj[v, 0]:
            return True
    return False

def find_hamiltonian_cycle_edges(adj):
    """Find a Hamiltonian cycle in a 6-node graph, return list of (a,b) edge pairs."""
    n_nodes = adj.shape[0]
    if n_nodes < 3:
        return None
    full_mask = (1 << n_nodes) - 1
    dp = np.zeros((1 << n_nodes, n_nodes), dtype=bool)
    parent = np.zeros((1 << n_nodes, n_nodes), dtype=np.int8) - 1
    
    for v in range(n_nodes):
        if adj[0, v]:
            dp[1 | (1 << v), v] = True
            parent[1 | (1 << v), v] = 0
    
    for mask in range(1, 1 << n_nodes):
        if not (mask & 1):
            continue
        for v in range(n_nodes):
            if not dp[mask, v]:
                continue
            for w in range(n_nodes):
                if adj[v, w] and not (mask & (1 << w)):
                    dp[mask | (1 << w), w] = True
                    parent[mask | (1 << w), w] = v
    
    for v in range(n_nodes):
        if dp[full_mask, v] and adj[v, 0]:
            # Reconstruct cycle
            path = []
            cur = v
            mask = full_mask
            while cur != 0:
                path.append(cur)
                prev = parent[mask, cur]
                if prev < 0:
                    break
                mask = mask & ~(1 << cur)
                cur = prev
            path.append(0)
            path.reverse()
            # path is 0 -> ... -> v, and v->0 closes the cycle
            edges = []
            for i in range(len(path) - 1):
                edges.append((path[i], path[i + 1]))
            edges.append((path[-1], path[0]))
            return edges
    return None
# EVOLVE-BLOCK-END
