import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

DEADLINE = time.time() + 270  # Safety margin

def _build_paley(q):
    """Return adjacency bitset list for Paley graph on q vertices (vertices 0..q-1)."""
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _has_k_clique(adj, n, k):
    """Check if graph has a clique of size k. Returns True if found."""
    # Simple backtracking search
    # For k=5, this is feasible for n=41
    
    def search(current_clique, candidates):
        if len(current_clique) == k:
            return True
        if len(candidates) < k - len(current_clique):
            return False
            
        for i, v in enumerate(candidates):
            # New candidates are those in candidates[i+1:] that are adjacent to v
            new_candidates = [u for u in candidates[i+1:] if (adj[v] & (1 << u))]
            if search(current_clique + [v], new_candidates):
                return True
        return False

    return search([], list(range(n)))

def _has_ind_clique(adj, n, k):
    """Check if graph has an independent set of size k. Returns True if found."""
    # Independent set in G is Clique in complement of G
    
    # Build complement adj
    comp_adj = [0] * n
    for i in range(n):
        # All bits set for n vertices
        all_bits = (1 << n) - 1
        # Remove self and neighbors
        comp_adj[i] = all_bits & ~(adj[i] | (1 << i))
        
    return _has_k_clique(comp_adj, n, k)

def _is_valid(adj, n, k=5):
    """Check if graph is (k,k)-free."""
    if _has_k_clique(adj, n, k):
        return False
    if _has_ind_clique(adj, n, k):
        return False
    return True

def _find_valid_graph(n, time_limit=5.0):
    """
    Try to find a (5,5)-free graph on n vertices using random repair.
    Returns adjacency list if found, None otherwise.
    """
    start_time = time.time()
    
    # Helper to find a specific clique/ind_set for repair
    def _find_k_clique(adj, n, k):
        def search(current_clique, candidates):
            if len(current_clique) == k:
                return current_clique
            if len(candidates) < k - len(current_clique):
                return None
                
            for i, v in enumerate(candidates):
                new_candidates = [u for u in candidates[i+1:] if (adj[v] & (1 << u))]
                res = search(current_clique + [v], new_candidates)
                if res:
                    return res
            return None

        return search([], list(range(n)))

    def _find_ind_set(adj, n, k):
        comp_adj = [0] * n
        for i in range(n):
            all_bits = (1 << n) - 1
            comp_adj[i] = all_bits & ~(adj[i] | (1 << i))
        return _find_k_clique(comp_adj, n, k)

    while time.time() - start_time < time_limit:
        # Initialize with random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        # Repair loop
        for _ in range(1000): # Inner loop limit
            if time.time() - start_time > time_limit:
                return None
                
            if _is_valid(adj, n, 5):
                return adj
                
            if _has_k_clique(adj, n, 5):
                clique = _find_k_clique(adj, n, 5)
                if clique:
                    u, v = random.sample(clique, 2)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            elif _has_ind_clique(adj, n, 5):
                ind_set = _find_ind_set(adj, n, 5)
                if ind_set:
                    u, v = random.sample(ind_set, 2)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            else:
                break
    return None

def construct():
    # Fallback: Paley(37)
    q = 37
    adj37 = _build_paley(q)
    edges37 = []
    for i in range(q):
        for j in range(i + 1, q):
            if adj37[i] & (1 << j):
                edges37.append((i, j))
    best_n = 37
    best_edges = edges37
    
    # Try to find a (5,5)-free graph on n vertices for n = 21, 20, 19
    # If found, construct 2n vertex graph (disjoint union)
    
    for target_base_n in [21, 20, 19]:
        if time.time() > DEADLINE - 10:
            break
            
        adj_base = _find_valid_graph(target_base_n, time_limit=5.0)
        
        if adj_base is not None:
            n_new = 2 * target_base_n
            adj_new = [0] * n_new
            
            for i in range(target_base_n):
                # First copy
                adj_new[i] = adj_base[i]
                
                # Second copy
                shifted_adj = 0
                for j in range(target_base_n):
                    if adj_base[i] & (1 << j):
                        shifted_adj |= (1 << (j + target_base_n))
                adj_new[i + target_base_n] = shifted_adj
                
            if _is_valid(adj_new, n_new, 5):
                best_n = n_new
                best_edges = []
                for i in range(n_new):
                    for j in range(i + 1, n_new):
                        if adj_new[i] & (1 << j):
                            best_edges.append((i, j))
                return best_n, best_edges
        
    return best_n, best_edges

# EVOLVE-BLOCK-END
