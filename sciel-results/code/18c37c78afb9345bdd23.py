import time
import random
from itertools import combinations

def _has_k_clique(adj, n, k):
    """
    Returns a clique of size k if found, else None.
    adj: list of ints (bitmasks)
    """
    if k == 1:
        for i in range(n):
            return [i] # Any vertex is a K1
    if k == 0:
        return []
    
    def _find_clique(candidates, k, path):
        if k == 1:
            # Return the first available vertex
            if candidates:
                lsb = candidates & -candidates
                v = lsb.bit_length() - 1
                return path + [v]
            return None
        
        if candidates == 0:
            return None
            
        # Heuristic: pick the vertex with the most neighbors in candidates? 
        # Or just the lowest index.
        # Let's pick the lowest index for simplicity, or random?
        # Random might help escape local minima? No, this is deterministic finding.
        # We just need to find ANY clique.
        
        lsb = candidates & -candidates
        v = lsb.bit_length() - 1
        
        # Option 1: Include v
        nbrs = adj[v] & candidates
        res = _find_clique(nbrs, k - 1, path + [v])
        if res:
            return res
            
        # Option 2: Exclude v
        candidates_without_v = candidates & ~lsb
        res = _find_clique(candidates_without_v, k, path)
        if res:
            return res
            
        return None

    full_mask = (1 << n) - 1
    return _find_clique(full_mask, k, [])

def _has_k_independent_set(adj, n, k):
    """
    Returns an independent set of size k if found, else None.
    Independent set in G is a clique in complement of G.
    """
    full_mask = (1 << n) - 1
    comp_adj = [full_mask ^ (1 << i) ^ adj[i] for i in range(n)]
    return _has_k_clique(comp_adj, n, k)

def _is_valid(adj, n, k=5):
    if _has_k_clique(adj, n, k):
        return False
    if _has_k_independent_set(adj, n, k):
        return False
    return True

def _local_search(n, max_time, seed=42):
    """
    Attempt to find a valid (5,5)-free graph on n vertices using local search.
    """
    random.seed(seed)
    start_time = time.time()
    deadline = start_time + max_time
    
    # Try multiple restarts
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        # Initialize random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Local search loop
        stuck_count = 0
        seen_states = set()
        
        while time.time() < deadline:
            # Check for violations
            clique = _has_k_clique(adj, n, 5)
            if clique:
                # Remove a random edge from the clique
                u, v = random.sample(clique, 2)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                stuck_count = 0
                continue
            
            ind_set = _has_k_independent_set(adj, n, 5)
            if ind_set:
                # Add a random edge between two non-adjacent vertices in the independent set
                # All pairs in an independent set are non-adjacent
                u, v = random.sample(ind_set, 2)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                stuck_count = 0
                continue
            
            # No violations found!
            # Verify one last time to be sure
            if _is_valid(adj, n):
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append((i, j))
                return n, edges
                
            # If we get here, it means _is_valid returned False but we didn't find a specific violation?
            # This shouldn't happen if _has_k_clique and _has_k_independent_set are correct.
            # But if it does, break and restart.
            break
            
    return None, None

def _paley_37():
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    return q, edges

def _paley_41():
    q = 41
    residues = {(x * x) % q for x in range(1, q)}
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    return q, edges

def _build_adjacency(n, edges):
    adj = [0] * n
    for i, j in edges:
        adj[i] |= (1 << j)
        adj[j] |= (1 << i)
    return adj

def construct():
    # 1. Try to find a 42-vertex graph via local search
    # We allocate most of the time to this.
    n_42, edges_42 = _local_search(42, max_time=250, seed=12345)
    if n_42 is not None:
        return n_42, edges_42
        
    # 2. Try Paley(41) - though it's likely invalid, it's fast to check
    n_41, edges_41 = _paley_41()
    adj_41 = _build_adjacency(n_41, edges_41)
    if _is_valid(adj_41, n_41):
        return n_41, edges_41
        
    # 3. Try to find a 41-vertex graph via local search with remaining time
    # This is unlikely to succeed if 42 failed, but 41 is easier.
    # However, if 42 failed due to time, 41 might also fail.
    # Let's try a quick search for 41 if we have time.
    # Actually, if 42 failed, it's probably because the search space is hard.
    # Paley(37) is guaranteed valid.
    
    # Let's try a quick local search for 41 with a short timeout
    n_41_ls, edges_41_ls = _local_search(41, max_time=30, seed=999)
    if n_41_ls is not None:
        return n_41_ls, edges_41_ls
        
    # 4. Fallback to Paley(37)
    n_37, edges_37 = _paley_37()
    return n_37, edges_37
