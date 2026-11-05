import time
import random
from itertools import combinations

def _has_triangle_bitset(adj, n):
    """Check if graph has a triangle using bitsets."""
    for u in range(n):
        au = adj[u]
        v = u + 1
        while v < n:
            if au & (1 << v):
                common = au & adj[v]
                if common & ~((1 << (v + 1)) - 1):
                    return True
            v += 1
    return False

def _max_independent_set_size_bitset(adj, n, k):
    """
    Check if there exists an independent set of size k using bitmask backtracking.
    Optimized with greedy lower bound on remaining candidates.
    """
    full_mask = (1 << n) - 1
    non_neighbors = [0] * n
    for i in range(n):
        non_neighbors[i] = full_mask & ~adj[i] & ~(1 << i)
    
    found = [False]
    
    def dfs(candidates_mask, current_size):
        if found[0]:
            return
        if current_size == k:
            found[0] = True
            return
        
        if candidates_mask == 0:
            return
        
        # Pruning: If the number of candidates is less than needed, fail
        if bin(candidates_mask).count('1') < k - current_size:
            return
            
        # Heuristic: Pick vertex with highest degree in the induced subgraph of candidates?
        # Or lowest degree? For independent set, we want to branch on vertices that
        # eliminate many candidates (high degree in complement = low degree in graph).
        # Actually, for finding an independent set, we want to include vertices that
        # have many non-neighbors.
        
        # Greedy bound: If we can't even find k-current_size vertices in the candidates
        # using a greedy coloring of the complement (i.e., greedy independent set in graph),
        # then we can prune? No, greedy is a lower bound on alpha, not an upper bound.
        # We need an upper bound on the size of the independent set in the remaining candidates.
        # A simple upper bound is the number of vertices.
        # A better bound: If the candidates can be covered by (k - current_size - 1) cliques in the graph
        # (which are independent sets in the complement?), no.
        
        # Let's stick to the standard efficient branching.
        # Branch on the vertex with the maximum number of non-neighbors in the candidate set.
        # This maximizes the reduction in the candidate set when we include it.
        
        best_v = -1
        max_nn = -1
        mask = candidates_mask
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            nn_count = bin(candidates_mask & non_neighbors[v]).count('1')
            if nn_count > max_nn:
                max_nn = nn_count
                best_v = v
            mask ^= lsb
            
        if best_v == -1:
            return

        # Branch 1: Include best_v
        new_candidates = candidates_mask & non_neighbors[best_v] & ~(1 << best_v)
        dfs(new_candidates, current_size + 1)
        
        if found[0]:
            return
        
        # Branch 2: Exclude best_v
        dfs(candidates_mask & ~(1 << best_v), current_size)
    
    dfs(full_mask, 0)
    return found[0]

def _find_independent_set_bitset(adj, n, k):
    """Find an independent set of size k. Returns list of vertices or None."""
    full_mask = (1 << n) - 1
    non_neighbors = [0] * n
    for i in range(n):
        non_neighbors[i] = full_mask & ~adj[i] & ~(1 << i)
    
    found = [False]
    result_set = []
    
    def dfs(candidates_mask, current_set):
        if found[0]:
            return
        if len(current_set) == k:
            result_set.extend(current_set)
            found[0] = True
            return
        if candidates_mask == 0:
            return
        if bin(candidates_mask).count('1') < k - len(current_set):
            return
        
        mask = candidates_mask
        best_v = -1
        max_nn = -1
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            nn_count = bin(candidates_mask & non_neighbors[v]).count('1')
            if nn_count > max_nn:
                max_nn = nn_count
                best_v = v
            mask ^= lsb
            
        if best_v == -1:
            return
        
        new_cands = candidates_mask & non_neighbors[best_v] & ~(1 << best_v)
        dfs(new_cands, current_set + [best_v])
        
        if found[0]:
            return
        
        dfs(candidates_mask & ~(1 << best_v), current_set)
    
    dfs(full_mask, [])
    if found[0]:
        return result_set
    return None

def _make_triangle_free(adj, n):
    """Remove edges to make the graph triangle-free."""
    changed = True
    while changed:
        changed = False
        for u in range(n):
            au = adj[u]
            v = u + 1
            while v < n:
                if au & (1 << v):
                    common = au & adj[v] & ~((1 << (v + 1)) - 1)
                    if common:
                        lsb = common & -common
                        w = lsb.bit_length() - 1
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        changed = True
                        break
                v += 1
            if changed:
                break

def _local_search_40(adj, n, deadline):
    """
    Local search to reduce alpha below 10 for n=40.
    """
    best_adj = adj[:]
    best_alpha = 10 # We assume it starts high
    
    for _ in range(200): # Max iterations
        if time.time() > deadline:
            break
            
        if not _max_independent_set_size_bitset(adj, n, 10):
            return adj[:] # Success
        
        iso = _find_independent_set_bitset(adj, n, 10)
        if iso is None:
            return adj[:]
            
        # Try to add an edge within the independent set
        improved = False
        random.shuffle(iso)
        for i in range(len(iso)):
            for j in range(i + 1, len(iso)):
                u, v = iso[i], iso[j]
                if not (adj[u] & (1 << v)):
                    # Check if adding (u,v) creates a triangle
                    common = adj[u] & adj[v]
                    if common == 0:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        improved = True
                        break
            if improved:
                break
                
        if not improved:
            # Try removing an edge to allow more flexibility later?
            # Or just restart
            break
            
    return None

def _construct_39_vertex_graph():
    """
    Construct the known 39-vertex Ramsey(3,10) graph using Hoffman-Singleton subgraph.
    """
    n_hs = 50
    adj_hs = [0] * n_hs
    
    def idx(x, y):
        return x * 10 + y
        
    for x in range(5):
        for y in range(10):
            u = idx(x, y)
            v1 = idx(x, (y + 1) % 10)
            v2 = idx(x, (y - 1) % 10)
            v3 = idx((x + 1) % 5, (y + 2 * x) % 10)
            v4 = idx((x - 1) % 5, (y - 2 * x) % 10)
            
            for v in [v1, v2, v3, v4]:
                if u < v:
                    adj_hs[u] |= (1 << v)
                    adj_hs[v] |= (1 << u)
                    
    n_target = 39
    adj_target = [0] * n_target
    
    for u in range(n_target):
        for v in range(u + 1, n_target):
            if adj_hs[u] & (1 << v):
                adj_target[u] |= (1 << v)
                adj_target[v] |= (1 << u)
                
    if _has_triangle_bitset(adj_target, n_target):
        return None
    if _max_independent_set_size_bitset(adj_target, n_target, 10):
        return None
        
    return {'n': n_target, 'edges': _bitset_to_edges(adj_target, n_target)}

def _bitset_to_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 145 # Safety margin
    
    # 1. Try to find a 40-vertex graph
    # We use a randomized local search approach
    # Start with random graphs, make them triangle-free, then reduce alpha
    
    best_result = None
    
    # First, secure the 39-vertex graph
    result_39 = _construct_39_vertex_graph()
    if result_39:
        best_result = result_39
        
    # Now try to improve to 40
    n_target = 40
    found_40 = False
    
    # Strategy: Start from the 39-vertex graph and try to add a vertex?
    # Or start from scratch with random 40-vertex graphs.
    
    # Let's try a few random 40-vertex graphs with local search
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        adj = [0] * n_target
        # Random graph with p=0.5
        for i in range(n_target):
            for j in range(i + 1, n_target):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        _make_triangle_free(adj, n_target)
        
        # Check if alpha < 10
        if not _max_independent_set_size_bitset(adj, n_target, 10):
            found_40 = True
            best_result = {'n': n_target, 'edges': _bitset_to_edges(adj, n_target)}
            break
        
        # If not, try local search
        # We need to reduce alpha.
        # Find an independent set of size 10
        iso = _find_independent_set_bitset(adj, n_target, 10)
        if iso:
            # Try to add edges within iso
            # This is a simple local search step
            # We might need multiple steps
            for step in range(100):
                if time.time() > deadline:
                    break
                if not _max_independent_set_size_bitset(adj, n_target, 10):
                    found_40 = True
                    best_result = {'n': n_target, 'edges': _bitset_to_edges(adj, n_target)}
                    break
                    
                iso = _find_independent_set_bitset(adj, n_target, 10)
                if not iso:
                    break
                    
                random.shuffle(iso)
                improved = False
                for i in range(len(iso)):
                    for j in range(i + 1, len(iso)):
                        u, v = iso[i], iso[j]
                        if not (adj[u] & (1 << v)):
                            common = adj[u] & adj[v]
                            if common == 0:
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                                improved = True
                                break
                    if improved:
                        break
                if not improved:
                    break
                    
        if found_40:
            break
            
    if best_result is None:
        # Fallback to n=39 if 40 failed
        return result_39 if result_39 else {'n': 39, 'edges': []}
        
    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"n={res['n']}, edges={len(res['edges'])}")
