"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

DEADLINE = 325.0  # safety margin before 330s kill

def _has_clique_k(adj, k, candidates):
    """Check if there's a clique of size k among candidates using adj (set of sets)."""
    if len(candidates) < k:
        return False
    candidates = list(candidates)
    if k == 1:
        return len(candidates) >= 1
    if k == 2:
        for i in range(len(candidates)):
            for j in range(i+1, len(candidates)):
                if candidates[j] in adj[candidates[i]]:
                    return True
        return False
    if k == 3:
        for i in range(len(candidates)):
            for j in range(i+1, len(candidates)):
                if candidates[j] in adj[candidates[i]]:
                    for m in range(j+1, len(candidates)):
                        if candidates[m] in adj[candidates[i]] and candidates[m] in adj[candidates[j]]:
                            return True
        return False
    if k == 4:
        # Check all 4-subsets
        for combo in itertools.combinations(candidates, k):
            ok = True
            for i in range(k):
                for j in range(i+1, k):
                    if combo[j] not in adj[combo[i]]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return True
        return False
    return False

def _has_indep_k(adj, n_verts, k, candidates):
    """Check if there's an independent set of size k among candidates."""
    if len(candidates) < k:
        return False
    candidates = list(candidates)
    if k == 1:
        return len(candidates) >= 1
    if k == 2:
        for i in range(len(candidates)):
            for j in range(i+1, len(candidates)):
                if candidates[j] not in adj[candidates[i]]:
                    return True
        return False
    if k == 3:
        for i in range(len(candidates)):
            for j in range(i+1, len(candidates)):
                if candidates[j] not in adj[candidates[i]]:
                    for m in range(j+1, len(candidates)):
                        if candidates[m] not in adj[candidates[i]] and candidates[m] not in adj[candidates[j]]:
                            return True
        return False
    if k == 4:
        for combo in itertools.combinations(candidates, k):
            ok = True
            for i in range(k):
                for j in range(i+1, k):
                    if combo[j] in adj[combo[i]]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return True
        return False
    return False

def _is_55_free(adj, n):
    """Check if graph on n vertices has no K5 and no independent 5-set."""
    for i in range(n):
        # Check K5 containing i: need K4 in neighbors of i
        if _has_clique_k(adj, 4, adj[i]):
            return False
        # Check independent 5-set containing i: need independent 4-set in non-neighbors of i
        non_neighbors = set(range(n)) - adj[i] - {i}
        if _has_indep_k(adj, n, 4, non_neighbors):
            return False
    return True

def _try_add_vertex(adj, n, new_v):
    """Try to find a neighborhood for new vertex new_v (index n) that keeps graph (5,5)-free.
    Returns the set of neighbors, or None if impossible."""
    # The new vertex's neighborhood N must satisfy:
    # 1. No K4 in N (otherwise K5 with new_v)
    # 2. No independent 4-set in (existing vertices not in N) (otherwise independent 5-set with new_v)
    # We search for a valid N.
    
    # Start with empty neighborhood and try to add vertices
    # Or start with full and remove
    
    # Strategy: try several random neighborhoods
    best_n = None
    
    # First try: no edges to new vertex (independent vertex)
    # Check: no independent 5-set. Non-neighbors = all existing vertices.
    # Need no independent 4-set in all existing vertices.
    if not _has_indep_k(adj, n, 4, range(n)):
        return set()
    
    # Try: all edges to new vertex
    # Need no K4 in all existing vertices
    if not _has_clique_k(adj, 4, range(n)):
        return set(range(n))
    
    # Try random subsets
    for attempt in range(200):
        if time.time() > DEADLINE:
            break
        # Random neighborhood
        N = set()
        for v in range(n):
            if random.random() < 0.5:
                N.add(v)
        
        # Check constraints
        if _has_clique_k(adj, 4, N):
            continue
        non_N = set(range(n)) - N
        if _has_indep_k(adj, n, 4, non_N):
            continue
        
        return N
    
    return None

def _extend_graph(adj, n, max_add=5, seed=None):
    """Try to extend the graph by adding vertices one at a time."""
    if seed is not None:
        random.seed(seed)
    
    for add_idx in range(max_add):
        if time.time() > DEADLINE:
            break
        new_v = n + add_idx
        N = _try_add_vertex(adj, n, new_v)
        if N is None:
            break
        # Add the new vertex
        adj[new_v] = set(N)
        for v in N:
            adj[v].add(new_v)
        n += 1
    
    return n

def _paley_graph_37():
    """Construct the Paley graph on 37 vertices."""
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [set() for _ in range(q)]
    for i in range(q):
        for j in range(i+1, q):
            if (j - i) % q in residues:
                adj[i].add(j)
                adj[j].add(i)
    return adj, q

def construct():
    start_time = time.time()
    
    # Start with Paley graph on 37 vertices
    adj, n = _paley_graph_37()
    
    best_n = n
    best_adj = [s.copy() for s in adj]
    
    # Try multiple random extensions
    for seed in range(500):
        if time.time() > DEADLINE:
            break
        
        adj_copy = [s.copy() for s in best_adj]
        n_copy = best_n
        
        # Try to extend
        n_new = _extend_graph(adj_copy, n_copy, max_add=8, seed=seed)
        
        if n_new > best_n:
            best_n = n_new
            best_adj = [s.copy() for s in adj_copy]
            print(f"Extended to {n_new} vertices with seed {seed}")
    
    # If we couldn't extend, try a different approach: start from scratch with greedy
    if best_n == 37:
        # Try greedy construction from scratch
        for seed in range(200):
            if time.time() > DEADLINE:
                break
            random.seed(seed)
            n_greedy = 37
            adj_greedy = [set() for _ in range(n_greedy)]
            # Use Paley as base
            for i in range(n_greedy):
                for j in range(i+1, n_greedy):
                    if (j - i) % 37 in {(x*x) % 37 for x in range(1, 37)}:
                        adj_greedy[i].add(j)
                        adj_greedy[j].add(i)
            
            n_greedy = _extend_graph(adj_greedy, n_greedy, max_add=10, seed=seed)
            if n_greedy > best_n:
                best_n = n_greedy
                best_adj = [s.copy() for s in adj_greedy]
    
    # Build edge list
    edges = []
    for i in range(best_n):
        for j in adj_best[i]:
            if i < j:
                edges.append((i, j))
    
    return best_n, edges

# Fix: use best_adj not adj_best
def construct():
    start_time = time.time()
    
    # Start with Paley graph on 37 vertices
    adj, n = _paley_graph_37()
    
    best_n = n
    best_adj = [s.copy() for s in adj]
    
    # Try multiple random extensions
    for seed in range(500):
        if time.time() > DEADLINE:
            break
        
        adj_copy = [s.copy() for s in best_adj]
        n_copy = best_n
        
        # Try to extend
        n_new = _extend_graph(adj_copy, n_copy, max_add=8, seed=seed)
        
        if n_new > best_n:
            best_n = n_new
            best_adj = [s.copy() for s in adj_copy]
    
    # Build edge list
    edges = []
    for i in range(best_n):
        for j in best_adj[i]:
            if i < j:
                edges.append((i, j))
    
    return best_n, edges
# EVOLVE-BLOCK-END
