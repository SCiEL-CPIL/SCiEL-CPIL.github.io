"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _has_k4(adj, n):
    """Check if graph contains a K4. adj is n x n bool matrix."""
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i,j]: continue
            for k in range(j+1, n):
                if not adj[i,k] or not adj[j,k]: continue
                # Check if any l > k is connected to i, j, k
                for l in range(k+1, n):
                    if adj[i,l] and adj[j,l] and adj[k,l]:
                        return True
    return False

def _has_indep_6(adj, n):
    """Check if graph has an independent set of size 6.
    Independent set = clique in complement graph."""
    # Complement: two vertices are adjacent in complement if NOT adjacent in original
    # We need to check if complement has a clique of size 6
    # Use backtracking with pruning
    comp_adj = ~adj
    np.fill_diagonal(comp_adj, False)
    
    # Branch and bound for clique of size 6
    found = [False]
    
    def _search(candidates, size):
        if found[0]:
            return
        if size == 6:
            found[0] = True
            return
        if len(candidates) < 6 - size:
            return
        # Try each candidate
        for idx in range(len(candidates)):
            v = candidates[idx]
            # New candidates: those after v in list that are connected to v in complement
            new_cands = [candidates[j] for j in range(idx+1, len(candidates)) if comp_adj[v, candidates[j]]]
            if size + len(new_cands) >= 6:
                _search(new_cands, size + 1)
                if found[0]:
                    return
    
    _search(list(range(n)), 0)
    return found[0]

def _is_valid(adj, n):
    """Check K4-free and no independent set of size 6."""
    if _has_k4(adj, n):
        return False
    if _has_indep_6(adj, n):
        return False
    return True

def _paley_graph(q):
    """Construct Paley graph on q vertices."""
    squares = set()
    for x in range(1, q):
        squares.add((x*x) % q)
    adj = np.zeros((q, q), dtype=bool)
    for i in range(q):
        for j in range(i+1, q):
            if (j - i) % q in squares:
                adj[i, j] = adj[j, i] = True
    return adj

def _try_extend(adj, n, max_attempts=100):
    """Try to add one vertex to the graph."""
    new_n = n + 1
    for attempt in range(max_attempts):
        new_adj = adj.copy()
        new_adj = np.zeros((new_n, new_n), dtype=bool)
        new_adj[:n, :n] = adj
        
        if attempt == 0:
            # Connect to even vertices
            for i in range(0, n, 2):
                new_adj[n, i] = new_adj[i, n] = True
        elif attempt == 1:
            # Connect to odd vertices
            for i in range(1, n, 2):
                new_adj[n, i] = new_adj[i, n] = True
        else:
            # Random connection pattern
            k = random.randint(n//4, 3*n//4)
            neighbors = random.sample(range(n), k)
            for i in neighbors:
                new_adj[n, i] = new_adj[i, n] = True
        
        # Quick check: no K4 involving new vertex
        ok = True
        for i in range(n):
            if not new_adj[n, i]: continue
            for j in range(i+1, n):
                if not new_adj[n, j] or not adj[i,j]: continue
                for k in range(j+1, n):
                    if new_adj[n, k] and adj[i,k] and adj[j,k]:
                        ok = False
                        break
                if not ok: break
            if not ok: break
        if not ok:
            continue
        
        # Full validation
        if _is_valid(new_adj, new_n):
            return new_adj
    return None

def _local_search(adj, n, deadline):
    """Local search to maximize n."""
    best_adj = adj.copy()
    best_n = n
    stall = 0
    
    while time.time() < deadline:
        # Try to extend
        new_adj = _try_extend(best_adj, best_n, max_attempts=50)
        if new_adj is not None:
            best_adj = new_adj
            best_n += 1
            stall = 0
            continue
        
        # Try edge flips
        if best_n < 2:
            break
        u, v = random.sample(range(best_n), 2)
        old = best_adj[u, v]
        best_adj[u, v] = best_adj[v, u] = not old
        
        if _is_valid(best_adj, best_n):
            stall += 1
            # Try extending after flip
            new_adj = _try_extend(best_adj, best_n, max_attempts=20)
            if new_adj is not None:
                best_adj = new_adj
                best_n += 1
                stall = 0
        else:
            best_adj[u, v] = best_adj[v, u] = old
            stall += 1
        
        if stall > 5000:
            break
    
    return best_adj, best_n

def construct():
    start = time.time()
    deadline = start + 135  # 15s margin
    
    best_adj = None
    best_n = 0
    
    # Generate candidate starting graphs
    starts = []
    
    # Paley graphs
    for q in [13, 17, 29]:
        if time.time() > deadline - 20:
            break
        adj = _paley_graph(q)
        if _is_valid(adj, q):
            starts.append((adj, q))
    
    # Random graphs
    if time.time() < deadline - 30:
        for n_try in [14, 16, 18, 20]:
            for p in [0.35, 0.45, 0.55, 0.65]:
                if time.time() > deadline - 15:
                    break
                adj = np.zeros((n_try, n_try), dtype=bool)
                for i in range(n_try):
                    for j in range(i+1, n_try):
                        if random.random() < p:
                            adj[i,j] = adj[j,i] = True
                if _is_valid(adj, n_try):
                    starts.append((adj, n_try))
    
    starts.sort(key=lambda x: -x[1])
    
    for adj, n in starts:
        if time.time() > deadline - 10:
            break
        new_adj, new_n = _local_search(adj, n, deadline)
        if new_n > best_n:
            best_n = new_n
            best_adj = new_adj
    
    # Fallback
    if best_adj is None:
        best_adj = _paley_graph(17)
        best_n = 17
    
    # Final check
    if not _is_valid(best_adj, best_n):
        best_adj = _paley_graph(17)
        best_n = 17
    
    edges = [[i, j] for i in range(best_n) for j in range(i+1, best_n) if best_adj[i, j]]
    return {'n': best_n, 'edges': edges}
# EVOLVE-BLOCK-END
