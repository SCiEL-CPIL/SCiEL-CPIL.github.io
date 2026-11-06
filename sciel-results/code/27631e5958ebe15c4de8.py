import time
import random
import itertools

_START = time.time()
_DEADLINE = 140.0  # safety margin

def _elapsed():
    return time.time() - _START

def _build_adj_from_edges(n, edges):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _has_k4(adj, n):
    """Check for K4 using bitmasks. Returns None if no K4 found."""
    for i in range(n):
        for j in range(i+1, n):
            if not (adj[i] >> j) & 1:
                continue
            common = adj[i] & adj[j] & ~((1 << (j+1)) - 1)
            if common == 0:
                continue
            # Get list of common neighbors
            bits = []
            c = common
            while c:
                lsb = c & -c
                bits.append(lsb.bit_length() - 1)
                c ^= lsb
            for a_idx in range(len(bits)):
                a = bits[a_idx]
                for b_idx in range(a_idx+1, len(bits)):
                    b = bits[b_idx]
                    if (adj[a] >> b) & 1:
                        return True
    return False

def _has_ind_6(adj, n):
    """Check for independent set of size 6. Returns True if found."""
    all_mask = (1 << n) - 1
    comp = [all_mask & ~(adj[i] | (1 << i)) for i in range(n)]
    
    def backtrack(candidates, current):
        if len(current) == 6:
            return True
        if candidates.bit_count() < 6 - len(current):
            return False
        
        # Pick vertex with minimum degree in induced subgraph
        best_v = -1
        best_cnt = n + 1
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            cnt = (candidates & comp[v]).bit_count()
            if cnt < best_cnt:
                best_cnt = cnt
                best_v = v
            c ^= lsb
        
        if best_v == -1:
            return False
        
        v = best_v
        if backtrack(candidates & comp[v], current + [v]):
            return True
        if backtrack(candidates & ~(1 << v), current):
            return True
        return False
    
    return backtrack(all_mask, [])

def _is_valid(n, adj):
    if _has_k4(adj, n):
        return False
    if _has_ind_6(adj, n):
        return False
    return True

def _schlaefli_edges():
    """Return edges of the Schläfli graph (35 vertices)."""
    edges = []
    # The Schläfli graph is a 16-regular graph on 35 vertices.
    # It can be constructed from the 7x7 grid with diagonals, or via the
    # Clebsch graph / other constructions.
    # Here we use a known adjacency: vertices (i,j) for i,j in 0..6,
    # adjacent if (i-j) mod 7 in {1,2,3,4} OR (i+j) mod 7 in {1,2,3,4}
    # Wait, that's not quite right. Let me use a different known construction.
    # 
    # Actually, the Schläfli graph has 35 vertices. One construction:
    # Label vertices by (a,b) where a,b in Z_7. Two vertices (a,b) and (c,d)
    # are adjacent if (a-c, b-d) is a nonzero vector in Z_7^2 that is NOT a
    # multiple of (1,0), (0,1), (1,1), (1,-1) by a nonzero scalar? 
    # That's getting complicated. Let me use a simpler known adjacency.
    #
    # I'll hardcode the Schläfli graph edges. The Schläfli graph is strongly
    # regular with parameters (35, 16, 6, 8). Each vertex has 16 neighbors.
    #
    # Let me use the construction based on the 7x7 grid with "knight's move"
    # type adjacency. Actually, let me just use a known valid graph.
    #
    # For simplicity and reliability, I'll use the Clebsch graph for n=16
    # and for n=35 I'll try to construct it properly.
    #
    # The Schläfli graph: vertices are the 35 edges of K_7. Two vertices
    # (edges of K_7) are adjacent if they share an endpoint or if the two
    # edges form a "parallel" pair (i.e., the four endpoints form a 4-cycle
    # in a specific way). Actually, the Schläfli graph is the complement of
    # the line graph of K_7 minus the 1-factor edges... 
    #
    # Let me just use a simpler approach: I'll generate random graphs and
    # use local search. The Schläfli graph is hard to construct from memory.
    
    # Actually, I know a construction: the Schläfli graph is the graph on
    # the 35 points of the projective plane of order 6... no, that's order 5.
    # 
    # Let me use this: vertices (i,j) for i in 0..6, j in 0..6 (49 vertices),
    # but we need 35. 
    #
    # OK, I'll just use local search for n=35 and hope it works.
    # For now, return empty for the Schläfli graph and rely on local search.
    return []

def _try_local_search(n, adj, time_limit, seed=None):
    """Local search to find a valid K4-free graph with no independent 6-set."""
    if seed is not None:
        random.seed(seed)
    start = time.time()
    adj = list(adj)
    
    while time.time() - start < time_limit:
        # Check K4
        if _has_k4(adj, n):
            # Find a K4 and remove an edge
            k4 = _find_k4(adj, n)
            if k4:
                pairs = [(k4[0], k4[1]), (k4[0], k4[2]), (k4[0], k4[3]),
                         (k4[1], k4[2]), (k4[1], k4[3]), (k4[2], k4[3])]
                u, v = random.choice(pairs)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                continue
            break
        
        # Check Ind 6
        if _has_ind_6(adj, n):
            ind6 = _find_ind_6(adj, n)
            if ind6:
                pairs = [(ind6[i], ind6[j]) for i in range(6) for j in range(i+1, 6)]
                u, v = random.choice(pairs)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                continue
            break
        
        # Valid!
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges
    
    return None

def _find_k4(adj, n):
    for i in range(n):
        for j in range(i+1, n):
            if not (adj[i] >> j) & 1:
                continue
            common = adj[i] & adj[j] & ~((1 << (j+1)) - 1)
            bits = []
            c = common
            while c:
                lsb = c & -c
                bits.append(lsb.bit_length() - 1)
                c ^= lsb
            for a_idx in range(len(bits)):
                a = bits[a_idx]
                for b_idx in range(a_idx+1, len(bits)):
                    b = bits[b_idx]
                    if (adj[a] >> b) & 1:
                        return [i, j, a, b]
    return None

def _find_ind_6(adj, n):
    all_mask = (1 << n) - 1
    comp = [all_mask & ~(adj[i] | (1 << i)) for i in range(n)]
    
    def backtrack(candidates, current):
        if len(current) == 6:
            return current[:]
        if candidates.bit_count() < 6 - len(current):
            return None
        
        best_v = -1
        best_cnt = n + 1
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            cnt = (candidates & comp[v]).bit_count()
            if cnt < best_cnt:
                best_cnt = cnt
                best_v = v
            c ^= lsb
        
        if best_v == -1:
            return None
        
        v = best_v
        res = backtrack(candidates & comp[v], current + [v])
        if res:
            return res
        res = backtrack(candidates & ~(1 << v), current)
        if res:
            return res
        return None
    
    return backtrack(all_mask, [])

def _random_adj(n, p):
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def construct():
    best_n = 1
    best_edges = []
    
    # Try from n=35 down to n=18
    for n in range(35, 17, -1):
        if _elapsed() > _DEADLINE - 10:
            break
        
        # Time allocation
        if n >= 35:
            time_slice = 35.0
        elif n >= 34:
            time_slice = 20.0
        elif n >= 33:
            time_slice = 12.0
        elif n >= 32:
            time_slice = 7.0
        else:
            time_slice = 4.0
        
        if _elapsed() + time_slice > _DEADLINE - 5:
            time_slice = _DEADLINE - 5 - _elapsed()
            if time_slice < 1.0:
                break
        
        t_start = time.time()
        found = False
        restarts = 5 if n < 33 else 3
        
        for r in range(restarts):
            if time.time() - t_start > time_slice:
                break
            rem = time_slice - (time.time() - t_start)
            if rem < 0.5:
                break
            
            p = 0.4 + 0.05 * r
            adj_init = _random_adj(n, p)
            edges = _try_local_search(n, adj_init, rem, seed=r*1000+n)
            if edges is not None:
                best_n = n
                best_edges = edges
                found = True
                break
        
        if found:
            return {'n': best_n, 'edges': best_edges}
    
    # Fallback
    if best_n < 17:
        n = 20
        adj_init = _random_adj(n, 0.5)
        edges = _try_local_search(n, adj_init, max(1.0, _DEADLINE - 5 - _elapsed()))
        if edges:
            best_n = n
            best_edges = edges
        else:
            best_n = 5
            best_edges = []
    
    return {'n': best_n, 'edges': best_edges}
