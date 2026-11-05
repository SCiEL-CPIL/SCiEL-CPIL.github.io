import time
import random
from itertools import combinations

_START = time.time()
_DEADLINE = 140  # seconds, leave 10s margin for the 150s limit

def _elapsed():
    return time.time() - _START

def _has_k4(adj, n):
    """Check if graph has a K4. Returns the K4 vertices if found, else None."""
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                continue
            # Common neighbors of i and j, with index > j
            common = adj[i] & adj[j] & ~((1 << (j + 1)) - 1)
            if common == 0:
                continue
            # Extract bits
            bits = []
            c = common
            while c:
                lsb = c & -c
                idx = lsb.bit_length() - 1
                bits.append(idx)
                c ^= lsb
            for k_idx in range(len(bits)):
                k = bits[k_idx]
                for l_idx in range(k_idx + 1, len(bits)):
                    l = bits[l_idx]
                    if adj[k] & (1 << l):
                        return [i, j, k, l]
    return None

def _has_ind_6(adj, n):
    """Check if graph has an independent set of size 6. Returns the set if found, else None."""
    def backtrack(candidates, current_set):
        if len(current_set) == 6:
            return current_set[:]
        if candidates.bit_count() < 6 - len(current_set):
            return None
        
        # Pick a vertex with fewest candidates after including it (to prune better)
        # Heuristic: pick vertex that minimizes |candidates \ N(v)|
        best_v = -1
        best_count = -1
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            new_cand = candidates & ~adj[v]
            cnt = new_cand.bit_count()
            if best_count == -1 or cnt < best_count:
                best_count = cnt
                best_v = v
                if cnt == 0 and len(current_set) < 5:
                    # If including v leaves no candidates, we can't reach 6
                    # but we still need to try excluding
                    pass
            c ^= lsb
        
        if best_v == -1:
            return None
            
        v = best_v
        new_candidates = candidates & ~((1 << v) | adj[v])
        
        # Branch 1: Include v
        res = backtrack(new_candidates, current_set + [v])
        if res:
            return res
        
        # Branch 2: Exclude v
        res = backtrack(candidates & ~(1 << v), current_set)
        if res:
            return res
        
        return None

    return backtrack((1 << n) - 1, [])

def _paley_graph(p):
    """Construct the Paley graph on p vertices."""
    if p % 4 != 1:
        return None
    squares = {x * x % p for x in range(1, p)}
    adj = [0] * p
    edges = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append([i, j])
    return adj, edges

def _count_k4s(adj, n):
    """Count number of K4s in the graph."""
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                continue
            common = adj[i] & adj[j] & ~((1 << (j + 1)) - 1)
            if common == 0:
                continue
            bits = []
            c = common
            while c:
                lsb = c & -c
                idx = lsb.bit_length() - 1
                bits.append(idx)
                c ^= lsb
            for k_idx in range(len(bits)):
                k = bits[k_idx]
                for l_idx in range(k_idx + 1, len(bits)):
                    l = bits[l_idx]
                    if adj[k] & (1 << l):
                        count += 1
    return count

def _count_ind_6s(adj, n):
    """Count number of independent 6-sets in the graph (upper bound via sampling or exact for small n)."""
    # For efficiency, we'll just check existence and return 1 or 0
    # But for local search, we might want a measure. Let's just use existence.
    return 1 if _has_ind_6(adj, n) else 0

def _try_local_search(n, time_limit, seed=None):
    """Try to find a valid graph on n vertices using local search."""
    if seed is not None:
        random.seed(seed)
    start = time.time()
    
    # Initialize random graph with edge probability tuned for K4-free
    # For K4-free, we want low density. For no independent 6, we want high density.
    # Balance: p=0.3 to 0.5
    p_edge = 0.4
    
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p_edge:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    best_valid = None
    
    # Local search
    while time.time() - start < time_limit:
        # Check for K4
        k4 = _has_k4(adj, n)
        if k4:
            # Remove a random edge from the K4
            pairs = [(k4[0], k4[1]), (k4[0], k4[2]), (k4[0], k4[3]),
                     (k4[1], k4[2]), (k4[1], k4[3]), (k4[2], k4[3])]
            u, v = random.choice(pairs)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            continue
        
        # Check for Ind 6
        ind6 = _has_ind_6(adj, n)
        if ind6:
            # Add a random edge between two vertices in the independent set
            pairs = [(ind6[i], ind6[j]) for i in range(6) for j in range(i + 1, 6)]
            u, v = random.choice(pairs)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            continue
        
        # If no K4 and no Ind 6, we found a valid graph
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges
    
    return None

def _try_extend_from_paley(base_n, target_n, time_limit):
    """Try to extend a Paley graph to a larger target_n by adding vertices and edges."""
    start = time.time()
    
    # Start with Paley graph
    p = base_n
    squares = {x * x % p for x in range(1, p)}
    adj = [0] * target_n
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # For new vertices, randomly connect to existing ones
    for i in range(p, target_n):
        for j in range(i):
            if random.random() < 0.4:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Local search
    while time.time() - start < time_limit:
        k4 = _has_k4(adj, target_n)
        if k4:
            pairs = [(k4[0], k4[1]), (k4[0], k4[2]), (k4[0], k4[3]),
                     (k4[1], k4[2]), (k4[1], k4[3]), (k4[2], k4[3])]
            u, v = random.choice(pairs)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            continue
        
        ind6 = _has_ind_6(adj, target_n)
        if ind6:
            pairs = [(ind6[i], ind6[j]) for i in range(6) for j in range(i + 1, 6)]
            u, v = random.choice(pairs)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            continue
        
        edges = []
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges
    
    return None

def construct():
    # Base case: Paley 17
    p = 17
    squares = {x * x % p for x in range(1, p)}
    best_edges = [[i, j] for i in range(p) for j in range(i + 1, p) if (j - i) % p in squares]
    best_n = p
    
    # Try to find larger graphs
    # Allocate time slices for each n
    
    for n in range(18, 36):
        if _elapsed() > _DEADLINE - 5:
            break
        
        # Time slice: decrease as n increases
        time_slice = 3.0
        if n > 25:
            time_slice = 1.5
        if n > 30:
            time_slice = 0.8
            
        # Try multiple strategies
        # Strategy 1: Random local search
        edges = _try_local_search(n, time_slice / 2)
        if edges is not None:
            best_n = n
            best_edges = edges
            continue
        
        # Strategy 2: Extend from Paley
        remaining = time_slice / 2
        if remaining > 0.1:
            edges = _try_extend_from_paley(17, n, remaining)
            if edges is not None:
                best_n = n
                best_edges = edges
                continue
    
    return {'n': best_n, 'edges': best_edges}
