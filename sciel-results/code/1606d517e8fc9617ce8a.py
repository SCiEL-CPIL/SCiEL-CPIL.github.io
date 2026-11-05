import time
import random

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
            # Iterate pairs in common
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
    # Backtracking search
    # candidates: bitmask of vertices that can be added
    # current_set: list of vertices in the independent set so far
    
    def backtrack(candidates, current_set):
        if len(current_set) == 6:
            return current_set[:]
        # Pruning: if not enough candidates
        if candidates.bit_count() < 6 - len(current_set):
            return None
        
        # Pick a vertex to branch on
        # Heuristic: pick the vertex with the fewest neighbors in candidates?
        # Or just the first one.
        # Let's pick the first available vertex
        lsb = candidates & -candidates
        v = lsb.bit_length() - 1
        candidates ^= lsb
        
        # Branch 1: Include v
        # New candidates must not be adjacent to v
        new_candidates = candidates & ~adj[v]
        res = backtrack(new_candidates, current_set + [v])
        if res:
            return res
        
        # Branch 2: Exclude v
        res = backtrack(candidates, current_set)
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

def _try_local_search(n, time_limit):
    """Try to find a valid graph on n vertices using local search."""
    start = time.time()
    
    # Initialize random graph
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
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
            # All pairs are non-adjacent
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

def construct():
    # Base case: Paley 17
    p = 17
    squares = {x * x % p for x in range(1, p)}
    best_edges = [[i, j] for i in range(p) for j in range(i + 1, p) if (j - i) % p in squares]
    best_n = p
    
    # Try to find larger graphs
    # Allocate time slices for each n
    # We have ~140s.
    # Let's try n from 18 to 35.
    # Give each n a few seconds.
    
    for n in range(18, 36):
        if _elapsed() > _DEADLINE - 5:
            break
        
        # Time slice: decrease as n increases? Or keep constant?
        # Let's give 2 seconds per n.
        time_slice = 2.0
        if n > 25:
            time_slice = 1.0
        if n > 30:
            time_slice = 0.5
            
        edges = _try_local_search(n, time_slice)
        if edges is not None:
            best_n = n
            best_edges = edges
            # Continue to try larger n
            # But we should probably use remaining time for larger n
            # The loop naturally does this.
            
    return {'n': best_n, 'edges': best_edges}
