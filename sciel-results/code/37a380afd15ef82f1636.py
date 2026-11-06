import time
import random
from itertools import combinations

N = 20
DEADLINE_FRACTION = 0.90

def construct():
    start_time = time.time()
    deadline = start_time + 120 * DEADLINE_FRACTION
    
    # adj[i] is an integer bitmask of neighbors of i
    def _popcount(x):
        return bin(x).count('1')
    
    def _is_valid_add(adj, u, v):
        # Check if adding edge (u, v) keeps graph K3,3-free
        # A K3,3 is formed if there exist x, y in N(v) such that
        # |N(u) & N(x) & N(y)| >= 2.
        # The vertices u, x, y form one partition, and v, b1, b2 form the other.
        # b1, b2 are the common neighbors of u, x, y.
        nv = adj[v]
        # Get list of neighbors of v
        neigh_v = [i for i in range(N) if (nv >> i) & 1]
        for x, y in combinations(neigh_v, 2):
            common = adj[u] & adj[x] & adj[y]
            if _popcount(common) >= 2:
                return False
        return True

    def _add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def _remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def _edge_count(adj):
        return sum(_popcount(a) for a in adj) // 2

    def _get_missing_edges(adj):
        missing = []
        for u in range(N):
            for v in range(u+1, N):
                if not ((adj[u] >> v) & 1):
                    missing.append((u, v))
        return missing

    def _get_existing_edges(adj):
        existing = []
        for u in range(N):
            for v in range(u+1, N):
                if (adj[u] >> v) & 1:
                    existing.append((u, v))
        return existing

    def _greedy_fill(adj):
        # Try to add any missing edge in random order
        missing = _get_missing_edges(adj)
        random.shuffle(missing)
        for u, v in missing:
            if _is_valid_add(adj, u, v):
                _add_edge(adj, u, v)
        return adj

    best_adj = None
    best_count = 0
    
    # Initial random start
    # Start with empty graph and greedy fill
    adj = [0] * N
    adj = _greedy_fill(adj)
    best_adj = adj[:]
    best_count = _edge_count(adj)
    
    stagnation = 0
    kick_size = 2
    
    while time.time() < deadline:
        # Perform a kick: remove kick_size random edges from best_adj
        current_adj = best_adj[:]
        existing = _get_existing_edges(current_adj)
        
        if len(existing) < kick_size:
            kick_size = 1
            if len(existing) < 1:
                break
        
        # Remove random edges
        edges_to_remove = random.sample(existing, min(kick_size, len(existing)))
        for u, v in edges_to_remove:
            _remove_edge(current_adj, u, v)
            
        # Greedy fill
        current_adj = _greedy_fill(current_adj)
        current_count = _edge_count(current_adj)
        
        if current_count > best_count:
            best_count = current_count
            best_adj = current_adj
            stagnation = 0
            kick_size = 2
        else:
            stagnation += 1
            if stagnation > 20:
                kick_size = min(kick_size * 2, 30)
                stagnation = 0

    if best_adj is None:
        return []

    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges
