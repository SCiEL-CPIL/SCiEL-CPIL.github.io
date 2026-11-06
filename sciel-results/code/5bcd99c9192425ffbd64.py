import time
import random
from itertools import combinations

N = 20
DEADLINE_FRACTION = 0.88

def construct():
    start_time = time.time()
    deadline = start_time + 120 * DEADLINE_FRACTION
    
    def _popcount(x):
        return bin(x).count('1')
    
    def _iter_bits(mask):
        """Yield indices of set bits in mask"""
        while mask:
            lsb = mask & (-mask)
            idx = lsb.bit_length() - 1
            yield idx
            mask ^= lsb

    def _is_valid_add(adj, u, v):
        # Check if adding edge (u, v) keeps graph K3,3-free
        # A K3,3 forms if there exist x, y in N(v) such that
        # |N(u) & N(x) & N(y)| >= 2
        nv = adj[v]
        # Collect neighbors of v
        neigh_v = list(_iter_bits(nv))
        for i in range(len(neigh_v)):
            x = neigh_v[i]
            for j in range(i+1, len(neigh_v)):
                y = neigh_v[j]
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

    def _local_search_swap(adj, max_swaps=50):
        """Try edge swaps: remove one edge, add a different one"""
        existing = _get_existing_edges(adj)
        missing = _get_missing_edges(adj)
        if not existing or not missing:
            return adj, False
        
        random.shuffle(existing)
        random.shuffle(missing)
        
        for rem_u, rem_v in existing[:max_swaps]:
            for add_u, add_v in missing[:20]:
                # Skip if same edge
                if (rem_u, rem_v) == (add_u, add_v) or (rem_u, rem_v) == (add_v, add_u):
                    continue
                # Try the swap
                _remove_edge(adj, rem_u, rem_v)
                if _is_valid_add(adj, add_u, add_v):
                    _add_edge(adj, add_u, add_v)
                    # Check if we can add more edges now
                    _greedy_fill(adj)
                    return adj, True
                else:
                    _add_edge(adj, rem_u, rem_v)
        return adj, False

    best_adj = None
    best_count = 0
    
    # Multiple restarts
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        
        # Initial random start: greedy fill from empty
        adj = [0] * N
        adj = _greedy_fill(adj)
        current_count = _edge_count(adj)
        
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
        
        # Local search with kicks
        stagnation = 0
        kick_size = 2
        local_best = current_count
        
        while time.time() < deadline:
            # Perform a kick: remove kick_size random edges
            current_adj = best_adj[:] if best_adj is not None else adj[:]
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
            
            # Try swaps if no improvement from kick+fill
            if current_count <= best_count:
                current_adj, swapped = _local_search_swap(current_adj, max_swaps=30)
                current_count = _edge_count(current_adj)
            
            if current_count > best_count:
                best_count = current_count
                best_adj = current_adj
                local_best = current_count
                stagnation = 0
                kick_size = 2
            else:
                stagnation += 1
                if stagnation > 15:
                    kick_size = min(kick_size * 2, 20)
                    stagnation = 0
            
            # If we've been stuck for a while, restart
            if stagnation > 40:
                break

        # Safety check
        if time.time() > deadline * 0.95:
            break

    if best_adj is None:
        return []

    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges
