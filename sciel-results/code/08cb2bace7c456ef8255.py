import random
import time
from itertools import combinations

N = 18
FULL_MASK = (1 << N) - 1

# Precompute masks by size
masks_by_size = [[] for _ in range(N + 1)]
for mask in range(1 << N):
    size = bin(mask).count('1')
    masks_by_size[size].append(mask)

def _is_k4_free(adj):
    """Check if graph is K4-free using bitmasks."""
    for u in range(N):
        au = adj[u]
        for v in range(u + 1, N):
            if (au >> v) & 1:
                common = au & adj[v]
                # Check if common neighbors contain an edge
                c = common
                while c:
                    low_bit = c & (-c)
                    i = low_bit.bit_length() - 1
                    c ^= low_bit
                    if adj[i] & common:
                        return False
    return True

def _is_triangle_free(mask, adj):
    """Check if the subset represented by mask is triangle-free."""
    m = mask
    while m:
        low_bit = m & (-m)
        i = low_bit.bit_length() - 1
        m ^= low_bit
        ni = adj[i] & mask
        # Check if ni is an independent set
        nn = ni
        while nn:
            low_bit2 = nn & (-nn)
            j = low_bit2.bit_length() - 1
            nn ^= low_bit2
            if adj[j] & ni:
                return False
    return True

def _largest_triangle_free(adj):
    """Find the largest triangle-free induced subgraph size."""
    # Check from largest size down
    for size in range(N, 0, -1):
        for mask in masks_by_size[size]:
            if _is_triangle_free(mask, adj):
                return size
    return 0

def _bits_to_edges(adj_bits):
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if (adj_bits[i] >> j) & 1:
                edges.append([i, j])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    rng = random.Random(42)
    
    # Phase 1: Greedy Construction of multiple random K4-free graphs
    best_adj = None
    best_score = N + 1  # Max possible is 18
    
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    def greedy_random_graph():
        adj = [0] * N
        edges_copy = all_edges[:]
        rng.shuffle(edges_copy)
        
        for u, v in edges_copy:
            # Try adding edge (u, v)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if not _is_k4_free(adj):
                # Revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        return adj

    # Generate 50 random graphs
    for _ in range(50):
        if time.time() > deadline:
            break
        adj = greedy_random_graph()
        score = _largest_triangle_free(adj)
        if score < best_score:
            best_score = score
            best_adj = adj[:]
        if best_score <= 9:
            break

    if best_adj is None:
        # Fallback: Empty graph is K4-free, score 18
        best_adj = [0] * N
        best_score = 18

    # Phase 2: Local Search / Simulated Annealing
    current_adj = best_adj[:]
    current_score = best_score
    T = 2.0
    T_min = 0.1
    cooling = 0.99
    
    # Precompute non-edges and edges for faster move selection?
    # Or just pick random edge and try flip.
    
    while time.time() < deadline:
        # Pick a random edge to flip
        u, v = rng.choice(all_edges)
        
        # Determine if it's currently an edge
        is_edge = (current_adj[u] >> v) & 1
        
        # Try flipping
        new_adj = current_adj[:]
        if is_edge:
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
        else:
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
        
        # Check K4-free
        if not _is_k4_free(new_adj):
            continue
        
        # Evaluate new score
        new_score = _largest_triangle_free(new_adj)
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(-delta/T)
        if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
            current_adj = new_adj
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_adj = current_adj[:]
                if best_score <= 9:
                    break
        
        T *= cooling
        if T < T_min:
            T = T_min

    return _bits_to_edges(best_adj)
