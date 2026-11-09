import random
import time
from itertools import combinations

N = 18
FULL_MASK = (1 << N) - 1

def _is_k4_free(adj_bits):
    """Check if graph is K4-free using bitmasks."""
    for a in range(N):
        na = adj_bits[a]
        for b in range(a + 1, N):
            if not (na >> b) & 1:
                continue
            common_ab = na & adj_bits[b]
            for c in range(b + 1, N):
                if not (common_ab >> c) & 1:
                    continue
                common_abc = common_ab & adj_bits[c]
                mask_above_c = FULL_MASK ^ ((1 << (c + 1)) - 1)
                if common_abc & mask_above_c:
                    return False
    return True

def _largest_triangle_free(adj_bits):
    """Find largest triangle-free induced subgraph size using bitmask enumeration."""
    
    def is_triangle_free(mask):
        # For each vertex i in mask, check that adj[i] & mask is an independent set
        m = mask
        while m:
            low_bit = m & (-m)
            i = low_bit.bit_length() - 1
            m ^= low_bit
            ni = adj_bits[i] & mask
            # Check if ni is an independent set
            nn = ni
            while nn:
                low_bit2 = nn & (-nn)
                j = low_bit2.bit_length() - 1
                nn ^= low_bit2
                if adj_bits[j] & ni:
                    return False
        return True
    
    # Check sizes from N down to 1
    for size in range(N, 0, -1):
        found = False
        for subset in combinations(range(N), size):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if is_triangle_free(mask):
                found = True
                break
        if found:
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
    deadline = start_time + 110  # Safety margin before 120s limit
    
    rng = random.Random(12345)
    
    # === Phase 1: Greedy construction ===
    # Start with empty graph, add edges greedily to maximize triangle-breaking
    # while maintaining K4-free property
    
    def greedy_construct():
        adj = [0] * N
        all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        rng2 = random.Random(rng.random())
        rng2.shuffle(all_edges)
        
        for (u, v) in all_edges:
            # Try adding edge (u,v)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if not _is_k4_free(adj):
                # Revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        return adj
    
    best_adj = greedy_construct()
    best_score = _largest_triangle_free(best_adj)
    
    if time.time() > deadline:
        return _bits_to_edges(best_adj)
    
    # === Phase 2: Simulated annealing with bitmask operations ===
    # Moves: add an edge (if K4-free), remove an edge, or swap
    
    def evaluate(adj_bits):
        return _largest_triangle_free(adj_bits)
    
    current_adj = best_adj[:]
    current_score = best_score
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    edge_set = set(all_edges)
    
    T = 5.0  # Initial temperature
    T_min = 0.01
    cooling = 0.995
    
    step = 0
    while time.time() < deadline:
        step += 1
        
        # Choose a random move
        move_type = rng.random()
        
        if move_type < 0.4:
            # Try adding a random non-edge
            non_edges = [(u, v) for (u, v) in all_edges if not (current_adj[u] >> v) & 1]
            if not non_edges:
                continue
            u, v = rng.choice(non_edges)
            new_adj = current_adj[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            if not _is_k4_free(new_adj):
                continue
            new_score = evaluate(new_adj)
            delta = new_score - current_score  # We want to minimize
            if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
                current_adj = new_adj
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj[:]
        elif move_type < 0.8:
            # Try removing a random edge
            existing_edges = [(u, v) for (u, v) in all_edges if (current_adj[u] >> v) & 1]
            if not existing_edges:
                continue
            u, v = rng.choice(existing_edges)
            new_adj = current_adj[:]
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
            new_score = evaluate(new_adj)
            delta = new_score - current_score  # We want to minimize
            if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
                current_adj = new_adj
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj[:]
        else:
            # Try a swap: remove one edge, add another
            existing_edges = [(u, v) for (u, v) in all_edges if (current_adj[u] >> v) & 1]
            non_edges = [(u, v) for (u, v) in all_edges if not (current_adj[u] >> v) & 1]
            if not existing_edges or not non_edges:
                continue
            u1, v1 = rng.choice(existing_edges)
            u2, v2 = rng.choice(non_edges)
            new_adj = current_adj[:]
            new_adj[u1] &= ~(1 << v1)
            new_adj[v1] &= ~(1 << u1)
            new_adj[u2] |= (1 << v2)
            new_adj[v2] |= (1 << u2)
            if not _is_k4_free(new_adj):
                continue
            new_score = evaluate(new_adj)
            delta = new_score - current_score
            if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
                current_adj = new_adj
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj[:]
        
        # Cool down
        T = max(T_min, T * cooling)
        
        # Occasionally reheat if stuck
        if step % 1000 == 0 and current_score == best_score:
            T = max(T, 2.0)
    
    return _bits_to_edges(best_adj)
