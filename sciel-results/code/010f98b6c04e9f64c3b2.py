import random
import time

def construct():
    n = 20
    N = [0] * n
    edges = set()
    
    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')
    
    def get_neighbors(mask):
        while mask:
            lsb = mask & -mask
            yield lsb.bit_length() - 1
            mask ^= lsb

    def is_valid_add(u, v):
        if u == v:
            return False
        if (N[u] >> v) & 1:
            return False
        
        common_uv = N[u] & N[v]
        if popcount(common_uv) > 2:
            return False
            
        for x in get_neighbors(N[v]):
            if x == u:
                continue
            if popcount(N[u] & N[x]) > 1:
                return False
                
        for y in get_neighbors(N[u]):
            if y == v:
                continue
            if popcount(N[v] & N[y]) > 1:
                return False
                
        return True

    def add_edge(u, v):
        N[u] |= (1 << v)
        N[v] |= (1 << u)
        edges.add((u, v))
        
    def remove_edge(u, v):
        N[u] &= ~(1 << v)
        N[v] &= ~(1 << u)
        edges.discard((u, v))

    def polish():
        """Greedy local search: add any valid edge."""
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not ((N[i] >> j) & 1):
                        if is_valid_add(i, j):
                            add_edge(i, j)
                            changed = True

    # --- Initialization ---
    random.seed(42)
    # Start with a slightly denser random graph
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.4:
                add_edge(i, j)
    polish()
    
    best_edges = set(edges)
    best_count = len(edges)
    
    start_time = time.time()
    deadline = 145.0
    
    T = 1.0
    T_min = 1e-5
    cooling_rate = 0.99995
    
    # To speed up random choice from edges, we can maintain a list
    # But set iteration is fine for n=20 (max 190 edges)
    
    while time.time() < deadline:
        move_type = random.random()
        
        if move_type < 0.8:
            # Try to add
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and not ((N[u] >> v) & 1):
                if is_valid_add(u, v):
                    add_edge(u, v)
                    if len(edges) > best_count:
                        best_count = len(edges)
                        best_edges = set(edges)
        else:
            # Try to remove
            if edges:
                u, v = random.choice(list(edges))
                remove_edge(u, v)
                if random.random() < (2.71828 ** (-1.0 / T)):
                    # Accepted
                    pass
                else:
                    # Rejected
                    add_edge(u, v)

        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    # Final polish on best state
    for e in list(edges):
        remove_edge(e[0], e[1])
    for e in best_edges:
        add_edge(e[0], e[1])
    polish()
    
    return [list(e) for e in edges]

# EVOLVE-BLOCK-END
