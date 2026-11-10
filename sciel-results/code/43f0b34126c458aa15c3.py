import time
import random
import numpy as np

def construct():
    N = 40
    # Safety margin: evaluator kills at 150s. Use 145s.
    deadline = time.time() + 145.0
    
    # Initialize with 8 disjoint K5s (104 edges)
    # This is a known C6-free structure.
    adj_mask = [0] * N
    edges = []
    
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                u = base + a
                v = base + b
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                edges.append((u, v))
    
    current_edge_count = len(edges)
    best_edge_count = current_edge_count
    best_adj_mask = adj_mask[:]
    
    # Fast C6 check using bitsets
    # Check if adding edge (u, v) creates a C6.
    # Path: u-w-x-y-z-v (length 5 edges, 6 vertices)
    # u, v are endpoints. w, x, y, z are distinct and not u, v.
    def has_c6(u, v):
        # w in N(u) \ {v}
        w_mask = adj_mask[u] & ~(1 << v)
        while w_mask:
            w_bit = w_mask & -w_mask
            w = w_bit.bit_length() - 1
            w_mask ^= w_bit
            
            # x in N(w) \ {u, v, w}
            x_mask = adj_mask[w] & ~(1 << u) & ~(1 << v) & ~(1 << w)
            while x_mask:
                x_bit = x_mask & -x_mask
                x = x_bit.bit_length() - 1
                x_mask ^= x_bit
                
                # y in N(x) \ {u, v, w, x}
                y_mask = adj_mask[x] & ~(1 << u) & ~(1 << v) & ~(1 << w) & ~(1 << x)
                while y_mask:
                    y_bit = y_mask & -y_mask
                    y = y_bit.bit_length() - 1
                    y_mask ^= y_bit
                    
                    # z in N(y) \ {u, v, w, x, y} AND z in N(v)
                    # z must be neighbor of v.
                    # z must be neighbor of y.
                    # z distinct from u, v, w, x, y.
                    z_mask = adj_mask[y] & adj_mask[v]
                    z_mask &= ~(1 << u) & ~(1 << v) & ~(1 << w) & ~(1 << x) & ~(1 << y)
                    if z_mask:
                        return True
        return False

    def add_edge(u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edges.append((u, v))
        
    def remove_edge(u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        edges.remove((u, v))

    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # Precompute all possible edges for random selection
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    while time.time() < deadline:
        # Pick a move
        # 70% chance to add, 30% chance to remove
        if random.random() < 0.7:
            u, v = random.choice(all_pairs)
            if (adj_mask[u] >> v) & 1:
                continue # Edge exists
            
            if has_c6(u, v):
                continue # Blocked
            
            # Add edge
            add_edge(u, v)
            new_count = len(edges)
            if new_count > best_edge_count:
                best_edge_count = new_count
                best_adj_mask = adj_mask[:]
                
        else:
            if not edges:
                continue
            e = random.choice(edges)
            u, v = e
            remove_edge(u, v)
            
            # Acceptance for worsening move
            # Delta = -1
            if random.random() < np.exp(-1.0 / T):
                pass # Accepted
            else:
                # Reject
                add_edge(u, v)
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Restart temperature occasionally to explore more
            if random.random() < 0.01:
                T = 0.5

    # Reconstruct edge list from best_adj_mask
    best_edges = []
    for u in range(N):
        m = best_adj_mask[u]
        while m:
            low = m & -m
            v = low.bit_length() - 1
            m ^= low
            if u < v:
                best_edges.append([u, v])
                
    return best_edges

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
