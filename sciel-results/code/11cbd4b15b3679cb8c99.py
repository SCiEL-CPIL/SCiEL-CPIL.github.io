import time
import random
import numpy as np

def construct():
    deadline = time.time() + 110
    n = 40
    # Initial state: 8 disjoint K5s (104 edges)
    # This is a known C6-free structure with high edge count.
    edges = []
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                edges.append((5 * k + a, 5 * k + b))
    
    # Bitset representation of the graph for fast C6 checking
    # adj_mask[i] is an integer where bit j is 1 if (i, j) is an edge
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Precompute all possible edges not in the initial set
    all_edges = [(a, b) for a in range(n) for b in range(a + 1, n)]
    initial_edge_set = set(edges)
    candidate_edges = [e for e in all_edges if e not in initial_edge_set]
    
    def has_c6_fast(adj_mask, u, v):
        """
        Check if adding edge (u, v) creates a C6.
        A C6 containing (u, v) is a path u-w-x-y-z-v of length 5 (edges) 
        where u, v, w, x, y, z are distinct.
        This is equivalent to checking if there is a path of length 4 
        from u to v in G - (u, v).
        Since (u, v) is not yet in the graph, we just check for a path of length 4.
        
        Path: u - w - x - y - z - v
        Edges: (u,w), (w,x), (x,y), (y,z), (z,v)
        
        We can iterate over w in N(u), then x in N(w), etc.
        Using bitsets makes the inner loops very fast.
        """
        mask_u = adj_mask[u]
        mask_v = adj_mask[v]
        
        # w must be in N(u) and w != v
        w_mask = mask_u & ~(1 << v)
        
        while w_mask:
            w_bit = w_mask & -w_mask
            w = w_bit.bit_length() - 1
            w_mask ^= w_bit
            
            # x must be in N(w), x not in {u, v, w}
            mask_w = adj_mask[w]
            x_mask = mask_w & ~(1 << u) & ~(1 << v) & ~(1 << w)
            
            while x_mask:
                x_bit = x_mask & -x_mask
                x = x_bit.bit_length() - 1
                x_mask ^= x_bit
                
                # y must be in N(x), y not in {u, v, w, x}
                mask_x = adj_mask[x]
                y_mask = mask_x & ~(1 << u) & ~(1 << v) & ~(1 << w) & ~(1 << x)
                
                # Precompute: for each y in y_mask, check if N(y) intersects N(v)
                # with proper exclusions
                while y_mask:
                    y_bit = y_mask & -y_mask
                    y = y_bit.bit_length() - 1
                    y_mask ^= y_bit
                    
                    # z must be in N(y), z not in {u, v, w, x, y}
                    # AND z must be in N(v)
                    mask_y = adj_mask[y]
                    z_mask = mask_y & ~(1 << u) & ~(1 << v) & ~(1 << w) & ~(1 << x) & ~(1 << y)
                    
                    if z_mask & mask_v:
                        return True
        
        return False

    def add_edge(u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    def remove_edge(u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)

    current_edges = set(edges)
    best_edges = set(edges)
    best_score = len(best_edges)
    
    # Maintain a list of missing edges for faster random selection
    missing_edges = list(candidate_edges)
    
    # Simulated Annealing Parameters
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.9985
    restart_interval = 500
    
    steps = 0
    random.shuffle(missing_edges)
    
    while time.time() < deadline:
        # Pick a random move: add or remove an edge
        # Bias towards adding edges since we want to maximize
        if random.random() < 0.75:
            # Try to add a random edge from missing_edges
            if not missing_edges:
                # All edges present, try removing
                if not current_edges:
                    continue
                e = random.choice(list(current_edges))
                u, v = e
                remove_edge(u, v)
                current_edges.remove(e)
                missing_edges.append(e)
                if random.random() < np.exp(-1.0 / T):
                    pass
                else:
                    add_edge(u, v)
                    current_edges.add(e)
                    missing_edges.remove(e)
            else:
                # Pick a random missing edge
                idx = random.randint(0, len(missing_edges) - 1)
                u, v = missing_edges[idx]
                
                # Check C6
                if has_c6_fast(adj_mask, u, v):
                    # Reject: swap with a random later element to keep list valid
                    if idx < len(missing_edges) - 1:
                        missing_edges[idx] = missing_edges[-1]
                        missing_edges.pop()
                    else:
                        missing_edges.pop()
                    continue
                
                # Move is valid (graph remains C6-free)
                # Add edge
                add_edge(u, v)
                current_edges.add((u, v))
                # Remove from missing_edges
                if idx < len(missing_edges) - 1:
                    missing_edges[idx] = missing_edges[-1]
                    missing_edges.pop()
                else:
                    missing_edges.pop()
                
                new_score = len(current_edges)
                
                if new_score > best_score:
                    best_score = new_score
                    best_edges = set(current_edges)
        
        else:
            # Try to remove a random edge (to escape local optima)
            if not current_edges:
                continue
            e = random.choice(list(current_edges))
            u, v = e
            remove_edge(u, v)
            current_edges.remove(e)
            missing_edges.append(e)
            
            # Acceptance criterion for worsening move
            if random.random() < np.exp(-1.0 / T):
                pass # Accepted
            else:
                # Reject, revert
                add_edge(u, v)
                current_edges.add(e)
                missing_edges.remove(e)
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Restart with higher temperature to escape deep local minima
            if steps % restart_interval == 0:
                T = 0.5
        
        steps += 1

    return [[a, b] for (a, b) in sorted(best_edges)]
