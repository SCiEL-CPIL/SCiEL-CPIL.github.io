import random
import time

def construct():
    n = 40
    adj = [0] * n
    
    def bit_count(x):
        return bin(x).count('1')
    
    def edge_count():
        return sum(bit_count(a) for a in adj) // 2

    def creates_c6(u, v):
        # Check if adding edge (u, v) creates a C6.
        # Path: u - a - b - c - d - v
        # a in N(u)
        a_mask = adj[u] & ~(1 << u) & ~(1 << v)
        while a_mask:
            a_bit = a_mask & -a_mask
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit
            
            # b in N(a) \ {u, v, a}
            b_mask = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << a)
            while b_mask:
                b_bit = b_mask & -b_mask
                b = b_bit.bit_length() - 1
                b_mask ^= b_bit
                
                # c in N(b) \ {u, a, v, b}
                c_mask = adj[b] & ~(1 << u) & ~(1 << a) & ~(1 << v) & ~(1 << b)
                while c_mask:
                    c_bit = c_mask & -c_mask
                    c = c_bit.bit_length() - 1
                    c_mask ^= c_bit
                    
                    # d in N(c) \ {u, a, b, v, c} AND d in N(v)
                    d_mask = adj[c] & adj[v]
                    d_mask &= ~(1 << u) & ~(1 << a) & ~(1 << b) & ~(1 << c) & ~(1 << v)
                    if d_mask:
                        return True
        return False

    def add_edge(u, v):
        if adj[u] & (1 << v):
            return False
        if creates_c6(u, v):
            return False
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return True

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    start_time = time.time()
    limit = 115.0
    
    best_edges = 0
    best_adj = [0] * n
    
    # Simulated Annealing
    # Initial state: Greedy random
    def greedy_init():
        nonlocal adj
        adj = [0] * n
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        for u, v in pairs:
            add_edge(u, v)
        return edge_count()
    
    # Run SA
    # Temperature schedule
    T = 1.0
    cooling = 0.999
    min_T = 1e-5
    
    # Multiple restarts
    for restart in range(20):
        if time.time() - start_time > limit:
            break
            
        curr_edges = greedy_init()
        if curr_edges > best_edges:
            best_edges = curr_edges
            best_adj = adj[:]
            
        T = 1.0
        steps = 0
        max_steps = 50000 # Adjust based on time
        
        while T > min_T and time.time() - start_time < limit - 1:
            steps += 1
            
            # Choose a move
            # 1. Add a random missing edge
            # 2. Remove a random existing edge
            # 3. Swap: Remove one, Add another
            
            # To be efficient, we can just try to add a random missing edge.
            # If it fails, try removing a random edge.
            
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            
            if adj[u] & (1 << v):
                # Edge exists, try to remove
                remove_edge(u, v)
                delta = -1
                # Acceptance: always accept removal? 
                # In SA, we might want to keep density. 
                # But removing allows adding better edges later.
                # Let's use Metropolis: accept if delta > 0 or with prob exp(delta/T)
                # Here delta = -1.
                if random.random() > (1.0 / T): # exp(-1/T)
                    # Reject
                    add_edge(u, v) # Revert
                else:
                    # Accept
                    curr_edges += delta
            else:
                # Edge missing, try to add
                if creates_c6(u, v):
                    # Cannot add
                    # Try a swap? Or just skip.
                    # For simplicity, skip.
                    pass
                else:
                    add_edge(u, v)
                    delta = 1
                    # Always accept improvement
                    curr_edges += delta
                    if curr_edges > best_edges:
                        best_edges = curr_edges
                        best_adj = adj[:]
            
            T *= cooling
            
    # Final check
    return [[u, v] for u in range(n) for v in range(u+1, n) if best_adj[u] & (1 << v)]
