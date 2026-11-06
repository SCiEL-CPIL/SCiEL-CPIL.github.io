import random
import time

def construct():
    n = 40
    m = 401
    
    # Use bitmasks for adjacency: adj[i] is a 40-bit integer where bit j is set if edge (i,j) exists
    # Start with K20,20 + one edge within part A
    # A = 0..19, B = 20..39
    adj = [0] * n
    for u in range(20):
        for v in range(20, 40):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    # Add edge (0,1) within part A
    adj[0] |= (1 << 1)
    adj[1] |= (1 << 0)
    
    def compute_max_book(adj):
        """Compute max book size using bitmasks. Book size of edge (u,v) = popcount(adj[u] & adj[v])."""
        max_book = 0
        for u in range(n):
            # Iterate over set bits in adj[u]
            mask = adj[u]
            while mask:
                # Get lowest set bit
                v_bit = mask & (-mask)
                v = v_bit.bit_length() - 1
                if v > u:  # Only count each edge once
                    common = (adj[u] & adj[v]).bit_count()
                    if common > max_book:
                        max_book = common
                mask ^= v_bit
        return max_book
    
    def try_move(adj, remove_u, remove_v, add_u, add_v):
        """Try removing edge (remove_u, remove_v) and adding edge (add_u, add_v).
        Returns new adj and whether it's valid."""
        # Check if remove edge exists
        if not (adj[remove_u] & (1 << remove_v)):
            return None, False
        # Check if add edge doesn't exist
        if adj[add_u] & (1 << add_v):
            return None, False
        
        new_adj = adj[:]
        # Remove edge
        new_adj[remove_u] &= ~(1 << remove_v)
        new_adj[remove_v] &= ~(1 << remove_u)
        # Add edge
        new_adj[add_u] |= (1 << add_v)
        new_adj[add_v] |= (1 << add_u)
        return new_adj, True
    
    current_adj = adj[:]
    current_max_book = compute_max_book(current_adj)
    best_adj = current_adj[:]
    best_max_book = current_max_book
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Pre-compute list of all possible edges for random selection
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Temperature for simulated annealing
    T = 0.5
    T_min = 0.01
    cooling = 0.9995
    
    # Phase 1: Targeted moves - try to reduce books by moving edges from between parts to within parts
    # This breaks the bipartite structure which has high book sizes for cross edges
    phase1_deadline = start_time + 60
    
    while time.time() < phase1_deadline:
        # Find the edge with maximum book size
        max_edge = None
        max_book_val = 0
        for u in range(n):
            mask = adj[u]
            while mask:
                v_bit = mask & (-mask)
                v = v_bit.bit_length() - 1
                if v > u:
                    common = (adj[u] & adj[v]).bit_count()
                    if common > max_book_val:
                        max_book_val = common
                        max_edge = (u, v)
                mask ^= v_bit
        
        if max_edge is None:
            break
            
        u, v = max_edge
        
        # Try to remove this edge and add a random edge within a part
        # Prefer adding edges within parts A or B to break bipartiteness
        improved = False
        for _ in range(50):
            # Choose which part to add the edge in
            if random.random() < 0.5:
                # Part A: 0..19
                new_u = random.randint(0, 19)
                new_v = random.randint(0, 19)
            else:
                # Part B: 20..39
                new_u = random.randint(20, 39)
                new_v = random.randint(20, 39)
            
            if new_u == new_v:
                continue
            if (new_u, new_v) == (u, v) or (new_u, new_v) == (v, u):
                continue
            
            new_adj, valid = try_move(current_adj, u, v, new_u, new_v)
            if not valid:
                continue
            
            new_max_book = compute_max_book(new_adj)
            
            if new_max_book < best_max_book:
                best_max_book = new_max_book
                best_adj = new_adj
                current_adj = new_adj
                current_max_book = new_max_book
                improved = True
            elif new_max_book < current_max_book:
                current_adj = new_adj
                current_max_book = new_max_book
                improved = True
            elif random.random() < T:
                current_adj = new_adj
                current_max_book = new_max_book
        
        if not improved:
            # Try random moves instead
            for _ in range(20):
                remove_edge = random.choice(all_edges)
                add_edge = random.choice(all_edges)
                if remove_edge == add_edge:
                    continue
                new_adj, valid = try_move(current_adj, remove_edge[0], remove_edge[1], add_edge[0], add_edge[1])
                if not valid:
                    continue
                new_max_book = compute_max_book(new_adj)
                if new_max_book < best_max_book:
                    best_max_book = new_max_book
                    best_adj = new_adj
                    current_adj = new_adj
                    current_max_book = new_max_book
                elif new_max_book < current_max_book:
                    current_adj = new_adj
                    current_max_book = new_max_book
                elif random.random() < T:
                    current_adj = new_adj
                    current_max_book = new_max_book
        
        T = max(T_min, T * cooling)
    
    # Phase 2: General simulated annealing with fast bitset evaluation
    while time.time() < deadline:
        # Random move
        remove_edge = random.choice(all_edges)
        add_edge = random.choice(all_edges)
        if remove_edge == add_edge:
            continue
        
        new_adj, valid = try_move(current_adj, remove_edge[0], remove_edge[1], add_edge[0], add_edge[1])
        if not valid:
            continue
        
        new_max_book = compute_max_book(new_adj)
        
        if new_max_book < best_max_book:
            best_max_book = new_max_book
            best_adj = new_adj
            current_adj = new_adj
            current_max_book = new_max_book
        elif new_max_book < current_max_book:
            current_adj = new_adj
            current_max_book = new_max_book
        elif random.random() < T:
            current_adj = new_adj
            current_max_book = new_max_book
        
        T = max(T_min, T * cooling)
    
    # Convert best_adj to list of edges
    result = []
    for u in range(n):
        mask = best_adj[u]
        while mask:
            v_bit = mask & (-mask)
            v = v_bit.bit_length() - 1
            if v > u:
                result.append([u, v])
            mask ^= v_bit
    
    return result
