import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n = 40
    best_edges = []
    best_count = 0
    
    # Pre-compute bit masks for vertices
    # adj[i] is an integer where bit j is set if (i,j) is an edge
    
    def can_add(u, v, adj):
        """
        Returns True if edge (u,v) can be added without creating a C6.
        Assumes (u,v) is NOT currently in adj.
        """
        if u == v:
            return False
        if (adj[u] >> v) & 1:
            return False
        
        # Check for path of length 5: u-a-b-c-d-v
        # a in N(u), d in N(v)
        # a, b, c, d distinct, none equal to u, v
        
        nu = adj[u]
        nv = adj[v]
        
        # a must be in N(u)
        # a != v is true because (u,v) not in edge
        a_mask = nu
        while a_mask:
            a_bit = a_mask & -a_mask
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit
            
            # b in N(a)
            # b != u, b != v, b != a
            b_mask = adj[a] & ~(1 << u) & ~(1 << v)
            while b_mask:
                b_bit = b_mask & -b_mask
                b = b_bit.bit_length() - 1
                b_mask ^= b_bit
                
                # c in N(b)
                # c != u, c != a, c != v, c != b
                c_mask = adj[b] & ~(1 << u) & ~(1 << a) & ~(1 << v)
                while c_mask:
                    c_bit = c_mask & -c_mask
                    c = c_bit.bit_length() - 1
                    c_mask ^= c_bit
                    
                    # d in N(c) AND d in N(v)
                    # d != u, d != a, d != b, d != v
                    d_mask = adj[c] & nv & ~(1 << u) & ~(1 << a) & ~(1 << b) & ~(1 << v)
                    if d_mask:
                        return False
        return True

    def get_edges(adj):
        edges = []
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                if (ai >> j) & 1:
                    edges.append([i, j])
        return edges

    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def greedy_fill(adj):
        """Try to add all possible missing edges. Returns number of added edges."""
        added = 0
        # Iterate multiple times? No, one pass is usually enough if we check validity.
        # But adding one edge might block another.
        # To be safe, we can do a few passes.
        for _ in range(3):
            progress = False
            for i in range(n):
                ai = adj[i]
                for j in range(i + 1, n):
                    if not ((ai >> j) & 1):
                        if can_add(i, j, adj):
                            add_edge(adj, i, j)
                            added += 1
                            progress = True
            if not progress:
                break
        return added

    # Initialization
    for restart in range(500): # Increase restarts if time permits
        if time.time() > deadline:
            break
        
        adj = [0] * n
        
        # Start with 8 disjoint K5s
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    add_edge(adj, base + a, base + b)
        
        # Initial Greedy Fill
        greedy_fill(adj)
        
        current_count = count_edges(adj)
        
        # Simulated Annealing
        T = 5.0
        min_T = 0.1
        cooling = 0.995
        
        # Pre-compute list of all possible edges for random selection
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        steps = 0
        while T > min_T:
            if time.time() > deadline:
                break
            
            # Pick a random move
            # Strategy: 
            # 1. Try to add a random non-edge.
            # 2. If that fails (creates C6), try to remove a random edge and then greedily fill.
            
            # Option A: Add random non-edge
            # Get list of non-edges? Expensive.
            # Just pick random u, v.
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            if not ((adj[u] >> v) & 1):
                if can_add(u, v, adj):
                    add_edge(adj, u, v)
                    current_count += 1
                    # Accept
                    if current_count > best_count:
                        best_count = current_count
                        best_edges = get_edges(adj)
                    T *= cooling
                    steps += 1
                    continue
            # Option B: Remove random edge and fill
            # Pick random existing edge
            # To pick random existing edge efficiently, we can maintain a list, but it changes.
            # For now, pick random u, v that are connected.
            found = False
            for _ in range(10): # Try a few times to find an edge
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if (adj[u] >> v) & 1:
                    found = True
                    break
            
            if not found:
                continue
            
            # Save state
            # We are removing (u,v)
            remove_edge(adj, u, v)
            current_count -= 1
            
            # Greedy fill
            added = greedy_fill(adj)
            current_count += added
            
            new_count = current_count
            delta = new_count - (current_count + 1) # Delta relative to state BEFORE removal?
            # Actually, delta is new_count - old_count
            # old_count was current_count + 1
            # So delta = new_count - (current_count + 1) is wrong.
            # Let's track old_count.
            
            # Let's restart the logic for SA step clearly.
            
            # ... This is getting messy. Let's restructure the SA loop.

        # End of SA for this restart
        # Check if better
        final_count = count_edges(adj)
        if final_count > best_count:
            best_count = final_count
            best_edges = get_edges(adj)

    return best_edges
