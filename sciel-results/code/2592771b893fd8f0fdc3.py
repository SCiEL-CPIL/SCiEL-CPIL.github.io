import numpy as np
import random
import time

def construct():
    n = 40
    # Bitset representation of adjacency
    # adj_bits[i] is an integer where bit j is 1 if (i,j) is an edge
    adj_bits = [0] * n
    
    def get_edge_count():
        return sum(bin(x).count('l') for x in adj_bits) // 2 # Wait, bin(x).count('1') is better
        
    def edge_count():
        return sum(bin(x).count('1') for x in adj_bits) // 2

    def _check_c6_add(u, v):
        """Check if adding edge (u,v) creates a C6.
        Assumes (u,v) is NOT currently in the graph.
        Path: u - a - b - c - d - v
        """
        nu = adj_bits[u]
        nv = adj_bits[v]
        
        # Iterate a in N(u)
        a_mask = nu
        while a_mask:
            a_bit = a_mask & -a_mask
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit
            if a == v: continue
            
            # Iterate d in N(v)
            d_mask = nv
            while d_mask:
                d_bit = d_mask & -d_mask
                d = d_bit.bit_length() - 1
                d_mask ^= d_bit
                if d == u: continue
                if a == d: continue
                
                # Candidates for b: neighbors of a, excluding u, v, d
                # Candidates for c: neighbors of d, excluding u, v, a
                # We need an edge between some b in CandB and c in CandC.
                
                # Exclude u, v, d from N(a)
                cand_b = adj_bits[a] & ~(1 << u) & ~(1 << v) & ~(1 << d)
                # Exclude u, v, a from N(d)
                cand_c = adj_bits[d] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                
                # Quick check: if cand_b or cand_c is 0, no path
                if cand_b == 0 or cand_c == 0:
                    continue
                
                # Check if there is an edge between any b in cand_b and c in cand_c
                # Union of neighbors of all b in cand_b
                union_nb = 0
                tmp = cand_b
                while tmp:
                    b_bit = tmp & -tmp
                    b = b_bit.bit_length() - 1
                    tmp ^= b_bit
                    union_nb |= adj_bits[b]
                
                if (union_nb & cand_c) != 0:
                    return True # C6 found
        
        return False

    def add_edge(u, v):
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)

    def remove_edge(u, v):
        adj_bits[u] &= ~(1 << v)
        adj_bits[v] &= ~(1 << u)

    def has_edge(u, v):
        return (adj_bits[u] >> v) & 1

    # 1. Greedy Construction
    # Generate all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.seed(42)
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if not has_edge(u, v):
            if not _check_c6_add(u, v):
                add_edge(u, v)
                
    current_edges = edge_count()
    
    # 2. Local Search
    start_time = time.time()
    time_limit = 100.0 # Leave 20s for overhead
    
    # Precompute list of non-edges? It changes.
    # We'll just pick random pairs.
    
    # To speed up, we can maintain a list of "promising" non-edges or just random.
    
    while time.time() - start_time < time_limit:
        # Try to add a random non-edge
        # Pick random u, v
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v: continue
        if has_edge(u, v): continue
        
        if not _check_c6_add(u, v):
            add_edge(u, v)
            current_edges += 1
            continue
            
        # If we can't add (u,v), try a swap
        # Remove a random edge (a,b) and try to add (u,v) and maybe another edge?
        # Simpler: Remove (a,b), then try to add (u,v). If successful, net 0.
        # We need net gain. So remove 1, add 2?
        
        # Let's try: Remove (a,b). Then try to add (u,v). 
        # If (u,v) can be added, we are back to same count.
        # But maybe removing (a,b) allows adding (u,v) AND (x,y)?
        
        # Strategy:
        # 1. Pick random edge (a,b) to remove.
        # 2. Remove it.
        # 3. Try to add (u,v).
        # 4. If added, try to add another random non-edge (x,y).
        # 5. If (x,y) added, accept (net +1).
        # 6. If not, try to add (x,y) again with different x,y?
        # 7. If we can't get net gain, revert (re-add (a,b)).
        
        # To make this efficient, we only do this if we are stuck.
        
        # Pick random edge to remove
        # We need a list of current edges.
        # For n=40, max 105 edges.
        current_edge_list = []
        for i in range(n):
            mask = adj_bits[i]
            while mask:
                bit = mask & -mask
                j = bit.bit_length() - 1
                if i < j:
                    current_edge_list.append((i, j))
                mask ^= bit
        
        if not current_edge_list:
            break
            
        a, b = random.choice(current_edge_list)
        
        # Temporarily remove (a,b)
        remove_edge(a, b)
        
        # Try to add (u,v)
        success_u_v = False
        if not _check_c6_add(u, v):
            add_edge(u, v)
            success_u_v = True
            
        # Try to add another random non-edge (x,y)
        success_x_y = False
        if success_u_v:
            # Try a few random non-edges
            for _ in range(10):
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x == y: continue
                if has_edge(x, y): continue
                if not _check_c6_add(x, y):
                    add_edge(x, y)
                    success_x_y = True
                    break
                    
        if success_x_y:
            # Net gain +1 (removed 1, added 2)
            current_edges += 1
        else:
            # Revert
            if success_u_v:
                remove_edge(u, v)
            add_edge(a, b)
            
    # Convert to list of edges
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if has_edge(u, v):
                edges.append([u, v])
                
    return edges
