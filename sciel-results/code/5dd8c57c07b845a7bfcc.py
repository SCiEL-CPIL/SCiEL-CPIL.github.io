"""Construct a C4-free graph with high minimum degree using structured seeding + local search."""
import time
import random

def _is_c4_free(n, adj_bitsets):
    """Check if graph is C4-free using bitsets."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj_bitsets[u] & adj_bitsets[v]
            # Count bits
            c = common
            cnt = 0
            while c:
                cnt += 1
                c &= c - 1
            if cnt >= 2:
                return False
    return True

def _min_degree_from_bitsets(n, adj_bitsets):
    """Compute minimum degree from bitsets."""
    min_deg = n
    for i in range(n):
        deg = bin(adj_bitsets[i]).count('1')
        if deg < min_deg:
            min_deg = deg
    return min_deg

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    best_n = 39
    best_edges = []
    
    def elapsed():
        return time.time() - start_time
    
    # Try to build graphs for n from 47 down to 40
    for target_n in range(47, 39, -1):
        if elapsed() > deadline:
            break
        min_deg_needed = target_n - 39
        
        found = False
        # Try multiple random seeds with structured initialization
        for seed in range(500):
            if elapsed() > deadline:
                break
            
            random.seed(seed)
            adj = [0] * target_n
            deg = [0] * target_n
            
            # Phase 1: Build initial graph with random edges, ensuring C4-free
            # Start with a sparse random graph, then add edges greedily
            
            # Generate all possible edges
            all_edges = []
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    all_edges.append((u, v))
            random.shuffle(all_edges)
            
            # Greedy edge addition with fast C4 check
            for u, v in all_edges:
                if elapsed() > deadline:
                    break
                
                # Check if adding (u,v) creates a C4
                # C4: u-v-w-x-u where w∈N(v)\{u}, x∈N(u)\{v}, w~x, w≠x
                A = adj[v] & ~(1 << u)  # neighbors of v except u
                B = adj[u] & ~(1 << v)  # neighbors of u except v
                
                creates_c4 = False
                # Check if there's an edge between A and B (with w≠x)
                # For each w in A, check if adj[w] & B has any bit (excluding w)
                a_mask = A
                while a_mask:
                    w_bit = a_mask & (-a_mask)
                    w = w_bit.bit_length() - 1
                    a_mask ^= w_bit
                    # Check if w is adjacent to any x in B, x≠w
                    if adj[w] & B & ~(1 << w):
                        creates_c4 = True
                        break
                
                if not creates_c4:
                    # Add edge
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    deg[u] += 1
                    deg[v] += 1
            
            # Phase 2: Local search to improve minimum degree
            # Try to increase minimum degree by adding edges or swapping
            
            # First check current min degree
            cur_min_deg = min(deg)
            
            # If min degree is already sufficient, verify and accept
            if cur_min_deg >= min_deg_needed:
                # Verify C4-free
                if _is_c4_free(target_n, adj):
                    edges = []
                    for u in range(target_n):
                        a_mask = adj[u]
                        while a_mask:
                            v_bit = a_mask & (-a_mask)
                            v = v_bit.bit_length() - 1
                            if u < v:
                                edges.append([u, v])
                            a_mask ^= v_bit
                    best_n = target_n
                    best_edges = edges
                    found = True
                    break
            
            # If not sufficient, try local search: add more edges
            # Try to add edges that don't create C4 and increase low-degree vertices
            for iteration in range(200):
                if elapsed() > deadline:
                    break
                
                # Find vertices with minimum degree
                min_d = min(deg)
                if min_d >= min_deg_needed:
                    break
                
                # Try to add an edge incident to a low-degree vertex
                low_verts = [i for i in range(target_n) if deg[i] == min_d]
                random.shuffle(low_verts)
                
                improved = False
                for u in low_verts:
                    if elapsed() > deadline:
                        break
                    
                    # Try all possible edges from u
                    candidates = []
                    for v in range(target_n):
                        if v != u and not (adj[u] >> v) & 1:
                            candidates.append(v)
                    random.shuffle(candidates)
                    
                    for v in candidates:
                        # Check if adding (u,v) creates C4
                        A = adj[v] & ~(1 << u)
                        B = adj[u] & ~(1 << v)
                        
                        creates_c4 = False
                        a_mask = A
                        while a_mask:
                            w_bit = a_mask & (-a_mask)
                            w = w_bit.bit_length() - 1
                            a_mask ^= w_bit
                            if adj[w] & B & ~(1 << w):
                                creates_c4 = True
                                break
                        
                        if not creates_c4:
                            adj[u] |= 1 << v
                            adj[v] |= 1 << u
                            deg[u] += 1
                            deg[v] += 1
                            improved = True
                            break
                    if improved:
                        break
                
                if not improved:
                    break
            
            # After local search, check again
            cur_min_deg = min(deg)
            if cur_min_deg >= min_deg_needed:
                if _is_c4_free(target_n, adj):
                    edges = []
                    for u in range(target_n):
                        a_mask = adj[u]
                        while a_mask:
                            v_bit = a_mask & (-a_mask)
                            v = v_bit.bit_length() - 1
                            if u < v:
                                edges.append([u, v])
                            a_mask ^= v_bit
                    best_n = target_n
                    best_edges = edges
                    found = True
                    break
        
        if found:
            break
    
    if best_n >= 40:
        return {'n': best_n, 'edges': best_edges}
    else:
        # Return a trivial valid graph
        return {'n': 39, 'edges': []}
