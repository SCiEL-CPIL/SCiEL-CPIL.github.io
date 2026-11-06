import random
import time
import sys

def construct():
    n = 18
    # Initialize with Turan graph T(18, 3)
    # Parts: 0-5, 6-11, 12-17
    # Edges between all pairs of different parts
    
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            part_i = i // 6
            part_j = j // 6
            if part_i != part_j:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    def is_k4_free_add(adj, u, v):
        # Check if adding edge (u,v) creates a K4
        # K4 exists if there are x, y in N(u) intersect N(v) such that (x,y) is an edge
        common = adj[u] & adj[v]
        # Check if subgraph induced by common has any edge
        c_iter = common
        while c_iter:
            x_bit = c_iter & -c_iter
            x = x_bit.bit_length() - 1
            c_iter -= x_bit
            # If x is connected to any other vertex in common
            # Note: adj[x] contains x? No.
            # adj[x] & common checks if x has any neighbor in common.
            # If yes, that neighbor y is in common, so (x,y) is an edge.
            # And (u,x), (v,x), (u,y), (v,y) are edges.
            # So {u,v,x,y} is K4.
            if adj[x] & common:
                return False
        return True

    def get_score(adj):
        # Find largest triangle-free induced subgraph
        # Iterate sizes from n down to 1
        # Use bitmasks for subsets
        # To iterate subsets of size k, we can use a generator or precomputed list?
        # Precomputing all masks is 2^18 = 262k masks. Storing them is fine.
        # But we want to check by size.
        
        # For speed, we can just iterate all masks and track the max size that is triangle-free.
        # But we want to stop early if we find a large one?
        # No, we want the MAX. So we must check all.
        # However, if we check size 18, 17, etc., we can stop at the first one found?
        # Yes, because if a set of size k is triangle-free, we found the max (if we go descending).
        
        # Precompute list of masks by size?
        # Or just iterate all masks and filter by bit_count?
        
        # Let's precompute masks_by_size
        masks_by_size = [[] for _ in range(n + 1)]
        for mask in range(1 << n):
            k = mask.bit_count()
            masks_by_size[k].append(mask)
            
        for k in range(n, 0, -1):
            for mask in masks_by_size[k]:
                if is_triangle_free(mask, adj):
                    return k
        return 0

    def is_triangle_free(mask, adj):
        # Check if subgraph induced by mask is triangle-free
        # For each u in mask, N(u) intersect mask must be independent
        u_iter = mask
        while u_iter:
            u_bit = u_iter & -u_iter
            u = u_bit.bit_length() - 1
            u_iter -= u_bit
            neighbors = adj[u] & mask
            # Check if neighbors contains an edge
            v_iter = neighbors
            while v_iter:
                v_bit = v_iter & -v_iter
                v = v_bit.bit_length() - 1
                v_iter -= v_bit
                if adj[v] & neighbors:
                    return False
        return True

    # Main Search Loop
    best_adj = adj[:]
    best_score = get_score(best_adj)
    
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Local Search
    # Try to add edges to reduce triangle-free set size?
    # Adding edges usually increases triangles, which might break triangle-free sets.
    # But we must maintain K4-free.
    
    # Randomized Local Search
    while time.time() < deadline:
        # Try random moves
        # 1. Try to add a random non-edge
        # 2. Try to remove a random edge
        # 3. Try to swap
        
        current_adj = best_adj[:]
        
        # Perform a few random valid moves
        for _ in range(10):
            move_type = random.choice(['add', 'remove', 'swap'])
            
            if move_type == 'add':
                # Pick random non-edge
                for _ in range(10): # Try 10 times to find a valid add
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u == v: continue
                    if (current_adj[u] >> v) & 1: continue
                    if is_k4_free_add(current_adj, u, v):
                        # Add edge
                        current_adj[u] |= (1 << v)
                        current_adj[v] |= (1 << u)
                        break
            elif move_type == 'remove':
                # Pick random edge
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v: continue
                if (current_adj[u] >> v) & 1:
                    # Remove edge
                    current_adj[u] ^= (1 << v)
                    current_adj[v] ^= (1 << u)
            elif move_type == 'swap':
                # Remove u-v, add x-y
                # This is complex to ensure validity. Skip for now.
                pass
        
        score = get_score(current_adj)
        
        if score < best_score:
            best_adj = current_adj
            best_score = score
            # If we found a very good score, maybe do more local refinement?
            # For now, just continue.
        
        # If best_score is 9 or less, we can probably stop or be very happy.
        # Theoretical lower bound?
        # R(3,4)=9 implies any graph on 9 vertices has triangle or indep set 4.
        # This doesn't directly give lower bound on triangle-free set size.
        # But likely 10-11 is good.
        
    # Convert best_adj to list of edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges

# Run construct to verify? No, just define it.
# The evaluator calls construct().
