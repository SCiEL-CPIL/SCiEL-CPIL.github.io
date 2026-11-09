import random
import time
import sys

def construct():
    n = 40
    m = 401
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initialize random graph
    # Vertices 0..39
    # Generate 401 unique edges
    edges = set()
    while len(edges) < m:
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u != v:
            if u > v: u, v = v, u
            edges.add((u, v))
    
    # Convert to list for iteration
    edge_list = list(edges)
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edge_list:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Build book matrix
    # book[u][v] is number of common neighbors of u and v
    book = [[0]*n for _ in range(n)]
    for u, v in edge_list:
        common = adj[u] & adj[v]
        count = common.bit_count()
        book[u][v] = count
        book[v][u] = count
        
    def get_max_book():
        max_b = 0
        for u, v in edge_list:
            if book[u][v] > max_b:
                max_b = book[u][v]
        return max_b
        
    current_max = get_max_book()
    best_max = current_max
    best_edges = set(edge_list)
    
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.99999
    
    while time.time() < deadline:
        # Pick random edge to remove
        u, v = random.choice(edge_list)
        
        # Pick random non-edge to add
        # We need a pair (x, y) not in edges
        # Efficient way: pick x, y randomly until not in edges
        # Since graph is sparse (401/780 ~ 51%), this is fast
        while True:
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x == y: continue
            if x > y: x, y = y, x
            if (x, y) not in edges:
                break
        
        # Check if swap is valid (it is, we remove one, add one)
        # Calculate delta max book?
        # Let's just apply, calculate new max, and revert if worse (with prob)
        
        # Apply swap
        # 1. Remove (u, v)
        edges.remove((u, v))
        edge_list.remove((u, v)) # O(N) removal, but N=401, ok. 
        # Actually, list removal is O(N). 401 is small.
        
        # Update adj
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)
        
        # Update book for neighbors of u,v
        common_uv = adj[u] & adj[v] # Before clearing bits? No, we cleared bits in adj first?
        # Wait, adj[u] and adj[v] are updated. 
        # The common neighbors of u and v in the NEW graph (without u-v) are adj[u] & adj[v].
        # But the triangles destroyed are those w where w is common neighbor in OLD graph.
        # Old common neighbors: old_adj[u] & old_adj[v].
        # We need to decrement book[u][w] and book[v][w] for w in old_common.
        
        # Let's redo the update logic carefully.
        
        # Revert to safe state for logic clarity, or just do it in place carefully.
        # Let's revert the edge list and edges set first to think clearly.
        edges.add((u, v))
        edge_list.append((u, v))
        
        # Correct Update Logic:
        # 1. Identify old common neighbors of u, v
        old_common_uv = adj[u] & adj[v]
        
        # 2. Remove edge (u, v) from adj
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)
        
        # 3. Decrement book values for triangles destroyed
        # For each w in old_common_uv:
        #   book[u][w] -= 1
        #   book[v][w] -= 1
        # Note: book[u][v] is no longer valid, set to 0
        book[u][v] = 0
        book[v][u] = 0
        
        w = old_common_uv
        while w:
            lsb = w & -w
            w_idx = lsb.bit_length() - 1
            book[u][w_idx] -= 1
            book[v][w_idx] -= 1
            w ^= lsb
            
        # 4. Identify new common neighbors of x, y
        # Note: adj[x] and adj[y] are current (without u-v)
        new_common_xy = adj[x] & adj[y]
        
        # 5. Add edge (x, y) to adj
        adj[x] ^= (1 << y)
        adj[y] ^= (1 << x)
        
        # 6. Increment book values for triangles created
        # For each z in new_common_xy:
        #   book[x][z] += 1
        #   book[y][z] += 1
        # book[x][y] = popcount(new_common_xy)
        
        z = new_common_xy
        while z:
            lsb = z & -z
            z_idx = lsb.bit_length() - 1
            book[x][z_idx] += 1
            book[y][z_idx] += 1
            z ^= lsb
            
        book[x][y] = new_common_xy.bit_count()
        book[y][x] = book[x][y]
        
        # 7. Update edge sets
        edges.remove((u, v))
        # Remove (u,v) from edge_list
        # Find index
        # O(N)
        for i, e in enumerate(edge_list):
            if e == (u, v):
                edge_list[i] = (x, y)
                break
        edges.add((x, y))
        
        # 8. Calculate new max
        new_max = get_max_book()
        
        # 9. Accept/Reject
        if new_max < current_max:
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_edges = set(edge_list)
                # Optional: restart or continue
        else:
            delta = new_max - current_max
            if delta == 0:
                current_max = new_max
            else:
                # Reject with probability 1 - exp(-delta/T)
                if random.random() < (1.0 - (2.718281828 ** (-delta / T))):
                    current_max = new_max
                else:
                    # Revert
                    # This is complex. 
                    # It's easier to just not apply the move if we decide to reject?
                    # But we already applied it.
                    # We should calculate acceptance BEFORE applying?
                    # No, we need new_max to decide.
                    # So we must revert.
                    
                    # Revert logic is same as swap back.
                    # Swap (x,y) back to (u,v)
                    
                    # Remove (x,y)
                    old_common_xy_revert = adj[x] & adj[y]
                    adj[x] ^= (1 << y)
                    adj[y] ^= (1 << x)
                    book[x][y] = 0
                    book[y][x] = 0
                    z = old_common_xy_revert
                    while z:
                        lsb = z & -z
                        z_idx = lsb.bit_length() - 1
                        book[x][z_idx] -= 1
                        book[y][z_idx] -= 1
                        z ^= lsb
                        
                    # Add (u,v)
                    new_common_uv_revert = adj[u] & adj[v]
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    u = u
                    v = v
                    w = new_common_uv_revert
                    while w:
                        lsb = w & -w
                        w_idx = lsb.bit_length() - 1
                        book[u][w_idx] += 1
                        book[v][w_idx] += 1
                        w ^= lsb
                    book[u][v] = new_common_uv_revert.bit_count()
                    book[v][u] = book[u][v]
                    
                    # Revert edge sets
                    edges.remove((x, y))
                    for i, e in enumerate(edge_list):
                        if e == (x, y):
                            edge_list[i] = (u, v)
                            break
                    edges.add((u, v))
                    
                    # current_max remains unchanged
            T *= cooling_rate
            
    # Return best edges found
    # Format: list of [u, v]
    return [[u, v] for u, v in best_edges]
