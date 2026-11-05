import time
from math import gcd

N = 512
DEADLINE = 85.0

def construct():
    start_time = time.time()
    
    # 1. Build conflict graph
    # adj[i] is a bitmask of j such that lcm(i+1, j+1) > N
    adj = [0] * N
    
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            g = gcd(a, b)
            if (a * b) // g > N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # 2. Reorder vertices by degree descending
    # This helps the greedy coloring and branching heuristics
    degrees = [adj[i].bit_count() for i in range(N)]
    order = sorted(range(N), key=lambda x: -degrees[x])
    
    # Map original indices to new indices
    # new_index[orig_idx] = new_idx
    new_index = [0] * N
    for new_idx, orig_idx in enumerate(order):
        new_index[orig_idx] = new_idx
        
    # Build adj in new order
    # adj_new[i] is bitmask of neighbors of vertex 'order[i]' in the new indexing
    adj_new = [0] * N
    for i in range(N):
        orig_i = order[i]
        mask = 0
        # Iterate through bits in adj[orig_i]
        m = adj[orig_i]
        while m:
            lsb = m & -m
            orig_j = lsb.bit_length() - 1
            new_j = new_index[orig_j]
            mask |= (1 << new_j)
            m ^= lsb
        adj_new[i] = mask

    # 3. Greedy Coloring Upper Bound
    # Returns the number of colors used in a greedy coloring of the subgraph induced by R
    def greedy_color_bound(R):
        colors = 0
        remaining = R
        while remaining:
            # Pick a vertex from remaining to start a color class
            # Heuristic: pick vertex with highest degree in 'remaining'
            # For speed, we can just pick the lowest set bit, or first one.
            # Let's try lowest set bit for speed, it's often sufficient.
            # If we want better bound, highest degree is better but slower.
            # Let's stick to lowest set bit for now to keep it fast.
            v_bit = remaining & -remaining
            v = v_bit.bit_length() - 1
            
            # Remove v from remaining
            remaining ^= v_bit
            
            # Find all vertices in 'remaining' that are NOT adjacent to v
            # These can share the same color
            candidates = remaining & ~adj_new[v]
            
            # We will greedily assign colors to all candidates that are independent
            # Actually, the standard greedy coloring algorithm:
            # 1. Start with empty color class C.
            # 2. While there are uncolored vertices:
            #    a. Pick a vertex v.
            #    b. Add v to C.
            #    c. Remove v and its neighbors from the pool of candidates for C.
            # 3. Repeat for next color.
            
            # Let's implement the standard greedy coloring properly.
            # The 'remaining' variable holds all uncolored vertices.
            # We are constructing one color class at a time.
            
            # Wait, the logic above `candidates = remaining & ~adj_new[v]` is for finding vertices compatible with v.
            # But we need to remove them from 'remaining' as we assign them to this color class.
            
            # Let's restart the inner loop logic.
            # We have 'remaining' (uncolored).
            # We pick v.
            # We want to assign as many vertices as possible to this color class.
            # A vertex u can be in this color class if it is not adjacent to any vertex already in this class.
            
            # Let's use a simpler approach:
            # color_class = 0
            # pool = remaining
            # while pool:
            #   pick v from pool
            #   color_class |= (1 << v)
            #   pool &= ~(1 << v)
            #   pool &= ~adj_new[v]  # remove neighbors of v from pool
            # 
            # Then remaining &= ~color_class
            # colors += 1
            
            # This is correct.
            
            # Re-implementing inside the while loop
            pass # Placeholder, will rewrite function below

    def get_color_bound(R):
        colors = 0
        remaining = R
        while remaining:
            # Pick a vertex v from remaining
            # Heuristic: lowest set bit
            v_bit = remaining & -remaining
            v = v_bit.bit_length() - 1
            
            # Start a new color class
            color_class = 0
            pool = remaining
            
            while pool:
                # Pick a vertex from pool
                u_bit = pool & -pool
                u = u_bit.bit_length() - 1
                
                # Add u to this color class
                color_class |= u_bit
                
                # Remove u and its neighbors from pool
                pool &= ~u_bit
                pool &= ~adj_new[u]
                
            # Remove all vertices in this color class from remaining
            remaining &= ~color_class
            colors += 1
            
        return colors

    # 4. Branch and Bound
    best_size = 0
    best_set = []
    
    # Initial heuristic solution: Divisors of 360
    # Map values to new indices
    init_vals = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36, 40, 45, 60, 72, 90, 120, 180, 360]
    # Verify validity (should be valid)
    # Convert to new indices
    init_mask = 0
    for val in init_vals:
        orig_idx = val - 1
        new_idx = new_index[orig_idx]
        init_mask |= (1 << new_idx)
        
    # Check if init_mask is independent
    # For each v in init_mask, adj_new[v] & init_mask should be 0
    valid = True
    m = init_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        if adj_new[v] & init_mask:
            valid = False
            break
        m ^= lsb
        
    if valid:
        best_size = len(init_vals)
        # Store best_set as original values
        best_set = init_vals[:]
    else:
        # Fallback to single element
        best_size = 1
        best_set = [1]
        
    def solve(R, size, current_set_indices):
        nonlocal best_size, best_set, start_time
        
        # Deadline check
        if time.time() - start_time > DEADLINE:
            return
            
        if R == 0:
            if size > best_size:
                best_size = size
                # Convert new indices back to original values
                best_set = [order[i] + 1 for i in current_set_indices]
            return
        
        # Bound
        bound = get_color_bound(R)
        if size + bound <= best_size:
            return
            
        # Select vertex to branch on
        # Heuristic: Highest degree in R
        # Iterate through bits in R
        best_v = -1
        max_deg = -1
        temp = R
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = (adj_new[v] & R).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp ^= lsb
            
        if best_v == -1:
            return
            
        # Branch 1: Include best_v
        # Remove best_v and its neighbors from R
        new_R_incl = R & ~(adj_new[best_v] | (1 << best_v))
        current_set_indices.append(best_v)
        solve(new_R_incl, size + 1, current_set_indices)
        current_set_indices.pop()
        
        # Branch 2: Exclude best_v
        new_R_excl = R & ~(1 << best_v)
        solve(new_R_excl, size, current_set_indices)
        
    initial_mask = (1 << N) - 1
    solve(initial_mask, 0, [])
    
    # Return the best set found
    if not best_set:
        return [1]
        
    return best_set

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    # Verify
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            a, b = res[i], res[j]
            if (a // gcd(a, b)) * b > 512:
                print(f"Invalid pair: {a}, {b}")
                valid = False
                break
        if not valid:
            break
    if valid:
        print("All pairs valid.")
