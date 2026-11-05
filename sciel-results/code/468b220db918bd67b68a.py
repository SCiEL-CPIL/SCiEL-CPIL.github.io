import time
from math import gcd

N = 512
DEADLINE = 82.0  # Safety margin

def construct():
    start_time = time.time()
    
    # 1. Build conflict graph
    adj = [0] * N
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            g = gcd(a, b)
            if (a * b) // g > N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # 2. Reorder vertices by degree descending for better B&B
    degrees = [adj[i].bit_count() for i in range(N)]
    order = sorted(range(N), key=lambda x: -degrees[x])
    
    new_index = [0] * N
    for new_idx, orig_idx in enumerate(order):
        new_index[orig_idx] = new_idx
    
    adj_new = [0] * N
    for i in range(N):
        orig_i = order[i]
        mask = 0
        m = adj[orig_i]
        while m:
            lsb = m & -m
            orig_j = lsb.bit_length() - 1
            new_j = new_index[orig_j]
            mask |= (1 << new_j)
            m ^= lsb
        adj_new[i] = mask

    # 3. Greedy coloring bound
    def get_color_bound(R):
        colors = 0
        remaining = R
        while remaining:
            v_bit = remaining & -remaining
            v = v_bit.bit_length() - 1
            color_class = 0
            pool = remaining
            while pool:
                u_bit = pool & -pool
                u = u_bit.bit_length() - 1
                color_class |= u_bit
                pool &= ~u_bit
                pool &= ~adj_new[u]
            remaining &= ~color_class
            colors += 1
        return colors

    # 4. Initial solution: divisors of 360 (24 divisors, all pairwise lcm <= 360 <= 512)
    init_vals = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36, 40, 45, 60, 72, 90, 120, 180, 360]
    init_mask = 0
    for val in init_vals:
        orig_idx = val - 1
        new_idx = new_index[orig_idx]
        init_mask |= (1 << new_idx)
    
    # Verify validity
    valid = True
    m = init_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        if adj_new[v] & init_mask:
            valid = False
            break
        m ^= lsb
    
    best_size = len(init_vals) if valid else 1
    best_set = init_vals[:] if valid else [1]
    
    # 5. Branch and Bound to find improvements
    def solve(R, size, current_set_indices):
        nonlocal best_size, best_set, start_time
        
        if time.time() - start_time > DEADLINE:
            return
        
        if R == 0:
            if size > best_size:
                best_size = size
                best_set = [order[i] + 1 for i in current_set_indices]
            return
        
        bound = get_color_bound(R)
        if size + bound <= best_size:
            return
        
        # Select vertex: highest degree in R
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
        new_R_incl = R & ~(adj_new[best_v] | (1 << best_v))
        current_set_indices.append(best_v)
        solve(new_R_incl, size + 1, current_set_indices)
        current_set_indices.pop()
        
        # Branch 2: Exclude best_v
        new_R_excl = R & ~(1 << best_v)
        solve(new_R_excl, size, current_set_indices)
        
    # Start B&B from the full set
    initial_mask = (1 << N) - 1
    solve(initial_mask, 0, [])
    
    if not best_set:
        return [1]
    return best_set

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
