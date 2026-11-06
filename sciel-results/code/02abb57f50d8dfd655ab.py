import time
import random
import sys

def construct():
    """
    Finds a 5-coloring of K_n without monochromatic C4 using Simulated Annealing
    with incremental cost updates.
    Maximizes n.
    """
    deadline = time.time() + 115.0  # Safety margin for 120s limit
    
    def is_valid(n, colors):
        """
        Verifies if the coloring has no monochromatic C4.
        """
        if len(colors) != n * (n - 1) // 2:
            return False
        
        neighbors = [[0] * n for _ in range(5)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                if c < 0 or c > 4:
                    return False
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
                idx += 1
        
        for c in range(5):
            for u in range(n):
                nu = neighbors[c][u]
                if nu == 0:
                    continue
                for v in range(u + 1, n):
                    nv = neighbors[c][v]
                    common = nu & nv
                    # Check if popcount(common) >= 2
                    if common & (common - 1):
                        return False
        return True

    def solve_sa(n, time_limit):
        """
        Runs Simulated Annealing to find a valid coloring for K_n.
        """
        total_edges = n * (n - 1) // 2
        
        # Initialize colors
        colors = [random.randint(0, 4) for _ in range(total_edges)]
        
        # Initialize adjacency bitsets
        adj = [[0] * n for _ in range(5)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
                
        # Calculate initial cost
        def calc_cost(adj):
            cost = 0
            for c in range(5):
                for u in range(n):
                    nu = adj[c][u]
                    if nu == 0:
                        continue
                    for v in range(u + 1, n):
                        nv = adj[c][v]
                        common = nu & nv
                        if common & (common - 1):
                            cost += 1
            return cost
        # Optimization: We can calculate initial cost more efficiently or just use the incremental logic from a zero state?
        # Easier to just compute it once.
        current_cost = calc_cost(adj)
        
        if current_cost == 0:
            return colors
            
        best_cost = current_cost
        best_colors = colors[:]
        
        # SA Parameters
        T = 1.0
        cooling_rate = 0.999
        start_time = time.time()
        
        # Precompute edge endpoints
        edge_map = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map.append((i, j))
                idx += 1
        
        while time.time() - start_time < time_limit:
            # Pick a random edge
            e_idx = random.randint(0, total_edges - 1)
            u, v = edge_map[e_idx]
            old_c = colors[e_idx]
            
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
                
            # Calculate delta cost
            delta = 0
            
            # 1. Remove from old_c
            # Update pairs (u, w) and (v, w)
            # w must be neighbor of v in old_c for (u, w) to change
            # w must be neighbor of u in old_c for (v, w) to change
            
            # Get neighbors of v in old_c
            nv_old = adj[old_c][v]
            # Iterate through set bits of nv_old
            # Exclude u
            nv_old &= ~(1 << u)
            
            while nv_old:
                lsb = nv_old & -nv_old
                w = lsb.bit_length() - 1
                nv_old ^= lsb
                
                # Check pair (u, w) in old_c
                # K_old(u, w) = popcount(adj[old_c][u] & adj[old_c][w])
                # Note: v is in adj[old_c][u] currently
                # If w is in adj[old_c][v], then v is a common neighbor of u and w.
                # So K_old includes v.
                # After removal, v is not in adj[old_c][u], so K decreases by 1.
                # We need to check if K_old == 2.
                
                nu_w = adj[old_c][u] & adj[old_c][w]
                k_uw = nu_w.bit_count()
                if k_uw == 2:
                    delta -= 1
                    
            # Get neighbors of u in old_c
            nu_old = adj[old_c][u]
            nu_old &= ~(1 << v)
            
            while nu_old:
                lsb = nu_old & -nu_old
                w = lsb.bit_length() - 1
                nu_old ^= lsb
                
                # Check pair (v, w) in old_c
                nv_w = adj[old_c][v] & adj[old_c][w]
                k_vw = nv_w.bit_count()
                if k_vw == 2:
                    delta -= 1

            # Apply removal
            adj[old_c][u] ^= (1 << v)
            adj[old_c][v] ^= (1 << u)
            
            # 2. Add to new_c
            # Update pairs (u, w) and (v, w)
            # w must be neighbor of v in new_c for (u, w) to change
            # w must be neighbor of u in new_c for (v, w) to change
            
            # Get neighbors of v in new_c
            nv_new = adj[new_c][v]
            nv_new &= ~(1 << u)
            
            while nv_new:
                lsb = nv_new & -nv_new
                w = lsb.bit_length() - 1
                nv_new ^= lsb
                
                # Check pair (u, w) in new_c
                # Currently v is NOT in adj[new_c][u].
                # K_new(u, w) = popcount(adj[new_c][u] & adj[new_c][w])
                # After adding v to adj[new_c][u], K increases by 1 (since w is neighbor of v).
                # We need to check if K_new == 1.
                
                nu_w = adj[new_c][u] & adj[new_c][w]
                k_uw = nu_w.bit_count()
                if k_uw == 1:
                    delta += 1

            # Get neighbors of u in new_c
            nu_new = adj[new_c][u]
            nu_new &= ~(1 << v)
            
            while nu_new:
                lsb = nu_new & -nu_new
                w = lsb.bit_length() - 1
                nu_new ^= lsb
                
                # Check pair (v, w) in new_c
                nv_w = adj[new_c][v] & adj[new_c][w]
                k_vw = nv_w.bit_count()
                if k_vw == 1:
                    delta += 1

            # Apply addition
            adj[new_c][u] |= (1 << v)
            adj[new_c][v] |= (1 << u)
            
            # Update cost
            new_cost = current_cost + delta
            current_cost = new_cost
            
            # Acceptance
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                colors[e_idx] = new_c
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
                    if best_cost == 0:
                        return best_colors
            else:
                # Reject: Revert changes
                colors[e_idx] = old_c
                # Revert adj
                adj[new_c][u] ^= (1 << v)
                adj[new_c][v] ^= (1 << u)
                adj[old_c][u] |= (1 << v)
                adj[old_c][v] |= (1 << u)
                current_cost -= delta # Revert cost
                
            T *= cooling_rate
            if T < 0.01:
                T = 1.0

        return best_colors if best_cost == 0 else None

    # Search plan
    # We know n=26 is possible.
    # Try 28, 27, 26.
    
    # Fallback
    def get_seed_solution():
        n = 11
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = min(j - i, n - (j - i))
                colors.append(d - 1)
        return n, colors
        
    best_n, best_colors = get_seed_solution()
    
    search_plan = [
        (28, 10.0),
        (27, 15.0),
        (26, 30.0),
        (25, 10.0),
    ]
    
    for n, t_alloc in search_plan:
        if time.time() > deadline - 2.0:
            break
            
        res = solve_sa(n, t_alloc)
        
        if res is not None:
            if is_valid(n, res):
                best_n = n
                best_colors = res
                return {'n': best_n, 'colors': best_colors}
            
    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
