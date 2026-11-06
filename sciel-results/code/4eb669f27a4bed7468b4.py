import time
import random
import sys

def construct():
    """
    Finds a 5-coloring of K_n without monochromatic C4.
    Maximizes n.
    """
    # Hard deadline to ensure we return before 120s limit
    # 150s wall clock limit mentioned in directive, but evaluator kills at 120s?
    # "Each evaluation must finish within 120 s."
    # "The evaluator KILLS the program after 150s wall-clock."
    # We should aim to return before 115s to be safe.
    deadline = time.time() + 110.0
    
    def get_seed_solution():
        n = 11
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = min(j - i, n - (j - i))
                colors.append(d - 1)
        return n, colors

    best_n, best_colors = get_seed_solution()

    def get_edge_indices(n):
        """
        Returns list of (u, v) for all edges i<j in lexicographic order.
        Also returns a map from (u,v) to index.
        """
        edges = []
        index_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
                index_map[(i, j)] = idx
                idx += 1
        return edges, index_map

    def count_violations(n, colors):
        """
        Fast count of monochromatic C4s using bitsets.
        """
        # neighbors[c][u] is bitmask of neighbors of u in color c
        neighbors = [[0] * n for _ in range(5)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
                idx += 1
        
        cost = 0
        # Check for each color
        for c in range(5):
            # For each pair of vertices u, v
            # Common neighbors = neighbors[c][u] & neighbors[c][v]
            # If popcount(common) >= 2, it's a violation.
            # We can iterate u < v
            for u in range(n):
                nu = neighbors[c][u]
                if nu == 0:
                    continue
                for v in range(u + 1, n):
                    nv = neighbors[c][v]
                    common = nu & nv
                    # Check if at least 2 bits are set
                    # x & (x-1) removes lowest set bit. If result != 0, there was >1 bit.
                    if common & (common - 1):
                        cost += 1
        return cost

    def solve_sa(n, time_limit):
        total_edges = n * (n - 1) // 2
        edges, index_map = get_edge_indices(n)
        
        # Initial random coloring
        colors = [random.randint(0, 4) for _ in range(total_edges)]
        current_cost = count_violations(n, colors)
        
        if current_cost == 0:
            return colors
            
        best_cost = current_cost
        best_colors = colors[:]
        
        T = 1.0
        cooling_rate = 0.9998
        start_time = time.time()
        
        # Keep track of time for cooling adjustment
        while time.time() - start_time < time_limit:
            # Pick random edge
            e_idx = random.randint(0, total_edges - 1)
            u, v = edges[e_idx]
            old_c = colors[e_idx]
            
            # Pick new color
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
                
            # Apply move
            colors[e_idx] = new_c
            
            # Calculate new cost
            new_cost = count_violations(n, colors)
            
            delta = new_cost - current_cost
            # Acceptance
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
                    if best_cost == 0:
                        return best_colors
            else:
                colors[e_idx] = old_c
                
            T *= cooling_rate
            
            # Reheat if stuck?
            # Simple reheat: if T < 0.01, reset T to 0.5 and reset cooling?
            # Or just restart?
            # For now, simple cooling.
            
        return best_colors if best_cost == 0 else None

    # Strategy:
    # We know n=26 is possible.
    # We try n=29, 28, 27, 26.
    # Allocate time wisely.
    
    # Time allocation:
    # n=29: 10s
    # n=28: 10s
    # n=27: 20s
    # n=26: 60s (most likely candidate)
    # Remaining: Seed 11.
    
    targets = [
        (29, 10.0),
        (28, 10.0),
        (27, 20.0),
        (26, 60.0)
    ]
    
    for n, t_alloc in targets:
        if time.time() > deadline - 5:
            break
            
        # Ensure we don't exceed deadline
        t_alloc = min(t_alloc, deadline - time.time() - 2.0)
        if t_alloc <= 0:
            break
            
        # Run multiple restarts within t_alloc?
        # Or one long run?
        # One long run with SA is usually better for this size.
        # However, if it gets stuck in a local minimum, it might fail.
        # Let's add a simple restart mechanism inside the time limit if no improvement for a long time?
        # For simplicity, we run solve_sa which handles its own loop.
        
        res = solve_sa(n, t_alloc)
        
        if res:
            # Verify
            if count_violations(n, res) == 0:
                best_n = n
                best_colors = res
                # Return immediately as we are going downwards
                return {'n': best_n, 'colors': best_colors}
                
    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
