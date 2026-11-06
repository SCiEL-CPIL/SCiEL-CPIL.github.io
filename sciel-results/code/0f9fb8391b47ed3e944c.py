import time
import random
import sys
import numpy as np

def construct():
    """
    Finds a 5-coloring of K_n without monochromatic C4.
    Maximizes n.
    """
    deadline = time.time() + 145.0  # Safety margin for 150s limit

    # Known seed solution for n=11
    def get_seed_solution():
        n = 11
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = min(j - i, n - (j - i))
                colors.append(d - 1)
        return n, colors

    best_n, best_colors = get_seed_solution()

    def get_edge_index(i, j, n):
        if i > j:
            i, j = j, i
        # Standard lexicographic index for (i, j) with i < j
        # Number of edges with first vertex < i: i * (i - 1) / 2
        # Edges with first vertex i and second vertex < j: j - i - 1
        return i * (i - 1) // 2 + (j - i - 1)

    def is_valid(n, colors):
        """
        Verifies if the coloring has no monochromatic C4.
        """
        if len(colors) != n * (n - 1) // 2:
            return False
        
        # Build neighbor bitsets for each color
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
        
        # Check for C4:
        for c in range(5):
            for u in range(n):
                nu = neighbors[c][u]
                if nu == 0:
                    continue
                for v in range(u + 1, n):
                    nv = neighbors[c][v]
                    common = nu & nv
                    if common & (common - 1):
                        return False
        return True

    def count_violations(n, colors):
        """
        Counts the total number of monochromatic C4s (pairs with >=2 common neighbors).
        """
        neighbors = [[0] * n for _ in range(5)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
                idx += 1
        
        cost = 0
        for c in range(5):
            for u in range(n):
                nu = neighbors[c][u]
                if nu == 0:
                    continue
                for v in range(u + 1, n):
                    nv = neighbors[c][v]
                    common = nu & nv
                    if common & (common - 1):
                        cost += 1
        return cost

    def solve_sa(n, time_limit):
        """
        Runs Simulated Annealing to find a valid coloring for K_n.
        """
        total_edges = n * (n - 1) // 2
        
        # Initial random coloring
        colors = [random.randint(0, 4) for _ in range(total_edges)]
        
        # Precompute edge endpoints for fast access
        edge_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[idx] = (i, j)
                idx += 1
        
        current_cost = count_violations(n, colors)
        
        if current_cost == 0:
            return colors
            
        best_cost = current_cost
        best_colors = colors[:]
        
        # SA Parameters
        T = 1.0
        cooling_rate = 0.9999  # Slower cooling for better exploration
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            # Pick a random edge
            e_idx = random.randint(0, total_edges - 1)
            u, v = edge_map[e_idx]
            old_c = colors[e_idx]
            
            # Pick a new color different from old
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
                
            # Apply move
            colors[e_idx] = new_c
            
            # Calculate new cost
            new_cost = count_violations(n, colors)
            
            # Acceptance probability
            delta = new_cost - current_cost
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
                    if best_cost == 0:
                        return best_colors
            else:
                # Reject
                colors[e_idx] = old_c
                
            T *= cooling_rate
            
            # Reheat if stuck
            if T < 0.01:
                T = 1.0

        return best_colors if best_cost == 0 else None

    # Strategy:
    # Try n from 29 down to 12.
    # For each n, run SA for a few seconds.
    # If found, return.
    
    # Time allocation
    # n=29, 28, 27: 5s each
    # n=26: 20s (known to be possible, but SA might need more time to find it from random)
    # n=25..12: 5s each
    
    # Note: SA from random start for n=26 might be hard.
    # We can improve by starting from a "structured" random start or using a better initialization.
    # For now, random is the baseline.
    
    # To improve success rate for n=26, we can try multiple restarts within the time limit.
    
    for n in range(29, 11, -1):
        if time.time() > deadline - 5:
            break
            
        if n >= 27:
            t_alloc = 5.0
        elif n == 26:
            t_alloc = 30.0
        else:
            t_alloc = 5.0
            
        t_alloc = min(t_alloc, deadline - time.time() - 2.0)
        if t_alloc <= 0:
            break
            
        # Run SA with multiple restarts
        # Each restart gets t_alloc / num_restarts
        # Let's do 5 restarts for n=26, 1 for others to keep it simple?
        # Actually, SA is stochastic. More restarts help.
        # Let's just run one long SA instance. The cooling schedule handles the exploration/exploitation balance.
        
        res = solve_sa(n, t_alloc)
        
        if res:
            # Verify
            if is_valid(n, res):
                best_n = n
                best_colors = res
                return {'n': best_n, 'colors': best_colors}
                
    # If all fail, return seed
    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
