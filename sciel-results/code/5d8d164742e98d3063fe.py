import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    def is_valid(colors, n):
        """
        Check if the coloring is valid (no monochromatic C4).
        """
        neighbors = [[0] * 5 for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                neighbors[i][c] |= (1 << j)
                neighbors[j][c] |= (1 << i)
                idx += 1
        
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = neighbors[u][c] & neighbors[v][c]
                    if common.bit_count() >= 2:
                        return False
        return True

    def count_violations(colors, n):
        """
        Count violations.
        """
        neighbors = [[0] * 5 for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                neighbors[i][c] |= (1 << j)
                neighbors[j][c] |= (1 << i)
                idx += 1
        
        cost = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = neighbors[u][c] & neighbors[v][c]
                    cnt = common.bit_count()
                    if cnt > 1:
                        cost += cnt * (cnt - 1) // 2
        return cost

    def solve_n(n, max_time=10):
        """
        Local search to find a valid coloring for K_n.
        """
        num_edges = n * (n - 1) // 2
        if num_edges == 0:
            return []
        
        endpoints = []
        for i in range(n):
            for j in range(i + 1, n):
                endpoints.append((i, j))
        
        best_cost = float('inf')
        best_colors = None
        
        for restart in range(100):
            if time.time() > start_time + max_time:
                break
                
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            
            neighbors = [[0] * 5 for _ in range(n)]
            for e_idx, (u, v) in enumerate(endpoints):
                c = colors[e_idx]
                neighbors[u][c] |= (1 << v)
                neighbors[v][c] |= (1 << u)
            
            current_cost = count_violations(colors, n)
            
            if current_cost == 0:
                return colors
            
            no_improve_count = 0
            max_no_improve = 2000
            
            while no_improve_count < max_no_improve:
                if time.time() > start_time + max_time:
                    break
                    
                e_idx = random.randint(0, num_edges - 1)
                u, v = endpoints[e_idx]
                old_c = colors[e_idx]
                new_c = random.randint(0, 4)
                
                if old_c == new_c:
                    continue
                
                neighbors[u][old_c] &= ~(1 << v)
                neighbors[v][old_c] &= ~(1 << u)
                neighbors[u][new_c] |= (1 << v)
                neighbors[v][new_c] |= (1 << u)
                colors[e_idx] = new_c
                
                new_cost = count_violations(colors, n)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    no_improve_count = 0
                    if current_cost == 0:
                        return colors
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors.copy()
                else:
                    neighbors[u][new_c] &= ~(1 << v)
                    neighbors[v][new_c] &= ~(1 << u)
                    neighbors[u][old_c] |= (1 << v)
                    neighbors[v][old_c] |= (1 << u)
                    colors[e_idx] = old_c
                    no_improve_count += 1
                    
        return best_colors

    # Try to solve for n=26, then 27
    for n in [26, 27]:
        if time.time() > start_time + 100:
            break
        res = solve_n(n, max_time=40)
        if res:
            return {'n': n, 'colors': res}
            
    # Fallback: try smaller n if 26/27 failed
    for n in [20, 15]:
        if time.time() > start_time + 110:
            break
        res = solve_n(n, max_time=5)
        if res:
            return {'n': n, 'colors': res}
            
    # Final fallback
    n = 7
    colors = [random.randint(0, 4) for _ in range(n * (n - 1) // 2)]
    return {'n': n, 'colors': colors}
