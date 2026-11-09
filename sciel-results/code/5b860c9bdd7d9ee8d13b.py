import time
import random
import sys

def construct():
    """
    Uses Simulated Annealing to find a 4-coloring of K_n with no monochromatic triangles.
    This approach allows escaping local minima, unlike the previous hill-climbing method.
    """
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer

    def get_idx(n, i, j):
        if i > j:
            i, j = j, i
        return i * n - i * (i + 1) // 2 + (j - i - 1)

    def count_initial_triangles(colors, n):
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                c_ij = colors[get_idx(n, i, j)]
                for k in range(j + 1, n):
                    c_ik = colors[get_idx(n, i, k)]
                    c_jk = colors[get_idx(n, j, k)]
                    if c_ij == c_ik == c_jk:
                        count += 1
        return count

    def solve_sa(n, time_budget):
        """
        Run Simulated Annealing for n vertices within time_budget.
        Returns (colors, triangle_count) or None if no solution found.
        """
        num_edges = n * (n - 1) // 2
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        # Precompute neighbors for fast triangle updates
        # For each edge (i,j), list of k's
        # Actually, we can just iterate k in range(n) and skip i,j.
        
        current_energy = count_initial_triangles(colors, n)
        
        if current_energy == 0:
            return colors, 0

        T = 10.0
        T_min = 0.01
        cooling = 0.9995
        
        # We want to minimize energy (number of monochromatic triangles)
        while T > T_min:
            if time.time() - start_time > time_budget:
                break
            
            # Pick a random edge
            idx = random.randint(0, num_edges - 1)
            # Reverse map idx to (i,j)
            # This is slow if done every time. 
            # Better: pick random i, j directly.
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old_c = colors[get_idx(n, i, j)]
            if old_c == 3:
                new_c = random.randint(0, 2)
            else:
                new_c = random.randint(0, 3)
                while new_c == old_c:
                    new_c = random.randint(0, 3)
            
            # Calculate delta energy
            delta = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                # Triangles (i,j,k)
                # Edges: (i,j), (i,k), (j,k)
                idx_ik = get_idx(n, i, k)
                idx_jk = get_idx(n, j, k)
                c_ik = colors[idx_ik]
                c_jk = colors[idx_jk]
                
                # Before: (i,j) is old_c
                if old_c == c_ik == c_jk:
                    delta -= 1 # We are removing a mono triangle (if new_c != c_ik == c_jk)
                    # Wait, delta is Energy_new - Energy_old
                    # If it was mono before, energy contribution was 1.
                    # If it is NOT mono after, energy contribution is 0.
                    # So delta += -1.
                
                # After: (i,j) is new_c
                if new_c == c_ik == c_jk:
                    delta += 1 # We are adding a mono triangle
            
            # Acceptance criterion
            if delta < 0:
                # Always accept improvement
                colors[get_idx(n, i, j)] = new_c
                current_energy += delta
            else:
                # Accept with probability exp(-delta / T)
                if random.random() < pow(2.71828, -delta / T):
                    colors[get_idx(n, i, j)] = new_c
                    current_energy += delta
            
            T *= cooling
            
            if current_energy == 0:
                return colors, 0

        return colors, current_energy

    best_n = 0
    best_colors = None
    
    # Try to find a solution for n=50, then 49, etc.
    # We allocate time based on target n.
    targets = [50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32]
    
    for n in targets:
        if time.time() - start_time > time_limit:
            break
        
        # Allocate some time for this n
        # If n is 50, give it more time.
        # Let's just run until time runs out for the first few, then less.
        # Actually, if we find a solution for 50, we stop.
        # If not, we try 49.
        
        # To ensure we don't waste all time on 50 if it's hard,
        # we can set a sub-limit. But 50 is the record, so it should be findable.
        # Let's give 50 60s, 49 20s, 48 10s, etc.
        if n == 50:
            budget = 60.0
        elif n == 49:
            budget = 20.0
        elif n == 48:
            budget = 10.0
        else:
            budget = 5.0
            
        if time.time() - start_time + budget > time_limit:
            budget = time_limit - (time.time() - start_time)
            if budget <= 0:
                break
        
        result = solve_sa(n, time.time() - start_time + budget)
        
        if result is not None:
            cols, energy = result
            if energy == 0:
                best_n = n
                best_colors = cols
                break # Found a valid coloring for this n, which is the max we tried.
            # If energy > 0, we didn't solve it. We can keep the best partial?
            # No, the task requires "No colour class may contain a triangle".
            # So we must have energy 0.
            
    if best_colors is not None:
        return {'n': best_n, 'colors': best_colors}
    
    # Fallback: Return a known valid small coloring (n=32 from seed logic or similar)
    # Re-implementing the seed's n=32 logic as a safe fallback
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': colors}
