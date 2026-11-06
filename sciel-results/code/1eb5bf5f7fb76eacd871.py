import numpy as np
import time
import random

def construct():
    # Deadline guard: 140 seconds to be safe within 150s limit
    start_time = time.time()
    deadline = start_time + 140
    
    # Helper to generate the seed n=32 solution
    def generate_seed_solution():
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

    best_result = generate_seed_solution()
    best_n = 32
    
    # Try to find a solution for n=50
    target_n = 50
    
    # Precompute edge indices for n=50
    n = target_n
    num_edges = n * (n - 1) // 2
    edge_idx = [[-1] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1
            
    # Precompute triangle neighbors for each edge
    # tri_neighbors[e] = list of (idx_uw, idx_ew) for all w != u, v
    tri_neighbors = [[] for _ in range(num_edges)]
    for u in range(n):
        for v in range(u + 1, n):
            e = edge_idx[u][v]
            for w in range(n):
                if w != u and w != v:
                    tri_neighbors[e].append((edge_idx[u][w], edge_idx[v][w]))

    def calculate_score(colors):
        score = 0
        for u in range(n):
            for v in range(u + 1, n):
                c_uv = colors[edge_idx[u][v]]
                for w in range(v + 1, n):
                    if c_uv == colors[edge_idx[v][w]] == colors[edge_idx[u][w]]:
                        score += 1
        return score

    def solve_sa(n, time_limit):
        num_edges = n * (n - 1) // 2
        # Re-compute structures for this n if n != 50? 
        # We only call this for 50.
        
        colors = np.random.randint(0, 4, num_edges, dtype=np.int8)
        current_score = calculate_score(colors)
        
        if current_score == 0:
            return colors.tolist()
            
        T = 100.0
        T_min = 0.01
        cooling_rate = 0.999
        
        # Precompute neighbors for this specific n (already done for 50)
        # But we need to ensure tri_neighbors matches this n.
        # Since we only call for 50, it's fine.
        
        while time.time() < time_limit:
            if current_score == 0:
                return colors.tolist()
                
            # Pick a random edge
            e = np.random.randint(num_edges)
            old_color = colors[e]
            new_color = np.random.randint(0, 4)
            if new_color == old_color:
                continue
                
            # Calculate delta
            delta = 0
            for idx_uw, idx_vw in tri_neighbors[e]:
                c_uw = colors[idx_uw]
                c_vw = colors[idx_vw]
                
                # Check old state
                if c_uw == c_vw == old_color:
                    delta -= 1 # Removing a mono triangle
                # Check new state
                if c_uw == c_vw == new_color:
                    delta += 1 # Adding a mono triangle
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                colors[e] = new_color
                current_score += delta
                
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
        return None

    # Try multiple restarts for n=50
    for i in range(5):
        if time.time() > deadline:
            break
            
        # Allocate some time per attempt
        # If we have 100s left, give 20s per attempt
        attempt_deadline = min(deadline, time.time() + 20)
        
        res = solve_sa(target_n, attempt_deadline)
        if res is not None:
            return {'n': target_n, 'colors': res}
            
    # If n=50 failed, try n=49?
    # It's likely 50 works. If not, 49 might be easier.
    # But 50 is the record.
    
    return best_result
