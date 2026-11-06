import random
import time
import sys

def construct():
    # Deadline guard
    start_time = time.time()
    time_limit = 110  # Leave 10s margin for 120s limit

    # Fallback solution: n=32
    def get_fallback():
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

    # Try to find n=50
    n = 50
    # Initialize random coloring
    colors = [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]
    
    # Helper to get/set color
    def get_idx(i, j):
        if i > j: i, j = j, i
        return i * (n - 1) - i * (i + 1) // 2 + j - i - 1

    def count_mono_triangles():
        count = 0
        # Efficient counting is hard, just check all triangles
        # O(n^3) = 125000 ops per check. 100s / 125000 is small.
        # We need faster updates.
        # For now, simple check.
        for i in range(n):
            for j in range(i + 1, n):
                c_ij = colors[get_idx(i, j)]
                for k in range(j + 1, n):
                    c_ik = colors[get_idx(i, k)]
                    c_jk = colors[get_idx(j, k)]
                    if c_ij == c_ik == c_jk:
                        count += 1
        return count

    # Local Search
    best_count = count_mono_triangles()
    current_temp = 10.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        # Find a random mono triangle
        # We can't list them all easily without O(n^3)
        # Just pick random triangles until we find one?
        # If best_count is 0, we are done.
        if best_count == 0:
            return {'n': n, 'colors': colors}
        
        # Find a mono triangle
        found = False
        while not found:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            k = random.randint(0, n - 1)
            if i == j or j == k or i == k: continue
            a, b, c = sorted([i, j, k])
            c_ab = colors[get_idx(a, b)]
            c_ac = colors[get_idx(a, c)]
            c_bc = colors[get_idx(b, c)]
            if c_ab == c_ac == c_bc:
                found = True
                # Pick random edge to flip
                edge_choice = random.randint(0, 2)
                if edge_choice == 0:
                    idx = get_idx(a, b)
                elif edge_choice == 1:
                    idx = get_idx(a, c)
                else:
                    idx = get_idx(b, c)
                old_color = colors[idx]
                new_color = random.choice([c for c in range(4) if c != old_color])
                colors[idx] = new_color
                
                # Calculate delta
                # This is expensive to do exactly.
                # Approximate: just accept if random
                # Or do a quick local check?
                # For simplicity, use Metropolis with a rough estimate or just random walk
                # Let's just do a random walk with restarts if stuck?
                # Better: Calculate new count.
                new_count = count_mono_triangles()
                delta = new_count - best_count
                if delta < 0 or random.random() < pow(2.718, -delta / current_temp):
                    best_count = new_count
                else:
                    colors[idx] = old_color
                break
        
        current_temp *= cooling
        if current_temp < 0.01:
            break

    if best_count == 0:
        return {'n': n, 'colors': colors}
    
    return get_fallback()
