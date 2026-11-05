import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Try to find a valid coloring for n=50
    n = 50
    
    # Greedy Initialization
    # colors[i][j] is the color of edge (i, j)
    # We maintain adjacency matrices for each color
    # A_c[i][j] = 1 if color(i, j) == c, else 0
    
    # For efficiency, we track the number of monochromatic triangles
    # T = sum_c trace(A_c^3)
    
    # We will use a list of 4 numpy arrays
    A = [np.zeros((n, n), dtype=np.int8) for _ in range(4)]
    colors = np.full((n, n), -1, dtype=np.int8)
    
    # Greedy fill
    # For each pair (i, j), choose color c that minimizes increase in triangles
    # Increase in triangles for color c when adding edge (i, j) with color c:
    # It forms triangles with k if (i, k) and (j, k) are both color c.
    # Count = sum_k A_c[i][k] * A_c[j][k]
    
    for i in range(n):
        for j in range(i + 1, n):
            best_c = 0
            best_count = float('inf')
            for c in range(4):
                # Count common neighbors in color c
                # A_c[i] and A_c[j] are rows
                count = np.dot(A[c][i], A[c][j])
                if count < best_count:
                    best_count = count
                    best_c = c
            
            colors[i, j] = best_c
            colors[j, i] = best_c
            A[best_c][i, j] = 1
            A[best_c][j, i] = 1
            
            # Check time
            if time.time() > deadline - 10:
                # Fall back to known valid n=32
                return _get_n32()

    # Local Search
    # Objective: Minimize T = sum_c trace(A_c^3)
    # We can approximate T or compute it exactly.
    # Computing T exactly is fast for n=50.
    
    def get_T():
        t = 0
        for c in range(4):
            # trace(A^3)
            t += np.trace(A[c] @ A[c] @ A[c])
        return t
    
    current_T = get_T()
    if current_T == 0:
        return _format_output(n, colors)

    # Simulated Annealing
    # Temperature decay
    T_start = 100.0
    T_end = 0.1
    # We don't know the number of steps, so we use time.
    # Or we can just do hill climbing with random restarts if stuck.
    # Given the structure, hill climbing often works.
    
    # Let's try simple Hill Climbing first, with random restarts if stuck.
    # Actually, Simulated Annealing is better for escaping local minima.
    
    # We will run for the remaining time.
    # Steps per second: measure roughly.
    
    # To speed up, we can precompute the "delta" for each edge change?
    # No, the state changes.
    # But we can compute the change in T for a specific move (u, v, c_old, c_new).
    
    # Change (u, v) from c_old to c_new:
    # 1. Remove edge (u, v) from A[c_old].
    # 2. Add edge (u, v) to A[c_new].
    # 3. Calculate new T.
    
    # Since n=50, matrix multiplication is fast.
    # Let's just recompute T after each move.
    # 125,000 ops * 4 colors = 500,000 ops per T calculation.
    # In Python/NumPy, this is ~0.1ms to 1ms.
    # So we can do ~1000-10000 moves per second.
    # 140 seconds * 1000 = 140,000 moves.
    # This should be sufficient.
    
    # To be safe, let's optimize the T calculation.
    # T = sum_c sum_i (A_c^3)_ii
    # We can just use np.trace.
    
    # We will use a simple loop.
    # To avoid global variable lookup overhead, keep A local.
    
    # We'll do a restart strategy.
    # If we don't find 0 in 10s, restart with a new random seed?
    # Or just continue.
    
    # Let's try a few restarts if the first one fails quickly.
    # But greedy init is deterministic.
    # So we need to randomize the greedy choice if counts are tied.
    
    # Let's modify greedy init to be random among best choices.
    
    # Restart loop
    max_restarts = 5
    for attempt in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        # Re-initialize with randomness
        A = [np.zeros((n, n), dtype=np.int8) for _ in range(4)]
        colors = np.full((n, n), -1, dtype=np.int8)
        
        for i in range(n):
            for j in range(i + 1, n):
                counts = [0, 0, 0, 0]
                for c in range(4):
                    counts[c] = np.dot(A[c][i], A[c][j])
                
                min_count = min(counts)
                # Find all colors with min count
                candidates = [c for c in range(4) if counts[c] == min_count]
                best_c = candidates[np.random.randint(len(candidates))]
                
                colors[i, j] = best_c
                colors[j, i] = best_c
                A[best_c][i, j] = 1
                A[best_c][j, i] = 1
                
                if time.time() > deadline - 5:
                    break
            if time.time() > deadline - 5:
                break
        
        if time.time() > deadline - 5:
            break

        current_T = get_T()
        
        # If already 0, return
        if current_T == 0:
            return _format_output(n, colors)
            
        # Local Search
        # We will run until time runs out or T becomes 0.
        # Use a simple acceptance criterion: accept if T decreases.
        # If stuck, accept with some probability (Annealing).
        
        temp = 50.0
        # We don't know total steps, so we decay temp based on time?
        # Or just keep it constant and rely on time limit.
        
        while time.time() < deadline - 1:
            # Pick random edge
            u = np.random.randint(n)
            v = np.random.randint(n)
            if u == v:
                continue
            
            c_old = colors[u, v]
            
            # Try all other colors
            best_c_new = -1
            best_T_new = current_T
            
            # We can estimate the change without full recompute?
            # Let's just recompute for simplicity, it's fast enough.
            
            # To be faster, we can try the color that minimizes local triangle count first.
            # But let's just iterate 0-3.
            
            for c_new in range(4):
                if c_new == c_old:
                    continue
                    
                # Make change
                A[c_old][u, v] = 0
                A[c_old][v, u] = 0
                A[c_new][u, v] = 1
                A[c_new][v, u] = 1
                
                T_new = get_T()
                
                # Undo change
                A[c_old][u, v] = 1
                A[c_old][v, u] = 1
                A[c_new][u, v] = 0
                A[c_new][v, u] = 0
                
                if T_new < best_T_new:
                    best_T_new = T_new
                    best_c_new = c_new
            
            if best_c_new != -1:
                # Accept best move
                A[c_old][u, v] = 0
                A[c_old][v, u] = 0
                A[best_c_new][u, v] = 1
                A[best_c_new][v, u] = 1
                colors[u, v] = best_c_new
                colors[v, u] = best_c_new
                current_T = best_T_new
                
                if current_T == 0:
                    return _format_output(n, colors)
            else:
                # No improving move found.
                # Accept a random move with probability exp(-delta/T)?
                # Delta is positive (increase in T).
                # We need to know the delta.
                # Let's just pick a random new color and accept if it's not too bad?
                # Or restart?
                # Given the time, if we are stuck, we might as well restart with a new seed.
                # But we are inside the while loop.
                # Let's break and restart.
                break
        
        # If we broke out of while, we restart the outer loop (new seed)
        
    # If we failed to find n=50, return n=32
    return _get_n32()

def _get_n32():
    # Greenwood-Gleason colouring of K16 ... doubled
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

def _format_output(n, colors):
    # colors is n x n matrix
    # Need list in lexicographic order (i, j) with i < j
    res = []
    for i in range(n):
        for j in range(i + 1, n):
            res.append(int(colors[i, j]))
    return {'n': n, 'colors': res}
