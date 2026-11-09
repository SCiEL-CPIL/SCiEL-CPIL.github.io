import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 16
    num_colors = 13
    k4s = list(itertools.combinations(range(n), 4))
    num_k4s = len(k4s)
    
    # Map each K4 to its 6 edges
    # edge_index(i, j) where i < j
    def edge_idx(i, j):
        if i > j: i, j = j, i
        return i * n + j # simple unique id, or precomputed map
    
    # Precompute edges for each K4
    k4_edges = []
    for k4 in k4s:
        edges = list(itertools.combinations(k4, 2))
        k4_edges.append(edges)

    # State: colors[i][j]
    colors = [[0] * n for _ in range(n)]
    
    def get_color(i, j):
        if i > j: i, j = j, i
        return colors[i][j]
        
    def set_color(i, j, c):
        if i > j: i, j = j, i
        colors[i][j] = c

    def count_distinct_colors(edges):
        cs = set()
        for u, v in edges:
            cs.add(get_color(u, v))
        return len(cs)

    def total_violations():
        v = 0
        for i in range(num_k4s):
            if count_distinct_colors(k4_edges[i]) < 5:
                v += 1
        return v

    # Initialize randomly
    for i in range(n):
        for j in range(i+1, n):
            colors[i][j] = random.randint(0, num_colors - 1)
            
    current_violations = total_violations()
    best_violations = current_violations
    best_state = [row[:] for row in colors]
    
    # If we start lucky, return
    if current_violations == 0:
        return [get_color(i, j) for i in range(n) for j in range(i+1, n)]

    # Local search
    while time.time() < deadline:
        # Pick random edge
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if i == j:
            continue
        if i > j: i, j = j, i
        
        old_color = colors[i][j]
        
        # Find K4s containing this edge
        # Instead of searching all K4s, we can precompute which K4s contain which edge
        # But 1820 is small, maybe just check? No, precompute is better.
        pass # We'll use a simpler approach: just evaluate change by checking affected K4s
        
        # To find affected K4s, we can precompute a list of K4 indices for each edge.
        # Let's do that outside the loop or once at start.
        
    # Precomputation for speed
    edge_to_k4s = [[] for _ in range(n*n)]
    for k4_idx, k4 in enumerate(k4s):
        for u, v in itertools.combinations(k4, 2):
            ei = u * n + v if u < v else v * n + u
            edge_to_k4s[ei].append(k4_idx)

    # Restart loop with precomputation
    while time.time() < deadline:
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if i == j: continue
        if i > j: i, j = j, i
        
        ei = i * n + j
        affected_k4s = edge_to_k4s[ei]
        
        old_color = colors[i][j]
        
        best_new_color = old_color
        best_new_violations = current_violations
        
        for new_c in range(num_colors):
            if new_c == old_color: continue
            
            # Temporarily set
            colors[i][j] = new_c
            
            delta_v = 0
            for k4_idx in affected_k4s:
                old_count = count_distinct_colors(k4_edges[k4_idx])
                # Wait, we need old violation status
                # This is inefficient. 
                # Better: store current distinct count for each K4?
                # Or just recalculate the whole K4?
                # Recalculating 6 edges is fast.
                # But we need to know if it was a violation before.
                
                # Let's just calculate new violation status and compare to old.
                # We didn't store old status.
                # Let's store `k4_is_valid` array.
                pass
            colors[i][j] = old_color # revert for next iteration of color loop
            
            # This approach is getting complicated to track delta efficiently without state.
            # Given N=16, recalculating ALL 1820 K4s for every candidate move might be too slow?
            # 1820 * 6 = ~10k ops.
            # 13 candidates * 10k = 130k ops per move.
            # 1000 moves = 130M ops. In Python, this is ~10-20 seconds.
            # So we can do ~5000-10000 moves in 150s.
            # This is likely enough to find a solution from a random start?
            # Random start with 13 colors: Probability a specific K4 is valid?
            # Number of ways to color K4 with < 5 colors is small.
            # Total colorings: 13^6 = 4.8M.
            # Valid colorings: those with >= 5 distinct colors.
            # Most colorings are valid.
            # So random start has very low violations.
            # Local search should fix the few violations quickly.
            
            # Let's implement the "recalculate all" approach for simplicity and robustness.
            
            # Actually, let's just try a few random restarts if we don't find it quickly.
            break # break inner loop to restructure
            
    # Restructured: 
    # 1. Try random restarts.
    # 2. For each restart, run local search for a fixed number of steps or time.
    
    for restart in range(50): # Max 50 restarts
        # Reset time check
        if time.time() > deadline: break
        
        # Random Init
        for i in range(n):
            for j in range(i+1, n):
                colors[i][j] = random.randint(0, num_colors - 1)
        
        current_violations = total_violations()
        if current_violations == 0:
            return [get_color(i, j) for i in range(n) for j in range(i+1, n)]
            
        # Local Search
        no_improve_steps = 0
        max_steps = 2000 # Adjust based on time
        
        while no_improve_steps < max_steps and time.time() < deadline:
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j: continue
            if i > j: i, j = j, i
            
            old_color = colors[i][j]
            best_c = old_color
            best_v = current_violations
            
            for new_c in range(num_colors):
                if new_c == old_color: continue
                colors[i][j] = new_c
                v = total_violations()
                if v < best_v:
                    best_v = v
                    best_c = new_c
                # Optional: accept neutral moves to escape local minima?
                # If v == best_v and random.random() < 0.1:
                #     best_c = new_c
            
            colors[i][j] = best_c
            
            if best_v < current_violations:
                current_violations = best_v
                no_improve_steps = 0
                if current_violations == 0:
                    return [get_color(i, j) for i in range(n) for j in range(i+1, n)]
            else:
                no_improve_steps += 1
                
    # If we run out of time/restarts, return the best found (or last)
    # We should track the best state globally
    # For now, if we didn't find 0, we return the last state which might be invalid.
    # The task says "Minimize the number of colours used".
    # If we can't find a valid 13-colouring, we should fall back to 14 or 15?
    # The evaluator measures the RAW VALUE (number of colours used).
    # If we return an invalid 13-colouring, it might be scored 0?
    # "A valid colouring certifies f(16) <= colours used."
    # "Invalid result ... scores 0."
    # So we MUST return a valid colouring.
    # If we fail to find a 13-colouring, we must return a valid 14 or 15 colouring.
    
    # Let's ensure we always return a valid colouring.
    # We can start with a known valid 15-colouring (e.g. (i+j)%15 or similar) and try to reduce to 13.
    # Or just run the search and if it fails, return the best valid one found.
    
    # To be safe, let's track the best valid colouring found.
    
    # Re-implementation with safety net:
    best_valid_colors = None
    best_valid_num_colors = 100
    
    # Try to find 13
    # If not found, try 14
    # If not found, try 15 (guaranteed to be easy to find, e.g. (i+j)%15 might work? 
    # Actually (i+j)%17 works for 17 colors. 
    # (i+j)%15: K4 vertices a,b,c,d. Colors a+b, a+c... 
    # Collisions? 
    # If (i+j)%15 is used, max colors 15.
    # Is it valid? 
    # In Z_15, differences... 
    # Let's just use the 17-color solution as a fallback, which is guaranteed valid.
    # Or find a valid 14/15 quickly.
    
    # Given the time limit, and that 13 is likely findable, 
    # I will implement the search for 13.
    # If it fails, I will return a valid 15-coloring constructed by (i+j)%15? 
    # Let's check if (i+j)%15 is valid for K16.
    # Vertices 0..15.
    # If we use mod 15, vertex 15 is 0 mod 15? No, vertices are 0..15.
    # If we map vertices to Z_15, we have a collision 0 and 15? No, 15 is distinct from 0 in Z_15? 
    # Z_15 has elements 0..14. 15 mod 15 = 0. 
    # So vertex 0 and vertex 15 have the same "value". 
    # Edge (0, 15) has color 0+15=15=0.
    # This might cause issues.
    
    # Safest fallback: The original 17-color solution.
    # But we want to minimize.
    # If we can't find 13, we should try to find 14.
    
    # Let's just do the 13 search. It's very likely to succeed.
    # If it fails, return the 17-color solution to ensure validity (score 1.0 or better if we find 13).
    # Wait, if we return 17, score is 1.0. If we return 13, score is > 1.0.
    # If we return invalid 13, score is 0.
    # So we must only return 13 if valid.
    
    # Final plan:
    # 1. Search for valid 13-coloring.
    # 2. If found, return it.
    # 3. If not found before deadline, return the valid 17-coloring (seed solution).
    
    # Actually, we can also try 14 if 13 fails?
    # 14 is easier to find than 13.
    
    # Let's stick to 13 first.

    # The code above had a flaw: it didn't track best valid state.
    # I will rewrite the loop to be clean.

    return [ (i + j) % 17 for i in range(16) for j in range(i + 1, 16) ] # Fallback

# Let's write the actual code properly now.

import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n = 16
    k4s = list(itertools.combinations(range(n), 4))
    num_k4s = len(k4s)
    k4_edges = [list(itertools.combinations(k4, 2)) for k4 in k4s]
    
    # Precompute edge to K4 mapping
    edge_to_k4s = [[] for _ in range(n*n)]
    for k4_idx, k4 in enumerate(k4s):
        for u, v in itertools.combinations(k4, 2):
            ei = u * n + v if u < v else v * n + u
            edge_to_k4s[ei].append(k4_idx)

    def get_color(colors, i, j):
        if i > j: i, j = j, i
        return colors[i][j]
        
    def total_violations(colors):
        v = 0
        for k4_idx in range(num_k4s):
            cs = set()
            for u, v in k4_edges[k4_idx]:
                cs.add(get_color(colors, u, v))
            if len(cs) < 5:
                v += 1
        return v

    # Try 13 colors
    target_colors = 13
    for restart in range(100):
        if time.time() > deadline: break
        
        colors = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                colors[i][j] = random.randint(0, target_colors - 1)
                
        current_v = total_violations(colors)
        if current_v == 0:
            return [get_color(colors, i, j) for i in range(n) for j in range(i+1, n)]
            
        no_improve = 0
        max_no_improve = 1000
        
        while no_improve < max_no_improve and time.time() < deadline:
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j: continue
            if i > j: i, j = j, i
            
            old_c = colors[i][j]
            best_c = old_c
            best_v = current_v
            
            for new_c in range(target_colors):
                if new_c == old_c: continue
                colors[i][j] = new_c
                v = total_violations(colors)
                if v < best_v:
                    best_v = v
                    best_c = new_c
                # Add slight noise to escape local minima
                elif v == best_v and random.random() < 0.05:
                    best_c = new_c
            
            colors[i][j] = best_c
            
            if best_v < current_v:
                current_v = best_v
                no_improve = 0
                if current_v == 0:
                    return [get_color(colors, i, j) for i in range(n) for j in range(i+1, n)]
            else:
                no_improve += 1

    # If 13 fails, try 14
    target_colors = 14
    for restart in range(50):
        if time.time() > deadline: break
        colors = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                colors[i][j] = random.randint(0, target_colors - 1)
                
        current_v = total_violations(colors)
        if current_v == 0:
            return [get_color(colors, i, j) for i in range(n) for j in range(i+1, n)]
            
        no_improve = 0
        while no_improve < 500 and time.time() < deadline:
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j: continue
            if i > j: i, j = j, i
            
            old_c = colors[i][j]
            best_c = old_c
            best_v = current_v
            
            for new_c in range(target_colors):
                if new_c == old_c: continue
                colors[i][j] = new_c
                v = total_violations(colors)
                if v < best_v:
                    best_v = v
                    best_c = new_c
            
            colors[i][j] = best_c
            
            if best_v < current_v:
                current_v = best_v
                no_improve = 0
                if current_v == 0:
                    return [get_color(colors, i, j) for i in range(n) for j in range(i+1, n)]
            else:
                no_improve += 1

    # Fallback: Valid 17-coloring
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

