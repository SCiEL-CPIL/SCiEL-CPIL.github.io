import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # 1. Algebraic Seed Construction
    # We use a construction based on GF(16) to generate a valid 4-coloring for n=32.
    # This is much better than random for n=50 because it starts with 0 conflicts
    # in the core structure, reducing the "distance" to a valid n=50 solution.
    
    def gf16_mul(a, b):
        # Multiplication in GF(16) with polynomial x^4 + x + 1 (0b10011)
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    def get_seed_32():
        # Log table for GF(16) multiplicative group
        log = {}
        x = 1
        for e in range(15):
            log[x] = e
            x = gf16_mul(x, 2)
        
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                # Block structure:
                # If i, j in same half (i//16 == j//16), use GF(16) difference
                # Else, use color 3
                if i // 16 == j // 16:
                    # (i%16) ^ (j%16) is not field subtraction, but for Paley-like 
                    # constructions on GF(2^4), we usually use (i-j) in the field.
                    # However, the seed provided in the prompt uses XOR.
                    # Let's stick to the prompt's seed logic which is known to work for n=32.
                    val = log[(i % 16) ^ (j % 16)] % 3
                else:
                    val = 3
                colors.append(val)
        return colors

    seed_colors_32 = get_seed_32()
    
    # 2. Expand to n=50
    # Strategy: Start with the 32-vertex core. Add 18 new vertices.
    # Initialize edges involving new vertices using a heuristic:
    # Try to assign colors such that they don't form triangles with existing structures.
    # A simple heuristic: Random, but we will immediately repair.
    
    n_target = 50
    c = np.zeros((n_target, n_target), dtype=np.int8)
    
    # Fill the 32x32 block
    idx = 0
    for i in range(32):
        for j in range(i + 1, 32):
            c[i, j] = c[j, i] = seed_colors_32[idx]
            idx += 1
            
    # Initialize edges for new vertices (32..49)
    # Random initialization for the rest
    for i in range(32, n_target):
        for j in range(i + 1, n_target):
            c[i, j] = c[j, i] = random.randint(0, 3)
    for i in range(32):
        for j in range(32, n_target):
            c[i, j] = c[j, i] = random.randint(0, 3)
            
    # 3. Conflict-Driven Local Search
    # Instead of SA, we find all monochromatic triangles and fix them.
    # This is deterministic and focuses search effort where it is needed.
    
    def find_conflicts(c, n):
        """
        Returns a list of (u, v, w, color) for each monochromatic triangle.
        Also returns a count of conflicts per edge to prioritize "hot" edges.
        """
        conflicts = []
        edge_conflict_count = np.zeros((n, n), dtype=np.int32)
        
        # Vectorized triangle detection per color
        # For each color k, find triangles in subgraph G_k
        for k in range(4):
            A = (c == k).astype(np.int8)
            np.fill_diagonal(A, 0)
            
            # Triangles are triples (u,v,w) where A[u,v]=A[v,w]=A[w,u]=1
            # We can iterate over edges (u,v) in G_k and check common neighbors
            # But for speed, let's use matrix multiplication or just loops since n=50 is small.
            # n=50 -> 1225 edges. Checking neighbors is O(n). 1225*50 = 61250 ops per color.
            # 4 colors -> ~250k ops. Very fast.
            
            for u in range(n):
                for v in range(u + 1, n):
                    if A[u, v]:
                        # Find common neighbors w > v (to avoid duplicates)
                        # Actually, we just need to find ANY w.
                        # Let's find all w such that A[u,w] and A[v,w].
                        # common = A[u] & A[v]
                        # We can do this with numpy
                        common = np.where((A[u] == 1) & (A[v] == 1))[0]
                        for w in common:
                            if w > v: # Ensure unique triangle (u,v,w) with u<v<w
                                conflicts.append((u, v, w, k))
                                edge_conflict_count[u, v] += 1
                                edge_conflict_count[v, u] += 1
                                edge_conflict_count[v, w] += 1
                                edge_conflict_count[w, v] += 1
                                edge_conflict_count[u, w] += 1
                                edge_conflict_count[w, u] += 1
        return conflicts, edge_conflict_count

    def try_fix_edge(c, u, v, n):
        """
        Try to change color of edge (u,v) to remove conflicts.
        Returns True if a change was made, False otherwise.
        """
        c_old = c[u, v]
        
        # Calculate delta in number of conflicts for each new color
        # A change from c_old to c_new removes conflicts where c_old was the triangle color
        # and adds conflicts where c_new is the triangle color.
        
        # For a specific edge (u,v), the triangles involving it are (u,v,w) for all w.
        # The color of triangle (u,v,w) is c[u,v] if c[u,w]==c[v,w]==c[u,v], else not monochromatic (or different color).
        # Actually, if c[u,v] changes, a triangle (u,v,w) becomes monochromatic in color k
        # if c[u,w]==k and c[v,w]==k and new_c[u,v]==k.
        
        # Let's just count current conflicts for each color if we set edge to k.
        # But we need to be careful: changing c[u,v] affects triangles (u,v,w).
        
        best_delta = 0
        best_color = -1
        
        # Current conflicts involving (u,v) with color c_old
        # We need to know how many triangles (u,v,w) are currently monochromatic in c_old.
        # And how many would become monochromatic in c_new.
        
        # Let's compute the "cost" of each color k for edge (u,v).
        # Cost(k) = number of w such that c[u,w]==k and c[v,w]==k.
        # If we set c[u,v]=k, we add Cost(k) new monochromatic triangles.
        # We remove the current ones.
        
        costs = [0, 0, 0, 0]
        for w in range(n):
            if w == u or w == v:
                continue
            if c[u, w] == c[v, w]:
                costs[c[u, w]] += 1
        
        # Current cost is costs[c_old].
        # New cost for color k is costs[k].
        # We want to minimize the total number of monochromatic triangles.
        # Note: This metric is local. It counts triangles involving (u,v).
        # It doesn't account for the fact that changing (u,v) might break other triangles?
        # No, triangles are unique to their 3 edges. Changing (u,v) ONLY affects triangles containing (u,v).
        # So minimizing local cost is globally optimal for that step.
        
        for k in range(4):
            if k == c_old:
                continue
            delta = costs[k] - costs[c_old]
            if delta < best_delta:
                best_delta = delta
                best_color = k
        
        if best_color != -1 and best_delta < 0:
            c[u, v] = best_color
            c[v, u] = best_color
            return True
        return False

    # Main Loop
    # Alternate between finding conflicts and fixing the "hottest" edges.
    
    max_iters = 5000
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        conflicts, edge_conflict_count = find_conflicts(c, n_target)
        
        if len(conflicts) == 0:
            # Valid!
            colors_list = []
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    colors_list.append(int(c[i, j]))
            return {'n': n_target, 'colors': colors_list}
            
        # Find the edge with the maximum conflict count
        # Flatten and find max
        flat_counts = edge_conflict_count.flatten()
        max_idx = np.argmax(flat_counts)
        u = max_idx // n_target
        v = max_idx % n_target
        
        if edge_conflict_count[u, v] == 0:
            # No conflicts found by edge count? Should not happen if conflicts list is non-empty
            break
            
        # Try to fix this edge
        fixed = try_fix_edge(c, u, v, n_target)
        
        if not fixed:
            # If we can't improve locally, we might be stuck in a local minimum
            # where no single edge flip reduces the count.
            # In this case, perform a random perturbation to escape.
            # Pick a random edge and flip it.
            ru = random.randint(0, n_target - 1)
            rv = random.randint(0, n_target - 1)
            if ru == rv:
                continue
            if ru > rv:
                ru, rv = rv, ru
            old_c = c[ru, rv]
            new_c = (old_c + 1 + random.randint(0, 2)) % 4
            c[ru, rv] = new_c
            c[rv, ru] = new_c
            
    # Fallback: Return the best valid solution we can construct.
    # If we didn't reach 0 conflicts, we return the seed (n=32) to guarantee validity.
    # But wait, the task says "Maximize n". If we can't get 50, we should return 32.
    # However, if we have a partial solution for 50 that is valid, we return 50.
    # If we have 49 valid, we should return 49.
    
    # Let's check if the current c is valid. If not, we need to find the largest valid prefix.
    # But checking validity of all prefixes is expensive.
    # Given the "conflict-driven" approach, if we didn't fix all conflicts,
    # it's likely we are stuck.
    
    # Let's do a final check on the full 50-vertex graph.
    final_conflicts, _ = find_conflicts(c, n_target)
    if len(final_conflicts) == 0:
        colors_list = []
        for i in range(n_target):
            for j in range(i + 1, n_target):
                colors_list.append(int(c[i, j]))
        return {'n': n_target, 'colors': colors_list}
    
    # If 50 is not valid, try to salvage 49, 48...
    # This is risky because removing a vertex might not fix the remaining graph if the conflicts were distributed.
    # However, the seed 32 is guaranteed valid.
    # Let's just return the seed 32 to be safe, unless we found a valid larger one earlier.
    
    # Actually, let's try to verify if the 32-core is still valid after perturbations.
    # The perturbations might have broken the core.
    # To be absolutely safe and score > 0, we should return the known valid seed if the search fails.
    
    # Re-construct the seed 32 to ensure it's valid
    seed_colors_32 = get_seed_32()
    
    # Quick check: is the seed 32 valid?
    c_32 = np.zeros((32, 32), dtype=np.int8)
    idx = 0
    for i in range(32):
        for j in range(i + 1, 32):
            c_32[i, j] = c_32[j, i] = seed_colors_32[idx]
            idx += 1
    conf_32, _ = find_conflicts(c_32, 32)
    if len(conf_32) == 0:
        return {'n': 32, 'colors': seed_colors_32}
    
    # If even the seed is broken (shouldn't happen), return n=1
    return {'n': 1, 'colors': []}
