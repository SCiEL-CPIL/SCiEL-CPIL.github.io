import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    # Target n values to try, descending
    # We know 17 is possible. 18 is the target for improvement.
    for n in [18, 17, 16, 15]:
        if time.time() > deadline - 10:
            continue
            
        result = solve_local_search(n, deadline)
        if result is not None:
            return result
            
    # Fallback: Return a valid coloring for n=17 if we have one cached? 
    # Or just return the best found in the loop.
    # If the loop fails, we should still return *something* valid if possible.
    # The cyclic coloring for n=17 might be valid? Unlikely to be 0-conflict immediately.
    # But the local search should find it.
    
    # If all failed, return n=1 (trivially valid) or a small n.
    # But we want to maximize score.
    # Let's ensure we at least try n=17 with a long enough time.
    
    # Fallback to a known small valid coloring if search fails
    # For n=5, K5 can be colored with 4 colors without C6 (no C6 exists).
    # So any coloring of K5 is valid.
    n = 5
    colors = [random.randint(0, 3) for _ in range(n*(n-1)//2)]
    return {'n': n, 'colors': colors}

def solve_local_search(n, deadline):
    start_time = time.time()
    
    # Precompute all C6 cycles
    # A cycle is defined by 6 distinct vertices v0..v5 in order.
    # To avoid duplicates, we can fix v0 as the smallest vertex?
    # No, for conflict checking, we just need to check all cyclic permutations?
    # Actually, if we check all ordered cycles (v0, v1, v2, v3, v4, v5) where v0 is the min vertex,
    # we cover each undirected cycle once?
    # No, a cycle (0,1,2,3,4,5) is the same as (0,5,4,3,2,1).
    # And (1,2,3,4,5,0) has min 0, but starts at 1.
    # If we require v0 = min(v0..v5), we still have 2 directions.
    # Let's just generate all permutations of 6 vertices? No, too many.
    # Generate all combinations of 6 vertices.
    # For each combination, generate all 60 distinct cycles.
    # Store the edge indices for each cycle.
    
    edges_map = {}
    edge_idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edges_map[(i, j)] = edge_idx
            edge_idx += 1
            
    num_edges = edge_idx
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    
    # Generate cycles
    # To save memory and time, we can generate them on the fly or precompute edge tuples.
    # Precomputing 1 million tuples of 6 ints is ~24MB. Acceptable.
    # But generating them takes time.
    # Let's precompute a list of (e0, e1, e2, e3, e4, e5)
    
    cycles = []
    vertices = list(range(n))
    for comb in itertools.combinations(vertices, 6):
        # 60 cycles for 6 vertices
        # Fix v0 = comb[0] to reduce symmetry?
        # The number of distinct cycles in K6 is 60.
        # We can generate them by permuting the remaining 5 vertices.
        # (v0, p1, p2, p3, p4, p5)
        # This generates each cycle twice (clockwise/counter).
        # So we only need 30 unique undirected cycles per combination.
        # But for simplicity, let's just check all 60? 
        # Or just 30 by fixing v1 < v5?
        # Let's just use all 60 permutations of the 5 non-min vertices.
        # Actually, simpler:
        # A cycle is a sequence.
        # We can iterate over all permutations of the 6 vertices? 6! = 720.
        # 720 / 12 (rotations and reflection) = 60.
        # So yes, 60 unique cycles.
        
        # Optimization: 
        # We only need to check if *all* edges in the cycle have the same color.
        # We can precompute the list of edge indices for each cycle.
        
        perms = list(itertools.permutations(comb[1:]))
        for p in perms:
            cycle_verts = [comb[0]] + list(p)
            # Check if this is a "canonical" cycle to avoid duplicates?
            # It doesn't matter for correctness, just redundancy.
            # Redundancy factor 2. Acceptable.
            
            e_idx = []
            valid = True
            for k in range(6):
                u = cycle_verts[k]
                v = cycle_verts[(k+1)%6]
                if u > v:
                    u, v = v, u
                e_idx.append(edges_map[(u, v)])
            
            cycles.append(tuple(e_idx))
            
    if time.time() > deadline:
        return None

    # Local Search
    # Objective: Minimize number of monochromatic C6s
    # We will just pick a random conflicting cycle and fix it.
    
    def get_conflict():
        # Pick a random cycle
        # Check if it's monochromatic
        # To be faster, we can maintain a count? 
        # No, just random sampling.
        # If we sample randomly, we might miss rare conflicts?
        # No, if there are many, we find one quickly.
        # If there are few, we might take longer.
        
        # Better: Iterate through a random subset of cycles?
        # Or just pick one random cycle.
        c = random.choice(cycles)
        c0 = colors[c[0]]
        if c0 == colors[c[1]] == colors[c[2]] == colors[c[3]] == colors[c[4]] == colors[c[5]]:
            return c
        return None

    def count_conflicts():
        count = 0
        for c in cycles:
            c0 = colors[c[0]]
            if c0 == colors[c[1]] == colors[c[2]] == colors[c[3]] == colors[c[4]] == colors[c[5]]:
                count += 1
        return count

    # Initial check
    # If 0, return
    if count_conflicts() == 0:
        return {'n': n, 'colors': colors}

    # Search
    # Use a simple hill climbing with random restarts if stuck
    max_iterations = 50000
    stuck_counter = 0
    best_conflicts = count_conflicts()
    
    for i in range(max_iterations):
        if time.time() > deadline:
            break
            
        # Find a conflict
        # To ensure we find one if it exists, we might need to scan more than 1 random pick
        # But for n=17, conflicts are likely abundant initially.
        
        found = False
        for _ in range(10): # Try 10 random picks
            c = get_conflict()
            if c is not None:
                found = True
                break
        
        if not found:
            # Maybe no conflicts?
            # Verify with full scan occasionally?
            # Or just assume 0?
            # Let's do a full scan every 1000 iterations to be safe
            if i % 1000 == 0:
                if count_conflicts() == 0:
                    return {'n': n, 'colors': colors}
            # If we didn't find one, pick a random edge and flip?
            # This is random walk.
            edge_idx = random.randint(0, num_edges - 1)
            old_color = colors[edge_idx]
            new_color = (old_color + 1 + random.randint(0, 2)) % 4
            colors[edge_idx] = new_color
            continue

        # We found a conflicting cycle c
        # Pick a random edge in it to flip
        edge_pos = random.randint(0, 5)
        edge_idx = c[edge_pos]
        
        old_color = colors[edge_idx]
        # Try all 3 other colors, pick one that doesn't create *more* immediate problems?
        # Or just random.
        new_color = (old_color + 1 + random.randint(0, 2)) % 4
        colors[edge_idx] = new_color
        
        # Check if we improved?
        # We can't easily know without recounting.
        # Let's just run.
        
    # Final check
    if count_conflicts() == 0:
        return {'n': n, 'colors': colors}
        
    return None
