import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 150.0
    
    n = 17
    num_edges = n * (n - 1) // 2
    
    # Map (i, j) to edge index
    def edge_index(i, j):
        if i > j: i, j = j, i
        # i < j
        # 0,1 -> 0
        # 0,2 -> 1
        # ...
        # 0,n-1 -> n-1
        # 1,2 -> n
        # ...
        # k, k+1 -> (n-1) + (n-2) + ... + (n-k)
        # Sum of arithmetic progression
        # Index = (n-1)*i - i*(i-1)//2 + (j-i-1)
        # Let's verify:
        # i=0, j=1: 0 + 0 = 0. Correct.
        # i=0, j=2: 0 + 1 = 1. Correct.
        # i=1, j=2: (n-1)*1 - 0 + 0 = n-1. Correct.
        # i=1, j=3: (n-1)*1 - 0 + 1 = n. Correct.
        return (n - 1) * i - i * (i - 1) // 2 + (j - i - 1)

    # Generate all 6-cycles
    # A 6-cycle is defined by 6 distinct vertices v0, v1, v2, v3, v4, v5
    # Edges: (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0)
    # We generate unique cycles to avoid duplicates.
    # We can fix v0=0, v1=1 to reduce symmetry? No, we need all cycles.
    # To avoid duplicates, we can enforce v0 < v1 < v2 ...? No, cycles are cyclic.
    # Standard way: Iterate over combinations of 6 vertices.
    # For each combination, there are (6-1)! / 2 = 60 unique cycles.
    # Total combinations: C(17, 6) = 12376.
    # Total cycles: 12376 * 60 = 742560.
    
    cycles = []
    edge_to_cycles = [[] for _ in range(num_edges)]
    
    # This generation might take a bit of time, but it's one-time.
    # 12376 * 60 = 742k iterations. Each iteration involves 6 edge index calculations.
    # This should be fast enough (< 1 second).
    
    import itertools
    
    for verts in itertools.combinations(range(n), 6):
        # Generate all unique permutations of these 6 vertices that form a cycle
        # Fix v0 to be the smallest vertex to break rotation symmetry?
        # Actually, itertools.permutations is 720. 
        # We can just generate all permutations and use a set of frozensets of edges to deduplicate.
        # But 12376 * 720 = 8.9M permutations. Might be slow.
        
        # Better: Generate permutations of v1..v5 given v0.
        # v0 is fixed as the first element of the combination? No, the cycle can start anywhere.
        # Let's just use permutations and a set of edge-tuples.
        
        seen_edges_in_cycle = set()
        for perm in itertools.permutations(verts):
            # To avoid duplicates, we can require perm[0] < perm[3] (opposite vertices)
            # or perm[0] is the smallest vertex.
            if perm[0] != min(verts):
                continue
            # Also, to avoid reflection symmetry, require perm[1] < perm[5]
            if perm[1] > perm[5]:
                continue
            
            # Now perm represents a unique cycle
            # Check if we've seen this set of edges
            edges = []
            for i in range(6):
                u, v = perm[i], perm[(i + 1) % 6]
                idx = edge_index(u, v)
                edges.append(idx)
            
            # The set of edges uniquely identifies the cycle
            edge_set = frozenset(edges)
            if edge_set in seen_edges_in_cycle:
                continue
            seen_edges_in_cycle.add(edge_set)
            
            cycle_id = len(cycles)
            cycles.append(edges)
            for e in edges:
                edge_to_cycles[e].append(cycle_id)
    
    # Initial random coloring
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    
    # Calculate initial mono_count
    mono_count = [0, 0, 0, 0]
    for c in cycles:
        # Check if all edges in cycle c have the same color
        col = colors[c[0]]
        is_mono = True
        for e in c[1:]:
            if colors[e] != col:
                is_mono = False
                break
        if is_mono:
            mono_count[col] += 1
            
    total_violations = sum(mono_count)
    
    best_violations = total_violations
    best_colors = colors.copy()
    
    T = 1.0
    cooling = 0.999
    
    while time.time() < deadline - 5.0:
        # Pick a random edge
        e_idx = random.randint(0, num_edges - 1)
        old_col = colors[e_idx]
        new_col = random.randint(0, 3)
        if new_col == old_col:
            continue
            
        # Calculate delta violations
        delta = 0
        for cycle_id in edge_to_cycles[e_idx]:
            c = cycles[cycle_id]
            
            # Check if cycle was mono in old_col
            # We need to check the other 5 edges
            # If all other 5 are old_col, then it was mono (since e_idx was old_col)
            was_mono_old = True
            for e in c:
                if e == e_idx:
                    continue
                if colors[e] != old_col:
                    was_mono_old = False
                    break
            if was_mono_old:
                delta -= 1 # Removing a violation
                
            # Check if cycle becomes mono in new_col
            # If all other 5 are new_col, then it becomes mono (since e_idx becomes new_col)
            becomes_mono_new = True
            for e in c:
                if e == e_idx:
                    continue
                if colors[e] != new_col:
                    becomes_mono_new = False
                    break
            if becomes_mono_new:
                delta += 1 # Adding a violation
        
        # Acceptance
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            colors[e_idx] = new_col
            total_violations += delta
            if total_violations < best_violations:
                best_violations = total_violations
                best_colors = colors.copy()
                
        T *= cooling
        
        if best_violations == 0:
            break
            
    # If we found a valid coloring for n=17, return it
    if best_violations == 0:
        return {'n': n, 'colors': best_colors}
        
    # If not, try to find the largest valid n
    # We can try n=16, 15, etc.
    # But the search above was for n=17.
    # If it failed, it's likely n=17 is hard.
    # We can try a greedy construction for smaller n.
    
    # Fallback: Return the known n=9 solution if we didn't find better
    # Or try to salvage a partial solution?
    # The problem asks to maximize n.
    # If we can't find n=17, we should try n=16.
    
    # Let's implement a quick greedy for n=16 if 17 failed?
    # Given time constraints, it's better to just return the best valid n we can guarantee.
    # The seed program returns n=9.
    # If our SA didn't find 0 violations for 17, we don't have a valid 17.
    # We can try to run the same SA for n=16.
    
    # To save time, we will just return the n=9 solution if we didn't succeed with 17.
    # This ensures we don't score 0.
    
    n_seed = 9
    colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    return {'n': n_seed, 'colors': colors_seed}
