import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # 100s budget
    
    # Helper to generate edge indices in lexicographic order
    def get_edges(n):
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

    def edge_index(n, i, j):
        # Map (i, j) to index in list of edges
        # Number of edges with first vertex < i is sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)/2
        # Edges with first vertex i and second < j is j - i - 1
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    # Try to find a valid coloring for N
    def try_find_coloring(n):
        edges = get_edges(n)
        e_count = len(edges)
        
        # Precompute cycles of length 6
        # We need to map vertices to edge indices
        # For a cycle v1-v2-v3-v4-v5-v6-v1, edges are (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v6), (v6,v1)
        
        # Generate all 6-cycles
        # To avoid duplicates, we can enforce v1 < v2 < ... but cycles are not ordered paths.
        # A cycle is a set of 6 edges.
        # We can generate all combinations of 6 vertices, and then all 6-cycles on them.
        # There are 60 distinct 6-cycles on 6 labeled vertices.
        
        cycles = []
        cycles_containing_edge = [[] for _ in range(e_count)]
        
        # Optimization: Only generate cycles for the specific n
        # This might be slow if done every time, but n is small.
        
        from itertools import combinations, permutations
        
        # We need to map vertex pairs to edge indices quickly
        # Create a dict or 2D array
        edge_idx_map = {}
        for idx, (u, v) in enumerate(edges):
            edge_idx_map[(u, v)] = idx
            edge_idx_map[(v, u)] = idx
            
        # Generate cycles
        # To reduce work, we can just pick random subsets if we don't need all?
        # No, we need to detect ANY C6. So we need the full set.
        
        # For n=18, this precomputation might take a few seconds.
        # Let's see if we can do it faster or skip if time is tight.
        
        if time.time() > deadline - 5:
            return None

        vertices = range(n)
        # Generate all 6-cycles
        # A 6-cycle is determined by 6 distinct vertices and a cyclic ordering.
        # To avoid duplicates, we fix the smallest vertex as v1.
        # Then v2, v3, v4, v5, v6 are the rest in some order.
        # But (v1,v2,v3,v4,v5,v6) is same as (v1,v6,v5,v4,v3,v2).
        # So we can fix v2 < v6? No.
        # Standard way: iterate over all 6-subsets. For each subset, iterate over (6-1)!/2 = 60 cycles.
        
        all_cycles = []
        for combo in combinations(vertices, 6):
            # Generate 60 unique cycles on these 6 vertices
            # Fix v0 = combo[0]
            v0 = combo[0]
            rest = combo[1:]
            # Permutations of rest: 5! = 120. Divide by 2 for reversal.
            # We can generate permutations and only take those where rest[0] < rest[-1]
            for perm in permutations(rest):
                if perm[0] < perm[-1]:
                    cycle_verts = (v0,) + perm
                    # Get edge indices
                    cycle_edges = []
                    valid = True
                    for k in range(6):
                        u = cycle_verts[k]
                        v = cycle_verts[(k + 1) % 6]
                        e_idx = edge_idx_map.get((u, v))
                        if e_idx is None:
                            valid = False
                            break
                        cycle_edges.append(e_idx)
                    if valid:
                        all_cycles.append(cycle_edges)
                        
        # Build reverse index
        for c_idx, cycle in enumerate(all_cycles):
            for e_idx in cycle:
                cycles_containing_edge[e_idx].append(c_idx)

        if time.time() > deadline - 2:
            return None

        # Local Search
        # Initialize random colors
        colors = np.random.randint(0, 4, size=e_count)
        
        # Count monochromatic cycles
        # mono_count[c] = number of cycles with all edges color c
        mono_count = [0, 0, 0, 0]
        
        # Initial count
        for cycle in all_cycles:
            c0 = colors[cycle[0]]
            if all(colors[e] == c0 for e in cycle[1:]):
                mono_count[c0] += 1
                
        total_mono = sum(mono_count)
        
        # If already valid
        if total_mono == 0:
            return {'n': n, 'colors': colors.tolist()}
            
        # Search
        while total_mono > 0 and time.time() < deadline - 5:
            # Pick a random color with monochromatic cycles
            c_pick = None
            weights = [mono_count[i] for i in range(4) if mono_count[i] > 0]
            if not weights:
                break
            # Weighted choice
            r = random.random() * sum(weights)
            acc = 0
            for i in range(4):
                if mono_count[i] > 0:
                    acc += mono_count[i]
                    if r <= acc:
                        c_pick = i
                        break
            if c_pick is None:
                continue
                
            # Pick a random cycle of this color
            # We need a list of cycles for each color? 
            # Or just scan?
            # Scanning 1.1M cycles is slow.
            # Let's maintain a list of mono cycles per color?
            # That's hard to update.
            
            # Alternative: Just pick a random edge, and if it's in a mono cycle, use that.
            # But we need to find a mono cycle.
            
            # Let's just scan the cycles for the picked color?
            # No, we don't have the list.
            
            # Let's re-structure: 
            # Instead of picking a color then a cycle, pick a random edge.
            # Check if it's in a mono cycle.
            
            # Actually, the previous plan was:
            # Pick a random mono cycle.
            # To do that efficiently, we need to know which cycles are mono.
            
            # Let's keep a set of mono cycle indices?
            # Updating it is O(1) if we know the cycle's color.
            
            mono_cycle_set = set()
            for c_idx, cycle in enumerate(all_cycles):
                c0 = colors[cycle[0]]
                if all(colors[e] == c0 for e in cycle[1:]):
                    mono_cycle_set.add(c_idx)
                    
            if not mono_cycle_set:
                return {'n': n, 'colors': colors.tolist()}
                
            # Pick random cycle from set
            c_idx = random.choice(list(mono_cycle_set))
            cycle = all_cycles[c_idx]
            
            # Pick random edge in cycle
            e_pos = random.randint(0, 5)
            e_idx = cycle[e_pos]
            
            old_color = colors[e_idx]
            new_color = random.choice([c for c in range(4) if c != old_color])
            
            # Flip
            colors[e_idx] = new_color
            
            # Update mono counts and set
            # 1. Remove old mono status
            if c_idx in mono_cycle_set:
                mono_count[old_color] -= 1
                mono_cycle_set.remove(c_idx)
                
            # 2. Check all cycles containing e_idx
            for other_c_idx in cycles_containing_edge[e_idx]:
                other_cycle = all_cycles[other_c_idx]
                
                # Check if it was mono old_color
                # We only know if it was in the set? No, we need to check current state.
                # But we just changed e_idx.
                
                # Check if other_cycle is now mono new_color
                # All edges except e_idx must be new_color
                # And e_idx is new_color
                is_mono_new = True
                for e in other_cycle:
                    if e == e_idx:
                        continue
                    if colors[e] != new_color:
                        is_mono_new = False
                        break
                if is_mono_new:
                    if other_c_idx not in mono_cycle_set:
                        mono_count[new_color] += 1
                        mono_cycle_set.add(other_c_idx)
                        
                # Check if other_cycle was mono old_color and is no longer
                # It was mono old_color if it was in mono_cycle_set before?
                # No, mono_cycle_set only contains cycles that are mono *now* (before flip).
                # If it was in mono_cycle_set, it was mono old_color (since e_idx was old_color).
                # Now e_idx is new_color, so it's no longer mono old_color.
                # We already handled the specific c_idx.
                # What about other cycles that were mono old_color?
                # They are in mono_cycle_set.
                # We need to remove them if they contain e_idx.
                
                # So:
                if other_c_idx in mono_cycle_set:
                    # It was mono some color. Since e_idx was old_color, it must have been old_color.
                    mono_count[old_color] -= 1
                    mono_cycle_set.remove(other_c_idx)
                    
            total_mono = sum(mono_count)
            
        if total_mono == 0:
            return {'n': n, 'colors': colors.tolist()}
            
        return None

    # Try N from 18 down to 9
    for n in range(18, 8, -1):
        if time.time() > deadline - 10:
            break
        res = try_find_coloring(n)
        if res:
            return res
            
    # Fallback to seed
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
