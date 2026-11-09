import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # We aim for n=17 first, as it is the known bound.
    # If we find a valid coloring for 17, we return it.
    # If we have time, we could try 18, but 17 is the target for improvement over seed (which likely returns 9 or similar small n if it fails 17).
    # The seed program in the prompt returns n=9 if it fails 17.
    # So finding n=17 is a massive jump.
    
    # To be safe and fast, we will focus entirely on n=17.
    # If we fail to find a valid coloring for 17 within the time, we return a fallback.
    
    n = 17
    
    # Precompute edge indices and map
    num_edges = n * (n - 1) // 2
    edge_indices = []
    edge_map = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_map[(i, j)] = idx
            edge_indices.append((i, j))
            idx += 1
            
    # Precompute all C6 cycles and their edge sets
    # A C6 is a sequence of 6 distinct vertices v0-v1-v2-v3-v4-v5-v0
    # To avoid duplicates:
    # 1. Fix v0 as the smallest vertex in the cycle.
    # 2. Fix v1 < v5 (to handle reversal symmetry).
    # This generates each distinct cycle exactly once.
    
    cycles = []
    cycle_edges = []
    
    # Optimization: Generate cycles efficiently
    # For n=17, C(17,6) = 12376 combinations.
    # For each combination, there are 60 distinct cycles.
    # Total ~ 742k cycles.
    
    for vertices in itertools.combinations(range(n), 6):
        v0 = vertices[0]
        others = vertices[1:]
        for perm in itertools.permutations(others):
            v1, v2, v3, v4, v5 = perm
            if v1 > v5:
                continue
            
            cycle_verts = [v0, v1, v2, v3, v4, v5]
            c_edges = []
            for k in range(6):
                u = cycle_verts[k]
                v = cycle_verts[(k + 1) % 6]
                if u > v:
                    u, v = v, u
                c_edges.append(edge_map[(u, v)])
            
            cycle_edges.append(c_edges)
            
    num_cycles = len(cycle_edges)
    
    # Build edge_to_cycles index
    edge_to_cycles = [[] for _ in range(num_edges)]
    for c_idx, c_edges in enumerate(cycle_edges):
        for e_idx in c_edges:
            edge_to_cycles[e_idx].append(c_idx)
            
    # Initialize coloring
    # Start with a random coloring
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    
    # Initialize cycle status
    # cycle_color_counts[c_idx][color] = number of edges in cycle c_idx with that color
    cycle_color_counts = [[0, 0, 0, 0] for _ in range(num_cycles)]
    for c_idx, c_edges in enumerate(cycle_edges):
        for e_idx in c_edges:
            cycle_color_counts[c_idx][colors[e_idx]] += 1
            
    # Identify currently violated cycles (monochromatic C6s)
    # A cycle is violated if any color count is 6.
    violated_cycles = set()
    for c_idx in range(num_cycles):
        counts = cycle_color_counts[c_idx]
        if any(c == 6 for c in counts):
            violated_cycles.add(c_idx)
            
    # Local Search: Conflict-Driven Repair
    # Strategy:
    # 1. If no violated cycles, return success.
    # 2. Pick a random violated cycle.
    # 3. Try to fix it by changing the color of one of its 6 edges.
    # 4. For each edge in the cycle and each new color (different from current):
    #    a. Calculate how many *new* violations this change would create.
    #    b. Calculate how many *existing* violations this change would fix (usually just the one we picked, unless it fixes others too).
    #    c. Choose the move with the minimum net increase in violations (or maximum decrease).
    # 5. If the best move increases violations, we might still accept it with low probability (simulated annealing style) or just pick the least bad.
    #    However, a simpler "Greedy Repair" is: pick the move that minimizes the total number of violated cycles after the move.
    
    best_n_found = 9 # Fallback
    best_colors_found = [min(j - i, n - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    while time.time() < deadline - 2:
        if len(violated_cycles) == 0:
            return {'n': n, 'colors': colors}
        
        # Pick a random violated cycle
        c_idx = random.choice(list(violated_cycles))
        c_edges = cycle_edges[c_idx]
        
        # Determine the color of the cycle
        # Since it's violated, one color has count 6.
        mono_color = -1
        for col in range(4):
            if cycle_color_counts[c_idx][col] == 6:
                mono_color = col
                break
        
        # We want to change one of the 6 edges to a different color.
        # For each edge e in c_edges:
        #   For each new_color in {0,1,2,3} \ {colors[e]}:
        #     Simulate the change.
        #     Count how many cycles become violated (that weren't before) and how many are fixed.
        
        best_move = None
        best_score = float('inf')
        
        # Optimization: Instead of simulating full counts for all 6 edges * 3 colors = 18 scenarios
        # we can just calculate the delta in the number of violated cycles.
        
        for e_idx in c_edges:
            old_color = colors[e_idx]
            for new_color in range(4):
                if new_color == old_color:
                    continue
                
                # Calculate impact
                # The cycle c_idx will be fixed (unless it becomes mono with new_color? No, we change 1 edge, so it can't be mono with new_color unless other 5 are new_color, which is impossible since they were old_color).
                # So c_idx is definitely fixed.
                
                # Check other cycles connected to e_idx
                # A cycle is "at risk" if it has 5 edges of new_color and 1 edge of old_color (which is e_idx).
                # If we change e_idx to new_color, it becomes 6 new_color -> Violated.
                # A cycle is "fixed" if it has 6 edges of old_color (which is e_idx).
                
                new_violations = 0
                fixed_violations = 1 # c_idx itself
                
                for other_c_idx in edge_to_cycles[e_idx]:
                    if other_c_idx == c_idx:
                        continue
                    
                    counts = cycle_color_counts[other_c_idx]
                    
                    # Check if this cycle becomes violated
                    if counts[new_color] == 5:
                        # It has 5 of new_color. The 6th edge is e_idx (old_color).
                        # Changing e_idx to new_color makes it 6 new_color.
                        new_violations += 1
                        
                    # Check if this cycle is fixed
                    if counts[old_color] == 6:
                        # It is currently violated (mono old_color).
                        # Changing e_idx (old_color) to new_color breaks it.
                        fixed_violations += 1
                
                # Net change in number of violated cycles
                # We want to minimize (new_violations - fixed_violations)
                # Actually, we want to minimize the total count.
                # Current total = len(violated_cycles)
                # New total = len(violated_cycles) + new_violations - fixed_violations
                # So we minimize (new_violations - fixed_violations)
                
                score = new_violations - fixed_violations
                
                if score < best_score:
                    best_score = score
                    best_move = (e_idx, new_color)
                    
                    # Early termination if we find a move that fixes violations without creating new ones
                    if best_score <= 0:
                        break
            if best_score <= 0:
                break
        
        # If we found a move that doesn't increase the number of violations (score <= 0), take it.
        # If all moves increase violations (score > 0), we are in a local minimum.
        # We can either:
        # a) Take the least bad move (greedy).
        # b) Randomly restart (expensive).
        # c) Accept a bad move with low probability (annealing).
        
        # Let's use a hybrid:
        # If best_score < 0, accept.
        # If best_score == 0, accept with 50% prob (to explore neutral moves).
        # If best_score > 0, accept with 1% prob (to escape).
        
        accept = False
        if best_score < 0:
            accept = True
        elif best_score == 0:
            if random.random() < 0.5:
                accept = True
        else:
            if random.random() < 0.01:
                accept = True
        
        if accept and best_move:
            e_idx, new_color = best_move
            old_color = colors[e_idx]
            colors[e_idx] = new_color
            
            # Update cycle counts
            for c_idx2 in edge_to_cycles[e_idx]:
                counts = cycle_color_counts[c_idx2]
                
                # Update violated status for c_idx2
                # Before change
                was_violated = any(c == 6 for c in counts)
                
                counts[old_color] -= 1
                counts[new_color] += 1
                
                # After change
                is_violated = any(c == 6 for c in counts)
                
                if was_violated and not is_violated:
                    violated_cycles.discard(c_idx2)
                elif not was_violated and is_violated:
                    violated_cycles.add(c_idx2)
        else:
            # Stuck. To escape, we can do a "shake": randomly change a few edges.
            # Or just restart the local search from a new random state?
            # Restarting is O(num_cycles) which is ~700k operations. Might be too slow if done often.
            # Let's try a small shake: change 5 random edges.
            for _ in range(5):
                e_idx = random.randint(0, num_edges - 1)
                old_color = colors[e_idx]
                new_color = random.choice([c for c in range(4) if c != old_color])
                colors[e_idx] = new_color
                
                for c_idx2 in edge_to_cycles[e_idx]:
                    counts = cycle_color_counts[c_idx2]
                    was_violated = any(c == 6 for c in counts)
                    
                    counts[old_color] -= 1
                    counts[new_color] += 1
                    
                    is_violated = any(c == 6 for c in counts)
                    
                    if was_violated and not is_violated:
                        violated_cycles.discard(c_idx2)
                    elif not was_violated and is_violated:
                        violated_cycles.add(c_idx2)

    # If we didn't find a valid coloring for 17, return the fallback
    return {'n': best_n_found, 'colors': best_colors_found}

if __name__ == "__main__":
    result = construct()
    print(f"Result: n={result['n']}")
