import time
import sys
import random
import itertools
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLORS = 120

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    # Precompute edge indices
    edge_idx = [[-1]*N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1

    # Precompute all 4-cycles and their edge indices
    cycles = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        cycles.append((edge_idx[a][b], edge_idx[b][c], edge_idx[c][d], edge_idx[d][a]))
        # Cycle 2: a-b-d-c-a
        cycles.append((edge_idx[a][b], edge_idx[b][d], edge_idx[d][c], edge_idx[c][a]))
        # Cycle 3: a-c-b-d-a
        cycles.append((edge_idx[a][c], edge_idx[c][b], edge_idx[b][d], edge_idx[d][a]))

    NUM_CYCLES = len(cycles)

    # Map edge index to list of cycle indices it belongs to
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for i, cyc in enumerate(cycles):
        for e in cyc:
            edge_to_cycles[e].append(i)

    # Precompute for each cycle, the positions of each edge within the cycle
    # cycle_edge_pos[cycle_idx][edge_idx] = position (0-3) or -1 if not in cycle
    # Actually, let's precompute for faster access:
    # For each cycle, store the 4 edge indices as a tuple

    # Fast validity check for changing one edge
    def is_valid_change(colors, edge, new_color):
        for c_idx in edge_to_cycles[edge]:
            c = cycles[c_idx]
            # Build color tuple with the changed edge
            col0 = new_color if c[0] == edge else colors[c[0]]
            col1 = new_color if c[1] == edge else colors[c[1]]
            col2 = new_color if c[2] == edge else colors[c[2]]
            col3 = new_color if c[3] == edge else colors[c[3]]
            # Check if 4 distinct
            if col0 != col1 and col0 != col2 and col0 != col3 and \
               col1 != col2 and col1 != col3 and \
               col2 != col3:
                return False
        return True

    # Count distinct colors efficiently
    def count_colors(colors):
        return len(set(colors))

    # Generate recursive partition seed
    def get_seed_colors():
        colors = [0] * NUM_EDGES
        color_id = 0
        def process_partition(vertices):
            nonlocal color_id
            if len(vertices) <= 1:
                return
            mid = len(vertices) // 2
            left = vertices[:mid]
            right = vertices[mid:]
            for u in left:
                for v in right:
                    idx = edge_idx[u][v]
                    colors[idx] = color_id
            color_id += 1
            process_partition(left)
            process_partition(right)
        process_partition(list(range(N)))
        return colors

    # Try different initial seeds
    def get_seed_colors_v2():
        """Alternative seed: group-based coloring"""
        colors = [0] * NUM_EDGES
        # Color by vertex pair sum mod some value, then adjust
        for i in range(N):
            for j in range(i+1, N):
                idx = edge_idx[i][j]
                colors[idx] = (i + j) % 20
        return colors

    def get_seed_colors_v3():
        """Another seed: hierarchical with different split"""
        colors = [0] * NUM_EDGES
        color_id = 0
        def process(vertices, offset):
            nonlocal color_id
            if len(vertices) <= 1:
                return
            mid = len(vertices) // 2
            left = vertices[:mid]
            right = vertices[mid:]
            for u in left:
                for v in right:
                    idx = edge_idx[u][v]
                    colors[idx] = offset
            offset += 1
            process(left, offset)
            process(right, offset)
        process(list(range(N)), 0)
        return colors

    # Main search function
    def search(initial_colors, time_budget):
        current = initial_colors.copy()
        best_score = count_colors(current)
        best_colors = current.copy()
        
        # Track color frequencies
        freq = [0] * MAX_COLORS
        for c in current:
            freq[c] += 1
        
        # Track which colors are used
        used_colors = set(current)
        
        rng = random.Random()
        
        search_start = time.time()
        steps = 0
        no_improve_steps = 0
        max_no_improve = 5000
        
        while time.time() - search_start < time_budget:
            steps += 1
            
            # Pick a random edge
            e = rng.randint(0, NUM_EDGES - 1)
            old_color = current[e]
            
            # Strategy: try to introduce a new color or modify existing
            # 70% chance: try a new color (possibly unused)
            # 30% chance: try a random existing color
            if rng.random() < 0.7:
                # Try an unused color if possible
                unused = [c for c in range(MAX_COLORS) if freq[c] == 0]
                if unused and rng.random() < 0.5:
                    new_color = rng.choice(unused)
                else:
                    new_color = rng.randint(0, MAX_COLORS - 1)
            else:
                new_color = rng.randint(0, MAX_COLORS - 1)
            
            if new_color == old_color:
                continue
            
            if is_valid_change(current, e, new_color):
                # Apply the change
                freq[old_color] -= 1
                freq[new_color] += 1
                current[e] = new_color
                
                # Update used colors
                if freq[old_color] == 0:
                    used_colors.discard(old_color)
                used_colors.add(new_color)
                
                score = len(used_colors)
                
                if score > best_score:
                    best_score = score
                    best_colors = current.copy()
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
                
                # Occasionally revert if stuck
                if no_improve_steps > max_no_improve and rng.random() < 0.1:
                    current = best_colors.copy()
                    # Rebuild freq and used_colors
                    freq = [0] * MAX_COLORS
                    for c in current:
                        freq[c] += 1
                    used_colors = set(current)
                    no_improve_steps = 0
            else:
                # Invalid move, try a slightly different approach
                # Try changing to a color that's already used by a neighbor edge
                # This is more likely to be valid
                if rng.random() < 0.3:
                    # Find a color used by edges in the same "region"
                    # Just try a few random existing colors
                    existing = [c for c in range(MAX_COLORS) if freq[c] > 0]
                    if existing:
                        new_color2 = rng.choice(existing)
                        if new_color2 != old_color and is_valid_change(current, e, new_color2):
                            freq[old_color] -= 1
                            freq[new_color2] += 1
                            current[e] = new_color2
                            if freq[old_color] == 0:
                                used_colors.discard(old_color)
                            used_colors.add(new_color2)
                            score = len(used_colors)
                            if score > best_score:
                                best_score = score
                                best_colors = current.copy()
                                no_improve_steps = 0
                            else:
                                no_improve_steps += 1
        
        return best_score, best_colors

    # Multi-start search
    seeds = [get_seed_colors(), get_seed_colors_v2(), get_seed_colors_v3()]
    
    best_overall_score = 0
    best_overall_colors = None
    
    # Allocate time: 40% to first seed, 30% to second, 30% to third
    time_alloc = [0.4, 0.3, 0.3]
    total_time_budget = deadline - start_time
    
    for i, seed in enumerate(seeds):
        if time.time() > deadline - 5:  # Leave 5s safety margin
            break
        
        t_budget = time_alloc[i] * total_time_budget
        score, colors = search(seed, t_budget)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_colors = colors
        
        # If we found a good score, spend remaining time polishing
        if best_overall_score >= 25:
            # Use remaining time for a final polish on the best solution
            remaining = deadline - start_time - 2
            if remaining > 2:
                final_score, final_colors = search(best_overall_colors, remaining)
                if final_score > best_overall_score:
                    best_overall_score = final_score
                    best_overall_colors = final_colors
            break
    
    # Final safety: ensure we have a valid result
    if best_overall_colors is None:
        best_overall_colors = seeds[0]
        best_overall_score = count_colors(best_overall_colors)
    
    return best_overall_colors
