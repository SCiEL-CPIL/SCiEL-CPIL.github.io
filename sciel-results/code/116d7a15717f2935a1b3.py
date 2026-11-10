"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import itertools
import time

N = 16
NUM_PAIRS = N * (N - 1) // 2  # 120
MAX_COLOR = 119

def edge_index(i, j):
    if i > j:
        i, j = j, i
    idx = 0
    for k in range(i):
        idx += (N - 1 - k)
    idx += (j - 1 - i)
    return idx

# Precompute all 4-cycles and their edge indices
all_cycles = []
# Also precompute for each edge, which cycles it is in
edge_to_cycles = [[] for _ in range(NUM_PAIRS)]

for quad in itertools.combinations(range(N), 4):
    a, b, c, d = quad
    # Cycle 1: a-b-c-d-a
    c1 = [edge_index(a, b), edge_index(b, c), edge_index(c, d), edge_index(d, a)]
    # Cycle 2: a-b-d-c-a
    c2 = [edge_index(a, b), edge_index(b, d), edge_index(d, c), edge_index(c, a)]
    # Cycle 3: a-c-b-d-a
    c3 = [edge_index(a, c), edge_index(c, b), edge_index(b, d), edge_index(d, a)]
    
    for cyc in [c1, c2, c3]:
        all_cycles.append(cyc)
        for e in cyc:
            edge_to_cycles[e].append(len(all_cycles) - 1)

def is_valid_local(colors, edge_idx):
    """Check if the coloring is valid, focusing on cycles involving edge_idx."""
    for cycle_idx in edge_to_cycles[edge_idx]:
        cycle = all_cycles[cycle_idx]
        # Check if 4 distinct colors
        c0 = colors[cycle[0]]
        c1 = colors[cycle[1]]
        c2 = colors[cycle[2]]
        c3 = colors[cycle[3]]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
    return True

def is_valid_full(colors):
    for cycle in all_cycles:
        c0 = colors[cycle[0]]
        c1 = colors[cycle[1]]
        c2 = colors[cycle[2]]
        c3 = colors[cycle[3]]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
    return True

def count_colors(colors):
    return len(set(colors))

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a coloring of K_16 edges with colors in 0..119 such that
    no 4-cycle has 4 distinct edge colors. Maximize number of distinct colors.
    """
    # Strategy: 
    # 1. Start with a valid coloring using a small number of colors (e.g., 2 or 3).
    # 2. Use local search to rearrange colors.
    # 3. Greedily introduce new colors if possible.
    
    # Initialize with a simple valid pattern
    # Using (i+j) % 2 gives 2 colors. Let's try to start with more structure.
    # A known valid coloring for avoiding rainbow K4 is using 3 colors in a specific pattern.
    # However, a random start with 2 colors is safe.
    
    best_colors = None
    best_count = 0
    
    random.seed(42)
    
    # Try multiple restarts
    for attempt in range(20):
        # Start with 2 colors
        colors = [0] * NUM_PAIRS
        for i in range(N):
            for j in range(i + 1, N):
                idx = edge_index(i, j)
                # Simple parity
                colors[idx] = (i + j) % 2
        
        # Shuffle colors slightly to break symmetry? 
        # Actually, let's just run local search.
        
        current_count = count_colors(colors)
        used_colors = set(colors)
        
        # Local search loop
        # We want to increase the number of distinct colors.
        # We do this by:
        # 1. Picking an edge.
        # 2. Trying to assign a NEW color to it.
        # 3. If it violates constraints, we try to "fix" the neighborhood by changing other edges.
        # A simpler heuristic: 
        # Iterate through edges. For each edge, try all unused colors. If one works, take it.
        # If none work, try to swap colors with other edges to make space?
        
        # Let's use a simulated annealing / hill climbing approach.
        # State: colors
        # Score: number of distinct colors.
        # Move: change color of one edge.
        
        # To be efficient, we track the set of used colors.
        
        # Greedy Phase: Add new colors
        # We maintain a list of "candidate" edges that might take a new color.
        
        # Let's do a simple hill climbing:
        # 1. Try to change an edge to a new color.
        # 2. If it fails, try to change an edge to an existing color to "relax" constraints? 
        #    Actually, changing to an existing color doesn't help add new colors directly, 
        #    but it might allow a different edge to take a new color later.
        
        # Better: 
        # Keep a list of all edges.
        # Repeat:
        #   For each edge (randomly shuffled):
        #      Try to assign a random NEW color (not in used_colors).
        #      If valid, update used_colors and continue.
        #      If invalid, try to assign a random EXISTING color? 
        #         This might break validity if we're not careful, but we check.
        #         If it becomes valid, we accept it (score doesn't change, but structure changes).
        #         This is a random walk on the valid configurations.
        
        # To ensure progress, we bias towards adding new colors.
        
        for step in range(5000):
            e = random.randint(0, NUM_PAIRS - 1)
            old_color = colors[e]
            
            # Option 1: Try a new color
            if len(used_colors) < MAX_COLOR + 1:
                # Pick a color not in used_colors
                # To avoid expensive set operations, we can keep a list of available colors?
                # Or just pick random and check.
                new_color = random.randint(0, MAX_COLOR)
                if new_color not in used_colors:
                    colors[e] = new_color
                    if is_valid_local(colors, e):
                        used_colors.add(new_color)
                        continue
                    else:
                        colors[e] = old_color
            
            # Option 2: Try an existing color (to shuffle)
            # Pick a random existing color
            if used_colors:
                new_color = random.choice(list(used_colors))
                if new_color != old_color:
                    colors[e] = new_color
                    if is_valid_local(colors, e):
                        # Check if old_color is still used
                        # This is expensive to check every time.
                        # We can defer the set update or just count at the end?
                        # For now, accept the move.
                        # We need to update used_colors.
                        # If old_color is no longer used, remove it.
                        # This requires a scan or a count array.
                        
                        # Let's maintain a count array for colors.
                        pass # We'll implement counts below
            
        # The above loop is a bit messy with state management.
        # Let's restart with a cleaner implementation using a color count array.
        
        break # Disable the messy loop above, implement clean one below.

    # Clean Implementation
    for attempt in range(50):
        # Initialize
        colors = [0] * NUM_PAIRS
        for i in range(N):
            for j in range(i + 1, N):
                idx = edge_index(i, j)
                colors[idx] = (i + j) % 2
        
        color_counts = [0] * (MAX_COLOR + 1)
        for c in colors:
            color_counts[c] += 1
        
        used_colors_count = sum(1 for c in color_counts if c > 0)
        
        # Local Search
        no_improvement = 0
        for step in range(20000):
            e = random.randint(0, NUM_PAIRS - 1)
            old_color = colors[e]
            
            # Try to introduce a new color
            if used_colors_count < MAX_COLOR + 1:
                # Find an unused color
                # Randomly pick a few candidates
                for _ in range(5):
                    new_color = random.randint(0, MAX_COLOR)
                    if color_counts[new_color] == 0:
                        colors[e] = new_color
                        if is_valid_local(colors, e):
                            color_counts[old_color] -= 1
                            if color_counts[old_color] == 0:
                                used_colors_count -= 1
                            color_counts[new_color] += 1
                            used_colors_count += 1
                            no_improvement = 0
                            break
                        else:
                            colors[e] = old_color
                else:
                    # No new color found easily, try to shuffle
                    pass
            
            # Try to shuffle existing colors
            # Pick a random color from used colors
            # To pick a random used color, we can just pick a random index and check count?
            # Or maintain a list of used colors?
            # Let's just pick a random color in 0..MAX_COLOR and see if it's used.
            new_color = random.randint(0, MAX_COLOR)
            
            if new_color == old_color:
                continue
            
            colors[e] = new_color
            if is_valid_local(colors, e):
                # Accept move
                color_counts[old_color] -= 1
                if color_counts[old_color] == 0:
                    used_colors_count -= 1
                color_counts[new_color] += 1
                if color_counts[new_color] == 1:
                    used_colors_count += 1
                no_improvement = 0
            else:
                colors[e] = old_color
                no_improvement += 1
            
            if no_improvement > 2000:
                break
        
        current_count = used_colors_count
        if current_count > best_count:
            best_count = current_count
            best_colors = colors[:]
        
        if best_count > 80: # Good enough?
            break

    if best_colors is None:
        return [0] * NUM_PAIRS
    
    return best_colors
# EVOLVE-BLOCK-END
