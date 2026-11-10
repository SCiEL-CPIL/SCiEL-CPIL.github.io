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
    best_colors = None
    best_count = 0
    
    random.seed(42)
    
    # Precompute for each edge, the list of "other 3 edges" for each cycle it belongs to.
    # This allows for faster validity checking.
    # For edge e in cycle C, let others = C \ {e}.
    # If the colors of 'others' are all distinct, then the new color for e MUST be one of them.
    
    # edge_others[e] is a list of tuples (idx1, idx2, idx3)
    edge_others = [[] for _ in range(NUM_PAIRS)]
    for cycle in all_cycles:
        for i in range(4):
            e = cycle[i]
            others = [cycle[j] for j in range(4) if j != i]
            edge_others[e].append(others)
    
    def is_valid_fast(colors, e, new_color):
        """
        Check if assigning new_color to edge e is valid.
        We only need to check cycles containing e.
        For each cycle, if the other 3 edges have distinct colors, new_color must match one of them.
        """
        for others in edge_others[e]:
            c1 = colors[others[0]]
            c2 = colors[others[1]]
            c3 = colors[others[2]]
            # Check if c1, c2, c3 are distinct
            if c1 != c2 and c1 != c3 and c2 != c3:
                # They are distinct, so new_color must be one of them
                if new_color != c1 and new_color != c2 and new_color != c3:
                    return False
        return True

    # Try multiple restarts
    for attempt in range(50):
        # Initialize with a valid 2-coloring
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
        # We want to maximize used_colors_count.
        # Strategy:
        # 1. Try to assign a NEW color (count 0) to a random edge.
        # 2. If that fails, try to assign a random EXISTING color to shuffle the structure.
        
        no_improvement = 0
        max_steps = 50000
        
        for step in range(max_steps):
            e = random.randint(0, NUM_PAIRS - 1)
            old_color = colors[e]
            
            # 80% chance to try a new color, 20% to shuffle
            if random.random() < 0.8:
                # Find an unused color
                # To make this efficient, we can keep a list of unused colors?
                # Or just pick random until we find one?
                # Since we have 120 colors and likely use < 50, random picking is fine.
                found_new = False
                for _ in range(10): # Try 10 random candidates
                    new_color = random.randint(0, MAX_COLOR)
                    if color_counts[new_color] == 0:
                        if is_valid_fast(colors, e, new_color):
                            # Accept
                            color_counts[old_color] -= 1
                            if color_counts[old_color] == 0:
                                used_colors_count -= 1
                            color_counts[new_color] += 1
                            used_colors_count += 1
                            colors[e] = new_color
                            no_improvement = 0
                            found_new = True
                            break
                        else:
                            # Invalid, try another
                            pass
                if found_new:
                    continue
            
            # Shuffle move
            new_color = random.randint(0, MAX_COLOR)
            if new_color == old_color:
                continue
            
            if is_valid_fast(colors, e, new_color):
                # Accept shuffle
                color_counts[old_color] -= 1
                if color_counts[old_color] == 0:
                    used_colors_count -= 1
                color_counts[new_color] += 1
                if color_counts[new_color] == 1:
                    used_colors_count += 1
                colors[e] = new_color
                # Shuffle doesn't necessarily improve score, but helps exploration
                # We don't reset no_improvement on shuffle, only on new color?
                # Actually, if shuffle allows a new color later, it's good.
                # Let's say any valid move resets a "stagnation" counter?
                # Or just run for max_steps?
                pass
            else:
                no_improvement += 1
                
            if no_improvement > 5000:
                break
        
        current_count = used_colors_count
        if current_count > best_count:
            best_count = current_count
            best_colors = colors[:]
        
        if best_count > 60: # Threshold to stop early if we find a very good one
            break

    if best_colors is None:
        return [0] * NUM_PAIRS
    
    return best_colors
# EVOLVE-BLOCK-END
