import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2
DEADLINE = 85  # seconds, safety margin from 90s limit

def edge_index(i, j):
    """Return index of edge (i,j) in lexicographic ordering."""
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def get_four_cycles():
    """Generate all 4-cycles (as tuples of 4 edge indices) for all 4-vertex subsets."""
    cycles = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Three 4-cycles on {a,b,c,d}:
        # (a,b,c,d): edges (a,b),(b,c),(c,d),(d,a)
        # (a,b,d,c): edges (a,b),(b,d),(d,c),(c,a)
        # (a,c,b,d): edges (a,c),(c,b),(b,d),(d,a)
        cycles.append((
            edge_index(a,b), edge_index(b,c), edge_index(c,d), edge_index(d,a)
        ))
        cycles.append((
            edge_index(a,b), edge_index(b,d), edge_index(d,c), edge_index(c,a)
        ))
        cycles.append((
            edge_index(a,c), edge_index(c,b), edge_index(b,d), edge_index(d,a)
        ))
    return cycles

def is_valid(colors, cycles):
    """Check if no 4-cycle has four distinct edge colors."""
    for e0, e1, e2, e3 in cycles:
        c0, c1, c2, c3 = colors[e0], colors[e1], colors[e2], colors[e3]
        if len({c0, c1, c2, c3}) == 4:
            return False
    return True

def count_colors(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    cycles = get_four_cycles()
    num_cycles = len(cycles)
    
    # Strategy: Start with all zeros (valid, 1 color), then try to introduce new colors
    # greedily while maintaining validity.
    
    best_colors = [0] * NUM_EDGES
    best_count = 1
    
    # Pre-compute for each edge, which cycles it participates in and its position
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, (e0, e1, e2, e3) in enumerate(cycles):
        for e in (e0, e1, e2, e3):
            edge_to_cycles[e].append(ci)
    
    # Pre-compute position of each edge within each cycle for faster lookup
    # edge_pos[edge_idx][cycle_idx] = position in that cycle (0-3), or -1 if not in cycle
    # But that's too big. Instead, for each edge, store list of (cycle_idx, position)
    edge_cycle_pos = [[] for _ in range(NUM_EDGES)]
    for ci, (e0, e1, e2, e3) in enumerate(cycles):
        for pos, e in enumerate((e0, e1, e2, e3)):
            edge_cycle_pos[e].append((ci, pos))
    
    # For each cycle, store the 4 edge indices as a tuple for fast access
    cycle_edges = [tuple(c) for c in cycles]
    
    def try_assign_fast(colors, edge_idx, color_val):
        """Try assigning color_val to edge_idx, return True if valid. Optimized."""
        for ci, pos in edge_cycle_pos[edge_idx]:
            e0, e1, e2, e3 = cycle_edges[ci]
            # Get the 4 colors, with the new value at the right position
            c0 = colors[e0] if pos != 0 else color_val
            c1 = colors[e1] if pos != 1 else color_val
            c2 = colors[e2] if pos != 2 else color_val
            c3 = colors[e3] if pos != 3 else color_val
            # Check if all 4 are distinct
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                return False
        return True
    
    # Greedy: try to assign new colors to edges
    # Use multiple random restarts with different edge orderings
    random.seed(42)
    
    for restart in range(300):
        if time.time() - start_time > DEADLINE:
            break
        
        colors = [0] * NUM_EDGES
        used_colors = {0}
        
        # Random edge ordering
        edges = list(range(NUM_EDGES))
        random.shuffle(edges)
        
        # Try to assign increasingly new colors
        for e in edges:
            if time.time() - start_time > DEADLINE:
                break
            # Try to assign a new color (max used + 1)
            new_color = max(used_colors) + 1
            if try_assign_fast(colors, e, new_color):
                colors[e] = new_color
                used_colors.add(new_color)
        
        curr_count = count_colors(colors)
        if curr_count > best_count:
            best_count = curr_count
            best_colors = colors[:]
        
        # Also try: start from best_colors and perturb
        if restart > 10 and best_count > 1:
            colors2 = best_colors[:]
            used2 = set(colors2)
            edges2 = list(range(NUM_EDGES))
            random.shuffle(edges2)
            improved = False
            for e in edges2:
                if time.time() - start_time > DEADLINE:
                    break
                # Try to change to a new color
                new_color = max(used2) + 1
                if try_assign_fast(colors2, e, new_color):
                    colors2[e] = new_color
                    used2.add(new_color)
                    improved = True
                # Also try to change to an existing color that's different
                # to free up the current color for reuse elsewhere
                curr = colors2[e]
                for alt in sorted(used2 - {curr}):
                    if try_assign_fast(colors2, e, alt):
                        colors2[e] = alt
                        # Remove curr if no longer used
                        if curr not in colors2:
                            used2.discard(curr)
                        break
            
            curr_count2 = count_colors(colors2)
            if curr_count2 > best_count:
                best_count = curr_count2
                best_colors = colors2[:]
    
    return best_colors
