"""Improved program: vectorized local search with multiple restarts and smart initialization."""
import random
import time
import itertools
import numpy as np

N = 16
E = N * (N - 1) // 2  # 120 edges
MAX_COLOR = 119

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (2 * N - 1 - i) // 2 + j - i - 1

def precompute():
    """
    Returns:
    - cycles: list of tuples of 4 edge indices forming a 4-cycle
    - edge_to_cycles: for each edge, list of cycle indices it belongs to
    - cycle_edges: numpy array of shape (num_cycles, 4)
    - For each edge, the list of (cycle_idx, pos) pairs
    """
    cycles = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b, b-c, c-d, d-a
        e1 = (edge_idx(a, b), edge_idx(b, c), edge_idx(c, d), edge_idx(d, a))
        # Cycle 2: a-b, b-d, d-c, c-a
        e2 = (edge_idx(a, b), edge_idx(b, d), edge_idx(d, c), edge_idx(c, a))
        # Cycle 3: a-c, c-b, b-d, d-a
        e3 = (edge_idx(a, c), edge_idx(c, b), edge_idx(b, d), edge_idx(d, a))
        
        for cyc in (e1, e2, e3):
            cycles.append(cyc)
    
    num_cycles = len(cycles)
    cycle_edges = np.array(cycles, dtype=np.int32)  # shape (num_cycles, 4)
    
    # For each edge, list of (cycle_idx, pos)
    edge_cycle_pos = [[] for _ in range(E)]
    for ci in range(num_cycles):
        for pos in range(4):
            e = cycle_edges[ci, pos]
            edge_cycle_pos[e].append((ci, pos))
    
    # For each edge, list of the other 3 edges in each cycle it belongs to
    # edge_other_edges[e] = list of tuples of 3 edge indices
    edge_other_edges = []
    for e in range(E):
        others_list = []
        for ci, pos in edge_cycle_pos[e]:
            others = tuple(cycle_edges[ci, p] for p in range(4) if p != pos)
            others_list.append(others)
        edge_other_edges.append(others_list)
    
    return cycles, cycle_edges, edge_cycle_pos, edge_other_edges

def is_valid(colors):
    """Check if no 4-cycle has 4 distinct colors."""
    # colors is a numpy array of length E
    # For each cycle, check if all 4 colors are distinct
    # cycle_colors has shape (num_cycles, 4)
    cycle_colors = colors[cycle_edges]
    # A cycle is invalid if all 4 colors are distinct
    # Check: c0 != c1, c0 != c2, c0 != c3, c1 != c2, c1 != c3, c2 != c3
    c0, c1, c2, c3 = cycle_colors[:, 0], cycle_colors[:, 1], cycle_colors[:, 2], cycle_colors[:, 3]
    invalid = (c0 != c1) & (c0 != c2) & (c0 != c3) & (c1 != c2) & (c1 != c3) & (c2 != c3)
    return not np.any(invalid)

def count_colors(colors):
    return len(set(colors.tolist()))

def construct():
    deadline = time.time() + 55.0
    
    cycles, cycle_edges, edge_cycle_pos, edge_other_edges = precompute()
    num_cycles = len(cycles)
    
    # Precompute for vectorized validation
    # For each cycle, we need to check 6 pairwise inequalities
    # We'll use a helper that takes colors and returns boolean array of invalid cycles
    
    def validate_all(colors):
        """Returns number of invalid cycles (0 means valid)."""
        cc = colors[cycle_edges]
        c0, c1, c2, c3 = cc[:, 0], cc[:, 1], cc[:, 2], cc[:, 3]
        invalid = (c0 != c1) & (c0 != c2) & (c0 != c3) & (c1 != c2) & (c1 != c3) & (c2 != c3)
        return int(np.sum(invalid))
    
    # Initial coloring: color(i,j) = i
    colors = np.zeros(E, dtype=np.int32)
    for i in range(N):
        for j in range(i + 1, N):
            colors[edge_idx(i, j)] = i
    
    best = colors.copy()
    best_count = count_colors(best)
    
    random.seed(42)
    
    # Precompute for each edge the list of other-edges tuples for fast constraint checking
    # We'll use a faster approach: for a proposed recolor of edge e to color c,
    # check all cycles containing e: for each such cycle, if the other 3 edges have
    # 3 distinct colors, then c must equal one of them.
    
    # For speed, precompute for each edge:
    # - list of the 3 other edges in each cycle (already in edge_other_edges)
    
    def can_recolor(colors, e, new_color):
        """Check if recoloring edge e to new_color maintains validity."""
        others_list = edge_other_edges[e]
        for others in others_list:
            c0 = colors[others[0]]
            c1 = colors[others[1]]
            c2 = colors[others[2]]
            # If all three are distinct, new_color must be one of them
            if c0 != c1 and c1 != c2 and c0 != c2:
                if new_color != c0 and new_color != c1 and new_color != c2:
                    return False
        return True
    
    # Sort edges by number of constraints (fewer first)
    edge_order = sorted(range(E), key=lambda e: len(edge_other_edges[e]))
    
    # Main search with multiple restarts
    best_overall = best.copy()
    best_overall_count = best_count
    
    for restart in range(50):
        if time.time() >= deadline:
            break
        
        # Reset colors to initial or a random valid start
        if restart == 0:
            colors = best_overall.copy()
        else:
            # Start from best and add some random perturbation
            colors = best_overall.copy()
            # Randomly recolor up to 10 edges to new valid colors
            for _ in range(10):
                e = random.choice(edge_order)
                old_c = colors[e]
                # Try a few random new colors
                for _ in range(5):
                    new_c = random.randint(0, MAX_COLOR)
                    if new_c == old_c:
                        continue
                    if can_recolor(colors, e, new_c):
                        colors[e] = new_c
                        break
        
        cur_count = count_colors(colors)
        if cur_count > best_overall_count:
            best_overall = colors.copy()
            best_overall_count = cur_count
        
        # Local search phase
        iterations = 0
        while time.time() < deadline - 0.5:
            iterations += 1
            
            # Pick a random edge
            e = random.choice(edge_order)
            old_color = colors[e]
            
            # Determine valid new colors
            # A color c is valid if for every cycle containing e,
            # either the other 3 colors are not all distinct, or c is one of them
            
            # Compute the set of "blocked" colors: colors that would create a rainbow
            # For each cycle with 3 distinct other colors, all colors except those 3 are blocked
            # So valid colors = intersection over all such cycles of {the 3 colors}
            # Plus colors that are always valid (if no constraint blocks them)
            
            valid_mask = np.ones(MAX_COLOR + 1, dtype=bool)
            for others in edge_other_edges[e]:
                c0 = colors[others[0]]
                c1 = colors[others[1]]
                c2 = colors[others[2]]
                if c0 != c1 and c1 != c2 and c0 != c2:
                    # Only c0, c1, c2 are valid from this constraint
                    mask = np.zeros(MAX_COLOR + 1, dtype=bool)
                    mask[c0] = True
                    mask[c1] = True
                    mask[c2] = True
                    valid_mask &= mask
            
            valid_colors = np.where(valid_mask)[0]
            valid_colors = [c for c in valid_colors if c != old_color]
            
            if not valid_colors:
                continue
            
            # Prefer unused colors to maximize diversity
            used_set = set(colors.tolist())
            unused_valid = [c for c in valid_colors if c not in used_set]
            
            if unused_valid:
                new_color = random.choice(unused_valid)
            else:
                new_color = random.choice(valid_colors)
            
            colors[e] = new_color
            
            nc = count_colors(colors)
            if nc > best_overall_count:
                best_overall = colors.copy()
                best_overall_count = nc
    
    return best_overall.tolist()
