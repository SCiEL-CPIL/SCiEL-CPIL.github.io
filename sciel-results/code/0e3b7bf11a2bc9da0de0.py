"""Combined approach: bit-parallel backtracking for C6-free 4-colorings of K_n."""
import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Fallback: n=9 seed
    n_seed = 9
    seed_colors = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    best = {'n': n_seed, 'colors': seed_colors}
    
    # Try n=18 first (most valuable), then n=17
    for target_n in [18, 17]:
        if time.time() > deadline - 30.0:
            break
        try:
            result = _backtrack_search(target_n, deadline - 20.0)
            if result is not None:
                return result
        except Exception:
            pass
    
    # If backtracking failed, try SA for n=17 as fallback
    if time.time() < deadline - 10.0:
        try:
            result = _sa_search(17, deadline - 5.0)
            if result is not None:
                return result
        except Exception:
            pass
    
    return best


def _precompute_cycles(n):
    """Precompute all C6 cycles and their edge sets."""
    edge_map = {}
    edge_list = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_map[(i, j)] = idx
            edge_list.append((i, j))
            idx += 1
    num_edges = len(edge_list)
    
    # Generate all unique C6 cycles
    cycles = []  # list of tuples of 6 edge indices
    edge_to_cycles = defaultdict(set)  # edge_idx -> set of cycle indices
    
    vertices = list(range(n))
    
    for subset in itertools.combinations(vertices, 6):
        # Generate all distinct 6-cycles on these 6 vertices
        # Fix vertex 0 as start, then permute the other 5
        v0 = subset[0]
        others = subset[1:]
        
        seen = set()
        for perm in itertools.permutations(others):
            cycle = (v0,) + perm
            # Canonicalize: rotate so min is first, then fix direction
            min_idx = cycle.index(min(cycle))
            rot = cycle[min_idx:] + cycle[:min_idx]
            # Fix direction: second vertex < last vertex
            if rot[1] > rot[-1]:
                rot = (rot[0],) + rot[:0:-1]
            
            if rot in seen:
                continue
            seen.add(rot)
            
            # Extract edges
            edges = []
            for k in range(6):
                u = rot[k]
                v = rot[(k + 1) % 6]
                if u > v:
                    u, v = v, u
                edges.append(edge_map[(u, v)])
            
            cycle_id = len(cycles)
            cycles.append(tuple(edges))
            for e in edges:
                edge_to_cycles[e].add(cycle_id)
    
    return cycles, edge_to_cycles, edge_map, num_edges


def _backtrack_search(n, deadline):
    """
    Backtracking search for C6-free 4-coloring of K_n.
    Uses constraint propagation with bit-parallel cycle checking.
    """
    if time.time() > deadline - 5.0:
        return None
    
    cycles, edge_to_cycles, edge_map, num_edges = _precompute_cycles(n)
    num_cycles = len(cycles)
    
    if num_cycles == 0:
        return {'n': n, 'colors': [0] * num_edges}
    
    # For each cycle, track which colors are "used" (have at least one edge in that color)
    # A cycle is violated if all 6 edges have the same color
    
    # Strategy: assign edges one by one, maintaining for each color a set of assigned edges
    # When assigning edge e to color c, check if this completes any C6 in color c
    
    # Precompute: for each color, which cycles would be completed if edge e is added
    # cycle_edge_count[c][cycle_id] = number of edges in cycle with color c
    
    # Use a more efficient approach:
    # For each edge, precompute which cycles it belongs to
    # When we assign an edge to a color, we check if any cycle now has all 6 edges in that color
    
    # State: colors[edge_idx] = 0-3 or -1 (unassigned)
    colors = [-1] * num_edges
    
    # For each color, maintain a bitmask of which edges are assigned to that color
    # Since num_edges can be up to 153, we use Python ints as bitmasks
    color_masks = [0] * 4  # bit e is set if edge e has color c
    
    # For each cycle and color, track how many edges in that cycle have that color
    cycle_color_counts = [[0] * 4 for _ in range(num_cycles)]
    
    # Precompute for each edge, the list of cycles it's in
    edge_cycle_list = [list(edge_to_cycles[e]) for e in range(num_edges)]
    
    # Order edges by degree (edges in more cycles first) for better pruning
    edge_order = sorted(range(num_edges), key=lambda e: -len(edge_cycle_list[e]))
    
    def check_violation(cycle_id, color):
        """Check if cycle has all 6 edges in given color."""
        return cycle_color_counts[cycle_id][color] == 6
    
    def assign(edge_idx, color):
        """Assign edge to color, update state. Returns True if no violation."""
        colors[edge_idx] = color
        color_masks[color] |= (1 << edge_idx)
        valid = True
        for c_id in edge_cycle_list[edge_idx]:
            cycle_color_counts[c_id][color] += 1
            if cycle_color_counts[c_id][color] == 6:
                valid = False
                break
        return valid
    
    def unassign(edge_idx, color):
        """Remove edge from color, update state."""
        colors[edge_idx] = -1
        color_masks[color] &= ~(1 << edge_idx)
        for c_id in edge_cycle_list[edge_idx]:
            cycle_color_counts[c_id][color] -= 1
    
    def backtrack(pos):
        """Recursive backtracking. pos is index in edge_order."""
        if time.time() > deadline - 2.0:
            return False
        
        if pos == num_edges:
            return True
        
        e = edge_order[pos]
        
        # Try each color
        for color in range(4):
            if assign(e, color):
                if backtrack(pos + 1):
                    return True
                unassign(e, color)
            # If time is running out, stop
            if time.time() > deadline - 2.0:
                return False
        
        return False
    
    # Use iterative deepening with randomization to avoid getting stuck
    # First try deterministic, then randomized restarts
    
    # Try deterministic first
    if backtrack(0):
        return {'n': n, 'colors': colors}
    
    # If deterministic failed, try randomized restarts
    random.seed(42)
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline - 2.0:
            break
        
        # Reset state
        colors = [-1] * num_edges
        color_masks = [0] * 4
        cycle_color_counts = [[0] * 4 for _ in range(num_cycles)]
        
        # Randomize edge order slightly
        edge_order_rand = edge_order[:]
        # Shuffle the last 30% of edges
        split = int(0.7 * num_edges)
        edge_order_rand[split:] = random.sample(edge_order_rand[split:], num_edges - split)
        
        # Redefine backtrack with randomized order
        def backtrack_rand(pos):
            if time.time() > deadline - 2.0:
                return False
            
            if pos == num_edges:
                return True
            
            e = edge_order_rand[pos]
            
            # Randomize color order
            color_order = list(range(4))
            random.shuffle(color_order)
            
            for color in color_order:
                if assign(e, color):
                    if backtrack_rand(pos + 1):
                        return True
                    unassign(e, color)
                if time.time() > deadline - 2.0:
                    return False
            
            return False
        
        if backtrack_rand(0):
            return {'n': n, 'colors': colors}
    
    return None


def _sa_search(n, deadline):
    """Simulated annealing fallback for finding valid colorings."""
    if time.time() > deadline - 5.0:
        return None
    
    cycles, edge_to_cycles, edge_map, num_edges = _precompute_cycles(n)
    num_cycles = len(cycles)
    
    if num_cycles == 0:
        return {'n': n, 'colors': [0] * num_edges}
    
    # Initialize random coloring
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    
    # Cycle violation counts
    cycle_counts = [[0, 0, 0, 0] for _ in range(num_cycles)]
    total_violations = 0
    
    for c_id in range(num_cycles):
        edges = cycles[c_id]
        for e in edges:
            c = colors[e]
            cycle_counts[c_id][c] += 1
        for c in range(4):
            if cycle_counts[c_id][c] == 6:
                total_violations += 1
                break
    
    # Simulated annealing
    T = 1.0
    cooling = 0.9995
    min_T = 0.01
    
    while time.time() < deadline - 1.0:
        if total_violations == 0:
            return {'n': n, 'colors': colors}
        
        # Pick random edge
        e = random.randint(0, num_edges - 1)
        old_color = colors[e]
        new_color = random.randint(0, 3)
        if new_color == old_color:
            continue
        
        # Calculate delta
        delta = 0
        affected_cycles = edge_to_cycles[e]
        
        for c_id in affected_cycles:
            cc = cycle_counts[c_id]
            old_count = cc[old_color]
            
            old_violation = 1 if old_count == 6 else 0
            
            cc[old_color] -= 1
            cc[new_color] += 1
            new_count = cc[new_color]
            
            new_violation = 1 if new_count == 6 else 0
            
            delta += new_violation - old_violation
        
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            colors[e] = new_color
            total_violations += delta
        else:
            # Revert
            for c_id in affected_cycles:
                cc = cycle_counts[c_id]
                cc[old_color] += 1
                cc[new_color] -= 1
        
        T *= cooling
        if T < min_T:
            T = min_T
    
    if total_violations == 0:
        return {'n': n, 'colors': colors}
    
    return None
# EVOLVE-BLOCK-END
