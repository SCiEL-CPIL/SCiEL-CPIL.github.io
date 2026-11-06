import time
import random
import sys

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLORS = 120

# EVOLVE-BLOCK-START

def _edge_index(i, j):
    """Map (i,j) with i<j to a 0-based index in the lexicographic edge list."""
    if i > j:
        i, j = j, i
    # Number of edges before vertex i: sum_{k=0}^{i-1} (N-1-k) = i*(N-1) - i*(i-1)//2
    # Then offset by (j - i - 1)
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def _edge_from_index(idx):
    """Map 0-based index back to (i, j) with i < j."""
    # Find i such that i*(N-1) - i*(i-1)//2 <= idx < (i+1)*(N-1) - (i+1)*i//2
    i = 0
    while True:
        next_start = (i + 1) * (N - 1) - (i + 1) * i // 2
        if idx < next_start:
            break
        i += 1
    start = i * (N - 1) - i * (i - 1) // 2
    j = i + 1 + (idx - start)
    return (i, j)

def _build_four_cycles():
    """
    Precompute all 4-cycles (as tuples of 4 edge indices).
    For each 4-vertex subset {a,b,c,d} with a<b<c<d, there are 3 cycles:
      C1: a-b-c-d-a  -> edges (a,b),(b,c),(c,d),(d,a)
      C2: a-b-d-c-a  -> edges (a,b),(b,d),(d,c),(c,a)
      C3: a-c-b-d-a  -> edges (a,c),(c,b),(b,d),(d,a)
    """
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Cycle 1: a-b-c-d-a
                    e1 = _edge_index(a, b)
                    e2 = _edge_index(b, c)
                    e3 = _edge_index(c, d)
                    e4 = _edge_index(a, d)
                    cycles.append((e1, e2, e3, e4))
                    # Cycle 2: a-b-d-c-a
                    e1 = _edge_index(a, b)
                    e2 = _edge_index(b, d)
                    e3 = _edge_index(c, d)
                    e4 = _edge_index(a, c)
                    cycles.append((e1, e2, e3, e4))
                    # Cycle 3: a-c-b-d-a
                    e1 = _edge_index(a, c)
                    e2 = _edge_index(b, c)
                    e3 = _edge_index(b, d)
                    e4 = _edge_index(a, d)
                    cycles.append((e1, e2, e3, e4))
    return cycles

def _build_edge_cycle_map(cycles):
    """Map each edge index to the list of cycle indices containing it."""
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cycle in enumerate(cycles):
        for e in cycle:
            edge_to_cycles[e].append(ci)
    return edge_to_cycles

# Precompute at module load time (fast, ~3264 cycles for N=16)
_FOUR_CYCLES = _build_four_cycles()
_EDGE_TO_CYCLES = _build_edge_cycle_map(_FOUR_CYCLES)
_NUM_CYCLES = len(_FOUR_CYCLES)

def _is_valid_coloring(colors):
    """Check if a full coloring has no rainbow 4-cycle."""
    for cycle in _FOUR_CYCLES:
        c0, c1, c2, c3 = (colors[e] for e in cycle)
        if len({c0, c1, c2, c3}) == 4:
            return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _local_search_add_colors(colors, time_limit):
    """
    Given a valid coloring, try to increase the number of distinct colors
    by reassigning individual edges to new (unused) colors, checking validity.
    Uses a greedy + random-restart approach within the time limit.
    """
    colors = list(colors)
    used = set(colors)
    deadline = time.time() + time_limit
    best_count = len(used)
    best_colors = list(colors)
    
    while time.time() < deadline - 0.5:
        # Try to add a new color to a random edge
        unused = [c for c in range(MAX_COLORS) if c not in used]
        if not unused:
            break
        
        progress = False
        for _ in range(200):  # limited attempts per outer iteration
            if time.time() > deadline - 0.5:
                break
            ei = random.randint(0, NUM_EDGES - 1)
            old_c = colors[ei]
            new_c = random.choice(unused)
            
            # Check if changing edge ei from old_c to new_c preserves validity
            valid = True
            for ci in _EDGE_TO_CYCLES[ei]:
                cycle = _FOUR_CYCLES[ci]
                col_set = set()
                for e in cycle:
                    if e == ei:
                        col_set.add(new_c)
                    else:
                        col_set.add(colors[e])
                if len(col_set) == 4:
                    valid = False
                    break
            
            if valid:
                colors[ei] = new_c
                used.add(new_c)
                unused.remove(new_c)
                progress = True
                if len(used) > best_count:
                    best_count = len(used)
                    best_colors = list(colors)
        
        if not progress:
            # Try some random reassignments among existing colors to unlock new possibilities
            for _ in range(100):
                if time.time() > deadline - 0.5:
                    break
                ei = random.randint(0, NUM_EDGES - 1)
                old_c = colors[ei]
                # Pick a random existing color (not the current one)
                existing_list = list(used)
                if len(existing_list) <= 1:
                    continue
                new_c = random.choice([c for c in existing_list if c != old_c])
                
                valid = True
                for ci in _EDGE_TO_CYCLES[ei]:
                    cycle = _FOUR_CYCLES[ci]
                    col_set = set()
                    for e in cycle:
                        if e == ei:
                            col_set.add(new_c)
                        else:
                            col_set.add(colors[e])
                    if len(col_set) == 4:
                        valid = False
                        break
                
                if valid:
                    colors[ei] = new_c
    
    if best_count >= _count_colors(colors):
        return best_colors
    return colors

def _construct_via_greedy_with_backtracking(deadline):
    """
    Structural approach: build a valid coloring using a recursive backtracking
    with MRV (Minimum Remaining Values) heuristic, preferring new colors to
    maximize diversity. This is a different search paradigm from pure local search.
    
    Key idea: assign edges in an order that maximizes constraint propagation.
    Use a dynamic edge ordering: at each step, pick the unassigned edge that
    participates in the most cycles where the other edges are already assigned
    (i.e., most constrained). Try new colors first to maximize diversity.
    """
    colors = [-1] * NUM_EDGES
    used = set()
    
    # Track for each cycle: how many edges assigned, and the set of colors used
    cycle_assigned_count = [0] * _NUM_CYCLES
    cycle_color_sets = [set() for _ in range(_NUM_CYCLES)]
    
    # For efficiency, also track per-cycle the list of assigned colors (with multiplicity not needed, just set)
    
    # Dynamic edge ordering: precompute for each edge its cycle count
    edge_cycle_count = [len(_EDGE_TO_CYCLES[e]) for e in range(NUM_EDGES)]
    
    # We'll use a priority queue concept: at each step, pick the unassigned edge
    # with the highest "constraint pressure" = number of cycles where 3+ other edges are assigned
    
    import heapq
    
    def constraint_pressure(ei):
        """Number of cycles containing ei where at least 3 of the other 3 edges are assigned."""
        pressure = 0
        for ci in _EDGE_TO_CYCLES[ei]:
            cycle = _FOUR_CYCLES[ci]
            assigned_others = sum(1 for e in cycle if e != ei and colors[e] != -1)
            if assigned_others >= 2:  # at least 2 others assigned means 3 total including ei
                pressure += 1
        return pressure
    
    # Build initial priority queue
    # We'll recompute dynamically; for simplicity, use a list and pick max each time
    # For 120 edges this is fine
    
    unassigned = list(range(NUM_EDGES))
    
    def backtrack(depth):
        if time.time() > deadline - 2:
            return False
        if not unassigned:
            return True
        
        # MRV: pick unassigned edge with highest constraint pressure
        best_ei = unassigned[0]
        best_pressure = -1
        for ei in unassigned:
            p = constraint_pressure(ei)
            if p > best_pressure:
                best_pressure = p
                best_ei = ei
        
        unassigned.remove(best_ei)
        
        # Try colors: new colors first (to maximize diversity), then existing
        new_colors = [c for c in range(MAX_COLORS) if c not in used]
        random.shuffle(new_colors)
        existing_colors = list(used)
        random.shuffle(existing_colors)
        
        domain = new_colors + existing_colors
        
        for c in domain:
            if time.time() > deadline - 2:
                unassigned.append(best_ei)
                return False
            
            # Check validity: for each cycle containing best_ei, if all 4 edges would be assigned,
            # ensure not all 4 colors are distinct
            valid = True
            affected = []
            for ci in _EDGE_TO_CYCLES[best_ei]:
                cycle = _FOUR_CYCLES[ci]
                # Check if all other 3 edges are assigned
                all_others_assigned = all(colors[e] != -1 for e in cycle if e != best_ei)
                if all_others_assigned:
                    col_set = {c}
                    for e in cycle:
                        if e != best_ei:
                            col_set.add(colors[e])
                    if len(col_set) == 4:
                        valid = False
                        break
                else:
                    # Partial: check if we already have 3 distinct colors among assigned + this
                    # and the 4th is unassigned -> the 4th must match one of the 3
                    # For now, just check if adding c creates 4 distinct among assigned subset
                    # Actually we can only fully check when all 4 are assigned
                    # But we can check: if 3 others assigned with 3 distinct colors, c must be one of them
                    assigned_others = [e for e in cycle if e != best_ei and colors[e] != -1]
                    if len(assigned_others) == 3:
                        other_colors = {colors[e] for e in assigned_others}
                        if len(other_colors) == 3 and c not in other_colors:
                            valid = False
                            break
            
            if valid:
                # Assign
                colors[best_ei] = c
                used.add(c)
                for ci in _EDGE_TO_CYCLES[best_ei]:
                    cycle_assigned_count[ci] += 1
                    cycle_color_sets[ci].add(c)
                
                if backtrack(depth + 1):
                    return True
                
                # Unassign
                colors[best_ei] = -1
                # Check if c is still used elsewhere
                still_used = any(colors[e] == c for e in range(NUM_EDGES) if e != best_ei)
                if not still_used:
                    used.discard(c)
                for ci in _EDGE_TO_CYCLES[best_ei]:
                    cycle_assigned_count[ci] -= 1
                    # Rebuild color set (expensive but rare due to backtracking)
                    cycle = _FOUR_CYCLES[ci]
                    cycle_color_sets[ci] = {colors[e] for e in cycle if colors[e] != -1}
                
                unassigned.append(best_ei)
        
        unassigned.append(best_ei)
        return False
    
    # Limit backtracking: if we go too deep without progress, bail out
    # For now, just try it with the time limit
    success = backtrack(0)
    
    if success:
        return colors
    
    # If backtracking failed (likely due to time), return a partial valid coloring
    # Fill unassigned edges with a safe color
    for ei in unassigned:
        # Find a color that doesn't create a rainbow
        # Use color 0 as fallback, then verify and repair
        colors[ei] = 0
    
    # Verify and repair
    if not _is_valid_coloring(colors):
        # Fallback: use a known valid coloring
        return _fallback_coloring()
    
    return colors

def _fallback_coloring():
    """A simple valid coloring: color edge (i,j) by i % 7. This uses 7 colors and is valid."""
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            colors[_edge_index(i, j)] = i % 7
    return colors

def _structural_block_coloring():
    """
    Alternative structural approach: partition the 16 vertices into blocks
    and use a hierarchical coloring scheme.
    
    Idea: Split vertices into 2 groups of 8. 
    - Edges within group 1: use a set of colors A
    - Edges within group 2: use a set of colors B  
    - Edges between groups: use a set of colors C
    
    To avoid rainbow 4-cycles: a 4-cycle that uses edges from all three types
    would need 4 distinct colors. We can structure this so that cross-group
    edges share colors in a way that prevents rainbows.
    
    More specifically, use a "bipartite-like" structure where cross edges
    between groups use a small palette, and intra-group edges use larger palettes.
    
    Let's try: 
    - Group 1: vertices 0-7, Group 2: vertices 8-15
    - Intra-group edges: assign colors from a pool, ensuring no rainbow within group
    - Cross-group edges: assign colors such that for any two vertices in group 1 and two in group 2,
      the 4-cycle formed doesn't have 4 distinct colors.
    
    For cross-group edges, if we color edge (u,v) with u in G1, v in G2 as f(u,v) = u mod k + v mod k,
    we can control the palette.
    
    Actually, let's use a simpler proven structure:
    - Color edge (i,j) as (i mod 4, j mod 4) mapped to a single color index.
    - This gives at most 16*16=256 combos but with i<j constraint.
    - Need to verify no rainbow 4-cycles.
    """
    # Let's use: color(i,j) = (i % 4) * 4 + (j % 4) for i<j
    # This uses at most 16 colors. Let's check validity.
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            colors[_edge_index(i, j)] = (i % 4) * 4 + (j % 4)
    
    if _is_valid_coloring(colors):
        return colors
    else:
        return _fallback_coloring()

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s kill
    
    random.seed(42)
    
    # Strategy 1: Try the structural block coloring first (fast, deterministic)
    result = _structural_block_coloring()
    count1 = _count_colors(result)
    
    # Strategy 2: Try greedy backtracking with time limit
    remaining = deadline - time.time()
    if remaining > 10:
        bt_result = _construct_via_greedy_with_backtracking(deadline)
        count2 = _count_colors(bt_result)
        if count2 > count1:
            result = bt_result
            count1 = count2
    
    # Strategy 3: Local search to add more colors
    remaining = deadline - time.time()
    if remaining > 5:
        result = _local_search_add_colors(result, remaining - 3)
    
    # Final verification
    if not _is_valid_coloring(result):
        result = _fallback_coloring()
    
    # Ensure all colors are in range [0, 119]
    result = [min(c, MAX_COLORS - 1) for c in result]
    
    return result

# EVOLVE-BLOCK-END
