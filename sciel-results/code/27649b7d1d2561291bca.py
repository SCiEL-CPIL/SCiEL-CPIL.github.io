import time
import random
import numpy as np
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

def _edge_index(i, j):
    """Map pair (i,j) with i<j to edge index 0..119 in lexicographic order."""
    # Edges before vertex i: sum_{k=0}^{i-1} (N-1-k) = i*(N-1) - i*(i-1)//2
    # Position within vertex i: j - i - 1
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Build all 4-cycles: for each 4-vertex subset, there are 3 distinct 4-cycles
# A 4-cycle in K4 (vertices a<b<c<d) omits one of the 3 perfect matchings.
# The 3 cycles are:
#   1. Omit {ab,cd}: uses ac, bd, ad, bc -> a-c-b-d-a
#   2. Omit {ac,bd}: uses ab, cd, ad, bc -> a-b-c-d-a
#   3. Omit {ad,bc}: uses ab, cd, ac, bd -> a-b-d-c-a

def _build_cycles():
    """Return list of cycles, each cycle is a tuple of 4 edge indices."""
    cycles = []
    for a in range(N - 3):
        for b in range(a + 1, N - 2):
            for c in range(b + 1, N - 1):
                for d in range(c + 1, N):
                    # Cycle 1: a-b-c-d-a (edges: ab, bc, cd, da)
                    cycles.append((_edge_index(a, b), _edge_index(b, c), _edge_index(c, d), _edge_index(a, d)))
                    # Cycle 2: a-b-d-c-a (edges: ab, bd, dc, ca)
                    cycles.append((_edge_index(a, b), _edge_index(b, d), _edge_index(c, d), _edge_index(a, c)))
                    # Cycle 3: a-c-b-d-a (edges: ac, cb, bd, da)
                    cycles.append((_edge_index(a, c), _edge_index(b, c), _edge_index(b, d), _edge_index(a, d)))
    return cycles

CYCLES = _build_cycles()
NUM_CYCLES = len(CYCLES)

# Build for each edge, the list of cycles it belongs to
def _build_edge_to_cycles():
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(CYCLES):
        for e in cyc:
            edge_to_cycles[e].append(ci)
    return edge_to_cycles

EDGE_TO_CYCLES = _build_edge_to_cycles()

def _count_violations(colors):
    """Count number of 4-cycles with 4 distinct edge colors."""
    violations = 0
    for cyc in CYCLES:
        c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
        if len({c0, c1, c2, c3}) == 4:
            violations += 1
    return violations

def _count_distinct_colors(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # State
    colors = np.zeros(NUM_EDGES, dtype=np.int32)
    
    # Track distinct colors
    color_counts = np.zeros(MAX_COLOR + 1, dtype=np.int32)
    distinct_colors = 1  # all zeros
    
    # Track violations per cycle
    cycle_is_violation = np.zeros(NUM_CYCLES, dtype=bool)
    total_violations = 0
    
    # Best tracking
    best_colors = colors.copy()
    best_distinct = 1
    
    # Helper to check if a cycle has 4 distinct colors
    def _is_violation(cyc, colors_arr):
        c0, c1, c2, c3 = colors_arr[cyc[0]], colors_arr[cyc[1]], colors_arr[cyc[2]], colors_arr[cyc[3]]
        return len({c0, c1, c2, c3}) == 4
    
    # Helper to update violations for an edge change
    def _update_violations_for_edge(e, old_color, new_color, colors_arr):
        nonlocal total_violations
        for ci in EDGE_TO_CYCLES[e]:
            cyc = CYCLES[ci]
            e0, e1, e2, e3 = cyc
            # Get current colors (before change)
            oc0, oc1, oc2, oc3 = colors_arr[e0], colors_arr[e1], colors_arr[e2], colors_arr[e3]
            
            # Determine new colors
            nc0, nc1, nc2, nc3 = oc0, oc1, oc2, oc3
            if e0 == e:
                nc0 = new_color
            elif e1 == e:
                nc1 = new_color
            elif e2 == e:
                nc2 = new_color
            elif e3 == e:
                nc3 = new_color
            
            old_viol = len({oc0, oc1, oc2, oc3}) == 4
            new_viol = len({nc0, nc1, nc2, nc3}) == 4
            
            if old_viol and not new_viol:
                total_violations -= 1
                cycle_is_violation[ci] = False
            elif not old_viol and new_viol:
                total_violations += 1
                cycle_is_violation[ci] = True
    
    # Helper to try a move: change edge e from old_color to new_color
    # Returns True if accepted (no new violations), False otherwise
    def _try_change(e, new_color, colors_arr):
        old_color = colors_arr[e]
        if old_color == new_color:
            return False
        
        # Check if this change would create any new violations
        for ci in EDGE_TO_CYCLES[e]:
            cyc = CYCLES[ci]
            e0, e1, e2, e3 = cyc
            oc0, oc1, oc2, oc3 = colors_arr[e0], colors_arr[e1], colors_arr[e2], colors_arr[e3]
            
            nc0, nc1, nc2, nc3 = oc0, oc1, oc2, oc3
            if e0 == e:
                nc0 = new_color
            elif e1 == e:
                nc1 = new_color
            elif e2 == e:
                nc2 = new_color
            elif e3 == e:
                nc3 = new_color
            
            if not (len({oc0, oc1, oc2, oc3}) == 4) and (len({nc0, nc1, nc2, nc3}) == 4):
                return False  # Would create a violation
        
        # Apply the change
        colors_arr[e] = new_color
        _update_violations_for_edge(e, old_color, new_color, colors_arr)
        return True
    
    # Phase 1: Greedy diversity increase
    # Start from all zeros, try to introduce new colors one by one
    # For each new color, try to assign it to edges that don't create violations
    
    # Precompute unused colors
    used_colors = set([0])
    
    # Try to add colors 1, 2, 3, ... in order
    for new_c in range(1, MAX_COLOR + 1):
        if time.time() > deadline:
            break
            
        # Try to find an edge to assign this new color
        # Prefer edges with low degree (fewer cycles) to minimize impact
        # Actually, prefer edges where the new color might be "safe"
        
        # Try all edges, pick one that works
        assigned = False
        # Randomize edge order for variety
        edge_order = list(range(NUM_EDGES))
        random.shuffle(edge_order)
        
        for e in edge_order:
            if _try_change(e, new_c, colors):
                # Update color counts
                old_c = 0  # This is tricky, we need to know old color
                # Actually, _try_change doesn't return old color. Let's fix.
                # For now, just update counts based on current state
                # This is inefficient, let's do it properly in a better way
                
                # Since we start from all zeros and only add new colors,
                # the old color is always 0 for the first pass.
                # But later, we might change colors.
                
                # Let's just recount distinct colors for simplicity in Phase 1
                distinct_colors = len(set(colors))
                used_colors.add(new_c)
                assigned = True
                break
        
        if not assigned:
            # Can't add this color, skip
            continue
            
    # Update best
    if total_violations == 0:
        best_distinct = len(set(colors))
        best_colors = colors.copy()
    
    # Phase 2: Local search to improve diversity
    # Use simulated annealing or hill climbing
    # Objective: maximize distinct colors while keeping violations = 0
    
    T = 1.0
    cooling = 0.9999
    min_T = 0.01
    
    current_distinct = len(set(colors))
    
    # Recount color counts
    color_counts = np.zeros(MAX_COLOR + 1, dtype=np.int32)
    for c in colors:
        color_counts[c] += 1
    
    while time.time() < deadline:
        if T < min_T:
            break
            
        # Pick a random edge
        e_idx = np.random.randint(0, NUM_EDGES)
        old_color = colors[e_idx]
        
        # Pick a new color
        # To encourage diversity, pick a color not currently used, or a random one
        if np.random.random() < 0.5:
            # Pick an unused color
            unused = [c for c in range(MAX_COLOR + 1) if color_counts[c] == 0]
            if unused:
                new_color = random.choice(unused)
            else:
                new_color = np.random.randint(0, MAX_COLOR + 1)
        else:
            new_color = np.random.randint(0, MAX_COLOR + 1)
            
        if new_color == old_color:
            continue
            
        # Try the change
        old_distinct = current_distinct
        old_viol = total_violations
        
        # Save state
        saved_colors = colors.copy()
        
        if _try_change(e_idx, new_color, colors):
            # Update color counts
            color_counts[old_color] -= 1
            color_counts[new_color] += 1
            
            new_distinct = current_distinct
            if color_counts[old_color] == 0:
                new_distinct -= 1
            if color_counts[new_color] == 1:
                new_distinct += 1
                
            new_viol = total_violations
            
            # Accept if no violations and distinct increased, or with probability if distinct decreased but no violations
            if new_viol == 0:
                if new_distinct > old_distinct:
                    # Always accept improvement
                    current_distinct = new_distinct
                    if new_distinct > best_distinct:
                        best_distinct = new_distinct
                        best_colors = colors.copy()
                else:
                    # Accept with probability exp((new_distinct - old_distinct) / T)
                    delta = new_distinct - old_distinct
                    if delta >= 0 or np.random.random() < np.exp(delta / T):
                        current_distinct = new_distinct
                    else:
                        # Revert
                        colors[:] = saved_colors
                        color_counts[old_color] += 1
                        color_counts[new_color] -= 1
                        total_violations = old_viol
                        # Recount cycle violations
                        for ci in range(NUM_CYCLES):
                            cycle_is_violation[ci] = _is_violation(CYCLES[ci], colors)
                        total_violations = np.sum(cycle_is_violation)
            else:
                # Revert
                colors[:] = saved_colors
                color_counts[old_color] += 1
                color_counts[new_color] -= 1
                total_violations = old_viol
                for ci in range(NUM_CYCLES):
                    cycle_is_violation[ci] = _is_violation(CYCLES[ci], colors)
                total_violations = np.sum(cycle_is_violation)
        else:
            # Change was rejected due to violation, no state change needed
            pass
            
        T *= cooling
        
    # Final check: ensure best_colors is valid
    if _count_violations(best_colors) != 0:
        # Fallback to all zeros
        return [0] * NUM_EDGES
        
    return best_colors.tolist()
