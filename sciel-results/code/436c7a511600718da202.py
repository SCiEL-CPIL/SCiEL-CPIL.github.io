import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

# Precompute edge index mapping
def _edge_index(i, j):
    """Map pair (i,j) with i<j to edge index 0..119 in lexicographic order."""
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Build all 4-cycles
def _build_cycles():
    """Return list of cycles, each cycle is a tuple of 4 edge indices."""
    cycles = []
    for a in range(N - 3):
        for b in range(a + 1, N - 2):
            for c in range(b + 1, N - 1):
                for d in range(c + 1, N):
                    # Three 4-cycles on {a,b,c,d}
                    # Cycle 1: a-b-c-d-a (edges: ab, bc, cd, da)
                    cycles.append((_edge_index(a,b), _edge_index(b,c), _edge_index(c,d), _edge_index(a,d)))
                    # Cycle 2: a-b-d-c-a (edges: ab, bd, dc, ca)
                    cycles.append((_edge_index(a,b), _edge_index(b,d), _edge_index(c,d), _edge_index(a,c)))
                    # Cycle 3: a-c-b-d-a (edges: ac, cb, bd, da)
                    cycles.append((_edge_index(a,c), _edge_index(b,c), _edge_index(b,d), _edge_index(a,d)))
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

def _valid(colors):
    return _count_violations(colors) == 0

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_colors = None
    best_distinct = 0
    
    random.seed(42)
    
    # Strategy: Start with a structured baseline using rank-1 algebraic coloring,
    # then use targeted local search to increase diversity while maintaining validity
    
    # Initial: Use a structured approach with rank-1 algebraic coloring
    # Color edge (i,j) by (i * j) mod K for some K
    # This tends to create structure that helps with 4-cycle constraints
    
    # Let's try a combination: use modular arithmetic to create initial structure
    # Then use local search to expand the color palette
    
    colors = np.zeros(NUM_EDGES, dtype=np.int32)
    
    # Precompute degree of each edge
    edge_deg = [len(EDGE_TO_CYCLES[e]) for e in range(NUM_EDGES)]
    
    # Track violations per cycle
    cycle_is_violation = np.zeros(NUM_CYCLES, dtype=bool)
    total_violations = 0
    
    # Initialize with a structured coloring: use (i+j) mod 8 to create some structure
    # This gives 8 colors initially, which is valid (no 4 distinct in any cycle)
    for e in range(NUM_EDGES):
        # Find which edge this is
        # We need to reverse the edge index to get (i,j)
        # Let's precompute this
        pass
    
    # Precompute edge to vertex mapping
    edge_to_vertices = []
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_to_vertices.append((i, j))
            idx += 1
    
    # Initialize with structured coloring
    for e in range(NUM_EDGES):
        i, j = edge_to_vertices[e]
        # Use (i * 4 + j) % 16 as initial color - creates structure
        colors[e] = (i * 4 + j) % 16
    
    # Count initial violations
    for ci, cyc in enumerate(CYCLES):
        c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
        if len({c0, c1, c2, c3}) == 4:
            cycle_is_violation[ci] = True
            total_violations += 1
    
    # If we have violations, repair them first
    if total_violations > 0:
        # Simple repair: recolor edges in violating cycles to match another edge
        for iteration in range(10000):
            if time.time() > deadline - 10.0:
                break
            if total_violations == 0:
                break
            
            # Find a violating cycle
            viol_cycles = [ci for ci in range(NUM_CYCLES) if cycle_is_violation[ci]]
            if not viol_cycles:
                break
            ci = random.choice(viol_cycles)
            cyc = CYCLES[ci]
            # Pick an edge to recolor
            e = random.choice(cyc)
            old_color = colors[e]
            # Try to recolor to match one of the other edges
            other_edges = [cyc[i] for i in range(4) if cyc[i] != e]
            for oe in other_edges:
                new_color = colors[oe]
                if new_color == old_color:
                    continue
                # Check if this change is valid
                valid = True
                for ci2 in EDGE_TO_CYCLES[e]:
                    cyc2 = CYCLES[ci2]
                    c0 = colors[cyc2[0]] if cyc2[0] != e else new_color
                    c1 = colors[cyc2[1]] if cyc2[1] != e else new_color
                    c2 = colors[cyc2[2]] if cyc2[2] != e else new_color
                    c3 = colors[cyc2[3]] if cyc2[3] != e else new_color
                    if len({c0, c1, c2, c3}) == 4:
                        valid = False
                        break
                if valid:
                    # Apply change
                    colors[e] = new_color
                    # Update violation counts
                    for ci2 in EDGE_TO_CYCLES[e]:
                        cyc2 = CYCLES[ci2]
                        c0, c1, c2, c3 = colors[cyc2[0]], colors[cyc2[1]], colors[cyc2[2]], colors[cyc2[3]]
                        new_viol = len({c0, c1, c2, c3}) == 4
                        if cycle_is_violation[ci2] and not new_viol:
                            total_violations -= 1
                            cycle_is_violation[ci2] = False
                        elif not cycle_is_violation[ci2] and new_viol:
                            total_violations += 1
                            cycle_is_violation[ci2] = True
                    break
    
    # Now use local search to increase diversity
    # Track used colors
    used_colors = set(colors.tolist())
    
    def _try_change(e, new_color, colors_arr):
        """Try to change edge e to new_color. Returns True if valid."""
        old_color = colors_arr[e]
        if old_color == new_color:
            return False
        
        # Check if this change would create any new violations
        would_create = 0
        for ci in EDGE_TO_CYCLES[e]:
            cyc = CYCLES[ci]
            e0, e1, e2, e3 = cyc
            # Compute new colors for this cycle
            nc = []
            for idx in range(4):
                if cyc[idx] == e:
                    nc.append(new_color)
                else:
                    nc.append(colors_arr[cyc[idx]])
            
            # Old colors
            oc = []
            for idx in range(4):
                if cyc[idx] == e:
                    oc.append(old_color)
                else:
                    oc.append(colors_arr[cyc[idx]])
            
            old_viol = len(set(oc)) == 4
            new_viol = len(set(nc)) == 4
            
            if not old_viol and new_viol:
                would_create += 1
                break
        
        if would_create == 0:
            # Apply the change
            colors_arr[e] = new_color
            # Update violation counts
            for ci in EDGE_TO_CYCLES[e]:
                cyc = CYCLES[ci]
                c0, c1, c2, c3 = colors_arr[cyc[0]], colors_arr[cyc[1]], colors_arr[cyc[2]], colors_arr[cyc[3]]
                new_viol = len({c0, c1, c2, c3}) == 4
                if cycle_is_violation[ci] and not new_viol:
                    total_violations -= 1
                    cycle_is_violation[ci] = False
                elif not cycle_is_violation[ci] and new_viol:
                    total_violations += 1
                    cycle_is_violation[ci] = True
            return True
        return False
    
    # Local search phase
    for iteration in range(50000):
        if time.time() > deadline - 5.0:
            break
        
        if total_violations > 0:
            # Repair: find a violating cycle and change one of its edges
            viol_cycles = [ci for ci in range(NUM_CYCLES) if cycle_is_violation[ci]]
            if not viol_cycles:
                total_violations = 0
                continue
            ci = random.choice(viol_cycles)
            cyc = CYCLES[ci]
            e = random.choice(cyc)
            old_color = colors[e]
            # Try to change to a color that reduces violations
            other_colors = [colors[cyc[i]] for i in range(4) if cyc[i] != e]
            for new_color in other_colors:
                if new_color != old_color:
                    if _try_change(e, new_color, colors):
                        break
            else:
                new_color = random.randint(0, MAX_COLOR)
                _try_change(e, new_color, colors)
            continue
        
        # Valid: try to increase diversity
        e = random.randint(0, NUM_EDGES - 1)
        old_color = colors[e]
        
        # Try to introduce a new color
        if len(used_colors) < MAX_COLOR + 1:
            unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
            if unused:
                for _ in range(5):
                    new_color = random.choice(unused)
                    if _try_change(e, new_color, colors):
                        used_colors.add(new_color)
                        break
                else:
                    other_used = [c for c in used_colors if c != old_color]
                    if other_used:
                        new_color = random.choice(other_used)
                        if _try_change(e, new_color, colors):
                            pass
        else:
            new_color = random.randint(0, MAX_COLOR)
            _try_change(e, new_color, colors)
        
        # Update used_colors
        used_colors = set(colors.tolist())
        
        # Track best
        current_distinct = len(used_colors)
        if current_distinct > best_distinct:
            best_distinct = current_distinct
            best_colors = colors.copy()
    
    # Final: if we have a valid coloring, return it; otherwise return best valid found
    if best_colors is not None and total_violations == 0:
        return best_colors.tolist()
    elif best_colors is not None:
        if _count_violations(best_colors.tolist()) == 0:
            return best_colors.tolist()
    
    # Fallback: all zeros (always valid)
    return [0] * NUM_EDGES
