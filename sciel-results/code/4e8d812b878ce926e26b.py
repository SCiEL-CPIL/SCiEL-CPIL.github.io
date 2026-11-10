import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

# Precompute edge index mapping
def _edge_index(i, j):
    """Map pair (i,j) with i<j to edge index 0..119 in lexicographic order."""
    # Lexicographic order: (0,1),(0,2),...,(0,15),(1,2),(1,3),...,(14,15)
    # For vertex i, the edges are (i, i+1), (i, i+2), ..., (i, N-1)
    # Number of edges before vertex i: i*(N-1) - i*(i-1)//2
    # Within vertex i, edge (i,j) is at position j - i - 1
    # Total: i*(N-1) - i*(i-1)//2 + (j - i - 1)
    # Simplify: i*N - i - i*(i-1)//2 + j - i - 1 = i*N - 2*i - i*(i-1)//2 + j - 1
    # Let's just compute directly:
    # Edges starting at vertex k: N-1-k edges
    # Edges before vertex i: sum_{k=0}^{i-1} (N-1-k) = i*(N-1) - i*(i-1)//2
    # Position within vertex i: j - i - 1
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Build all 4-cycles: for each 4-vertex subset, there are 3 distinct 4-cycles
# A 4-cycle on vertices {a,b,c,d} (a<b<c<d) can be:
#   a-b-c-d-a (edges: ab, bc, cd, da)
#   a-b-d-c-a (edges: ab, bd, dc, ca)  
#   a-c-b-d-a (edges: ac, cb, bd, da)
# Wait, let me think more carefully. A 4-cycle is a cycle of length 4 in K4.
# In K4 with vertices a<b<c<d, the three 4-cycles (up to reversal) are:
#   a-b-c-d-a: edges ab, bc, cd, da
#   a-b-d-c-a: edges ab, bd, dc, ca
#   a-c-b-d-a: edges ac, cb, bd, da
# But actually a 4-cycle is determined by which two pairs of opposite edges we use.
# In K4, the 3 perfect matchings are: {ab,cd}, {ac,bd}, {ad,bc}
# A 4-cycle uses two of the three perfect matchings' complement... 
# Actually, a 4-cycle in K4 uses 4 edges. K4 has 6 edges. A 4-cycle omits 2 edges.
# The 2 omitted edges must form a perfect matching (opposite edges in the cycle).
# So the 3 cycles correspond to omitting one of the 3 perfect matchings:
#   Omit {ab,cd}: cycle uses ac, bd, ad, bc -> a-c-b-d-a
#   Omit {ac,bd}: cycle uses ab, cd, ad, bc -> a-b-c-d-a
#   Omit {ad,bc}: cycle uses ab, cd, ac, bd -> a-b-d-c-a

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
    
    # Precompute for fast violation counting: for each cycle, track the set of colors
    # Use numpy for speed where possible
    
    random.seed(42)
    
    # Strategy: Start with a valid low-diversity coloring, then use targeted moves
    # to increase diversity while maintaining validity
    
    # Initial: all zeros (valid, 1 color)
    colors = np.zeros(NUM_EDGES, dtype=np.int32)
    
    # Build a structure to quickly check if changing an edge color creates a violation
    # For each cycle, we need to know the multiset of colors on its 4 edges
    
    # Precompute: for each edge, which cycles contain it (already done in EDGE_TO_CYCLES)
    
    # Helper: given current colors and a cycle, return the count of each color in that cycle
    # For speed, we'll maintain a data structure
    
    # Let's use a different approach: maintain for each cycle the list of colors
    # and track violations incrementally
    
    # For each cycle, store the 4 edge indices and the current color counts
    # cycle_color_counts[ci] = dict or array counting colors in cycle ci
    
    # Actually, let's use a simpler but effective approach:
    # 1. Maintain current colors
    # 2. When we change edge e from old_color to new_color:
    #    - For each cycle containing e, check if it was a violation before and after
    #    - This is O(deg(e)) per move, where deg(e) is the number of cycles containing e
    
    # Precompute degree of each edge
    edge_deg = [len(EDGE_TO_CYCLES[e]) for e in range(NUM_EDGES)]
    
    # Track violations per cycle
    cycle_is_violation = np.zeros(NUM_CYCLES, dtype=bool)
    total_violations = 0
    
    # Initialize: all zeros, no violations
    # (all cycles have 4 edges of color 0, so 1 distinct color, not a violation)
    
    def _update_violations_for_edge(e, old_color, new_color, colors_arr):
        """Update violation status for cycles containing edge e."""
        nonlocal total_violations
        for ci in EDGE_TO_CYCLES[e]:
            cyc = CYCLES[ci]
            # Get the 4 edge indices
            e0, e1, e2, e3 = cyc
            # Compute old and new color sets for this cycle
            # Old colors:
            oc = [colors_arr[e0], colors_arr[e1], colors_arr[e2], colors_arr[e3]]
            # But edge e has old_color, and new_color is not yet applied
            # So oc already has old_color at position of e
            # New colors: replace old_color with new_color at position of e
            nc = oc[:]
            # Find position of e in cycle
            for idx in range(4):
                if cyc[idx] == e:
                    nc[idx] = new_color
                    break
            
            old_viol = len(set(oc)) == 4
            new_viol = len(set(nc)) == 4
            
            if old_viol and not new_viol:
                total_violations -= 1
                cycle_is_violation[ci] = False
            elif not old_viol and new_viol:
                total_violations += 1
                cycle_is_violation[ci] = True
    
    def _try_change(e, new_color, colors_arr):
        """Try to change edge e to new_color. Returns True if valid (no new violations)."""
        old_color = colors_arr[e]
        if old_color == new_color:
            return False
        
        # Check if this change would create any new violations
        would_create = 0
        for ci in EDGE_TO_CYCLES[e]:
            cyc = CYCLES[ci]
            e0, e1, e2, e3 = cyc
            oc = [colors_arr[e0], colors_arr[e1], colors_arr[e2], colors_arr[e3]]
            nc = oc[:]
            for idx in range(4):
                if cyc[idx] == e:
                    nc[idx] = new_color
                    break
            if not (len(set(oc)) == 4) and (len(set(nc)) == 4):
                would_create += 1
                break
        
        if would_create == 0:
            # Apply the change
            colors_arr[e] = new_color
            _update_violations_for_edge(e, old_color, new_color, colors_arr)
            return True
        return False
    
    # Phase 1: Greedy diversity increase
    # Repeatedly try to increase the number of distinct colors
    # Strategy: pick a random edge, try to assign it a new color (one not yet used, or 
    # one that doesn't create violations)
    
    used_colors = set(colors.tolist())
    
    for iteration in range(100000):
        if time.time() > deadline - 5.0:
            break
        
        if total_violations > 0:
            # Need to repair: find a violating cycle and change one of its edges
            # Pick a random violating cycle
            viol_cycles = [ci for ci in range(NUM_CYCLES) if cycle_is_violation[ci]]
            if not viol_cycles:
                total_violations = 0
                continue
            ci = random.choice(viol_cycles)
            cyc = CYCLES[ci]
            # Pick a random edge in this cycle to change
            e = random.choice(cyc)
            old_color = colors[e]
            # Try to change to a color that reduces violations
            # Try colors from the other 3 edges in the cycle
            other_colors = [colors[cyc[i]] for i in range(4) if cyc[i] != e]
            # Try one of the other colors to reduce distinct count
            for new_color in other_colors:
                if new_color != old_color:
                    if _try_change(e, new_color, colors):
                        break
            else:
                # Try a random color
                new_color = random.randint(0, MAX_COLOR)
                _try_change(e, new_color, colors)
            continue
        
        # Valid: try to increase diversity
        # Pick a random edge and try to give it a new color
        e = random.randint(0, NUM_EDGES - 1)
        old_color = colors[e]
        
        # Try to use a color not yet in the used set
        # But first, let's try to expand: pick a new color
        # To maximize diversity, we want to introduce new colors
        # Try colors from MAX_COLOR down to 0 that aren't used yet
        # But this is slow. Instead, try a random new color or a used color
        
        # Strategy: try to assign a color that is not currently on this edge
        # and prefer colors not in used_colors to expand diversity
        if len(used_colors) < MAX_COLOR + 1:
            # Try to introduce a new color
            # Pick a random unused color
            unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
            if unused:
                # Try a few random unused colors
                for _ in range(5):
                    new_color = random.choice(unused)
                    if _try_change(e, new_color, colors):
                        used_colors.add(new_color)
                        break
                else:
                    # Try a used color (different from current)
                    other_used = [c for c in used_colors if c != old_color]
                    if other_used:
                        new_color = random.choice(other_used)
                        if _try_change(e, new_color, colors):
                            pass
        else:
            # All colors used, just shuffle
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
        # Verify best_colors is valid
        if _count_violations(best_colors.tolist()) == 0:
            return best_colors.tolist()
    
    # Fallback: all zeros (always valid)
    return [0] * NUM_EDGES
