"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _edge_index(i, j):
    """Map pair (i,j) with 0<=i<j<16 to flat index in [0,120)."""
    return 15 * i - i * (i - 1) // 2 + (j - i - 1)

def _all_k4s():
    """Precompute all K4s as lists of 6 edge indices."""
    k4s = []
    for quad in combinations(range(16), 4):
        edges = []
        for a, b in combinations(quad, 2):
            edges.append(_edge_index(a, b))
        k4s.append(tuple(edges))
    return k4s

_K4S = _all_k4s()
_N_K4S = len(_K4S)

# Precompute for each edge, which K4s contain it
_EDGE_TO_K4S = [[] for _ in range(120)]
for _k4_idx, _k4 in enumerate(_K4S):
    for _e in _k4:
        _EDGE_TO_K4S[_e].append(_k4_idx)

# Precompute edge pairs within each K4 for fast violation checking
_K4_EDGES = [list(k4) for k4 in _K4S]

def _count_violations(colours):
    """Count K4s with fewer than 5 distinct colours."""
    violations = 0
    for k4 in _K4S:
        if len(set(colours[e] for e in k4)) < 5:
            violations += 1
    return violations

def _count_colours(colours):
    """Count number of distinct colours used."""
    return len(set(colours))

def _normalise_colours(colours):
    """Relabel colours to 0..k-1 preserving order of first appearance."""
    mapping = {}
    next_c = 0
    for c in colours:
        if c not in mapping:
            mapping[c] = next_c
            next_c += 1
    return [mapping[c] for c in colours]

def _try_merge_colour(colours, target, source, k4s):
    """Try to merge all edges of colour 'source' into colour 'target'.
    Returns (new_colours, valid) where valid is True if constraint holds."""
    new_colours = list(colours)
    changed = False
    for idx, c in enumerate(new_colours):
        if c == source:
            new_colours[idx] = target
            changed = True
    if not changed:
        return colours, False
    # Check all K4s
    for k4 in k4s:
        if len(set(new_colours[e] for e in k4)) < 5:
            return colours, False
    return new_colours, True

def _backtracking_construct(deadline, max_colours=15):
    """
    Backtracking search to construct a valid edge-coloring of K16
    with at most max_colours colors.
    
    We assign edges in a fixed order. For each edge, we try colors 0..max_colours-1.
    Forward checking: after assigning an edge, we check that no K4 containing this
    edge becomes "impossible" (i.e., has < 5 colors with some edges unassigned).
    
    For efficiency, we only check K4s that are now "complete" (all 6 edges assigned)
    or "nearly complete" (5 edges assigned).
    """
    # Edge assignment order: by vertex pairs, but we'll use a heuristic order
    # that groups edges by the smaller vertex to enable early pruning
    edge_order = []
    for i in range(16):
        for j in range(i + 1, 16):
            edge_order.append((i, j))
    
    # Map from (i,j) to position in edge_order
    edge_pos = {}
    for pos, (i, j) in enumerate(edge_order):
        edge_pos[(i, j)] = pos
    
    # For each K4, track which edges are assigned and what colors they have
    # k4_edge_colors[k4_idx] = dict mapping edge_idx -> color (only assigned edges)
    
    colours = [None] * 120
    k4_edge_colors = [dict() for _ in range(_N_K4S)]
    
    # Count how many edges in each K4 are assigned
    k4_assigned_count = [0] * _N_K4S
    
    # For forward checking: for each K4, track the set of colors used among assigned edges
    # and the number of unassigned edges
    # A K4 is "violated" if: all 6 edges assigned and < 5 distinct colors
    # A K4 is "impossible" if: assigned edges have < 5 colors AND unassigned edges <= (5 - len(assigned_colors))
    #   because even if all unassigned edges get new colors, we can't reach 5
    
    def check_k4_feasible(k4_idx):
        """Check if K4 k4_idx can still reach 5 colors."""
        assigned_colors = set(k4_edge_colors[k4_idx].values())
        n_assigned = k4_assigned_count[k4_idx]
        n_unassigned = 6 - n_assigned
        # We need at least 5 distinct colors total
        # Current distinct: len(assigned_colors)
        # Max additional distinct: min(n_unassigned, max_colours - len(assigned_colors))
        # But we can only add colors not already used, up to n_unassigned new colors
        max_possible = len(assigned_colors) + min(n_unassigned, max_colours - len(assigned_colors))
        if max_possible < 5:
            return False
        return True
    
    def check_all_feasible():
        """Check that all K4s are still feasible."""
        for k4_idx in range(_N_K4S):
            if k4_assigned_count[k4_idx] > 0:  # Only check if any edge assigned
                if not check_k4_feasible(k4_idx):
                    return False
        return True
    
    def check_complete_k4s():
        """Check that all complete K4s (all 6 edges assigned) have >= 5 colors."""
        for k4_idx in range(_N_K4S):
            if k4_assigned_count[k4_idx] == 6:
                if len(set(k4_edge_colors[k4_idx].values())) < 5:
                    return False
        return True
    
    # Backtracking state
    def backtrack(pos):
        """Try to assign edges from position pos onwards."""
        if time.time() > deadline:
            return False
        if pos == 120:
            return True
        
        i, j = edge_order[pos]
        e_idx = _edge_index(i, j)
        
        # Try each color
        for color in range(max_colours):
            # Assign this edge
            colours[e_idx] = color
            
            # Update K4s that contain this edge
            affected_k4s = _EDGE_TO_K4S[e_idx]
            for k4_idx in affected_k4s:
                k4_edge_colors[k4_idx][e_idx] = color
                k4_assigned_count[k4_idx] += 1
            
            # Check feasibility
            feasible = True
            for k4_idx in affected_k4s:
                if not check_k4_feasible(k4_idx):
                    feasible = False
                    break
            
            if not feasible:
                # Undo
                colours[e_idx] = None
                for k4_idx in affected_k4s:
                    del k4_edge_colors[k4_idx][e_idx]
                    k4_assigned_count[k4_idx] -= 1
                continue
            
            # If all edges in a K4 are now assigned, check it's valid
            all_k4s_complete = True
            for k4_idx in affected_k4s:
                if k4_assigned_count[k4_idx] == 6:
                    if len(set(k4_edge_colors[k4_idx].values())) < 5:
                        all_k4s_complete = False
                        break
            
            if not all_k4s_complete:
                # Undo
                colours[e_idx] = None
                for k4_idx in affected_k4s:
                    del k4_edge_colors[k4_idx][e_idx]
                    k4_assigned_count[k4_idx] -= 1
                continue
            
            # Recursive call
            if backtrack(pos + 1):
                return True
            
            # Undo
            colours[e_idx] = None
            for k4_idx in affected_k4s:
                del k4_edge_colors[k4_idx][e_idx]
                k4_assigned_count[k4_idx] -= 1
        
        return False
    
    if backtrack(0):
        return _normalise_colours(colours)
    return None

def _local_search_reduce(colours, deadline, target_count=None):
    """Local search to reduce the number of colors used."""
    if target_count is None:
        target_count = _count_colours(colours)
    
    best_colours = list(colours)
    best_count = _count_colours(colours)
    
    # Try to merge colors
    used_colours = sorted(set(colours))
    current_colours = list(colours)
    
    # Systematic merge attempts
    progress = True
    while progress and time.time() < deadline:
        progress = False
        used = sorted(set(current_colours))
        if len(used) <= best_count:
            break
        for src in reversed(used):
            if time.time() > deadline:
                break
            for tgt in used:
                if tgt >= src:
                    continue
                new_colours, valid = _try_merge_colour(current_colours, tgt, src, _K4S)
                if valid:
                    current_colours = new_colours
                    cur_count = _count_colours(current_colours)
                    if cur_count < best_count:
                        best_colours = list(current_colours)
                        best_count = cur_count
                    progress = True
                    break
            if progress:
                break
    
    # Random perturbations to enable further merges
    if time.time() < deadline - 5:
        for _ in range(200):
            if time.time() > deadline:
                break
            edge_idx = random.randint(0, 119)
            old_colour = current_colours[edge_idx]
            # Try changing to a random existing color
            used = sorted(set(current_colours))
            if not used:
                continue
            new_colour = random.choice(used)
            if new_colour == old_colour:
                continue
            current_colours[edge_idx] = new_colour
            
            # Try to merge
            used_after = sorted(set(current_colours))
            if len(used_after) <= best_count:
                found = False
                for src in reversed(used_after):
                    if time.time() > deadline:
                        break
                    for tgt in used_after:
                        if tgt >= src:
                            continue
                        nc, v = _try_merge_colour(current_colours, tgt, src, _K4S)
                        if v:
                            current_colours = nc
                            cur_count = _count_colours(current_colours)
                            if cur_count < best_count:
                                best_colours = list(current_colours)
                                best_count = cur_count
                            found = True
                            break
                    if found:
                        break
            else:
                # Revert
                current_colours[edge_idx] = old_colour
    
    return best_colours, best_count

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    best_colours = None
    best_count = 999
    
    # Strategy 1: Try backtracking with small palettes
    for max_c in [13, 14, 15]:
        if time.time() > deadline - 30:
            break
        result = _backtracking_construct(deadline - 20, max_c)
        if result is not None:
            count = _count_colours(result)
            if count < best_count:
                best_colours = result
                best_count = count
            if best_count <= max_c:
                break  # Can't do better than max_c
    
    # Strategy 2: If backtracking failed, use the seed coloring
    if best_colours is None:
        colours = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
        colours = _normalise_colours(colours)
        best_colours = list(colours)
        best_count = _count_colours(colours)
    
    # Strategy 3: Local search to reduce colors
    if time.time() < deadline - 10:
        polished, count = _local_search_reduce(best_colours, deadline - 5)
        if count < best_count:
            best_colours = polished
            best_count = count
    
    return best_colours

# EVOLVE-BLOCK-END

