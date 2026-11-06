# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_not_2_colorable_fast(n, edges):
    """
    Check if hypergraph is NOT 2-colorable using bitmask-based DPLL.
    Returns True if NOT 2-colorable.
    """
    if not edges:
        return False
    
    # Compress vertices
    active = set()
    for e in edges:
        active.update(e)
    m = len(active)
    if m == 0:
        return False
    v_map = {v: i for i, v in enumerate(sorted(active))}
    comp_edges = [tuple(v_map[v] for v in e) for e in edges]
    
    # Build edge masks: each edge as a bitmask of its 5 vertices
    edge_masks = []
    for e in comp_edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    # For each edge, precompute the 5 vertex positions
    edge_verts = [list(e) for e in comp_edges]
    
    # Unit propagation using bitmasks
    def unit_propagate(assign0, assign1):
        """
        assign0: bitmask of vertices assigned color 0
        assign1: bitmask of vertices assigned color 1
        Returns (assign0, assign1, ok) where ok=False means contradiction.
        """
        changed = True
        while changed:
            changed = False
            for i, mask in enumerate(edge_masks):
                verts = edge_verts[i]
                c0 = 0
                c1 = 0
                unassigned = []
                for v in verts:
                    bit = 1 << v
                    if assign0 & bit:
                        c0 += 1
                    elif assign1 & bit:
                        c1 += 1
                    else:
                        unassigned.append(v)
                
                if c0 == 5 or c1 == 5:
                    return assign0, assign1, False
                
                if c0 == 4 and len(unassigned) == 1:
                    v = unassigned[0]
                    bit = 1 << v
                    if assign1 & bit:
                        return assign0, assign1, False
                    assign1 |= bit
                    changed = True
                elif c1 == 4 and len(unassigned) == 1:
                    v = unassigned[0]
                    bit = 1 << v
                    if assign0 & bit:
                        return assign0, assign1, False
                    assign0 |= bit
                    changed = True
        return assign0, assign1, True
    
    def backtrack(assign0, assign1):
        """Returns True if a valid 2-coloring exists."""
        # Find first unassigned vertex
        assigned_mask = assign0 | assign1
        unassigned_mask = ((1 << m) - 1) & ~assigned_mask
        if unassigned_mask == 0:
            # All assigned, verify no monochromatic edge
            for mask in edge_masks:
                if (mask & assign0) == mask or (mask & assign1) == mask:
                    return False
            return True
        
        # Pick vertex: use first unassigned for simplicity, or heuristic
        v = (unassigned_mask & -unassigned_mask).bit_length() - 1
        
        # Try color 0
        new0 = assign0 | (1 << v)
        new1 = assign1
        a0, a1, ok = unit_propagate(new0, new1)
        if ok and backtrack(a0, a1):
            return True
        
        # Try color 1
        new0 = assign0
        new1 = assign1 | (1 << v)
        a0, a1, ok = unit_propagate(new0, new1)
        if ok and backtrack(a0, a1):
            return True
        
        return False
    
    a0, a1, ok = unit_propagate(0, 0)
    if not ok:
        return True  # Contradiction means not colorable
    
    colorable = backtrack(a0, a1)
    return not colorable

def _minimize_edges(n, edges, time_limit):
    """Aggressive greedy edge removal with multiple passes."""
    start_time = time.time()
    current = [list(e) for e in edges]
    
    improved = True
    while improved:
        improved = False
        random.shuffle(current)
        i = 0
        while i < len(current):
            if time.time() - start_time > time_limit:
                return current
            test = current[:i] + current[i+1:]
            if _is_not_2_colorable_fast(n, test):
                current = test
                improved = True
            else:
                i += 1
    
    return current

def construct():
    """
    Find a small non-2-colorable 5-uniform hypergraph.
    Strategy: try multiple n values, build dense hypergraphs, then minimize.
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    best_edges = None
    best_n = None
    best_count = float('inf')
    
    # Try n values from small to larger
    for n in range(9, 17):
        if time.time() > deadline - 20:
            break
        
        all_edges = list(combinations(range(n), 5))
        random.shuffle(all_edges)
        
        # Start with a dense subset to ensure non-2-colorability
        # Then minimize aggressively
        for start_frac in [1.0, 0.8, 0.6, 0.5]:
            if time.time() > deadline - 15:
                break
            
            num_start = int(len(all_edges) * start_frac)
            initial = all_edges[:num_start]
            
            # Check if non-2-colorable
            if _is_not_2_colorable_fast(n, initial):
                # Minimize
                remaining_time = deadline - time.time() - 5
                if remaining_time < 2:
                    break
                minimized = _minimize_edges(n, initial, remaining_time)
                
                if len(minimized) < best_count:
                    best_count = len(minimized)
                    best_n = n
                    best_edges = minimized
                
                # If we're at or below 51, we're near optimal
                if best_count <= 51:
                    break
        
        # If we found something good, try to improve with more minimization time
        if best_count <= 60:
            remaining = deadline - time.time() - 10
            if remaining > 5 and best_edges is not None:
                improved = _minimize_edges(best_n, best_edges, remaining)
                if len(improved) < best_count:
                    best_count = len(improved)
                    best_edges = improved
    
    # Fallback: complete hypergraph on 9 vertices (126 edges)
    if best_edges is None:
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    
    return {'n': best_n, 'edges': [list(e) for e in best_edges]}
# EVOLVE-BLOCK-END
