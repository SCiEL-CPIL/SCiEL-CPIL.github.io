"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
import numpy as np

N = 256

def _build_conflict_masks():
    """Build bitset masks: conflict[i] has bit j set if {i+1, j+1} participates in a 3-AP."""
    # For each pair (a, b), check if they form part of a 3-AP with some c in [1, N]
    # A 3-AP: x, x+d, x+2d for d > 0
    # Pairs that conflict: (x, x+d), (x, x+2d), (x+d, x+2d)
    masks = [0] * N
    for d in range(1, N // 2 + 1):
        for x in range(1, N - 2 * d + 1):
            a, b, c = x - 1, x + d - 1, x + 2 * d - 1
            # All three pairs conflict
            masks[a] |= (1 << b) | (1 << c)
            masks[b] |= (1 << a) | (1 << c)
            masks[c] |= (1 << a) | (1 << b)
    return masks

def _greedy_color(candidates_mask, conflict_masks):
    """
    Greedy coloring of the subgraph induced by candidates_mask.
    Returns list of colors (integers) for each vertex in candidates.
    The number of distinct colors is an upper bound on the independence number.
    """
    # Get list of vertices in candidates
    verts = []
    m = candidates_mask
    while m:
        # Find lowest set bit
        low = m & (-m)
        idx = low.bit_length() - 1
        verts.append(idx)
        m ^= low
    
    if not verts:
        return []
    
    # Color assignment: greedy by degree in candidate subgraph
    # Order by descending degree within candidates
    deg = {}
    for v in verts:
        deg[v] = (conflict_masks[v] & candidates_mask).bit_count()
    
    ordered = sorted(verts, key=lambda v: -deg[v])
    
    colors = {}
    next_color = 0
    used_colors = [set() for _ in range(len(verts) + 10)]  # color -> set of vertices
    
    for v in ordered:
        # Find first color not used by neighbors
        neighbor_colors = set()
        conf = conflict_masks[v] & candidates_mask
        for u in verts:
            if conf & (1 << u):
                if u in colors:
                    neighbor_colors.add(colors[u])
        
        c = 0
        while c in neighbor_colors:
            c += 1
        colors[v] = c
        next_color = max(next_color, c + 1)
    
    return next_color

def _branch_and_bound(candidates_mask, current_sum, current_count, conflict_masks, 
                      reciprocal_weights, deadline, best_state):
    """
    Branch and bound for maximum weight independent set.
    candidates_mask: bitset of vertices still available
    current_sum: sum of reciprocals of selected vertices
    current_count: number of selected vertices
    Returns (best_sum, best_mask)
    """
    if time.time() > deadline:
        return best_state
    
    if candidates_mask == 0:
        if current_sum > best_state[0]:
            best_state[0] = current_sum
            best_state[1] = 0  # no more to add
        return best_state
    
    # Upper bound: current_sum + sum of all remaining reciprocals
    # Compute sum of reciprocals of remaining candidates
    rem_sum = 0.0
    m = candidates_mask
    while m:
        low = m & (-m)
        idx = low.bit_length() - 1
        rem_sum += reciprocal_weights[idx]
        m ^= low
    
    if current_sum + rem_sum <= best_state[0] + 1e-15:
        return best_state
    
    # Get coloring bound
    num_colors = _greedy_color(candidates_mask, conflict_masks)
    # The independence number is at most num_colors
    # Upper bound: current_sum + sum of top num_colors reciprocals
    # For simplicity, use: current_sum + rem_sum as a looser but faster bound
    # A tighter bound: take the num_colors largest reciprocals from candidates
    
    # Get all candidate weights and sort
    cand_weights = []
    m = candidates_mask
    while m:
        low = m & (-m)
        idx = low.bit_length() - 1
        cand_weights.append((reciprocal_weights[idx], idx))
        m ^= low
    
    cand_weights.sort(reverse=True)
    tight_bound = current_sum + sum(w for w, _ in cand_weights[:num_colors])
    if tight_bound <= best_state[0] + 1e-15:
        return best_state
    
    # Branch: pick a vertex to branch on
    # Heuristic: pick vertex with highest degree in candidates
    best_vertex = -1
    max_deg = -1
    m = candidates_mask
    while m:
        low = m & (-m)
        idx = low.bit_length() - 1
        deg = (conflict_masks[idx] & candidates_mask).bit_count()
        if deg > max_deg:
            max_deg = deg
            best_vertex = idx
        m ^= low
    
    if best_vertex == -1:
        return best_state
    
    # Branch 1: include best_vertex
    new_mask = candidates_mask & ~(1 << best_vertex)
    # Remove all neighbors of best_vertex
    new_mask &= ~conflict_masks[best_vertex]
    best_state = _branch_and_bound(new_mask, current_sum + reciprocal_weights[best_vertex], 
                                    current_count + 1, conflict_masks, reciprocal_weights,
                                    deadline, best_state)
    
    if time.time() > deadline:
        return best_state
    
    # Branch 2: exclude best_vertex
    new_mask = candidates_mask & ~(1 << best_vertex)
    best_state = _branch_and_bound(new_mask, current_sum, current_count,
                                    conflict_masks, reciprocal_weights, deadline, best_state)
    
    return best_state

def _greedy_initial_solution(conflict_masks, reciprocal_weights):
    """
    Greedy construction: repeatedly pick the vertex with highest reciprocal weight
    that doesn't conflict with already-selected vertices.
    Returns (sum, mask)
    """
    selected_mask = 0
    available = (1 << N) - 1
    total = 0.0
    
    while available:
        # Find vertex with max reciprocal weight in available
        best = -1
        best_w = -1
        m = available
        while m:
            low = m & (-m)
            idx = low.bit_length() - 1
            if reciprocal_weights[idx] > best_w:
                best_w = reciprocal_weights[idx]
                best = idx
            m ^= low
        
        if best == -1:
            break
        
        selected_mask |= (1 << best)
        total += best_w
        # Remove best and its neighbors from available
        available &= ~(1 << best)
        available &= ~conflict_masks[best]
    
    return total, selected_mask

def _local_search(selected_mask, conflict_masks, reciprocal_weights, deadline):
    """
    Local search: try single swaps to improve the solution.
    Returns (best_sum, best_mask)
    """
    current_mask = selected_mask
    current_sum = sum(reciprocal_weights[i] for i in range(N) if (current_mask >> i) & 1)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding each unselected vertex
        m = ~current_mask & ((1 << N) - 1)
        while m:
            low = m & (-m)
            idx = low.bit_length() - 1
            # Check if idx conflicts with any selected vertex
            if (conflict_masks[idx] & current_mask) == 0:
                new_mask = current_mask | (1 << idx)
                new_sum = current_sum + reciprocal_weights[idx]
                if new_sum > current_sum + 1e-15:
                    current_mask = new_mask
                    current_sum = new_sum
                    improved = True
                    break
            m ^= low
        
        if not improved:
            # Try swaps: remove one, add one
            for i in range(N):
                if time.time() > deadline:
                    break
                if not ((current_mask >> i) & 1):
                    continue
                # Try removing i and adding some j
                without_i = current_mask & ~(1 << i)
                # Find best j that can be added
                m = ~without_i & ((1 << N) - 1)
                best_j = -1
                best_gain = 0
                while m:
                    low = m & (-m)
                    j = low.bit_length() - 1
                    if (conflict_masks[j] & without_i) == 0:
                        gain = reciprocal_weights[j] - reciprocal_weights[i]
                        if gain > best_gain + 1e-15:
                            best_gain = gain
                            best_j = j
                    m ^= low
                
                if best_j != -1:
                    new_mask = without_i | (1 << best_j)
                    new_sum = current_sum + best_gain
                    current_mask = new_mask
                    current_sum = new_sum
                    improved = True
                    break
    
    return current_sum, current_mask

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    # Build conflict masks
    conflict_masks = _build_conflict_masks()
    
    # Reciprocal weights
    reciprocal_weights = [1.0 / (i + 1) for i in range(N)]
    
    # Phase 1: Greedy initial solution
    greedy_sum, greedy_mask = _greedy_initial_solution(conflict_masks, reciprocal_weights)
    
    # Phase 2: Local search from greedy
    ls_sum, ls_mask = _local_search(greedy_mask, conflict_masks, reciprocal_weights, deadline)
    
    best_sum = ls_sum
    best_mask = ls_mask
    
    # Phase 3: Branch and bound to try to improve
    # Start from the current best and try to add more vertices
    # Actually, B&B should explore the full space. Let's run it with a time limit.
    # We'll use the local search result as the initial bound.
    
    if time.time() < deadline:
        best_state = [best_sum, best_mask]
        # Run B&B from scratch (full search)
        full_mask = (1 << N) - 1
        _branch_and_bound(full_mask, 0.0, 0, conflict_masks, reciprocal_weights, 
                         deadline, best_state)
        best_sum, best_mask = best_state
    
    # Convert mask to list
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
    
    # Verify no 3-AP
    result_set = set(result)
    for a in result:
        for b in result:
            if a == b:
                continue
            c = 2 * b - a
            if c in result_set and c != a and c != b:
                # Found a 3-AP, need to fix
                # Remove c
                result.remove(c)
                result_set.discard(c)
    
    if not result:
        # Fallback: just return {1}
        result = [1]
    
    return result
# EVOLVE-BLOCK-END
