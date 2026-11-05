"""Few edges in 5-uniform hypergraphs without Property B.

Strategy: Greedy set cover on the dual (colorings) with bitmask representation,
then aggressive local search (remove + swap) to minimize edge count.
For n=10: 1024 colorings, 252 edges. Each edge kills 64 colorings.
"""
import time
import random
from itertools import combinations

_DEADLINE = 145.0
_START = time.monotonic()


def _elapsed():
    return time.monotonic() - _START


def _build_edge_masks(n):
    """
    For each 5-edge, compute a bitmask of colorings it kills (makes monochromatic).
    Returns (edges_list, edge_masks) where edge_masks[i] is an int with bit c set
    iff coloring c makes edge i monochromatic.
    """
    edges = list(combinations(range(n), 5))
    total_colorings = 1 << n
    num_edges = len(edges)
    
    # For each edge, compute which colorings make it monochromatic
    edge_masks = [0] * num_edges
    
    for ei, edge in enumerate(edges):
        edge_set = set(edge)
        free_positions = [i for i in range(n) if i not in edge_set]
        num_free = len(free_positions)
        
        mask = 0
        for free_mask in range(1 << num_free):
            # Build coloring where edge vertices are all 0, free vertices per free_mask
            c0 = 0
            for j, pos in enumerate(free_positions):
                if free_mask & (1 << j):
                    c0 |= (1 << pos)
            # c0: edge vertices are 0, free vertices set
            mask |= (1 << c0)
            
            # c1: edge vertices are all 1, free vertices same
            c1 = c0
            for v in edge:
                c1 |= (1 << v)
            mask |= (1 << c1)
        
        edge_masks[ei] = mask
    
    return edges, edge_masks


def _verify_not_2_colorable(n, edges):
    """
    Verify hypergraph is NOT 2-colorable using backtracking with unit propagation.
    Returns True if NOT 2-colorable (Property B fails).
    """
    m = len(edges)
    if m == 0:
        return False
    
    # Build incidence
    v_edges = [[] for _ in range(n)]
    for ei, e in enumerate(edges):
        for v in e:
            v_edges[v].append(ei)
    
    # Vertex ordering by degree (highest first)
    v_order = sorted(range(n), key=lambda v: -len(v_edges[v]))
    
    color = [-1] * n
    pos_count = [0] * m
    neg_count = [0] * m
    uncount = [5] * m
    
    def _propagate():
        changed = True
        while changed:
            changed = False
            for ei in range(m):
                pc = pos_count[ei]
                nc = neg_count[ei]
                uc = uncount[ei]
                if pc == 5 or nc == 5:
                    return False
                if uc == 0:
                    continue
                if pc == 4 and uc == 1:
                    for v in edges[ei]:
                        if color[v] == -1:
                            color[v] = 1
                            neg_count[ei] += 1
                            uncount[ei] -= 1
                            for e2 in v_edges[v]:
                                if e2 != ei:
                                    neg_count[e2] += 1
                                    uncount[e2] -= 1
                            changed = True
                            break
                elif nc == 4 and uc == 1:
                    for v in edges[ei]:
                        if color[v] == -1:
                            color[v] = 0
                            pos_count[ei] += 1
                            uncount[ei] -= 1
                            for e2 in v_edges[v]:
                                if e2 != ei:
                                    pos_count[e2] += 1
                                    uncount[e2] -= 1
                            changed = True
                            break
        return True
    
    def _search(idx):
        while idx < n and color[v_order[idx]] != -1:
            idx += 1
        if idx == n:
            return True
        
        v = v_order[idx]
        color_snap = color[:]
        pos_snap = pos_count[:]
        neg_snap = neg_count[:]
        unc_snap = uncount[:]
        
        for c in (0, 1):
            color[v] = c
            valid = True
            for ei in v_edges[v]:
                if c == 0:
                    pos_count[ei] += 1
                else:
                    neg_count[ei] += 1
                uncount[ei] -= 1
                if pos_count[ei] == 5 or neg_count[ei] == 5:
                    valid = False
                    break
            if valid and _propagate():
                if _search(idx + 1):
                    return True
            color[:] = color_snap
            pos_count[:] = pos_snap
            neg_count[:] = neg_snap
            uncount[:] = unc_snap
        
        return False
    
    return not _search(0)


def _known_51_edges():
    """The known 51-edge construction: even 5-subsets of 10 vertices."""
    evenset = frozenset((0, 2, 4, 6, 8))
    edges = []
    for c in combinations(range(10), 5):
        if sum(1 for v in c if v in evenset) % 2 == 0:
            edges.append(list(c))
    return edges


def _greedy_cover(edges, edge_masks, n):
    """
    Greedy set cover: repeatedly pick the edge that covers the most uncovered colorings.
    Uses bitmask representation for speed.
    """
    total_colorings = 1 << n
    all_mask = (1 << total_colorings) - 1
    uncovered = all_mask
    
    selected = []
    selected_set = set()
    
    # Precompute: for each edge, which colorings it covers (already in edge_masks)
    # Greedy: pick edge covering most uncovered colorings
    
    remaining = list(range(len(edges)))
    rng = random.Random(12345)
    
    while uncovered != 0 and remaining:
        if _elapsed() > _DEADLINE - 30:
            break
        
        best_ei = -1
        best_count = 0
        
        for ei in remaining:
            if ei in selected_set:
                continue
            count = bin(edge_masks[ei] & uncovered).count('1')
            if count > best_count:
                best_count = count
                best_ei = ei
                if count == bin(uncovered).count('1'):
                    break
        
        if best_ei == -1 or best_count == 0:
            break
        
        selected.append(best_ei)
        selected_set.add(best_ei)
        remaining.remove(best_ei)
        uncovered &= ~edge_masks[best_ei]
    
    return selected


def _local_search_remove(edges, edge_masks, n, selected, rng, time_limit=20.0):
    """
    Aggressively remove redundant edges. An edge is removable if its colorings
    are all covered by other selected edges.
    """
    total_colorings = 1 << n
    result = list(selected)
    result_set = set(result)
    
    # Compute coverage: union of masks of selected edges
    def _compute_coverage(edge_list):
        cov = 0
        for ei in edge_list:
            cov |= edge_masks[ei]
        return cov
    
    current_cov = _compute_coverage(result)
    all_mask = (1 << total_colorings) - 1
    
    if (current_cov & all_mask) != all_mask:
        return result  # Not fully covered, don't remove
    
    changed = True
    while changed and _elapsed() < time_limit:
        changed = False
        indices = list(range(len(result)))
        rng.shuffle(indices)
        
        for idx in indices:
            ei = result[idx]
            # Check if removing ei still covers all
            new_cov = current_cov & ~edge_masks[ei]
            # Actually we need: is every coloring still covered by at least one other edge?
            # new_cov = coverage without ei
            # But we need coverage WITH all other edges
            other_cov = 0
            for j, ej in enumerate(result):
                if j != idx:
                    other_cov |= edge_masks[ej]
            
            if (other_cov & all_mask) == all_mask:
                # Can remove
                current_cov = other_cov
                result.pop(idx)
                result_set.discard(ei)
                changed = True
                break
    
    return result


def _local_search_swap(edges, edge_masks, n, selected, rng, time_limit=20.0):
    """
    Try to replace pairs of edges with a single edge that covers the same colorings.
    Also try single-edge replacement.
    """
    total_colorings = 1 << n
    all_mask = (1 << total_colorings) - 1
    result = list(selected)
    
    def _compute_coverage(edge_list):
        cov = 0
        for ei in edge_list:
            cov |= edge_masks[ei]
        return cov
    
    current_cov = _compute_coverage(result)
    if (current_cov & all_mask) != all_mask:
        return result
    
    # Try removing one edge and adding another
    for _round in range(50):
        if _elapsed() > time_limit:
            break
        
        improved = False
        indices = list(range(len(result)))
        rng.shuffle(indices)
        
        for idx in indices:
            if _elapsed() > time_limit:
                break
            
            ei = result[idx]
            # Try removing ei
            other_result = [x for x in result if x != ei]
            other_cov = _compute_coverage(other_result)
            
            # Find colorings only covered by ei
            only_by_ei = edge_masks[ei] & ~other_cov
            
            if only_by_ei == 0:
                # ei is redundant
                result = other_result
                current_cov = other_cov
                improved = True
                break
            
            # Try adding a single edge to cover only_by_ei
            for ej in range(len(edges)):
                if ej == ei or ej in result:
                    continue
                if (edge_masks[ej] & only_by_ei) == only_by_ei:
                    # ej covers all colorings that only ei covered
                    new_result = other_result + [ej]
                    new_cov = _compute_coverage(new_result)
                    if (new_cov & all_mask) == all_mask:
                        if len(new_result) < len(result):
                            result = new_result
                            current_cov = new_cov
                            improved = True
                            break
        
        if not improved:
            break
    
    return result


def construct():
    if _elapsed() > _DEADLINE - 10:
        return {'n': 10, 'edges': _known_51_edges()}
    
    rng = random.Random(42)
    best_edges = _known_51_edges()
    best_n = 10
    best_count = len(best_edges)
    
    # Try n=10 first (most promising for small edge count)
    for n in [10, 11]:
        if _elapsed() > _DEADLINE - 25:
            break
        
        try:
            edges_list, edge_masks = _build_edge_masks(n)
        except Exception:
            continue
        
        total_colorings = 1 << n
        
        # Greedy set cover
        t0 = time.monotonic()
        selected = _greedy_cover(edges_list, edge_masks, n)
        t1 = time.monotonic()
        
        if len(selected) < best_count:
            sel_edges = [list(edges_list[ei]) for ei in selected]
            if _verify_not_2_colorable(n, sel_edges):
                best_edges = sel_edges
                best_n = n
                best_count = len(sel_edges)
        
        # Local search: remove redundant edges
        if _elapsed() < _DEADLINE - 20:
            improved = _local_search_remove(
                edges_list, edge_masks, n, selected, rng,
                time_limit=min(15.0, _DEADLINE - 20 - _elapsed())
            )
            if len(improved) < best_count:
                sel_edges = [list(edges_list[ei]) for ei in improved]
                if _verify_not_2_colorable(n, sel_edges):
                    best_edges = sel_edges
                    best_n = n
                    best_count = len(sel_edges)
        
        # Local search: swap
        if _elapsed() < _DEADLINE - 15:
            swapped = _local_search_swap(
                edges_list, edge_masks, n, improved, rng,
                time_limit=min(15.0, _DEADLINE - 15 - _elapsed())
            )
            if len(swapped) < best_count:
                sel_edges = [list(edges_list[ei]) for ei in swapped]
                if _verify_not_2_colorable(n, sel_edges):
                    best_edges = sel_edges
                    best_n = n
                    best_count = len(sel_edges)
        
        # Multiple restarts with different random seeds for greedy
        if _elapsed() < _DEADLINE - 15:
            for seed in [7, 13, 21, 33]:
                if _elapsed() > _DEADLINE - 15:
                    break
                rng2 = random.Random(seed)
                # Slightly different greedy: randomize tie-breaking
                remaining = list(range(len(edges_list)))
                uncovered = (1 << total_colorings) - 1
                selected2 = []
                selected_set2 = set()
                
                while uncovered != 0 and remaining:
                    if _elapsed() > _DEADLINE - 15:
                        break
                    
                    # Score each remaining edge
                    scores = []
                    for ei in remaining:
                        if ei in selected_set2:
                            continue
                        count = bin(edge_masks[ei] & uncovered).count('1')
                        scores.append((count, ei))
                    
                    if not scores:
                        break
                    
                    # Pick from top candidates with some randomness
                    max_score = max(s[0] for s in scores)
                    candidates = [s[1] for s in scores if s[0] >= max_score * 0.9]
                    
                    if not candidates:
                        best_ei = max(scores, key=lambda x: x[0])[1]
                    else:
                        best_ei = rng2.choice(candidates)
                    
                    selected2.append(best_ei)
                    selected_set2.add(best_ei)
                    remaining.remove(best_ei)
                    uncovered &= ~edge_masks[best_ei]
                
                if len(selected2) < best_count:
                    sel_edges = [list(edges_list[ei]) for ei in selected2]
                    if _verify_not_2_colorable(n, sel_edges):
                        best_edges = sel_edges
                        best_n = n
                        best_count = len(sel_edges)
                    
                    # Local search on this
                    if _elapsed() < _DEADLINE - 15:
                        improved2 = _local_search_remove(
                            edges_list, edge_masks, n, selected2, rng2,
                            time_limit=min(5.0, _DEADLINE - 15 - _elapsed())
                        )
                        if len(improved2) < best_count:
                            sel_edges = [list(edges_list[ei]) for ei in improved2]
                            if _verify_not_2_colorable(n, sel_edges):
                                best_edges = sel_edges
                                best_n = n
                                best_count = len(sel_edges)
    
    # Final verification
    if _verify_not_2_colorable(best_n, best_edges):
        return {'n': best_n, 'edges': best_edges}
    
    # Fallback
    return {'n': 10, 'edges': _known_51_edges()}
