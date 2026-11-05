"""Rewritten program: greedy + enhanced local search with swaps for max distinct colors on C4-free edge coloring."""
import time
import random
from itertools import combinations

N = 16
M = N * (N - 1) // 2  # 120 edges

# Precompute edge indices in lexicographic order
_EDGE_LIST = [(i, j) for i in range(N) for j in range(i + 1, N)]
_EDGE_IDX = {}
for _k, (_i, _j) in enumerate(_EDGE_LIST):
    _EDGE_IDX[(_i, _j)] = _k

def _get_edge_idx(u, v):
    if u > v:
        u, v = v, u
    return _EDGE_IDX[(u, v)]

# Precompute all 4-cycles (sets of 4 edge indices)
_FOUR_CYCLES = []
_seen_cycles = set()
for quad in combinations(range(N), 4):
    a, b, c, d = quad
    for cyc in [
        [(a, b), (b, c), (c, d), (d, a)],
        [(a, b), (b, d), (d, c), (c, a)],
        [(a, c), (c, b), (b, d), (d, a)],
    ]:
        idx_tuple = tuple(sorted(_get_edge_idx(u, v) for u, v in cyc))
        if idx_tuple not in _seen_cycles:
            _seen_cycles.add(idx_tuple)
            _FOUR_CYCLES.append(idx_tuple)

# For each edge, precompute the list of 4-cycles it participates in
_EDGE_CYCLES = [[] for _ in range(M)]
for _cyc in _FOUR_CYCLES:
    for _e in _cyc:
        _EDGE_CYCLES[_e].append(_cyc)

def _is_rainbow_conflict(coloring, edge_idx, new_color):
    """Check if assigning new_color to edge_idx creates any rainbow 4-cycle."""
    for cyc in _EDGE_CYCLES[edge_idx]:
        c1 = c2 = c3 = -1
        cnt = 0
        for e in cyc:
            if e == edge_idx:
                continue
            c = coloring[e]
            if c == -1:
                break
            if cnt == 0:
                c1 = c
            elif cnt == 1:
                c2 = c
            elif cnt == 2:
                c3 = c
            cnt += 1
        if cnt < 3:
            continue
        # Check 4 distinct: c1,c2,c3 all distinct and all != new_color
        if c1 != c2 and c1 != c3 and c2 != c3 and new_color != c1 and new_color != c2 and new_color != c3:
            return True
    return False

def _count_distinct(coloring):
    return len(set(c for c in coloring if c != -1))

def _validate(coloring):
    """Full validation: no 4-cycle has 4 distinct colors."""
    for cyc in _FOUR_CYCLES:
        colors = [coloring[e] for e in cyc]
        if len(set(colors)) == 4:
            return False
    return True

def _greedy_construct():
    """Greedy: process edges in order, assign smallest valid new color."""
    coloring = [-1] * M
    used_colors = set()
    for e in range(M):
        found = False
        for c in range(120):
            if c in used_colors:
                continue
            if not _is_rainbow_conflict(coloring, e, c):
                coloring[e] = c
                used_colors.add(c)
                found = True
                break
        if not found:
            for c in range(120):
                if c not in used_colors:
                    continue
                if not _is_rainbow_conflict(coloring, e, c):
                    coloring[e] = c
                    found = True
                    break
        if not found:
            coloring[e] = 0
            used_colors.add(0)
    return coloring

def _try_add_new_color(coloring, deadline):
    """Try to introduce a brand new color not currently in the palette."""
    current_colors = set(c for c in coloring if c != -1)
    best_score = _count_distinct(coloring)
    improved = False
    
    for e in range(M):
        if time.time() > deadline:
            break
        old_c = coloring[e]
        # Try colors not in current_colors
        for new_c in range(120):
            if new_c in current_colors:
                continue
            if _is_rainbow_conflict(coloring, e, new_c):
                continue
            coloring[e] = new_c
            new_score = _count_distinct(coloring)
            if new_score > best_score:
                best_score = new_score
                improved = True
                # Keep this change and continue looking for more
                current_colors.add(new_c)
                break
            else:
                coloring[e] = old_c
    return coloring, best_score, improved

def _try_swap_and_recolor(coloring, deadline):
    """Try edge swaps (exchange colors between two edges) then recolor to add new colors."""
    best_score = _count_distinct(coloring)
    improved = False
    
    for _iter in range(200):
        if time.time() > deadline:
            break
        # Pick two random edges with different colors
        e1 = random.randint(0, M - 1)
        e2 = random.randint(0, M - 1)
        if e1 == e2:
            continue
        c1 = coloring[e1]
        c2 = coloring[e2]
        if c1 == c2:
            continue
        
        # Tentatively swap
        coloring[e1] = c2
        coloring[e2] = c1
        
        # Check validity
        valid = True
        for cyc in _EDGE_CYCLES[e1]:
            colors = [coloring[e] for e in cyc]
            if len(set(colors)) == 4:
                valid = False
                break
        if valid:
            for cyc in _EDGE_CYCLES[e2]:
                colors = [coloring[e] for e in cyc]
                if len(set(colors)) == 4:
                    valid = False
                    break
        
        if not valid:
            coloring[e1] = c1
            coloring[e2] = c2
            continue
        
        # After valid swap, try to add a new color
        coloring, score, add_improved = _try_add_new_color(coloring, deadline)
        if add_improved and score > best_score:
            best_score = score
            improved = True
        elif not add_improved:
            # Check if swap alone helps (it shouldn't for distinct count, but might enable future)
            # Revert if no improvement
            pass  # Keep the swap as it might help future iterations
    
    return coloring, best_score, improved

def _local_search_enhanced(coloring, deadline):
    """Enhanced local search: recolor + swap + new color introduction."""
    coloring = list(coloring)
    best_score = _count_distinct(coloring)
    
    # Phase A: Standard recoloring to try new colors
    for _iteration in range(100):
        if time.time() > deadline:
            break
        improved = False
        for e in range(M):
            if time.time() > deadline:
                return coloring, best_score
            old_c = coloring[e]
            current_colors = set(coloring)
            for new_c in range(120):
                if new_c == old_c:
                    continue
                if new_c in current_colors:
                    continue
                if _is_rainbow_conflict(coloring, e, new_c):
                    continue
                coloring[e] = new_c
                new_score = _count_distinct(coloring)
                if new_score > best_score:
                    best_score = new_score
                    improved = True
                    break
                else:
                    coloring[e] = old_c
        if not improved:
            break
    
    # Phase B: Swap + recolor to break local optima
    if time.time() < deadline:
        coloring, score, _ = _try_swap_and_recolor(coloring, deadline)
        if score > best_score:
            best_score = score
    
    # Phase C: Another round of recoloring after swaps
    if time.time() < deadline:
        for _iteration in range(50):
            if time.time() > deadline:
                break
            improved = False
            for e in range(M):
                if time.time() > deadline:
                    return coloring, best_score
                old_c = coloring[e]
                current_colors = set(coloring)
                for new_c in range(120):
                    if new_c == old_c:
                        continue
                    if new_c in current_colors:
                        continue
                    if _is_rainbow_conflict(coloring, e, new_c):
                        continue
                    coloring[e] = new_c
                    new_score = _count_distinct(coloring)
                    if new_score > best_score:
                        best_score = new_score
                        improved = True
                        break
                    else:
                        coloring[e] = old_c
            if not improved:
                break
    
    return coloring, best_score

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin within 90s limit
    
    # Phase 1: Greedy construction
    coloring = _greedy_construct()
    score = _count_distinct(coloring)
    
    # Validate
    if not _validate(coloring):
        return [0] * M
    
    # Phase 2: Enhanced local search
    if time.time() < deadline:
        coloring, score = _local_search_enhanced(coloring, deadline)
    
    # Final validation
    if not _validate(coloring):
        return [0] * M
    
    return coloring
# EVOLVE-BLOCK-END
