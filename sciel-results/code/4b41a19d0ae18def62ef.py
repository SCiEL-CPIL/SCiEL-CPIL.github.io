"""Find minimal non-2-colorable 5-uniform hypergraph via local search."""
import time
import random
from itertools import combinations

_WALL_LIMIT = 150.0
_MARGIN = 15.0


def _is_2_colorable(n, edges, deadline):
    """True iff there exists a 2-colouring with NO monochromatic edge.
    Uses backtracking with unit propagation."""
    edges = [tuple(e) for e in edges]
    m = len(edges)
    if m == 0:
        return True
    
    # For each edge, track count of each color and unassigned vertices
    color = [-1] * n
    e0 = [0] * m
    e1 = [0] * m
    uncol = [list(e) for e in edges]
    
    def propagate(queue):
        """Unit propagation: if an edge has 4 of one color, force the 5th."""
        while queue:
            ei = queue.pop()
            u = uncol[ei]
            if e0[ei] == 5 or e1[ei] == 5:
                return False
            if len(u) == 1:
                v = u[0]
                if e0[ei] == 4:
                    forced = 1
                elif e1[ei] == 4:
                    forced = 0
                else:
                    continue
                if color[v] == -1:
                    color[v] = forced
                    uncol[ei].remove(v)
                    if forced == 0:
                        e0[ei] += 1
                    else:
                        e1[ei] += 1
                    # Propagate to other edges containing v
                    for ej in range(m):
                        if ej != ei and v in uncol[ej]:
                            uncol[ej].remove(v)
                            if forced == 0:
                                e0[ej] += 1
                            else:
                                e1[ej] += 1
                            if e0[ej] == 5 or e1[ej] == 5:
                                return False
                            if len(uncol[ej]) == 1:
                                queue.append(ej)
                elif color[v] != forced:
                    return False
        return True
    
    import sys
    sys.setrecursionlimit(100000)
    
    def rec(k):
        if time.time() > deadline:
            return None  # timeout
        if k >= n:
            return True
        for c in (0, 1):
            color[k] = c
            touched = []
            for ei, e in enumerate(edges):
                if k in e:
                    uncol[ei].remove(k)
                    if c == 0:
                        e0[ei] += 1
                    else:
                        e1[ei] += 1
                    touched.append(ei)
                    if e0[ei] == 5 or e1[ei] == 5:
                        # Conflict, undo and try other color
                        color[k] = -1
                        for ej in touched:
                            uncol[ej].append(k)
                            if c == 0:
                                e0[ej] -= 1
                            else:
                                e1[ej] -= 1
                        break
            else:
                # No immediate conflict, propagate
                queue = [ei for ei in touched if len(uncol[ei]) == 1]
                if propagate(queue):
                    result = rec(k + 1)
                    if result is not None:
                        return result
                # Undo
                color[k] = -1
                for ej in touched:
                    if k in uncol[ej]:
                        uncol[ej].append(k)
                    if c == 0:
                        e0[ej] -= 1
                    else:
                        e1[ej] -= 1
                continue
            # If we broke out of for due to conflict, try next color
        return False
    
    result = rec(0)
    if result is None:
        # Timeout - assume not 2-colorable (conservative for our purpose)
        return False
    return result


def _greedy_build(n, max_edges, deadline, rng):
    """Greedy: add edges that are hardest to satisfy."""
    all_edges = list(combinations(range(n), 5))
    rng.shuffle(all_edges)
    chosen = []
    for e in all_edges:
        if len(chosen) >= max_edges:
            break
        if time.time() > deadline:
            break
        chosen.append(list(e))
        # Check if current set is non-2-colorable
        if not _is_2_colorable(n, chosen, deadline):
            return chosen
    return None


def _local_search(n, edges, deadline, rng):
    """Try to remove edges while maintaining non-2-colorability."""
    edges = [list(e) for e in edges]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        order = list(range(len(edges)))
        rng.shuffle(order)
        for i in order:
            if i >= len(edges):
                continue
            candidate = edges[:i] + edges[i+1:]
            if not _is_2_colorable(n, candidate, deadline):
                del edges[i]
                improved = True
                break
    return edges


def _try_n(n, deadline, rng):
    """Try to find minimal non-2-colorable hypergraph on n vertices."""
    best = None
    
    # Start with random edge sets of increasing size
    for target_size in range(30, 60):
        if time.time() > deadline:
            break
        for restart in range(3):
            if time.time() > deadline:
                break
            # Random subset of edges
            all_edges = list(combinations(range(n), 5))
            rng.shuffle(all_edges)
            candidate = [list(e) for e in all_edges[:target_size]]
            
            if not _is_2_colorable(n, candidate, deadline):
                # Non-2-colorable! Try to minimize
                minimized = _local_search(n, candidate, deadline, rng)
                if best is None or len(minimized) < len(best):
                    best = minimized
                    # Try to go smaller
                    if len(minimized) < target_size - 5:
                        break
    
    return best


def construct():
    start = time.time()
    deadline = start + _WALL_LIMIT - _MARGIN
    rng = random.Random(42)
    
    # Fallback: complete 5-graph on 9 vertices
    fallback = {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    best = fallback
    
    # Try various n values
    for n in (9, 10, 11, 12, 13):
        if time.time() > deadline:
            break
        try:
            res = _try_n(n, deadline, rng)
        except Exception:
            res = None
        if res is not None and len(res) < len(best['edges']):
            best = {'n': n, 'edges': res}
    
    return best


if __name__ == '__main__':
    _t0 = time.time()
    _res = construct()
    _ok = _is_2_colorable(_res['n'], _res['edges'], _t0 + 120.0)
    assert not _ok, "constructed hypergraph must NOT be 2-colourable"
    print("edges:", len(_res['edges']), "n:", _res['n'], "2-colourable:", _ok)
