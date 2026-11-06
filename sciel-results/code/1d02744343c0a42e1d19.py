"""Dense 4-cycle-rainbow edge-coloured graph on 20 vertices."""
import time
import random
from itertools import combinations

N = 20
DEADLINE_MARGIN = 5.0  # seconds to spare before the 150s wall-clock kill

def _all_4_cycles():
    """Generate all 4-cycles (sets of 4 edges) in K_n.
    A 4-cycle is determined by 4 vertices; there are 3 ways to pair them into a cycle.
    For vertices a<b<c<d, the three 4-cycles are:
      (a,b,c,d), (a,b,d,c), (a,c,b,d)
    Each cycle is a frozenset of 4 (u,v) edges with u<v.
    """
    cycles = []
    for a, b, c, d in combinations(range(N), 4):
        # Three ways to form a 4-cycle from 4 vertices
        for perm in [(a, b, c, d), (a, b, d, c), (a, c, b, d)]:
            x, y, z, w = perm
            edges = set()
            for u, v in [(x, y), (y, z), (z, w), (w, x)]:
                edges.add((min(u, v), max(u, v)))
            cycles.append(frozenset(edges))
    return cycles

def _is_valid(edges_set, colors, cycles):
    """Check if the current edge set + coloring is valid (all 4-cycles are rainbow)."""
    for cyc in cycles:
        # Check all 4 edges exist
        if not all(e in edges_set for e in cyc):
            continue
        cols = [colors[e] for e in cyc]
        if len(set(cols)) < 4:
            return False
    return True

def _count_bad_cycles(edges_set, colors, cycles):
    """Return a dict: edge -> number of bad 4-cycles it participates in."""
    bad_count = {}
    for cyc in cycles:
        if not all(e in edges_set for e in cyc):
            continue
        cols = [colors[e] for e in cyc]
        if len(set(cols)) < 4:
            for e in cyc:
                bad_count[e] = bad_count.get(e, 0) + 1
    return bad_count

def _try_greedy_remove(initial_colors, cycles, time_limit):
    """Greedy: start with K_n and given coloring, remove edges that break rainbow property."""
    all_edges = frozenset(combinations(range(N), 2))
    edges_set = set(all_edges)
    colors = dict(initial_colors)
    
    t_start = time.time()
    
    while True:
        if time.time() - t_start > time_limit:
            break
        bad = _count_bad_cycles(edges_set, colors, cycles)
        if not bad:
            break
        # Remove the edge in the most bad cycles
        worst_edge = max(bad, key=lambda e: bad[e])
        edges_set.remove(worst_edge)
        del colors[worst_edge]
    
    result = [[u, v, colors[(u, v)]] for (u, v) in sorted(edges_set)]
    return result

def construct():
    t0 = time.time()
    deadline = 150.0 - DEADLINE_MARGIN
    
    cycles = _all_4_cycles()
    all_edges = list(combinations(range(N), 2))
    
    best_edges = 0
    best_result = []
    
    # Try several deterministic coloring schemes
    colorings = []
    
    # Scheme 1: (u + v) mod 20
    c1 = {(u, v): (u + v) % 20 for u, v in all_edges}
    colorings.append(c1)
    
    # Scheme 2: (u - v) mod 20
    c2 = {(u, v): (u - v) % 20 for u, v in all_edges}
    colorings.append(c2)
    
    # Scheme 3: (u * v) mod 20
    c3 = {(u, v): (u * v) % 20 for u, v in all_edges}
    colorings.append(c3)
    
    # Scheme 4: (u + 2*v) mod 20
    c4 = {(u, v): (u + 2 * v) % 20 for u, v in all_edges}
    colorings.append(c4)
    
    # Scheme 5: (2*u + v) mod 20
    c5 = {(u, v): (2 * u + v) % 20 for u, v in all_edges}
    colorings.append(c5)
    
    # Random colorings
    for seed in range(20):
        random.seed(seed)
        cr = {}
        for e in all_edges:
            cr[e] = random.randint(0, 19)
        colorings.append(cr)
    
    remaining_time = lambda: deadline - time.time()
    
    for i, coloring in enumerate(colorings):
        if remaining_time() < 2.0:
            break
        # Allocate time per attempt, shrinking as we go
        time_left = remaining_time()
        attempt_time = min(time_left * 0.3, max(0.5, time_left / (len(colorings) - i)))
        
        result = _try_greedy_remove(coloring, cycles, attempt_time)
        if len(result) > best_edges:
            best_edges = len(result)
            best_result = result
    
    # If we have time left, do a final pass: try to add back edges that were removed
    if remaining_time() > 3.0 and best_result:
        current_edges = set()
        current_colors = {}
        for u, v, c in best_result:
            e = (min(u, v), max(u, v))
            current_edges.add(e)
            current_colors[e] = c
        
        # Try adding back each missing edge with a color that doesn't create a bad cycle
        missing = set(all_edges) - current_edges
        t_add_start = time.time()
        for e in missing:
            if time.time() - t_add_start > remaining_time() - 2.0:
                break
            u, v = e
            for c in range(20):
                # Tentatively add
                test_edges = current_edges | {e}
                test_colors = current_colors
                test_colors[e] = c
                # Check validity - this is expensive, only do it if promising
                # Quick check: only check cycles that include this edge
                bad = False
                for cyc in cycles:
                    if e not in cyc:
                        continue
                    if not all(ed in test_edges for ed in cyc):
                        continue
                    cols = [test_colors[ed] for ed in cyc]
                    if len(set(cols)) < 4:
                        bad = True
                        break
                if not bad:
                    current_edges.add(e)
                    current_colors[e] = c
                    break
        if len(current_edges) > best_edges:
            best_edges = len(current_edges)
            best_result = [[u, v, current_colors[(u, v)]] for (u, v) in sorted(current_edges)]
    
    return best_result if best_result else [[0, v, 0] for v in range(1, N)]
