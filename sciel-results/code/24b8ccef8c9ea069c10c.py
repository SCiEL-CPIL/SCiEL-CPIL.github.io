"""Rewritten program: search for a small 5-uniform hypergraph without Property B."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

START_TIME = time.time()
TIME_LIMIT = 115.0  # seconds

def time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def check_2_colorable(n, edges):
    """
    Check if the hypergraph is 2-colorable using backtracking with unit propagation.
    Returns True if 2-colorable (has Property B), False if not.
    """
    if not edges:
        return True

    # Preprocess
    unique_edges = []
    seen = set()
    for e in edges:
        e_sorted = tuple(sorted(e))
        if e_sorted not in seen:
            seen.add(e_sorted)
            unique_edges.append(list(e_sorted))
    edges = unique_edges
    m = len(edges)
    if m == 0:
        return True

    # Build adjacency: for each vertex, list of edge indices containing it
    vert_to_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vert_to_edges[v].append(i)

    def propagate(colors):
        """
        Apply unit propagation. Returns True if no contradiction, False otherwise.
        colors: list of -1, 0, or 1
        """
        changed = True
        while changed:
            changed = False
            for e in edges:
                counts = [0, 0]
                unassigned = []
                for v in e:
                    if colors[v] == -1:
                        unassigned.append(v)
                    else:
                        counts[colors[v]] += 1

                if len(unassigned) == 0:
                    if counts[0] == 5 or counts[1] == 5:
                        return False

                if counts[0] == 4 and len(unassigned) == 1:
                    colors[unassigned[0]] = 1
                    changed = True
                elif counts[1] == 4 and len(unassigned) == 1:
                    colors[unassigned[0]] = 0
                    changed = True
        return True

    def backtrack(colors):
        # Find most constrained unassigned vertex
        best_v = -1
        best_count = -1
        for v in range(n):
            if colors[v] == -1:
                # Count how many edges are "threatening" (4 of one color, 1 unassigned)
                threat = 0
                for ei in vert_to_edges[v]:
                    e = edges[ei]
                    c0 = sum(1 for u in e if colors[u] == 0)
                    c1 = sum(1 for u in e if colors[u] == 1)
                    un = sum(1 for u in e if colors[u] == -1)
                    if c0 + c1 == 4 and un == 1:
                        threat += 1
                if threat > best_count:
                    best_count = threat
                    best_v = v
                    if threat > 0:
                        break  # Good enough

        if best_v == -1:
            # All assigned, verify
            for e in edges:
                c = colors[e[0]]
                mono = True
                for u in e[1:]:
                    if colors[u] != c:
                        mono = False
                        break
                if mono:
                    return False
            return True

        # Try 0 then 1
        for val in [0, 1]:
            backup = colors.copy()
            colors[best_v] = val
            if propagate(colors):
                if backtrack(colors):
                    return True
            colors[:] = backup

        return False

    colors = [-1] * n
    colors[0] = 0  # Symmetry breaking
    if not propagate(colors):
        return False
    return backtrack(colors)

def minimize_edges(n, edges):
    """Try to remove edges while maintaining non-2-colorability."""
    current = list(edges)
    improved = True
    while improved and time_left() > 3:
        improved = False
        idxs = list(range(len(current)))
        random.shuffle(idxs)
        for i in idxs:
            if time_left() < 3:
                break
            test = current[:i] + current[i+1:]
            if not check_2_colorable(n, test):
                current = test
                improved = True
                break
    return current

def search_small_hypergraph():
    best_edges = None
    best_n = 9
    best_m = 126

    # Strategy: Start with dense hypergraphs, check if non-2-colorable, then minimize.
    # Try various n values and random subsets of edges.
    for n in range(9, 16):
        if time_left() < 5:
            break

        all_edges = list(combinations(range(n), 5))
        total = len(all_edges)

        # Try several random subsets of decreasing size
        for target_m in [min(total, 60), min(total, 45), min(total, 35), min(total, 30)]:
            if time_left() < 5:
                break

            for trial in range(200):
                if time_left() < 3:
                    break

                # Randomly pick target_m edges
                random.shuffle(all_edges)
                edges = [list(e) for e in all_edges[:target_m]]

                if not check_2_colorable(n, edges):
                    # Non-2-colorable! Minimize
                    minimized = minimize_edges(n, edges)
                    if best_edges is None or len(minimized) < best_m:
                        best_edges = minimized
                        best_n = n
                        best_m = len(minimized)
                    if best_m <= 30:
                        break
            if best_m <= 30:
                break

    if best_edges is not None:
        return {'n': best_n, 'edges': best_edges}

    # Fallback: complete graph on 9 vertices
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}

def construct():
    try:
        result = search_small_hypergraph()
        return result
    except Exception:
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}

# EVOLVE-BLOCK-END
