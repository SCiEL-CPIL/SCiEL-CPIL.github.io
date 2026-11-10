"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _is_2_colorable(n, edge_bitmasks):
    for coloring in range(1 << n):
        valid = True
        for em in edge_bitmasks:
            red = 0
            blue = 0
            for v in range(n):
                if (em >> v) & 1:
                    if (coloring >> v) & 1:
                        blue += 1
                    else:
                        red += 1
            if red == 5 or blue == 5:
                valid = False
                break
        if valid:
            return True
    return False

def construct():
    deadline = time.time() + 140
    n = 9
    all_edges = list(combinations(range(n), 5))
    edge_bitmasks = []
    for e in all_edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_bitmasks.append(mask)
    current_edges = list(range(len(all_edges)))
    for _pass in range(5):
        if time.time() > deadline:
            break
        i = 0
        while i < len(current_edges) and time.time() < deadline:
            edge_idx = current_edges[i]
            test_edges = [edge_bitmasks[e] for e in current_edges if e != edge_idx]
            if not _is_2_colorable(n, test_edges):
                current_edges = [e for e in current_edges if e != edge_idx]
            else:
                i += 1
    result_edges = [list(all_edges[e]) for e in current_edges]
    return {'n': n, 'edges': result_edges}
# EVOLVE-BLOCK-END
