import random
import time
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Returns True if the hypergraph is 2-colorable, False otherwise.
    Uses backtracking with unit propagation.
    """
    colors = [-1] * n
    vertex_edges = [[] for _ in range(n)]
    for i, edge in enumerate(edges):
        for v in edge:
            vertex_edges[v].append(i)
    count_0 = [0] * len(edges)
    count_1 = [0] * len(edges)
    
    def assign(idx, color, c, c0, c1):
        if c[idx] != -1:
            return c[idx] == color
        c[idx] = color
        for e_idx in vertex_edges[idx]:
            if color == 0:
                c0[e_idx] += 1
                if c0[e_idx] == 5:
                    return False
            else:
                c1[e_idx] += 1
                if c1[e_idx] == 5:
                    return False
            if c0[e_idx] == 4:
                for v in edges[e_idx]:
                    if c[v] == -1:
                        if not assign(v, 1, c, c0, c1):
                            return False
            elif c1[e_idx] == 4:
                for v in edges[e_idx]:
                    if c[v] == -1:
                        if not assign(v, 0, c, c0, c1):
                            return False
        return True
    
    def solve(state_colors, state_count_0, state_count_1):
        idx = -1
        for i in range(n):
            if state_colors[i] == -1:
                idx = i
                break
        if idx == -1:
            return True
        new_c = state_colors.copy()
        new_c0 = state_count_0.copy()
        new_c1 = state_count_1.copy()
        if assign(idx, 0, new_c, new_c0, new_c1):
            if solve(new_c, new_c0, new_c1):
                return True
        new_c = state_colors.copy()
        new_c0 = state_count_0.copy()
        new_c1 = state_count_1.copy()
        if assign(idx, 1, new_c, new_c0, new_c1):
            if solve(new_c, new_c0, new_c1):
                return True
        return False

    return solve([-1]*n, [0]*len(edges), [0]*len(edges))

def construct():
    best_edges = None
    best_count = 1000
    start_time = time.time()
    
    for n in range(9, 13):
        if time.time() - start_time > 100:
            break
        all_possible_edges = list(combinations(range(n), 5))
        num_edges = len(all_possible_edges)
        
        # Enumerate colorings (fix vertex 0 to 0 to break symmetry)
        colorings = []
        for mask in range(1 << (n-1)):
            c = [0] * n
            c[0] = 0
            for i in range(1, n):
                c[i] = (mask >> (i-1)) & 1
            colorings.append(c)
        
        num_colorings = len(colorings)
        
        # For each edge, determine which colorings it violates (makes monochromatic)
        edge_hits = []
        for e in all_possible_edges:
            hits = set()
            for ci, c in enumerate(colorings):
                colors_in_edge = [c[v] for v in e]
                if len(set(colors_in_edge)) == 1:
                    hits.add(ci)
            edge_hits.append(hits)
        
        # Greedy set cover: pick edges that hit the most uncovered colorings
        all_indices = set(range(num_colorings))
        covered = set()
        selected = []
        remaining = all_indices.copy()
        
        # Sort edges by hit count descending for greedy
        edge_order = sorted(range(num_edges), key=lambda ei: len(edge_hits[ei]), reverse=True)
        
        for ei in edge_order:
            if not remaining:
                break
            if edge_hits[ei] & remaining:
                selected.append(ei)
                covered.update(edge_hits[ei])
                remaining -= edge_hits[ei]
        
        # Local search: try to remove redundant edges
        improved = True
        while improved:
            improved = False
            for i in range(len(selected)):
                test_selected = selected[:i] + selected[i+1:]
                test_covered = set()
                for ej in test_selected:
                    test_covered.update(edge_hits[ej])
                if test_covered == all_indices:
                    selected = test_selected
                    improved = True
                    break
        
        # Try swapping: replace one edge with another to enable further removal
        for _ in range(100):  # limited iterations
            best_swap = None
            best_size = len(selected)
            for i in range(len(selected)):
                for j in range(num_edges):
                    if j in selected:
                        continue
                    test_selected = selected[:i] + [j] + selected[i+1:]
                    test_covered = set()
                    for ej in test_selected:
                        test_covered.update(edge_hits[ej])
                    if test_covered != all_indices:
                        continue
                    # Try to remove from test_selected
                    for k in range(len(test_selected)):
                        if test_selected[k] == j:
                            continue
                        test2 = test_selected[:k] + test_selected[k+1:]
                        test2_covered = set()
                        for ej in test2:
                            test2_covered.update(edge_hits[ej])
                        if test2_covered == all_indices and len(test2) < best_size:
                            best_size = len(test2)
                            best_swap = test2
            if best_swap:
                selected = best_swap
                # Remove redundant again
                improved = True
                while improved:
                    improved = False
                    for i in range(len(selected)):
                        test_selected = selected[:i] + selected[i+1:]
                        test_covered = set()
                        for ej in test_selected:
                            test_covered.update(edge_hits[ej])
                        if test_covered == all_indices:
                            selected = test_selected
                            improved = True
                            break
            else:
                break
        
        if len(selected) < best_count:
            best_count = len(selected)
            best_edges = [list(all_possible_edges[ei]) for ei in sorted(selected)]
            if not is_2_colorable(n, best_edges):
                break
            else:
                best_edges = None
                best_count = 1000
    
    if best_edges:
        return {'n': n, 'edges': best_edges}
    else:
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
