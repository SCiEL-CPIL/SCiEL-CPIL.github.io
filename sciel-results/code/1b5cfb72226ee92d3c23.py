import time
import random
import itertools
import sys
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # Phase 1: Get a valid n=17 coloring using Z_17 + local refinement
    n17 = 17
    r17 = _z17_refine(deadline - 60.0)
    
    if r17 is None:
        r17 = _sa_search(17, deadline - 50.0, 1)
    
    if r17 is None:
        n_small = 9
        colors_small = [min(j - i, n_small - j + i) - 1 for i in range(n_small) for j in range(i + 1, n_small)]
        return {'n': n_small, 'colors': colors_small}
    
    best = r17
    
    # Phase 2: Try to extend to n=18 with constraint propagation + backtracking
    if time.time() < deadline - 15.0:
        r18 = _extend_with_propagation(17, r17['colors'], deadline - 5.0)
        if r18 is not None:
            return r18
    
    # Phase 3: If n=18 failed, try multiple SA restarts for n=18
    if time.time() < deadline - 10.0:
        r18b = _sa_search(18, deadline - 5.0, 42)
        if r18b is not None:
            return r18b
    
    return best


def _z17_refine(deadline):
    """Z_17 structured init + QUBO-inspired penalty-guided local refinement."""
    n = 17
    num_edges = n * (n - 1) // 2
    
    # Build edge map
    edge_map = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_map[(i, j)] = idx
            idx += 1
    
    # Precompute all C6 cycles
    cycles = []
    edge_to_cycles = {e: [] for e in range(num_edges)}
    for subset in itertools.combinations(range(n), 6):
        seen = set()
        for perm in itertools.permutations(subset):
            mn = min(perm)
            mi = perm.index(mn)
            rot = perm[mi:] + perm[:mi]
            if rot[1] > rot[-1]:
                rot = (rot[0],) + rot[:0:-1]
            if rot in seen:
                continue
            seen.add(rot)
            edges = []
            for k in range(6):
                u, v = rot[k], rot[(k + 1) % 6]
                a, b = min(u, v), max(u, v)
                edges.append(edge_map[(a, b)])
            cid = len(cycles)
            cycles.append(edges)
            for e in edges:
                edge_to_cycles[e].append(cid)
    
    if not cycles:
        return {'n': n, 'colors': [0] * num_edges}
    
    # Z_17 initialization
    color_of_diff = {}
    powers = [1, 3, 9, 10, 13, 5, 15, 11, 16, 14, 8, 7, 4, 12, 2, 6]
    for k, d in enumerate(powers):
        c = k // 4
        color_of_diff[d] = c
        color_of_diff[17 - d] = c
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            if d in color_of_diff:
                colors.append(color_of_diff[d])
            else:
                colors.append(0)
    
    # Compute initial violation count
    cycle_counts = [[0, 0, 0, 0] for _ in range(len(cycles))]
    total_viol = 0
    for c_id in range(len(cycles)):
        for e in cycles[c_id]:
            cycle_counts[c_id][colors[e]] += 1
        for c in range(4):
            if cycle_counts[c_id][c] == 6:
                total_viol += 1
                break
    
    if total_viol == 0:
        return {'n': n, 'colors': colors}
    
    # QUBO-inspired penalty-guided local refinement
    # Penalty for edge e if coloring it with color c creates a monochromatic C6
    # We use a greedy + iterated local search approach
    random.seed(0)
    
    best_colors = list(colors)
    best_viol = total_viol
    
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        if best_viol == 0:
            return {'n': n, 'colors': best_colors}
        
        # Greedy pass: for each edge, pick the color that minimizes penalty
        colors = list(best_colors)
        cycle_counts = [[0, 0, 0, 0] for _ in range(len(cycles))]
        for c_id in range(len(cycles)):
            for e in cycles[c_id]:
                cycle_counts[c_id][colors[e]] += 1
        
        # Compute penalty for each edge-color pair
        # penalty(e, c) = number of cycles where e is colored c and all other 5 edges are also c
        # We approximate by checking if coloring e with c would complete a C6
        edge_order = list(range(num_edges))
        random.shuffle(edge_order)
        
        for e in edge_order:
            if time.time() > deadline:
                break
            best_c = colors[e]
            best_pen = float('inf')
            for c in range(4):
                pen = 0
                for c_id in edge_to_cycles[e]:
                    cc = cycle_counts[c_id]
                    # If we color e with c, how many other edges in this cycle are c?
                    # If 5 others are c, then we'd have a monochromatic C6
                    others_c = cc[c] - (1 if colors[e] == c else 0)
                    if others_c == 5:
                        pen += 1
                if pen < best_pen:
                    best_pen = pen
                    best_c = c
            if best_c != colors[e]:
                oc = colors[e]
                for c_id in edge_to_cycles[e]:
                    cycle_counts[c_id][oc] -= 1
                    cycle_counts[c_id][best_c] += 1
                colors[e] = best_c
        
        # Count violations
        total_viol = 0
        for c_id in range(len(cycles)):
            cc = cycle_counts[c_id]
            for c in range(4):
                if cc[c] == 6:
                    total_viol += 1
                    break
        
        if total_viol < best_viol:
            best_viol = total_viol
            best_colors = list(colors)
        
        if best_viol == 0:
            return {'n': n, 'colors': best_colors}
        
        # Perturbation: randomly flip some edges
        num_flips = max(1, num_edges // 20)
        for _ in range(num_flips):
            e = random.randint(0, num_edges - 1)
            nc = random.randint(0, 3)
            oc = colors[e]
            if nc != oc:
                for c_id in edge_to_cycles[e]:
                    cycle_counts[c_id][oc] -= 1
                    cycle_counts[c_id][nc] += 1
                colors[e] = nc
        
        # Recompute violation
        total_viol = 0
        for c_id in range(len(cycles)):
            cc = cycle_counts[c_id]
            for c in range(4):
                if cc[c] == 6:
                    total_viol += 1
                    break
        
        if total_viol < best_viol:
            best_viol = total_viol
            best_colors = list(colors)
    
    if best_viol == 0:
        return {'n': n, 'colors': best_colors}
    return None


def _sa_search(n, deadline, seed=0):
    """Simulated annealing to find a C6-free 4-coloring of K_n."""
    random.seed(seed)
    edge_map = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_map[(i, j)] = idx
            idx += 1
    num_edges = n * (n - 1) // 2
    
    cycles = []
    edge_to_cycles = {e: [] for e in range(num_edges)}
    for subset in itertools.combinations(range(n), 6):
        seen = set()
        for perm in itertools.permutations(subset):
            mn = min(perm)
            mi = perm.index(mn)
            rot = perm[mi:] + perm[:mi]
            if rot[1] > rot[-1]:
                rot = (rot[0],) + rot[:0:-1]
            if rot in seen:
                continue
            seen.add(rot)
            edges = []
            for k in range(6):
                u, v = rot[k], rot[(k + 1) % 6]
                a, b = min(u, v), max(u, v)
                edges.append(edge_map[(a, b)])
            cid = len(cycles)
            cycles.append(edges)
            for e in edges:
                edge_to_cycles[e].append(cid)
    
    if not cycles:
        return {'n': n, 'colors': [0] * num_edges}
    
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    cycle_counts = [[0, 0, 0, 0] for _ in range(len(cycles))]
    total_viol = 0
    for c_id in range(len(cycles)):
        for e in cycles[c_id]:
            cycle_counts[c_id][colors[e]] += 1
        for c in range(4):
            if cycle_counts[c_id][c] == 6:
                total_viol += 1
                break
    
    T = 2.0
    cooling = 0.9995
    min_T = 0.05
    
    while time.time() < deadline:
        if total_viol == 0:
            return {'n': n, 'colors': colors}
        e = random.randint(0, num_edges - 1)
        oc = colors[e]
        nc = random.randint(0, 3)
        if nc == oc:
            continue
        delta = 0
        for c_id in edge_to_cycles[e]:
            cc = cycle_counts[c_id]
            old_v = 1 if cc[oc] == 6 else 0
            cc[oc] -= 1
            cc[nc] += 1
            new_v = 1 if cc[nc] == 6 else 0
            delta += new_v - old_v
        if delta < 0 or random.random() < 2.718281828 ** (-delta / T):
            colors[e] = nc
            total_viol += delta
        else:
            for c_id in edge_to_cycles[e]:
                cc = cycle_counts[c_id]
                cc[oc] += 1
                cc[nc] -= 1
        T = max(min_T, T * cooling)
    
    if total_viol == 0:
        return {'n': n, 'colors': colors}
    return None


def _extend_with_propagation(n_old, old_colors, deadline):
    """Extend coloring to n_old+1 using constraint propagation + backtracking."""
    n_new = n_old + 1
    n_old_edges = n_old * (n_old - 1) // 2
    n_new_edges = n_new * (n_new - 1) // 2
    
    edge_map = {}
    idx = 0
    for i in range(n_new):
        for j in range(i + 1, n_new):
            edge_map[(i, j)] = idx
            idx += 1
    
    new_edge_indices = [edge_map[(j, n_new - 1)] for j in range(n_old)]
    new_edge_set = set(new_edge_indices)
    
    cycles = []
    edge_to_cycles = {e: [] for e in range(n_new_edges)}
    for subset in itertools.combinations(range(n_new), 6):
        seen = set()
        for perm in itertools.permutations(subset):
            mn = min(perm)
            mi = perm.index(mn)
            rot = perm[mi:] + perm[:mi]
            if rot[1] > rot[-1]:
                rot = (rot[0],) + rot[:0:-1]
            if rot in seen:
                continue
            seen.add(rot)
            edges = []
            for k in range(6):
                u, v = rot[k], rot[(k + 1) % 6]
                a, b = min(u, v), max(u, v)
                edges.append(edge_map[(a, b)])
            cid = len(cycles)
            cycles.append(edges)
            for e in edges:
                edge_to_cycles[e].append(cid)
    
    colors = list(old_colors)
    for e in new_edge_indices:
        colors[e] = -1
    
    # Domain for each new edge
    domains = {e: set([0, 1, 2, 3]) for e in new_edge_indices}
    
    # Constraint propagation: for each new edge, remove colors that would create a C6
    # A color c is invalid for edge e if there exists a cycle containing e where
    # all other 5 edges are already colored c
    def propagate():
        changed = True
        while changed:
            changed = False
            for e in new_edge_indices:
                if len(domains[e]) == 0:
                    return False
                new_domain = set()
                for c in domains[e]:
                    valid = True
                    for c_id in edge_to_cycles[e]:
                        cyc = cycles[c_id]
                        # Check if all other edges in this cycle are colored c
                        all_c = True
                        for edge in cyc:
                            if edge == e:
                                continue
                            if colors[edge] != c:
                                all_c = False
                                break
                            # If it's a new edge, check its domain
                            if edge in new_edge_set:
                                if c not in domains[edge]:
                                    all_c = False
                                    break
                        if all_c:
                            valid = False
                            break
                    if valid:
                        new_domain.add(c)
                if len(new_domain) < len(domains[e]):
                    domains[e] = new_domain
                    changed = True
                    if len(domains[e]) == 0:
                        return False
        return True
    
    if not propagate():
        return None
    
    # Order new edges by domain size (MRV heuristic)
    ordered = sorted(new_edge_indices, key=lambda e: len(domains[e]))
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == len(ordered):
            return True
        e = ordered[idx]
        for c in sorted(domains[e]):
            colors[e] = c
            # Check if this assignment is consistent
            consistent = True
            for c_id in edge_to_cycles[e]:
                cyc = cycles[c_id]
                for edge in cyc:
                    if edge == e:
                        continue
                    if colors[edge] != -1 and colors[edge] != c:
                        continue
                    if colors[edge] != -1 and colors[edge] == c:
                        # Check if all edges in this cycle are now c
                        all_c = True
                        for edge2 in cyc:
                            if edge2 == e:
                                continue
                            if colors[edge2] != c:
                                all_c = False
                                break
                        if all_c:
                            consistent = False
                            break
                if not consistent:
                    break
            if consistent:
                # Propagate
                saved_domains = {e2: set(d2) for e2, d2 in domains.items()}
                domains[e] = {c}
                if propagate():
                    if backtrack(idx + 1):
                        return True
                # Restore domains
                domains.clear()
                domains.update(saved_domains)
            colors[e] = -1
        return False
    
    if backtrack(0):
        return {'n': n_new, 'colors': colors}
    return None


# EVOLVE-BLOCK-END
