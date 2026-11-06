import time
import random
from itertools import combinations
from collections import Counter, defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    n = 20
    num_colors = 20
    vertices = list(range(n))
    
    all_4cycles = []
    for quad in combinations(vertices, 4):
        a, b, c, d = quad
        cycles = [
            ((a,b), (b,c), (c,d), (d,a)),
            ((a,b), (b,d), (d,c), (c,a)),
            ((a,c), (c,b), (b,d), (d,a)),
        ]
        for cyc in cycles:
            norm = tuple(sorted(tuple(sorted(e)) for e in cyc))
            all_4cycles.append(norm)
    
    all_4cycles = list(set(all_4cycles))
    num_cycles = len(all_4cycles)
    
    all_edges = list(combinations(vertices, 2))
    num_edges = len(all_edges)
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    
    cycle_edges = []
    for cyc in all_4cycles:
        cycle_edges.append([edge_to_idx[tuple(sorted(e))] for e in cyc])
    
    edge_cycles = defaultdict(list)
    for ci, cycle in enumerate(cycle_edges):
        for ei in cycle:
            edge_cycles[ei].append(ci)
    
    edge_present = [True] * num_edges
    edge_color = [random.randint(0, num_colors-1) for _ in range(num_edges)]
    
    cycle_violated = [False] * num_cycles
    num_violations = 0
    
    def is_violated(ci):
        cycle = cycle_edges[ci]
        if not all(edge_present[i] for i in cycle):
            return False
        c0 = edge_color[cycle[0]]
        c1 = edge_color[cycle[1]]
        c2 = edge_color[cycle[2]]
        c3 = edge_color[cycle[3]]
        return not (c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3)
    
    for ci in range(num_cycles):
        if is_violated(ci):
            cycle_violated[ci] = True
            num_violations += 1
    
    def update_cycle(ci):
        nonlocal num_violations
        was = cycle_violated[ci]
        is_v = is_violated(ci)
        cycle_violated[ci] = is_v
        if is_v and not was:
            num_violations += 1
        elif not is_v and was:
            num_violations -= 1
    
    def recolor(ei, new_c):
        nonlocal num_violations
        if edge_color[ei] == new_c:
            return
        edge_color[ei] = new_c
        for ci in edge_cycles[ei]:
            update_cycle(ci)
    
    def remove_edge(ei):
        nonlocal num_violations
        if not edge_present[ei]:
            return
        edge_present[ei] = False
        for ci in edge_cycles[ei]:
            update_cycle(ci)
    
    # Phase 1: Local search
    max_iter = 200000
    for it in range(max_iter):
        if time.time() > deadline - 20:
            break
        if num_violations == 0:
            break
        
        ci = -1
        for _ in range(20):
            c = random.randint(0, num_cycles-1)
            if cycle_violated[c]:
                ci = c
                break
        if ci == -1:
            for i in range(num_cycles):
                if cycle_violated[i]:
                    ci = i
                    break
            if ci == -1:
                break
        
        cycle = cycle_edges[ci]
        if not all(edge_present[i] for i in cycle):
            continue
        
        cols = [edge_color[i] for i in cycle]
        cc = Counter(cols)
        dups = [c for c, cnt in cc.items() if cnt > 1]
        if not dups:
            continue
        
        dup_c = random.choice(dups)
        edges_dup = [i for i in cycle if edge_color[i] == dup_c]
        ei = random.choice(edges_dup)
        
        old_c = edge_color[ei]
        best_c = -1
        best_delta = 1
        for _ in range(8):
            nc = random.randint(0, num_colors-1)
            if nc == old_c:
                continue
            edge_color[ei] = nc
            delta = 0
            for cj in edge_cycles[ei]:
                was = cycle_violated[cj]
                is_v = is_violated(cj)
                if is_v and not was:
                    delta += 1
                elif not is_v and was:
                    delta -= 1
            edge_color[ei] = old_c
            if delta < best_delta:
                best_delta = delta
                best_c = nc
        
        if best_c >= 0 and best_delta < 0:
            recolor(ei, best_c)
    
    if num_violations == 0:
        result = []
        for i, e in enumerate(all_edges):
            if edge_present[i]:
                result.append([e[0], e[1], edge_color[i]])
        return result
    
    # Phase 2: Remove edges
    while time.time() < deadline - 5:
        if num_violations == 0:
            break
        
        violated_cis = [ci for ci in range(num_cycles) if cycle_violated[ci]]
        if not violated_cis:
            break
        
        evc = [0] * num_edges
        for ci in violated_cis:
            for ei in cycle_edges[ci]:
                if edge_present[ei]:
                    evc[ei] += 1
        
        max_c = 0
        max_ei = -1
        for ei in range(num_edges):
            if edge_present[ei] and evc[ei] > max_c:
                max_c = evc[ei]
                max_ei = ei
        
        if max_ei == -1:
            present = [i for i in range(num_edges) if edge_present[i]]
            if not present:
                break
            max_ei = random.choice(present)
        
        remove_edge(max_ei)
        
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            if num_violations == 0:
                break
            ci = -1
            for _ in range(15):
                c = random.randint(0, num_cycles-1)
                if cycle_violated[c]:
                    ci = c
                    break
            if ci == -1:
                for i in range(num_cycles):
                    if cycle_violated[i]:
                        ci = i
                        break
                if ci == -1:
                    break
            cycle = cycle_edges[ci]
            if not all(edge_present[i] for i in cycle):
                continue
            cols = [edge_color[i] for i in cycle]
            cc = Counter(cols)
            dups = [c for c, cnt in cc.items() if cnt > 1]
            if not dups:
                continue
            dup_c = random.choice(dups)
            edges_dup = [i for i in cycle if edge_color[i] == dup_c]
            ei = random.choice(edges_dup)
            old_c = edge_color[ei]
            for _ in range(10):
                nc = random.randint(0, num_colors-1)
                if nc == old_c:
                    continue
                edge_color[ei] = nc
                delta = 0
                for cj in edge_cycles[ei]:
                    was = cycle_violated[cj]
                    is_v = is_violated(cj)
                    if is_v and not was:
                        delta += 1
                    elif not is_v and was:
                        delta -= 1
                if delta < 0:
                    break
                edge_color[ei] = old_c
            else:
                continue
    
    result = []
    for i, e in enumerate(all_edges):
        if edge_present[i]:
            result.append([e[0], e[1], edge_color[i]])
    return result
