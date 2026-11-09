import numpy as np
import time
import random
from itertools import combinations

def construct():
    n = 20
    deadline = time.time() + 115
    random.seed(42)
    np.random.seed(42)
    
    # Precompute all 4-cycles as (u, v, x, y) where u-v-x-y-u is a cycle
    # A 4-cycle is defined by 4 distinct vertices with edges u-v, v-x, x-y, y-u
    all_4cycles = []
    for u in range(n):
        for v in range(u+1, n):
            for x in range(n):
                if x == u or x == v:
                    continue
                for y in range(n):
                    if y == u or y == v or y == x:
                        continue
                    # Check that u-v-x-y-u forms a valid 4-cycle (all 4 edges possible)
                    # We need edges: (u,v), (v,x), (x,y), (y,u)
                    # To avoid duplicates, enforce ordering: u < v and u < y and v < x or similar
                    # Actually, let's just collect all and dedupe later
                    all_4cycles.append((u, v, x, y))
    
    # Deduplicate: a 4-cycle (u,v,x,y) is the same as (v,x,y,u), (x,y,u,v), (y,u,v,x)
    # and their reverses. Canonical form: sort the 4 vertices, then check the cycle structure
    seen = set()
    unique_4cycles = []
    for (u, v, x, y) in all_4cycles:
        # The cycle is u-v-x-y-u
        # Canonical: the set of 4 edges {(u,v),(v,x),(x,y),(y,u)}
        edges_in_cycle = frozenset([
            frozenset([u, v]),
            frozenset([v, x]),
            frozenset([x, y]),
            frozenset([y, u])
        ])
        if edges_in_cycle not in seen:
            seen.add(edges_in_cycle)
            unique_4cycles.append((u, v, x, y))
    
    # Store edge indices for each 4-cycle
    edge_index = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_index[(i, j)] = idx
            idx += 1
    total_edges = idx
    
    # For each 4-cycle, store the 4 edge indices
    cycle_edge_indices = []
    for (u, v, x, y) in unique_4cycles:
        e1 = edge_index.get((min(u,v), max(u,v)))
        e2 = edge_index.get((min(v,x), max(v,x)))
        e3 = edge_index.get((min(x,y), max(x,y)))
        e4 = edge_index.get((min(y,u), max(y,u)))
        if e1 is not None and e2 is not None and e3 is not None and e4 is not None:
            cycle_edge_indices.append((e1, e2, e3, e4))
    
    # Convert to numpy array for vectorized operations
    cycle_arr = np.array(cycle_edge_indices, dtype=np.int32) if cycle_edge_indices else np.empty((0,4), dtype=np.int32)
    num_cycles = len(cycle_arr)
    
    # Build adjacency list for each edge (which cycles it participates in)
    edge_to_cycles = [[] for _ in range(total_edges)]
    for ci, (e1, e2, e3, e4) in enumerate(cycle_arr):
        for e in (e1, e2, e3, e4):
            edge_to_cycles[e].append(ci)
    
    def is_valid(colors, present):
        """Check if the coloring is valid. colors: array of length total_edges, present: boolean array"""
        if num_cycles == 0:
            return True
        # For each cycle, check all 4 edges are present and have distinct colors
        for ci in range(num_cycles):
            e1, e2, e3, e4 = cycle_arr[ci]
            if not (present[e1] and present[e2] and present[e3] and present[e4]):
                continue
            c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
            if c1 == c2 or c1 == c3 or c1 == c4 or c2 == c3 or c2 == c4 or c3 == c4:
                return False
        return True
    
    def count_valid_colors_for_edge(colors, present, edge_idx):
        """Count how many colors are valid for the given edge (if it were present)"""
        forbidden = set()
        for ci in edge_to_cycles[edge_idx]:
            e1, e2, e3, e4 = cycle_arr[ci]
            # Check if all other 3 edges in this cycle are present
            other_edges = [e for e in (e1, e2, e3, e4) if e != edge_idx]
            if all(present[oe] for oe in other_edges):
                for oe in other_edges:
                    forbidden.add(colors[oe])
        return [c for c in range(n) if c not in forbidden]
    
    def greedy_build(order_seed):
        """Greedy build with given vertex ordering"""
        random.seed(order_seed)
        colors = np.zeros(total_edges, dtype=np.int32)
        present = np.zeros(total_edges, dtype=bool)
        
        vertices = list(range(n))
        random.shuffle(vertices)
        
        # Build edges in order of the vertex permutation
        for i in range(1, n):
            u = vertices[i]
            # Try to connect u to previous vertices
            candidates = list(range(i))
            random.shuffle(candidates)
            for j in candidates:
                v = vertices[j]
                eidx = edge_index[(min(u,v), max(u,v))]
                valid_colors = count_valid_colors_for_edge(colors, present, eidx)
                if valid_colors:
                    c = random.choice(valid_colors)
                    colors[eidx] = c
                    present[eidx] = True
        
        # Second pass: try to add more edges
        all_e = list(range(total_edges))
        random.shuffle(all_e)
        for eidx in all_e:
            if present[eidx]:
                continue
            valid_colors = count_valid_colors_for_edge(colors, present, eidx)
            if valid_colors:
                c = random.choice(valid_colors)
                colors[eidx] = c
                present[eidx] = True
        
        return colors, present
    
    def local_search(colors, present, time_limit=5.0):
        """Local search: try to add edges and recolor"""
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            improved = False
            
            # Try to add missing edges
            missing = [e for e in range(total_edges) if not present[e]]
            random.shuffle(missing)
            for eidx in missing:
                if time.time() - start_time > time_limit:
                    break
                valid_colors = count_valid_colors_for_edge(colors, present, eidx)
                if valid_colors:
                    c = random.choice(valid_colors)
                    colors[eidx] = c
                    present[eidx] = True
                    improved = True
            
            # Try recoloring existing edges to free up colors
            if not improved:
                present_edges = [e for e in range(total_edges) if present[e]]
                random.shuffle(present_edges)
                for eidx in present_edges[:50]:  # Limit for speed
                    if time.time() - start_time > time_limit:
                        break
                    old_c = colors[eidx]
                    for new_c in range(n):
                        if new_c == old_c:
                            continue
                        # Check if recoloring is valid
                        valid = True
                        for ci in edge_to_cycles[eidx]:
                            e1, e2, e3, e4 = cycle_arr[ci]
                            if not all(present[oe] for oe in (e1, e2, e3, e4)):
                                continue
                            new_colors = {e1: colors[e1], e2: colors[e2], e3: colors[e3], e4: colors[e4]}
                            new_colors[eidx] = new_c
                            cs = [new_colors[e1], new_colors[e2], new_colors[e3], new_colors[e4]]
                            if len(set(cs)) < 4:
                                valid = False
                                break
                        if valid:
                            colors[eidx] = new_c
                            improved = True
                            break
            
            if not improved:
                break
        
        return colors, present
    
    def remove_and_refill(colors, present, time_limit=3.0):
        """Remove a random edge and refill greedily to escape local minima"""
        start_time = time.time()
        present_edges = [e for e in range(total_edges) if present[e]]
        if not present_edges:
            return colors, present
        
        # Remove a random edge
        eidx = random.choice(present_edges)
        present[eidx] = False
        
        # Refill greedily
        all_e = list(range(total_edges))
        random.shuffle(all_e)
        for e in all_e:
            if time.time() - start_time > time_limit:
                break
            if present[e]:
                continue
            valid_colors = count_valid_colors_for_edge(colors, present, e)
            if valid_colors:
                c = random.choice(valid_colors)
                colors[e] = c
                present[e] = True
        
        return colors, present
    
    best_edges = 0
    best_colors = None
    best_present = None
    
    # Multiple restarts with different strategies
    for restart in range(200):
        if time.time() > deadline - 10:
            break
        
        if restart < 100:
            # Standard greedy build
            colors, present = greedy_build(restart * 7 + 1)
        elif restart < 150:
            # Greedy build then local search
            colors, present = greedy_build(restart * 13 + 3)
            colors, present = local_search(colors, present, time_limit=2.0)
        else:
            # Remove and refill strategy
            colors, present = greedy_build(restart * 17 + 5)
            colors, present = remove_and_refill(colors, present, time_limit=2.0)
            colors, present = local_search(colors, present, time_limit=2.0)
        
        cnt = np.sum(present)
        if cnt > best_edges:
            best_edges = cnt
            best_colors = colors.copy()
            best_present = present.copy()
        
        if best_edges > 160:
            # Try to improve further with local search
            if time.time() < deadline - 15:
                colors, present = local_search(best_colors, best_present, time_limit=3.0)
                cnt = np.sum(present)
                if cnt > best_edges:
                    best_edges = cnt
                    best_colors = colors.copy()
                    best_present = present.copy()
    
    # Final polish
    if time.time() < deadline - 5 and best_present is not None:
        best_colors, best_present = local_search(best_colors, best_present, time_limit=5.0)
        best_edges = np.sum(best_present)
    
    # Build result
    res = []
    if best_present is None:
        # Fallback: just add a star
        for v in range(1, n):
            res.append([0, v, v % n])
        return res
    
    for i in range(n):
        for j in range(i+1, n):
            eidx = edge_index[(i, j)]
            if best_present[eidx]:
                res.append([i, j, int(best_colors[eidx])])
    
    return res
