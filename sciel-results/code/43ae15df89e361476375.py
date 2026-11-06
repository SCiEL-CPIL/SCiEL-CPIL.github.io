import time
import random
import sys
import itertools
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin (evaluator kills at 150s)
    
    def get_fallback(n):
        colors = [0] * (n * (n - 1) // 2)
        return {'n': n, 'colors': colors}
    
    best_result = get_fallback(5)
    
    def precompute(n):
        edge_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[(i, j)] = idx
                idx += 1
        
        def edge_index(u, v):
            if u > v:
                u, v = v, u
            return edge_map[(u, v)]
        
        vertices = list(range(n))
        cycles = []
        edge_to_cycles = [[] for _ in range(n * (n - 1) // 2)]
        
        for combo in itertools.combinations(vertices, 6):
            v0 = combo[0]
            others = combo[1:]
            for perm in itertools.permutations(others):
                if perm[0] > perm[-1]:
                    continue
                
                path = [v0] + list(perm)
                cycle_edges = []
                for k in range(6):
                    u = path[k]
                    v = path[(k + 1) % 6]
                    e = edge_index(u, v)
                    cycle_edges.append(e)
                
                # Check for duplicate edges (invalid cycle)
                if len(set(cycle_edges)) != 6:
                    continue
                    
                cycle_id = len(cycles)
                cycles.append(cycle_edges)
                for e in cycle_edges:
                    edge_to_cycles[e].append(cycle_id)
        
        return edge_map, cycles, edge_to_cycles

    def search_n(n, time_budget):
        if time.time() > deadline - 5:
            return None
            
        try:
            edge_map, cycles, edge_to_cycles = precompute(n)
        except Exception:
            return None
            
        m = n * (n - 1) // 2
        num_cycles = len(cycles)
        
        if num_cycles == 0:
            colors = [0] * m
            return {'n': n, 'colors': colors}
        
        end_time = time.time() + time_budget
        
        while time.time() < end_time:
            if time.time() > deadline - 5:
                break
            
            colors = [random.randint(0, 3) for _ in range(m)]
            counts = [[0, 0, 0, 0] for _ in range(num_cycles)]
            bad_count = 0
            
            for cid, ce in enumerate(cycles):
                c_counts = [0, 0, 0, 0]
                for e in ce:
                    c_counts[colors[e]] += 1
                counts[cid] = c_counts
                if any(c == 6 for c in c_counts):
                    bad_count += 1
            
            if bad_count == 0:
                return {'n': n, 'colors': colors}
            
            best_colors = colors[:]
            best_bad = bad_count
            
            T = 3.0
            cool_rate = 0.9995
            iter_start = time.time()
            
            while time.time() < iter_start + 2.5:
                if time.time() > deadline - 5:
                    break
                
                e = random.randint(0, m - 1)
                c_old = colors[e]
                
                c_new = random.randint(0, 3)
                while c_new == c_old:
                    c_new = random.randint(0, 3)
                
                delta = 0
                for cid in edge_to_cycles[e]:
                    c_counts = counts[cid]
                    old_cnt = c_counts[c_old]
                    new_cnt = c_counts[c_new]
                    
                    if old_cnt == 6:
                        delta -= 1
                    if new_cnt == 5:
                        delta += 1
                    
                    c_counts[c_old] -= 1
                    c_counts[c_new] += 1
                
                accept = False
                if delta < 0:
                    accept = True
                else:
                    if delta > 0:
                        if delta / T > 50:
                            accept = False
                        else:
                            prob = math.exp(-delta / T)
                            if random.random() < prob:
                                accept = True
                
                if accept:
                    colors[e] = c_new
                    bad_count += delta
                    if bad_count < best_bad:
                        best_bad = bad_count
                        best_colors = colors[:]
                        if best_bad == 0:
                            return {'n': n, 'colors': best_colors}
                else:
                    for cid in edge_to_cycles[e]:
                        c_counts = counts[cid]
                        c_counts[c_old] += 1
                        c_counts[c_new] -= 1
                
                T *= cool_rate
                if T < 0.1:
                    T = 1.5
        
        return None

    def try_cyclic_n17(time_budget):
        """Exploit F_17 structure: color edge (i,j) by f[(j-i) mod 17] where f maps F_17\\{0} -> {0,1,2,3}.
        Monochromatic C6 in cyclic coloring corresponds to 6-term arithmetic progressions in F_17."""
        if time.time() > deadline - 5:
            return None
        
        n = 17
        p = 17
        m = n * (n - 1) // 2
        
        # Precompute edge indices for cyclic coloring
        edge_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[(i, j)] = idx
                idx += 1
        
        def edge_index(u, v):
            if u > v:
                u, v = v, u
            return edge_map[(u, v)]
        
        # Precompute all C6 cycles
        vertices = list(range(n))
        cycles = []
        edge_to_cycles = [[] for _ in range(m)]
        
        for combo in itertools.combinations(vertices, 6):
            v0 = combo[0]
            others = combo[1:]
            for perm in itertools.permutations(others):
                if perm[0] > perm[-1]:
                    continue
                
                path = [v0] + list(perm)
                cycle_edges = []
                valid = True
                for k in range(6):
                    u = path[k]
                    v = path[(k + 1) % 6]
                    e = edge_index(u, v)
                    cycle_edges.append(e)
                
                if len(set(cycle_edges)) != 6:
                    continue
                    
                cycle_id = len(cycles)
                cycles.append(cycle_edges)
                for e in cycle_edges:
                    edge_to_cycles[e].append(cycle_id)
        
        num_cycles = len(cycles)
        if num_cycles == 0:
            return {'n': n, 'colors': [0] * m}
        
        # For cyclic coloring: edge (i,j) has diff d = (j-i) mod 17 (or 17-(j-i) for i>j)
        # Actually for i<j, diff = j-i which is in {1,...,16}
        # The color of edge (i,j) = f[(j-i) mod 17] where we use the positive residue
        # But we need to handle: for i<j, diff = j-i in {1,...,16}
        # For the "other direction", diff' = 17 - (j-i)
        # In a cyclic coloring, we assign color based on the difference modulo p
        # Edge (i,j) with i<j: we can use diff = (j-i) % p, but this gives values 1..16
        # However, in F_p, the edge between i and j can be characterized by (j-i) mod p
        # Since i<j, j-i is in 1..16, and 17-(j-i) is also in 1..16
        # For a proper cyclic coloring, we need f[d] = f[p-d] for consistency? No,
        # actually the edge {i,j} has two directed differences: (j-i) and (i-j) = p-(j-i)
        # In the undirected edge, we just pick one. Let's use d = (j-i) % p for i<j.
        
        # So the coloring is determined by f: {1,...,16} -> {0,1,2,3}
        # Edge (i,j), i<j: color = f[j-i]
        
        # Precompute for each cycle, which differences it uses
        # A cycle v0,v1,...,v5 has edges (v0,v1),(v1,v2),...,(v5,v0)
        # For each edge (a,b) in the cycle, the difference is (b-a) % 17 if a<b, else 17-(a-b)
        # Wait, let me be more careful. For edge (a,b) with a<b, diff = b-a.
        # For edge (a,b) with a>b, diff = a-b. But in our cycle, we have undirected edges.
        # Let's just compute the diff for each edge in the cycle.
        
        cycle_diffs = []
        for ce in cycles:
            diffs = []
            for e_idx in ce:
                # Find which edge this is
                # We need to recover (i,j) from edge index
                # This is inefficient; let's precompute edge_to_pair
                pass
        
        # Better: precompute edge_to_pair
        edge_to_pair = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_to_pair[idx] = (i, j)
                idx += 1
        
        cycle_diffs = []
        for ce in cycles:
            diffs = []
            for e_idx in ce:
                i, j = edge_to_pair[e_idx]
                d = j - i  # always positive since i < j
                diffs.append(d)
            cycle_diffs.append(diffs)
        
        # Now, a cycle is monochromatic if all its edges have the same color.
        # Edge with diff d has color f[d].
        # So cycle is monochromatic if f[d1] = f[d2] = ... = f[d6] for the 6 diffs in the cycle.
        # Note: diffs can repeat in a cycle!
        
        # Objective: minimize number of monochromatic cycles.
        # f is a function from {1,...,16} to {0,1,2,3}, so 4^16 possible assignments.
        # We'll use local search on f.
        
        end_time = time.time() + time_budget
        
        while time.time() < end_time:
            if time.time() > deadline - 5:
                break
            
            # Random initialization of f
            f = [random.randint(0, 3) for _ in range(16)]  # f[d-1] = color for diff d
            
            def count_mono_cycles(f):
                mono_count = 0
                for diffs in cycle_diffs:
                    # Check if all edges in this cycle have the same color
                    c0 = f[diffs[0] - 1]
                    all_same = True
                    for d in diffs[1:]:
                        if f[d - 1] != c0:
                            all_same = False
                            break
                    if all_same:
                        mono_count += 1
                return mono_count
            
            bad = count_mono_cycles(f)
            
            if bad == 0:
                # Construct the coloring
                colors = [0] * m
                for i in range(n):
                    for j in range(i + 1, n):
                        d = j - i
                        colors[edge_index(i, j)] = f[d - 1]
                return {'n': n, 'colors': colors}
            
            best_f = f[:]
            best_bad = bad
            
            T = 2.0
            cool_rate = 0.9997
            iter_start = time.time()
            
            while time.time() < iter_start + 2.0:
                if time.time() > deadline - 5:
                    break
                
                # Pick a random difference to change
                d = random.randint(1, 16)
                old_c = f[d - 1]
                new_c = random.randint(0, 3)
                while new_c == old_c:
                    new_c = random.randint(0, 3)
                
                # Compute delta: how many cycles become/stop being monochromatic
                delta = 0
                for diffs in cycle_diffs:
                    # Count how many edges in this cycle have diff d
                    # and check if changing f[d] affects monochromaticity
                    c_old_counts = {}
                    c_new_counts = {}
                    for dd in diffs:
                        if dd == d:
                            c_old_counts[old_c] = c_old_counts.get(old_c, 0) + 1
                            c_new_counts[new_c] = c_new_counts.get(new_c, 0) + 1
                        else:
                            c = f[dd - 1]
                            c_old_counts[c] = c_old_counts.get(c, 0) + 1
                            c_new_counts[c] = c_new_counts.get(c, 0) + 1
                    
                    was_mono = (len(c_old_counts) == 1)
                    is_mono = (len(c_new_counts) == 1)
                    
                    if was_mono and not is_mono:
                        delta -= 1
                    elif not was_mono and is_mono:
                        delta += 1
                
                accept = False
                if delta < 0:
                    accept = True
                else:
                    if delta > 0:
                        if delta / T > 50:
                            accept = False
                        else:
                            prob = math.exp(-delta / T)
                            if random.random() < prob:
                                accept = True
                
                if accept:
                    f[d - 1] = new_c
                    bad += delta
                    if bad < best_bad:
                        best_bad = bad
                        best_f = f[:]
                        if best_bad == 0:
                            colors = [0] * m
                            for i in range(n):
                                for j in range(i + 1, n):
                                    dd = j - i
                                    colors[edge_index(i, j)] = best_f[dd - 1]
                            return {'n': n, 'colors': colors}
                else:
                    pass  # no change
                
                T *= cool_rate
                if T < 0.1:
                    T = 1.5
        
        return None

    # First try the structured cyclic approach for n=17
    if time.time() < deadline - 10:
        result = try_cyclic_n17(50.0)
        if result:
            return result
    
    # Fall back to general search
    targets = [17, 16, 15, 14, 13, 12, 11]
    budgets = {
        17: 30.0,
        16: 20.0,
        15: 15.0,
        14: 10.0,
        13: 5.0,
        12: 3.0,
        11: 1.0
    }
    
    for n in targets:
        if time.time() > deadline - 10:
            break
        
        budget = budgets.get(n, 5.0)
        if time.time() + budget > deadline - 5:
            budget = deadline - 5 - time.time()
            if budget < 1.0:
                continue
                 
        result = search_n(n, budget)
        
        if result:
            return result
    
    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
