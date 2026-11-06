import time
import random
import sys
import itertools
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Fallback valid coloring: n=5
    def get_fallback(n):
        colors = [0] * (n * (n - 1) // 2)
        return {'n': n, 'colors': colors}
    
    best_result = get_fallback(5)
    
    # Target n values to try in descending order of ambition
    targets = [15, 14, 13, 12, 11, 10]
    
    for n in targets:
        if time.time() > deadline - 10:
            break
        
        # Precompute edge map
        edge_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[(i, j)] = idx
                idx += 1
        m = idx
        
        # Precompute cycles
        vertices = list(range(n))
        cycles = []
        edge_to_cycles = [[] for _ in range(m)]
        
        for combo in itertools.combinations(vertices, 6):
            v0 = combo[0]
            others = combo[1:]
            seen_cycles = set()
            for perm in itertools.permutations(others):
                if perm[0] > perm[-1]:
                    continue
                path = [v0] + list(perm)
                cycle_edges = []
                for k in range(6):
                    u = path[k]
                    v = path[(k + 1) % 6]
                    if u > v:
                        u, v = v, u
                    cycle_edges.append(edge_map[(u, v)])
                
                key = frozenset(cycle_edges)
                if key in seen_cycles:
                    continue
                seen_cycles.add(key)
                
                cycles.append(cycle_edges)
                for e in cycle_edges:
                    edge_to_cycles[e].append(len(cycles) - 1)
        
        # Initial Coloring: Random
        colors = [random.randint(0, 3) for _ in range(m)]
        
        # Function to count monochromatic C6s
        def count_bad(colors):
            bad = 0
            for ce in cycles:
                c0 = colors[ce[0]]
                if all(colors[e] == c0 for e in ce[1:]):
                    bad += 1
            return bad
        
        current_bad = count_bad(colors)
        best_colors = colors[:]
        best_bad = current_bad
        
        # Local Search with Simulated Annealing
        T = 5.0
        cool_rate = 0.995
        max_iter = 50000
        
        # Pre-extract for speed
        e2c = edge_to_cycles
        cyc = cycles
        cols = colors
        
        for i in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            e = random.randint(0, m - 1)
            c_old = cols[e]
            c_new = random.choice([c for c in range(4) if c != c_old])
            
            delta = 0
            for ci in e2c[e]:
                ce = cyc[ci]
                was_mono = True
                for ed in ce:
                    if ed != e and cols[ed] != c_old:
                        was_mono = False
                        break
                
                will_be_mono = True
                for ed in ce:
                    if ed != e and cols[ed] != c_new:
                        will_be_mono = False
                        break
                
                if was_mono:
                    delta -= 1
                if will_be_mono:
                    delta += 1
            
            if delta < 0 or (T > 0.1 and random.random() < math.exp(-delta / T)):
                cols[e] = c_new
                current_bad += delta
                if current_bad < best_bad:
                    best_bad = current_bad
                    best_colors = cols[:]
                    if best_bad == 0:
                        return {'n': n, 'colors': best_colors}
            
            T *= cool_rate
        
    return best_result
