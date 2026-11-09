import time
import random
import itertools
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def solve_n_sa(n, time_limit):
        num_edges = n * (n - 1) // 2
        edge_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[(i, j)] = idx
                idx += 1
        
        cycles = []
        cycle_edges = []
        
        for vertices in itertools.combinations(range(n), 6):
            v0 = vertices[0]
            others = vertices[1:]
            for perm in itertools.permutations(others):
                v1, v2, v3, v4, v5 = perm
                if v1 > v5:
                    continue
                cycle_verts = [v0, v1, v2, v3, v4, v5]
                c_edges = []
                for k in range(6):
                    u = cycle_verts[k]
                    v = cycle_verts[(k + 1) % 6]
                    if u > v:
                        u, v = v, u
                    c_edges.append(edge_map[(u, v)])
                cycles.append(cycle_verts)
                cycle_edges.append(c_edges)
        
        num_cycles = len(cycles)
        
        edge_to_cycles = [[] for _ in range(num_edges)]
        for c_idx, c_edges in enumerate(cycle_edges):
            for e_idx in c_edges:
                edge_to_cycles[e_idx].append(c_idx)
        
        max_restarts = 100
        for r in range(max_restarts):
            if time.time() > time_limit:
                break
                
            colors = [random.randint(0, 3) for _ in range(num_edges)]
            
            cycle_counts = [[0, 0, 0, 0] for _ in range(num_cycles)]
            for c_idx, c_edges in enumerate(cycle_edges):
                for e_idx in c_edges:
                    cycle_counts[c_idx][colors[e_idx]] += 1
            
            mono_count = 0
            for c_idx in range(num_cycles):
                counts = cycle_counts[c_idx]
                if any(c == 6 for c in counts):
                    mono_count += 1
            
            if mono_count == 0:
                return colors
            
            T = 1.0
            T_min = 0.01
            cooling = 0.999
            steps_since_improve = 0
            best_mono = mono_count
            
            while time.time() < time_limit:
                if mono_count == 0:
                    return colors
                
                e_idx = random.randint(0, num_edges - 1)
                old_color = colors[e_idx]
                new_color = random.choice([c for c in range(4) if c != old_color])
                
                delta = 0
                for c_idx in edge_to_cycles[e_idx]:
                    counts = cycle_counts[c_idx]
                    if counts[old_color] == 6:
                        delta -= 1
                    if counts[new_color] == 5:
                        delta += 1
                
                if delta < 0:
                    accept = True
                else:
                    if random.random() < math.exp(-delta / T):
                        accept = True
                    else:
                        accept = False
                
                if accept:
                    colors[e_idx] = new_color
                    mono_count += delta
                    for c_idx in edge_to_cycles[e_idx]:
                        cycle_counts[c_idx][old_color] -= 1
                        cycle_counts[c_idx][new_color] += 1
                    
                    if mono_count < best_mono:
                        best_mono = mono_count
                        steps_since_improve = 0
                    else:
                        steps_since_improve += 1
                        
                    T = max(T_min, T * cooling)
                    
                    if steps_since_improve > 1000:
                        break
                else:
                    steps_since_improve += 1
                    T = max(T_min, T * cooling)
                    if steps_since_improve > 1000:
                        break
            
        return None

    for n in [19, 18, 17, 16, 15, 14]:
        if time.time() > deadline - 5:
            break
            
        if n >= 18:
            alloc = 25
        elif n == 17:
            alloc = 40
        else:
            alloc = 10
            
        end_time = time.time() + alloc
        if end_time > deadline - 5:
            end_time = deadline - 5
            
        res = solve_n_sa(n, end_time)
        if res:
            return {'n': n, 'colors': res}
            
    n = 10
    colors = [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]
    return {'n': n, 'colors': colors}
