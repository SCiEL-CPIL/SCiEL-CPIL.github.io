import time
import random
import sys
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s buffer for safety

    def edge_index(n, i, j):
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def precompute_cycles(n):
        cycles = []
        vertices = list(range(n))
        for combo in combinations(vertices, 6):
            v0 = min(combo)
            others = [v for v in combo if v != v0]
            seen = set()
            for perm in permutations(others):
                if perm[0] < perm[4]:
                    cycle_vertices = (v0, perm[0], perm[1], perm[2], perm[3], perm[4])
                    edges = []
                    for k in range(6):
                        u = cycle_vertices[k]
                        v = cycle_vertices[(k + 1) % 6]
                        if u > v:
                            u, v = v, u
                        idx = edge_index(n, u, v)
                        edges.append(idx)
                    edge_set = frozenset(edges)
                    if edge_set not in seen:
                        seen.add(edge_set)
                        cycles.append(tuple(edges))
        return cycles

    def solve_local_search(n):
        num_edges = n * (n - 1) // 2
        cycles = precompute_cycles(n)
        num_cycles = len(cycles)
        
        edge_to_cycles = [[] for _ in range(num_edges)]
        for i, cycle in enumerate(cycles):
            for e in cycle:
                edge_to_cycles[e].append(i)
        
        def init_random():
            return [random.randint(0, 3) for _ in range(num_edges)]

        def count_mono(colors):
            count = 0
            for cycle in cycles:
                c0 = colors[cycle[0]]
                mono = True
                for e in cycle[1:]:
                    if colors[e] != c0:
                        mono = False
                        break
                if mono:
                    count += 1
            return count

        best_score = float('inf')
        best_colors = None
        
        for restart in range(50):
            if time.time() > deadline:
                break
            
            colors = init_random()
            current_penalty = count_mono(colors)
            
            if current_penalty == 0:
                return colors
            
            T = 1.0
            cooling = 0.999
            steps = 5000
            
            for step in range(steps):
                if time.time() > deadline:
                    break
                
                e = random.randint(0, num_edges - 1)
                old_c = colors[e]
                new_c = random.choice([c for c in range(4) if c != old_c])
                
                delta = 0
                for c_idx in edge_to_cycles[e]:
                    cycle = cycles[c_idx]
                    is_mono_old = True
                    for ce in cycle:
                        if ce == e:
                            continue
                        if colors[ce] != old_c:
                            is_mono_old = False
                            break
                    
                    if is_mono_old:
                        delta -= 1
                        continue
                    
                    is_potential_mono_new = True
                    for ce in cycle:
                        if ce == e:
                            continue
                        if colors[ce] != new_c:
                            is_potential_mono_new = False
                            break
                    
                    if is_potential_mono_new:
                        delta += 1
                
                new_penalty = current_penalty + delta
                
                if new_penalty < current_penalty:
                    colors[e] = new_c
                    current_penalty = new_penalty
                    if current_penalty == 0:
                        return colors
                    if current_penalty < best_score:
                        best_score = current_penalty
                        best_colors = colors[:]
                else:
                    if delta == 0:
                        if random.random() < 0.5:
                            colors[e] = new_c
                    else:
                        prob = 2.71828 ** (-delta / T)
                        if random.random() < prob:
                            colors[e] = new_c
                            current_penalty = new_penalty
                            if current_penalty < best_score:
                                best_score = current_penalty
                                best_colors = colors[:]
                
                T *= cooling
        
        if best_colors is not None and count_mono(best_colors) == 0:
            return best_colors
            
        return None

    # Try n=17, 16, 15...
    for n in [17, 16, 15, 14, 13, 12, 11, 10, 9]:
        if time.time() > deadline:
            break
        
        # Give more time to higher n
        # We just run until deadline or solution found
        result = solve_local_search(n)
        
        if result is not None:
            return {'n': n, 'colors': result}
        
        # If we didn't find a solution for n, try next
        # To avoid spending all time on n=17 if it's hard, we could limit it,
        # but finding n=17 is the goal.
        # If n=17 fails, we fall back to n=16.
        
    # Fallback
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}

