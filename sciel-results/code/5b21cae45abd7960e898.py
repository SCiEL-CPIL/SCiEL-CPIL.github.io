import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    for n in range(34, 9, -1):
        if time.time() > deadline - 5.0:
            break
            
        if n == 34:
            time_limit = 80.0
        elif n >= 30:
            time_limit = 30.0
        else:
            time_limit = 10.0
            
        remaining = deadline - time.time()
        if time_limit > remaining:
            time_limit = remaining
            if time_limit < 1.0:
                break

        solution = solve_fast(n, time_limit)
        if solution is not None:
            return solution

    return {'n': 3, 'colors': [0] * 1}

def solve_fast(n, time_limit):
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_index = {t: i for i, t in enumerate(triples)}
    
    k4s = []
    k4_map = [[] for _ in range(num_triples)]
    for quad in combinations(range(n), 4):
        edges = []
        for tri in combinations(quad, 3):
            idx = triple_index[tri]
            edges.append(idx)
        k4_idx = len(k4s)
        k4s.append(tuple(edges))
        for e in edges:
            k4_map[e].append(k4_idx)
            
    k5s = []
    k5_map = [[] for _ in range(num_triples)]
    for quint in combinations(range(n), 5):
        edges = []
        for tri in combinations(quint, 3):
            idx = triple_index[tri]
            edges.append(idx)
        k5_idx = len(k5s)
        k5s.append(tuple(edges))
        for e in edges:
            k5_map[e].append(k5_idx)
            
    def run_sa(time_budget):
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        total_violations = 0
        for edges in k4s:
            if all(colors[e] == 0 for e in edges):
                total_violations += 1
        for edges in k5s:
            if all(colors[e] == 1 for e in edges):
                total_violations += 1
                
        best_colors = colors[:]
        best_violations = total_violations
        
        if total_violations == 0:
            return colors
            
        start = time.time()
        T = 1.0
        cooling = 0.999
        steps_since_improve = 0
        max_steps_no_improve = 2000
        
        while time.time() - start < time_budget:
            idx = random.randint(0, num_triples - 1)
            old_val = colors[idx]
            new_val = 1 - old_val
            
            delta = 0
            
            for k4_idx in k4_map[idx]:
                edges = k4s[k4_idx]
                if old_val == 0:
                    cnt = 0
                    for e in edges:
                        if e != idx and colors[e] == 0:
                            cnt += 1
                    if cnt == 3:
                        delta -= 1
                else:
                    cnt = 0
                    for e in edges:
                        if e != idx and colors[e] == 0:
                            cnt += 1
                    if cnt == 3:
                        delta += 1
                        
            for k5_idx in k5_map[idx]:
                edges = k5s[k5_idx]
                if old_val == 1:
                    cnt = 0
                    for e in edges:
                        if e != idx and colors[e] == 1:
                            cnt += 1
                    if cnt == 9:
                        delta -= 1
                else:
                    cnt = 0
                    for e in edges:
                        if e != idx and colors[e] == 1:
                            cnt += 1
                    if cnt == 9:
                        delta += 1
                        
            new_violations = total_violations + delta
            
            if new_violations < total_violations:
                accept = True
            else:
                if T > 1e-6:
                    prob = math.exp(-delta / T)
                    accept = random.random() < prob
                else:
                    accept = False
                    
            if accept:
                colors[idx] = new_val
                total_violations = new_violations
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_colors = colors[:]
                    steps_since_improve = 0
                    if best_violations == 0:
                        return colors
                else:
                    steps_since_improve += 1
            else:
                colors[idx] = old_val
                steps_since_improve += 1
                
            T *= cooling
            
            if steps_since_improve > max_steps_no_improve:
                # Restart
                return None
                
        if best_violations == 0:
            return best_colors
        return None

    # Run multiple restarts
    restart_time = time_limit / 5.0
    for _ in range(5):
        if time.time() > start_time + time_limit:
            break
        res = run_sa(restart_time)
        if res is not None:
            return {'n': n, 'colors': res}
            
    return None
