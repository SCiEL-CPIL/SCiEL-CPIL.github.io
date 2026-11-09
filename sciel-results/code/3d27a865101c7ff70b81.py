import numpy as np
import itertools
import time
import random

def construct():
    n = 16
    deadline = time.time() + 140.0
    
    k4s = list(itertools.combinations(range(n), 4))
    num_k4s = len(k4s)
    
    edge_to_k4s = [[[] for _ in range(n)] for _ in range(n)]
    for idx, quad in enumerate(k4s):
        a, b, c, d = quad
        edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        for u, v in edges:
            if u < v:
                edge_to_k4s[u][v].append(idx)
            else:
                edge_to_k4s[v][u].append(idx)

    def evaluate_coloring(colors, k):
        violations = 0
        distinct_counts = np.zeros(num_k4s, dtype=np.int8)
        violating_set = set()
        for idx, quad in enumerate(k4s):
            a, b, c, d = quad
            c1 = colors[a][b]
            c2 = colors[a][c]
            c3 = colors[a][d]
            c4 = colors[b][c]
            c5 = colors[b][d]
            c6 = colors[c][d]
            d = len({c1, c2, c3, c4, c5, c6})
            distinct_counts[idx] = d
            if d < 5:
                violations += 1
                violating_set.add(idx)
        return violations, distinct_counts, violating_set

    def search(k, time_limit):
        start_time = time.time()
        
        colors = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                c = random.randint(0, k-1)
                colors[i][j] = c
                colors[v][u] = c # Wait, v is not defined in this scope? No, j is.
                colors[j][i] = c
        
        violations, distinct_counts, violating_set = evaluate_coloring(colors, k)
        
        if violations == 0:
            return True, colors

        hist = [[0]*k for _ in range(num_k4s)]
        for idx, quad in enumerate(k4s):
            a, b, c, d = quad
            edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
            for u, v in edges:
                col = colors[u][v]
                hist[idx][col] += 1
        
        best_violations = violations
        best_colors = [row[:] for row in colors]
        
        T = 1.0
        decay = 0.999
        
        while time.time() - start_time < time_limit:
            # Select edge
            if violating_set:
                # Pick a random violating K4
                idx = random.choice(list(violating_set))
                quad = k4s[idx]
                # Pick a random edge in this K4
                edge = random.choice(list(itertools.combinations(quad, 2)))
                u, v = edge
            else:
                # Random edge
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue

            old_col = colors[u][v]
            new_col = random.randint(0, k-1)
            if old_col == new_col:
                continue
            
            affected_k4s = edge_to_k4s[min(u,v)][max(u,v)]
            
            delta_violations = 0
            changes = []
            
            for idx in affected_k4s:
                old_count = hist[idx][old_col]
                new_count = hist[idx][new_col]
                
                old_distinct = distinct_counts[idx]
                
                hist[idx][old_col] -= 1
                hist[idx][new_col] += 1
                
                new_distinct = old_distinct
                if old_count == 1:
                    new_distinct -= 1
                if new_count == 0:
                    new_distinct += 1
                
                old_is_violation = (old_distinct < 5)
                new_is_violation = (new_distinct < 5)
                
                if old_is_violation and not new_is_violation:
                    delta_violations -= 1
                elif not old_is_violation and new_is_violation:
                    delta_violations += 1
                
                changes.append((idx, old_distinct, new_distinct))
                distinct_counts[idx] = new_distinct
                
                # Update violating_set
                if old_is_violation and not new_is_violation:
                    violating_set.discard(idx)
                elif not old_is_violation and new_is_violation:
                    violating_set.add(idx)
            
            accept = False
            if delta_violations < 0:
                accept = True
            elif delta_violations == 0:
                accept = True
            else:
                if T > 0.01:
                    prob = np.exp(-delta_violations / T)
                    if random.random() < prob:
                        accept = True
            
            if accept:
                colors[u][v] = new_col
                colors[v][u] = new_col
                violations += delta_violations
                if violations < best_violations:
                    best_violations = violations
                    best_colors = [row[:] for row in colors]
                    if violations == 0:
                        return True, best_colors
            else:
                for idx, old_d, new_d in changes:
                    distinct_counts[idx] = old_d
                    # Revert violating_set
                    old_is_violation = (old_d < 5)
                    new_is_violation = (new_d < 5)
                    if old_is_violation and not new_is_violation:
                        # We removed it, so add it back
                        violating_set.add(idx)
                    elif not old_is_violation and new_is_violation:
                        # We added it, so remove it
                        violating_set.discard(idx)
                for idx in affected_k4s:
                    hist[idx][old_col] += 1
                    hist[idx][new_col] -= 1
            
            T *= decay
        
        return False, best_colors

    valid, colors_13 = search(13, 40.0)
    if valid:
        return [int(colors_13[i][j]) for i in range(16) for j in range(i+1, 16)]
    
    if time.time() < deadline - 20:
        valid, colors_14 = search(14, 40.0)
        if valid:
            return [int(colors_14[i][j]) for i in range(16) for j in range(i+1, 16)]
        
    if time.time() < deadline - 10:
        valid, colors_15 = search(15, 30.0)
        if valid:
            return [int(colors_15[i][j]) for i in range(16) for j in range(i+1, 16)]
            
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
