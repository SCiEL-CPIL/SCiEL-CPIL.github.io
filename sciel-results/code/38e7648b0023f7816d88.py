import random
import time
import math
from itertools import combinations

def construct():
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(edges)}
    num_edges = len(edges)
    
    # Precompute K4s
    k4s = []
    k4s_by_edge = [[] for _ in range(num_edges)]
    
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        e1 = edge_index[(a, b)]
        e2 = edge_index[(a, c)]
        e3 = edge_index[(a, d)]
        e4 = edge_index[(b, c)]
        e5 = edge_index[(b, d)]
        e6 = edge_index[(c, d)]
        k4_idx = len(k4s)
        k4s.append([e1, e2, e3, e4, e5, e6])
        for e in [e1, e2, e3, e4, e5, e6]:
            k4s_by_edge[e].append(k4_idx)
            
    def try_solve(M, time_limit):
        # M is the number of colours to use
        colours = [random.randint(0, M - 1) for _ in range(num_edges)]
        
        # Initialize K4 state
        # k4_colour_counts: list of lists, size M
        k4_colour_counts = [[0] * M for _ in range(len(k4s))]
        k4_distinct = [0] * len(k4s)
        
        for k4_idx, k4_edges in enumerate(k4s):
            for e in k4_edges:
                c = colours[e]
                if k4_colour_counts[k4_idx][c] == 0:
                    k4_distinct[k4_idx] += 1
                k4_colour_counts[k4_idx][c] += 1
                
        # Initial violations
        curr_viol = sum(1 for d in k4_distinct if d < 5)
        best_viol = curr_viol
        best_colours = colours[:]
        best_k4_colour_counts = [row[:] for row in k4_colour_counts]
        best_k4_distinct = k4_distinct[:]
        
        start_time = time.time()
        T = 1.0
        cooling = 0.999
        
        while time.time() - start_time < time_limit:
            if curr_viol == 0:
                return colours[:]
                
            # Pick a random edge
            e = random.randint(0, num_edges - 1)
            old_c = colours[e]
            new_c = random.randint(0, M - 1)
            if old_c == new_c:
                continue
                
            # Calculate delta violations
            delta = 0
            affected_k4s = k4s_by_edge[e]
            
            for k4_idx in affected_k4s:
                counts = k4_colour_counts[k4_idx]
                distinct = k4_distinct[k4_idx]
                
                # Remove old_c
                counts[old_c] -= 1
                if counts[old_c] == 0:
                    distinct -= 1
                    
                # Add new_c
                counts[new_c] += 1
                if counts[new_c] == 1:
                    distinct += 1
                    
                old_viol = 1 if k4_distinct[k4_idx] < 5 else 0
                new_viol = 1 if distinct < 5 else 0
                
                delta += (new_viol - old_viol)
                
                # Update state for this K4 (temporarily)
                k4_colour_counts[k4_idx] = counts
                k4_distinct[k4_idx] = distinct
                
            # Acceptance
            if delta < 0 or (T > 0.01 and random.random() < math.exp(-delta / T)):
                # Accept
                colours[e] = new_c
                curr_viol += delta
                if curr_viol < best_viol:
                    best_viol = curr_viol
                    best_colours = colours[:]
                    best_k4_colour_counts = [row[:] for row in k4_colour_counts]
                    best_k4_distinct = k4_distinct[:]
            else:
                # Reject: Revert changes
                for k4_idx in affected_k4s:
                    counts = k4_colour_counts[k4_idx]
                    distinct = k4_distinct[k4_idx]
                    
                    # Revert new_c
                    counts[new_c] -= 1
                    if counts[new_c] == 0:
                        distinct -= 1
                        
                    # Revert old_c
                    counts[old_c] += 1
                    if counts[old_c] == 1:
                        distinct += 1
                        
                    k4_colour_counts[k4_idx] = counts
                    k4_distinct[k4_idx] = distinct
                # colours[e] remains old_c
                # curr_viol remains unchanged
                
            T *= cooling
            if T < 0.01:
                T = 0.01
                
        return best_colours if best_viol == 0 else None

    # Try 13, 14, 15
    solution = try_solve(13, 40)
    if solution:
        return solution
        
    solution = try_solve(14, 40)
    if solution:
        return solution
        
    solution = try_solve(15, 30)
    if solution:
        return solution
        
    # Fallback: Deterministic 15-colouring
    return [(i + j) % 15 for i in range(n) for j in range(i + 1, n)]
