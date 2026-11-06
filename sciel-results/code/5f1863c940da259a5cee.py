import random
import time
import math
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 16
    k4s = []
    k4_edges = []
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        edges = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
        k4s.append(quad)
        k4_edges.append(edges)
        
    num_k4 = len(k4s)
    
    edge_to_k4s = {}
    for i, quad in enumerate(k4s):
        for e in k4_edges[i]:
            if e not in edge_to_k4s:
                edge_to_k4s[e] = []
            edge_to_k4s[e].append(i)

    # Helper to get flat list of edges in lex order
    def get_flat(C):
        return [C[i][j] for i in range(n) for j in range(i+1, n)]

    # Try K from 13 to 15
    for K in [13, 14, 15]:
        if time.time() > deadline - 10:
            break
            
        # Initialize
        C = [[-1]*n for _ in range(n)]
        colour_counts = [0] * K
        for i in range(n):
            for j in range(i+1, n):
                c = random.randint(0, K-1)
                C[i][j] = c
                C[j][i] = c
                colour_counts[c] += 1
                
        # Precompute violation status for each K4
        # viol[k4_idx] = 1 if violation, 0 else
        viol = [0] * num_k4
        total_violations = 0
        
        for i in range(num_k4):
            edges = k4_edges[i]
            cols = set()
            for u, v in edges:
                cols.add(C[u][v])
            if len(cols) < 5:
                viol[i] = 1
                total_violations += 1
                
        distinct_colours = sum(1 for c in colour_counts if c > 0)
        
        # State
        best_local_violations = total_violations
        best_local_distinct = distinct_colours
        best_local_C = [row[:] for row in C]
        best_local_counts = colour_counts[:]
        
        T = 50.0
        T_min = 0.01
        alpha = 0.9995
        
        while time.time() < deadline - 5 and T > T_min:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if u > v: u, v = v, u
            
            old_c = C[u][v]
            new_c = random.randint(0, K-1)
            if old_c == new_c: continue
            
            # Calculate delta violations
            delta_violations = 0
            affected = edge_to_k4s.get((u, v), [])
            for k4_idx in affected:
                edges = k4_edges[k4_idx]
                
                # Current set of colours
                # Optimization: We can just check the 6 edges.
                # To be fast, we can store the current set of colours for each K4?
                # No, updating sets is expensive.
                # Just iterate 6 edges.
                
                old_set = set()
                for a, b in edges:
                    old_set.add(C[a][b])
                old_size = len(old_set)
                old_v = 1 if old_size < 5 else 0
                
                # New set
                # Remove old_c, add new_c
                # If old_c is in the set, and it's the only occurrence, it disappears.
                # But we are just simulating.
                new_set = set()
                for a, b in edges:
                    if (a == u and b == v) or (a == v and b == u):
                        new_set.add(new_c)
                    else:
                        new_set.add(C[a][b])
                new_size = len(new_set)
                new_v = 1 if new_size < 5 else 0
                
                if old_v != new_v:
                    delta_violations += new_v - old_v
            
            # Calculate delta distinct
            delta_distinct = 0
            # old_c count decreases by 1
            if colour_counts[old_c] == 1:
                delta_distinct -= 1
            # new_c count increases by 1
            if colour_counts[new_c] == 0:
                delta_distinct += 1
            
            # Cost change
            # We prioritize minimizing violations.
            # If violations decrease, accept.
            # If violations increase, accept with prob.
            # If violations same, minimize distinct.
            
            if delta_violations < 0:
                accept = True
            elif delta_violations > 0:
                # Penalty for violations is high
                # Let's say cost = 1000 * viol + distinct
                delta_cost = 1000 * delta_violations + delta_distinct
                if delta_cost < 0:
                    accept = True
                else:
                    accept = random.random() < math.exp(-delta_cost / T)
            else:
                # delta_violations == 0
                if delta_distinct < 0:
                    accept = True
                elif delta_distinct > 0:
                    accept = random.random() < math.exp(-delta_distinct / T)
                else:
                    accept = True # No change in cost, move anyway? Or skip?
                    # Let's accept to explore
            
            if accept:
                # Update C
                C[u][v] = new_c
                C[v][u] = new_c
                
                # Update counts
                colour_counts[old_c] -= 1
                colour_counts[new_c] += 1
                
                # Update violations
                total_violations += delta_violations
                for k4_idx in affected:
                    # Recalculate viol status for this K4
                    # We already know old_v and new_v? 
                    # We calculated them in the loop above. 
                    # But we didn't store them. 
                    # Let's just recalculate viol[k4_idx]
                    edges = k4_edges[k4_idx]
                    cols = set()
                    for a, b in edges:
                        cols.add(C[a][b])
                    viol[k4_idx] = 1 if len(cols) < 5 else 0
                    # Note: total_violations was updated by delta, which is consistent.
                    
                distinct_colours += delta_distinct
                
                # Update best
                if total_violations < best_local_violations or (total_violations == best_local_violations and distinct_colours < best_local_distinct):
                    best_local_violations = total_violations
                    best_local_distinct = distinct_colours
                    best_local_C = [row[:] for row in C]
                    best_local_counts = colour_counts[:]
                    
            T *= alpha
            
        # After SA for this K
        if best_local_violations == 0:
            # We found a valid solution with K colours
            return get_flat(best_local_C)
            
    # If no valid solution found with K < 17, return seed
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
