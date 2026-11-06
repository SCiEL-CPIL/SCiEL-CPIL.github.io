import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 115.0  # Leave 5s for safety
    
    # Target n values to try in descending order
    target_ns = [34, 33, 32, 31, 30, 25, 20, 15, 10]
    
    for n in target_ns:
        if time.time() - start_time > time_limit - 5.0:
            break
            
        # Precompute structures for this n
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Precompute K4s and K5s
        k4_edges = []
        for quad in combinations(range(n), 4):
            edges = []
            for tri in combinations(quad, 3):
                edges.append(triple_to_idx[tri])
            k4_edges.append(edges)
            
        k5_edges = []
        for quint in combinations(range(n), 5):
            edges = []
            for tri in combinations(quint, 3):
                edges.append(triple_to_idx[tri])
            k5_edges.append(edges)
            
        num_k4 = len(k4_edges)
        num_k5 = len(k5_edges)
        
        # For each edge, which K4s and K5s does it belong to?
        edge_to_k4s = [[] for _ in range(num_triples)]
        for i, edges in enumerate(k4_edges):
            for e in edges:
                edge_to_k4s[e].append(i)
                
        edge_to_k5s = [[] for _ in range(num_triples)]
        for i, edges in enumerate(k5_edges):
            for e in edges:
                edge_to_k5s[e].append(i)
        
        # Constraint Propagation / Urgency-based Search
        # We maintain counts of monochromatic edges in each K4 and K5.
        # red_k4_count[i] = number of red edges in K4 i (out of 4)
        # blue_k5_count[i] = number of blue edges in K5 i (out of 10)
        
        # We will use a priority queue or just scan for the most "urgent" uncolored edge.
        # However, maintaining a dynamic priority queue with changing urgency is complex.
        # Instead, we can use a "min-conflicts" style but guided by urgency.
        # Or, we can use a backtracking search with forward checking, but that might be slow for n=34.
        
        # Let's try a hybrid: Start with random, then use a greedy repair step that picks
        # the edge whose flip reduces the total "near-miss" count the most.
        # Near-miss count:
        # For K4: if 3 red, 1 blue, flipping the blue to red creates a violation. Flipping a red to blue breaks the near-miss.
        # We want to avoid creating violations.
        
        # Let's stick to Local Search but with a better initialization and move selection.
        # The previous SA was too slow because it picked random edges.
        # Let's pick edges that are part of the most violations or near-violations.
        
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit - 5.0:
                break
                
            # Random initialization
            colors = np.random.randint(0, 2, size=num_triples)
            
            # Calculate initial violation counts
            # k4_red_violations[i] = 1 if all 4 red, else 0
            # k5_blue_violations[i] = 1 if all 10 blue, else 0
            
            # To speed up delta calculation, we precompute the neighbor lists.
            
            # Let's try a simpler, faster local search:
            # 1. Compute total cost.
            # 2. For each edge, compute the "potential gain" of flipping it.
            # 3. Flip the edge with the highest gain.
            # 4. Repeat until no gain or time limit.
            
            # This is essentially a greedy hill climbing.
            # To avoid local minima, we can add some randomization (kick).
            
            def get_cost_and_neighbors(colors):
                # This is a placeholder for a more efficient update
                pass
            
            # Efficient delta calculation is key.
            # Let's reuse the logic from the previous program but optimize the inner loop.
            
            # Precompute for each edge:
            # k4_neighbors[e] = list of (k4_idx, position_in_k4)
            # k5_neighbors[e] = list of (k5_idx, position_in_k5)
            
            # Actually, the previous program's delta calculation was O(degree).
            # Degree of an edge in K4s is C(n-3, 1) = n-3.
            # Degree of an edge in K5s is C(n-3, 2) = (n-3)(n-4)/2.
            # For n=34, degree_K5 ~ 31*30/2 = 465.
            # So delta calc is ~500 ops.
            # 50000 iters * 500 ops = 25M ops. Should be < 1s in Python if optimized?
            # But Python loops are slow. 25M simple ops might take 2-5 seconds.
            # So we can do ~20-50 restarts in 115s? Maybe.
            
            # The previous program failed likely because it didn't find a solution for n=34
            # within the time limit per restart, or the SA temperature schedule was bad.
            
            # Let's try a "WalkSAT" style approach:
            # If stuck, pick a random edge involved in a violation and flip it.
            
            cost = 0
            # Initialize violation flags
            k4_viol = np.zeros(num_k4, dtype=bool)
            k5_viol = np.zeros(num_k5, dtype=bool)
            
            for i, edges in enumerate(k4_edges):
                if np.all(colors[edges] == 0):
                    k4_viol[i] = True
                    cost += 1
                    
            for i, edges in enumerate(k5_edges):
                if np.all(colors[edges] == 1):
                    k5_viol[i] = True
                    cost += 1
            
            if cost == 0:
                return {'n': n, 'colors': colors.tolist()}
            
            max_iter = 20000
            for i in range(max_iter):
                if time.time() - start_time > time_limit - 5.0:
                    break
                    
                if cost == 0:
                    return {'n': n, 'colors': colors.tolist()}
                
                # Pick an edge
                # Strategy: Pick an edge involved in a violation, or random if none?
                # If there are violations, pick one at random from the violating cliques.
                # Otherwise, pick random.
                
                if cost > 0:
                    # Pick a random violation
                    if random.random() < 0.5 and np.any(k4_viol):
                        viol_idx = np.where(k4_viol)[0][random.randint(0, np.sum(k4_viol)-1)]
                        edges = k4_edges[viol_idx]
                        e = edges[random.randint(0, 3)]
                    elif np.any(k5_viol):
                        viol_idx = np.where(k5_viol)[0][random.randint(0, np.sum(k5_viol)-1)]
                        edges = k5_edges[viol_idx]
                        e = edges[random.randint(0, 9)]
                    else:
                        e = random.randint(0, num_triples - 1)
                else:
                    e = random.randint(0, num_triples - 1)
                
                # Flip edge e
                old_color = colors[e]
                new_color = 1 - old_color
                colors[e] = new_color
                
                # Update violations for K4s containing e
                for k4_idx in edge_to_k4s[e]:
                    edges = k4_edges[k4_idx]
                    # Check if was violation
                    was_viol = k4_viol[k4_idx]
                    # Check if is violation
                    is_viol = np.all(colors[edges] == 0)
                    
                    if was_viol and not is_viol:
                        cost -= 1
                        k4_viol[k4_idx] = False
                    elif not was_viol and is_viol:
                        cost += 1
                        k4_viol[k4_idx] = True
                
                # Update violations for K5s containing e
                for k5_idx in edge_to_k5s[e]:
                    edges = k5_edges[k5_idx]
                    was_viol = k5_viol[k5_idx]
                    is_viol = np.all(colors[edges] == 1)
                    
                    if was_viol and not is_viol:
                        cost -= 1
                        k5_viol[k5_idx] = False
                    elif not was_viol and is_viol:
                        cost += 1
                        k5_viol[k5_idx] = True
                        
    # Fallback: n=3
    triples_3 = list(combinations(range(3), 3))
    return {'n': 3, 'colors': [0] * len(triples_3)}

# EVOLVE-BLOCK-END
