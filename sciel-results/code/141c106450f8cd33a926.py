import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 115.0  # Leave 5s for safety
    
    # Target n values to try in descending order
    target_ns = [34, 33, 32, 31, 30, 25, 20, 15, 10]
    
    # We will try to find a valid coloring for each n
    # If we find one for a larger n, we stop and return it.
    # If we run out of time, we return the best valid one found so far.
    # If we find none, we must return a trivially valid one (e.g. n=3).
    
    best_result = None
    best_n = -1
    
    for n in target_ns:
        if time.time() - start_time > time_limit - 5.0:
            break
            
        # Precompute structures for this n
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Precompute K4s and K5s as numpy arrays for vectorized operations
        k4_list = []
        for quad in combinations(range(n), 4):
            edges = []
            for tri in combinations(quad, 3):
                edges.append(triple_to_idx[tri])
            k4_list.append(edges)
            
        k5_list = []
        for quint in combinations(range(n), 5):
            edges = []
            for tri in combinations(quint, 3):
                edges.append(triple_to_idx[tri])
            k5_list.append(edges)
            
        num_k4 = len(k4_list)
        num_k5 = len(k5_list)
        
        # Convert to numpy arrays
        if num_k4 > 0:
            k4_edges = np.array(k4_list, dtype=np.int32)
        else:
            k4_edges = np.empty((0, 4), dtype=np.int32)
            
        if num_k5 > 0:
            k5_edges = np.array(k5_list, dtype=np.int32)
        else:
            k5_edges = np.empty((0, 10), dtype=np.int32)
        
        # For each edge, which K4s and K5s does it belong to?
        edge_to_k4s = [[] for _ in range(num_triples)]
        for i in range(num_k4):
            for e in k4_edges[i]:
                edge_to_k4s[e].append(i)
                
        edge_to_k5s = [[] for _ in range(num_triples)]
        for i in range(num_k5):
            for e in k5_edges[i]:
                edge_to_k5s[e].append(i)
        
        # Convert to numpy arrays for faster access
        edge_to_k4s_arr = []
        for lst in edge_to_k4s:
            if lst:
                edge_to_k4s_arr.append(np.array(lst, dtype=np.int32))
            else:
                edge_to_k4s_arr.append(np.empty(0, dtype=np.int32))
                
        edge_to_k5s_arr = []
        for lst in edge_to_k5s:
            if lst:
                edge_to_k5s_arr.append(np.array(lst, dtype=np.int32))
            else:
                edge_to_k5s_arr.append(np.empty(0, dtype=np.int32))
                
        # Local Search with vectorized cost calculation
        max_restarts = 100
        current_time = time.time()
        
        found = False
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit - 5.0:
                break
                
            # Random initialization
            colors = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
            
            # Calculate initial cost using vectorized operations
            def calc_cost_vectorized(colors):
                cost = 0
                if num_k4 > 0:
                    # k4_edges is shape (num_k4, 4)
                    # colors[k4_edges] is shape (num_k4, 4)
                    # We need to check if all 4 are 0
                    k4_colors = colors[k4_edges]
                    cost += np.sum(np.all(k4_colors == 0, axis=1))
                
                if num_k5 > 0:
                    # k5_edges is shape (num_k5, 10)
                    # colors[k5_edges] is shape (num_k5, 10)
                    # We need to check if all 10 are 1
                    k5_colors = colors[k5_edges]
                    cost += np.sum(np.all(k5_colors == 1, axis=1))
                
                return cost
                
            cost = calc_cost_vectorized(colors)
            
            if cost == 0:
                return {'n': n, 'colors': colors.tolist()}
                
            # Simulated Annealing / Hill Climbing
            T = 1.0
            T_min = 0.01
            cooling = 0.999
            max_iter_per_restart = 50000
            
            for i in range(max_iter_per_restart):
                if time.time() - start_time > time_limit - 5.0:
                    break
                    
                # Pick random edge
                e = random.randint(0, num_triples - 1)
                
                # Calculate delta cost
                delta = 0
                old_color = colors[e]
                new_color = 1 - old_color
                
                # Check K4s containing e using vectorized operations
                k4_indices = edge_to_k4s_arr[e]
                if len(k4_indices) > 0:
                    # Get the relevant K4 edges
                    relevant_k4s = k4_edges[k4_indices]  # shape (num_relevant, 4)
                    k4_colors = colors[relevant_k4s]  # shape (num_relevant, 4)
                    
                    # Check which K4s were all red (all 0)
                    was_red_mask = np.all(k4_colors == 0, axis=1)
                    delta -= np.sum(was_red_mask)
                    
                    # Check which K4s become all red after flip
                    # Flip the color of edge e in the relevant K4s
                    # e is at some position in each K4's edge list
                    # We need to create a modified copy
                    modified_k4_colors = k4_colors.copy()
                    for j, k4_idx in enumerate(k4_indices):
                        # Find position of e in k4_edges[k4_idx]
                        pos = np.where(k4_edges[k4_idx] == e)[0][0]
                        modified_k4_colors[j, pos] = new_color
                    
                    becomes_red_mask = np.all(modified_k4_colors == 0, axis=1)
                    delta += np.sum(becomes_red_mask)
                
                # Check K5s containing e using vectorized operations
                k5_indices = edge_to_k5s_arr[e]
                if len(k5_indices) > 0:
                    # Get the relevant K5 edges
                    relevant_k5s = k5_edges[k5_indices]  # shape (num_relevant, 10)
                    k5_colors = colors[relevant_k5s]  # shape (num_relevant, 10)
                    
                    # Check which K5s were all blue (all 1)
                    was_blue_mask = np.all(k5_colors == 1, axis=1)
                    delta -= np.sum(was_blue_mask)
                    
                    # Check which K5s become all blue after flip
                    modified_k5_colors = k5_colors.copy()
                    for j, k5_idx in enumerate(k5_indices):
                        # Find position of e in k5_edges[k5_idx]
                        pos = np.where(k5_edges[k5_idx] == e)[0][0]
                        modified_k5_colors[j, pos] = new_color
                    
                    becomes_blue_mask = np.all(modified_k5_colors == 1, axis=1)
                    delta += np.sum(becomes_blue_mask)
                        
                # Accept/Reject
                if delta < 0:
                    colors[e] = new_color
                    cost += delta
                else:
                    if random.random() < np.exp(-delta / T):
                        colors[e] = new_color
                        cost += delta
                        
                T *= cooling
                if T < T_min:
                    break
                    
            if cost == 0:
                return {'n': n, 'colors': colors.tolist()}
            
    # If we didn't find a valid coloring for large n, we need to return a valid one.
    # We can easily construct a valid one for small n.
    # For n=1,2,3: any coloring is valid.
    # Let's just return n=3 with all 0s.
    
    # Fallback: n=3
    triples_3 = list(combinations(range(3), 3))
    return {'n': 3, 'colors': [0] * len(triples_3)}

# EVOLVE-BLOCK-END
