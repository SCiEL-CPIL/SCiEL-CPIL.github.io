import time
import random
import numpy as np

def construct():
    """
    Constructs a C4-free subgraph of Q7 maximizing edge count.
    Uses a randomized greedy approach with multiple restarts and local search.
    """
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill
    
    d = 7
    n = 2 ** d  # 128 vertices
    
    # Precompute all edges: (u, v, dim) where u < v and dim is the differing bit
    all_edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y, i))
    
    E = len(all_edges)  # 448
    
    # For each edge, precompute the 3 other edges in each of the 6 C4s it belongs to
    # Edge (x, y) with dim i: C4 is formed by flipping bit j (j != i)
    # The C4 vertices are: x, x^j, y, y^j
    # The 4 edges are: (x, y) [dim i], (x, x^j) [dim j], (y, y^j) [dim j], (x^j, y^j) [dim i]
    
    # Build edge index: map (min(u,v), max(u,v)) -> index in all_edges
    edge_index = {}
    for idx, (u, v, dim) in enumerate(all_edges):
        edge_index[(u, v)] = idx
    
    # For each edge, find all C4s it belongs to and the other 3 edges in each
    edge_c4s = [[] for _ in range(E)]  # edge_c4s[e] = list of (other1, other2, other3)
    
    for idx, (x, y, i) in enumerate(all_edges):
        for j in range(d):
            if j == i:
                continue
            xj = x ^ (1 << j)
            yj = y ^ (1 << j)
            # The 4 edges of this C4:
            e1 = (x, y)  # this edge
            e2 = (min(x, xj), max(x, xj))  # dim j
            e3 = (min(y, yj), max(y, yj))  # dim j
            e4 = (min(xj, yj), max(xj, yj))  # dim i
            
            idx2 = edge_index[e2]
            idx3 = edge_index[e3]
            idx4 = edge_index[e4]
            
            edge_c4s[idx].append((idx2, idx3, idx4))
    
    best_count = 0
    best_edges = []
    
    def greedy_randomized(seed):
        """Randomized greedy: process edges in random order, add if no C4 is completed."""
        random.seed(seed)
        order = list(range(E))
        random.shuffle(order)
        
        chosen = [False] * E
        count = 0
        
        for idx in order:
            # Check if adding this edge would complete any C4
            ok = True
            for (o1, o2, o3) in edge_c4s[idx]:
                if chosen[o1] and chosen[o2] and chosen[o3]:
                    ok = False
                    break
            if ok:
                chosen[idx] = True
                count += 1
        
        return chosen, count
    
    def simulated_annealing(initial_chosen, initial_count, seed, time_budget):
        """
        Simulated Annealing to improve the edge set.
        Moves: Add an edge, Remove an edge, Swap an edge.
        """
        random.seed(seed)
        chosen = initial_chosen[:]
        count = initial_count
        
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.999
        
        end_time = time.time() + time_budget
        best_local_chosen = chosen[:]
        best_local_count = count
        
        while time.time() < end_time and T > T_min:
            # Choose a move type
            move_type = random.random()
            
            if move_type < 0.4:
                # Add an edge
                idx = random.randint(0, E - 1)
                if chosen[idx]:
                    continue
                
                # Check if adding idx is valid
                valid = True
                for (o1, o2, o3) in edge_c4s[idx]:
                    if chosen[o1] and chosen[o2] and chosen[o3]:
                        valid = False
                        break
                
                if valid:
                    chosen[idx] = True
                    count += 1
                    if count > best_local_count:
                        best_local_count = count
                        best_local_chosen = chosen[:]
                        
            elif move_type < 0.7:
                # Remove an edge
                idx = random.randint(0, E - 1)
                if not chosen[idx]:
                    continue
                
                chosen[idx] = False
                new_count = count - 1
                
                # Acceptance probability for worsening move
                # Delta = -1
                # Prob = exp(-1 / T)
                if random.random() < np.exp(-1.0 / T):
                    count = new_count
                else:
                    # Reject
                    chosen[idx] = True
                    
            else:
                # Swap: Remove one, Add another
                idx_out = random.randint(0, E - 1)
                if not chosen[idx_out]:
                    continue
                
                # Pick idx_in from not chosen
                idx_in = random.randint(0, E - 1)
                if chosen[idx_in]:
                    continue
                
                # Tentatively remove idx_out
                chosen[idx_out] = False
                
                # Check if idx_in can be added
                valid = True
                for (o1, o2, o3) in edge_c4s[idx_in]:
                    if chosen[o1] and chosen[o2] and chosen[o3]:
                        valid = False
                        break
                
                if valid:
                    chosen[idx_in] = True
                    # Count remains same
                    # Accept swap to explore
                else:
                    # Reject
                    chosen[idx_out] = True
                
            T *= cooling_rate
            
        return best_local_chosen, best_local_count
    
    # Multiple restarts
    seed = 42
    while time.time() - start_time < time_limit - 5:
        chosen, count = greedy_randomized(seed)
        
        if count > best_count:
            best_count = count
            best_edges = [all_edges[i][0] for i in range(E) if chosen[i]]
            best_edges = [[all_edges[i][0], all_edges[i][1]] for i in range(E) if chosen[i]]
        
        # Local Search: Simulated Annealing
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 10:
            sa_time = remaining_time * 0.8
            improved_chosen, improved_count = simulated_annealing(chosen, count, seed, sa_time)
            if improved_count > best_count:
                best_count = improved_count
                best_edges = [[all_edges[i][0], all_edges[i][1]] for i in range(E) if improved_chosen[i]]
        
        seed += 1
    
    # Return the best result
    return best_edges
