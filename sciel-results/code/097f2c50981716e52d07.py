"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Normalize edges to frozensets for easy lookup
    original_edge_set = set()
    for u, v in edges:
        original_edge_set.add(frozenset((u, v)))
        
    def greedy_solve(remaining_edges):
        if not remaining_edges:
            return []
            
        rem = set(remaining_edges)
        bicliques = []
        
        while rem:
            # Calculate degrees
            degrees = {i: 0 for i in range(n)}
            for e in rem:
                u, v = tuple(e)
                degrees[u] += 1
                degrees[v] += 1
                
            # Pick edge with max degree sum
            best_edge = None
            best_score = -1
            for e in rem:
                u, v = tuple(e)
                score = degrees[u] + degrees[v]
                if score > best_score:
                    best_score = score
                    best_edge = e
                    
            if not best_edge:
                break
                
            u, v = tuple(best_edge)
            
            # Find maximal biclique containing (u, v)
            L = {u}
            R = {v}
            
            changed = True
            while changed:
                changed = False
                
                # Add to L
                candidates_L = []
                for cand in range(n):
                    if cand in L or cand in R:
                        continue
                    can_add = True
                    for r in R:
                        if frozenset((cand, r)) not in rem:
                            can_add = False
                            break
                    if can_add:
                        deg = sum(1 for e in rem if cand in e)
                        candidates_L.append((deg, cand))
                        
                if candidates_L:
                    candidates_L.sort(reverse=True)
                    _, best_u = candidates_L[0]
                    L.add(best_u)
                    changed = True
                    
                # Add to R
                candidates_R = []
                for cand in range(n):
                    if cand in L or cand in R:
                        continue
                    can_add = True
                    for l in L:
                        if frozenset((l, cand)) not in rem:
                            can_add = False
                            break
                    if can_add:
                        deg = sum(1 for e in rem if cand in e)
                        candidates_R.append((deg, cand))
                        
                if candidates_R:
                    candidates_R.sort(reverse=True)
                    _, best_v = candidates_R[0]
                    R.add(best_v)
                    changed = True
                    
            # Remove edges covered
            for lu in L:
                for rv in R:
                    e = frozenset((lu, rv))
                    if e in rem:
                        rem.remove(e)
                        
            bicliques.append([L, R])
            
        return bicliques

    # Initial solution
    best_solution = greedy_solve(original_edge_set)
    best_len = len(best_solution)
    
    # Iterated Local Search
    start_time = time.time()
    timeout = 55.0
    
    while time.time() - start_time < timeout:
        if best_len == 0:
            break
            
        num_remove = random.randint(1, 2)
        if best_len <= num_remove:
            num_remove = max(1, best_len - 1)
            
        indices_to_remove = set()
        while len(indices_to_remove) < num_remove:
            indices_to_remove.add(random.randint(0, best_len - 1))
            
        current_rem = set(original_edge_set)
        
        for i in range(best_len):
            if i in indices_to_remove:
                continue
            L, R = best_solution[i]
            for lu in L:
                for rv in R:
                    e = frozenset((lu, rv))
                    if e in current_rem:
                        current_rem.remove(e)
                        
        new_bicliques = greedy_solve(current_rem)
        
        new_solution = []
        for i in range(best_len):
            if i in indices_to_remove:
                continue
            new_solution.append(best_solution[i])
        new_solution.extend(new_bicliques)
        
        if len(new_solution) < best_len:
            best_len = len(new_solution)
            best_solution = new_solution
            
    result = []
    for L, R in best_solution:
        L_list = [int(x) for x in sorted(L)]
        R_list = [int(x) for x in sorted(R)]
        result.append([L_list, R_list])
        
    return result
# EVOLVE-BLOCK-END
