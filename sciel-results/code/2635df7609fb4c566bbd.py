import time
from math import lcm
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 85  # Safety margin for 90s limit
    
    N = 256
    values = list(range(2, N + 1))
    n = len(values)
    
    # Precompute conflict: conflict[i][j] = True if lcm(values[i], values[j]) <= 256
    # Use 0-indexed: values[0]=2, values[1]=3, ..., values[254]=256
    conflict = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if lcm(values[i], values[j]) <= 256:
                conflict[i][j] = True
                conflict[j][i] = True
    
    # Weight of vertex i is 1.0 / values[i]
    weights = [1.0 / values[i] for i in range(n)]
    
    def is_conflict_with_all_in_set(i, S):
        """Check if vertex i conflicts with any vertex in set S (set of indices)"""
        for j in S:
            if conflict[i][j]:
                return True
        return False
    
    def greedy_construct(order):
        S = []
        S_set = set()
        for i in order:
            if not is_conflict_with_all_in_set(i, S_set):
                S.append(i)
                S_set.add(i)
        return S
    
    def score(S):
        return sum(weights[i] for i in S)
    
    def local_search(S, max_iterations=1000):
        S_list = list(S)
        S_set = set(S_list)
        best_score = score(S_list)
        
        improved = True
        iteration = 0
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            if time.time() - start_time > timeout:
                break
            
            # Try 1-move: remove one, add one
            # For each u in S, try to find best v not in S that is compatible with S\{u}
            candidates_not_in_S = [i for i in range(n) if i not in S_set]
            
            best_swap_gain = -1.0
            best_swap_u = None
            best_swap_v = None
            
            for u in list(S_set):
                # If we remove u, what's the best v to add?
                # v must be compatible with all in S_set - {u}
                # Gain = weight[v] - weight[u]
                # We want max weight[v] among compatible v
                best_v_for_u = None
                best_v_weight = -1.0
                for v in candidates_not_in_S:
                    if not is_conflict_with_all_in_set(v, S_set - {u}):
                        if weights[v] > best_v_weight:
                            best_v_weight = weights[v]
                            best_v_for_u = v
                
                if best_v_for_u is not None:
                    gain = weights[best_v_for_u] - weights[u]
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap_u = u
                        best_swap_v = best_v_for_u
            
            if best_swap_u is not None and best_swap_gain > 1e-12:
                S_set.remove(best_swap_u)
                S_set.add(best_swap_v)
                S_list = list(S_set)
                best_score += best_swap_gain
                improved = True
                candidates_not_in_S.remove(best_swap_v)
                candidates_not_in_S.append(best_swap_u)
                continue
            
            # Try add-only: add any compatible vertex
            for v in candidates_not_in_S:
                if not is_conflict_with_all_in_set(v, S_set):
                    S_set.add(v)
                    S_list = list(S_set)
                    best_score += weights[v]
                    candidates_not_in_S.remove(v)
                    improved = True
                    break
            
            # Try remove-only: if removing a vertex allows adding a better one, 
            # that's handled by swap. But maybe removing alone helps? No, it decreases score.
            
        return S_list
    
    # Generate diverse initial solutions
    solutions = []
    
    # 1. Greedy by ascending value (smallest first, highest weight)
    order1 = list(range(n))  # values[0]=2, values[1]=3, ...
    solutions.append(greedy_construct(order1))
    
    # 2. Greedy by descending value
    order2 = list(range(n - 1, -1, -1))
    solutions.append(greedy_construct(order2))
    
    # 3. Greedy by weight/conflict_degree
    conflict_degree = [sum(conflict[i][j] for j in range(n)) for i in range(n)]
    order3 = sorted(range(n), key=lambda i: (conflict_degree[i], -weights[i]))
    solutions.append(greedy_construct(order3))
    
    # 4. Greedy by weight/conflict_degree ratio
    order4 = sorted(range(n), key=lambda i: weights[i] / (conflict_degree[i] + 1), reverse=True)
    solutions.append(greedy_construct(order4))
    
    # 5. Random restarts with bias towards high weight
    for _ in range(20):
        if time.time() - start_time > timeout:
            break
        order = list(range(n))
        random.shuffle(order)
        solutions.append(greedy_construct(order))
    
    # Local search on each
    best_S = max(solutions, key=score)
    best_val = score(best_S)
    
    for S in solutions:
        if time.time() - start_time > timeout:
            break
        S_opt = local_search(S)
        val = score(S_opt)
        if val > best_val:
            best_val = val
            best_S = S_opt
    
    # Additional local search iterations on best solution
    if time.time() - start_time < timeout - 5:
        best_S = local_search(best_S, max_iterations=5000)
    
    # Convert back to values
    return [values[i] for i in best_S]
# EVOLVE-BLOCK-END
