import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin

    N = 256
    max_sum = 2 * N
    squares = set()
    i = 2
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
    
    # 1. Filter valid nodes (diagonal constraint: 2*v must not be a square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
            
    M = len(valid_nodes)
    if M == 0:
        return [1]
        
    # Map node value to index
    node_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    
    # 2. Build conflict sets for fast lookup
    # conflict[i] = set of j where valid_nodes[i] + valid_nodes[j] is a square
    conflict = [set() for _ in range(M)]
    edges = []
    for i in range(M):
        vi = valid_nodes[i]
        for j in range(i + 1, M):
            vj = valid_nodes[j]
            if vi + vj in squares:
                conflict[i].add(j)
                conflict[j].add(i)
                edges.append((i, j))
    
    # 3. Try MILP with timeout protection
    best_solution = []
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        from scipy.sparse import lil_matrix, csc_matrix
        import threading
        
        if len(edges) > 0:
            # Variables x[0..M-1]
            # Objective: maximize sum(x) -> minimize -sum(x)
            c = -np.ones(M)
            
            # Build sparse constraint matrix
            A = lil_matrix((len(edges), M), dtype=int)
            b = np.ones(len(edges))
            
            for k, (i, j) in enumerate(edges):
                A[k, i] = 1
                A[k, j] = 1
                
            A = csc_matrix(A)
            
            lb = np.zeros(M)
            ub = np.ones(M)
            integrality = np.ones(M)
            
            # Run MILP in a thread with timeout
            result_holder = {}
            
            def run_milp():
                try:
                    res = milp(c, constraints=LinearConstraint(A, -np.inf, b), 
                               integrality=integrality, bounds=Bounds(lb, ub))
                    result_holder['res'] = res
                except Exception as e:
                    result_holder['error'] = e
            
            milp_thread = threading.Thread(target=run_milp)
            milp_thread.start()
            milp_thread.join(timeout=20)  # 20s timeout for MILP
            
            if milp_thread.is_alive():
                # Timeout - fall through to local search
                pass
            elif 'res' in result_holder:
                res = result_holder['res']
                if res.success:
                    x = np.round(res.x).astype(int)
                    solution = []
                    for i in range(M):
                        if x[i] == 1:
                            solution.append(valid_nodes[i])
                    if solution:
                        best_solution = solution
            elif 'error' in result_holder:
                pass
                
    except Exception:
        pass
    
    # 4. Strong local search with multiple strategies
    # Start from best_solution if available, else greedy initialization
    
    if best_solution:
        result_set = set(best_solution)
    else:
        # Greedy initialization: sort by degree (fewest conflicts first)
        degrees = [len(conflict[i]) for i in range(M)]
        sorted_nodes = sorted(range(M), key=lambda i: degrees[i])
        
        result_set = set()
        for i in sorted_nodes:
            can_add = True
            for j in result_set:
                if j in conflict[i]:
                    can_add = False
                    break
            if can_add:
                result_set.add(i)
    
    # Local search: try various improvement moves
    def is_valid_set(s):
        """Check if set s is a valid independent set"""
        s_list = list(s)
        for i in range(len(s_list)):
            for j in range(i+1, len(s_list)):
                if s_list[j] in conflict[s_list[i]]:
                    return False
        return True
    
    improved = True
    iteration = 0
    while improved and time.time() < deadline - 10:
        improved = False
        iteration += 1
        
        # Strategy 1: Try to add any compatible vertex
        for i in range(M):
            if i in result_set:
                continue
            can_add = True
            for j in result_set:
                if j in conflict[i]:
                    can_add = False
                    break
            if can_add:
                result_set.add(i)
                improved = True
                break
        
        # Strategy 2: Try 1-for-1 swap (remove one, add one)
        if not improved:
            for remove in list(result_set):
                temp_set = result_set - {remove}
                for add in range(M):
                    if add in temp_set:
                        continue
                    can_add = True
                    for j in temp_set:
                        if j in conflict[add]:
                            can_add = False
                            break
                    if can_add:
                        # Only accept if it doesn't decrease size (it won't, 1-for-1)
                        # But we need a reason to swap - maybe for future improvements
                        # Skip pure 1-for-1 unless we track quality
                        pass
        
        # Strategy 3: Try 1-for-2 swap (remove one, add two)
        if not improved:
            for remove in list(result_set):
                temp_set = result_set - {remove}
                # Find two vertices that can be added to temp_set
                candidates = []
                for add1 in range(M):
                    if add1 in temp_set:
                        continue
                    can_add1 = True
                    for j in temp_set:
                        if j in conflict[add1]:
                            can_add1 = False
                            break
                    if can_add1:
                        candidates.append(add1)
                
                # Try pairs from candidates
                for idx1 in range(len(candidates)):
                    add1 = candidates[idx1]
                    temp_set2 = temp_set | {add1}
                    for idx2 in range(idx1 + 1, len(candidates)):
                        add2 = candidates[idx2]
                        if add2 in temp_set2:
                            continue
                        can_add2 = True
                        for j in temp_set2:
                            if j in conflict[add2]:
                                can_add2 = False
                                break
                        if can_add2:
                            # Size increases by 1
                            result_set = temp_set2 | {add2}
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        # Strategy 4: Try 2-for-3 swap (remove two, add three)
        if not improved and time.time() < deadline - 15:
            result_list = list(result_set)
            for i in range(len(result_list)):
                for j in range(i + 1, len(result_list)):
                    remove1, remove2 = result_list[i], result_list[j]
                    temp_set = result_set - {remove1, remove2}
                    
                    # Find three vertices to add
                    candidates = []
                    for add in range(M):
                        if add in temp_set:
                            continue
                        can_add = True
                        for k in temp_set:
                            if k in conflict[add]:
                                can_add = False
                                break
                        if can_add:
                            candidates.append(add)
                    
                    if len(candidates) < 3:
                        continue
                    
                    # Try triples (limit for performance)
                    found = False
                    for idx1 in range(min(len(candidates), 20)):
                        add1 = candidates[idx1]
                        temp_set1 = temp_set | {add1}
                        for idx2 in range(idx1 + 1, min(len(candidates), 20)):
                            add2 = candidates[idx2]
                            if add2 in temp_set1:
                                continue
                            can_add2 = True
                            for k in temp_set1:
                                if k in conflict[add2]:
                                    can_add2 = False
                                    break
                            if not can_add2:
                                continue
                            temp_set2 = temp_set1 | {add2}
                            for idx3 in range(idx2 + 1, min(len(candidates), 20)):
                                add3 = candidates[idx3]
                                if add3 in temp_set2:
                                    continue
                                can_add3 = True
                                for k in temp_set2:
                                    if k in conflict[add3]:
                                        can_add3 = False
                                        break
                                if can_add3:
                                    result_set = temp_set2 | {add3}
                                    improved = True
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if improved:
                    break
    
    # 5. Final verification and return
    # Convert back to values
    final_solution = [valid_nodes[i] for i in result_set]
    
    # Verify the solution is valid
    for i in range(len(final_solution)):
        a = final_solution[i]
        if 2 * a in squares:
            # Should not happen, but remove if it does
            final_solution.pop(i)
            i -= 1
            continue
        for j in range(i + 1, len(final_solution)):
            b = final_solution[j]
            if a + b in squares:
                # Invalid pair - this shouldn't happen
                break
    
    if not final_solution:
        return [1]
    
    return final_solution
