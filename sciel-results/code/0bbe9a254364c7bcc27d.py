import math
import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = 128
    elements = list(range(1, n + 1))
    
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    # 1. Precompute bad triples
    conflicts = {x: [] for x in elements}
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm(a, b)
            for c in range(b + 1, n + 1):
                if lcm(a, c) == lab and lcm(b, c) == lab:
                    conflicts[a].append((b, c))
                    conflicts[b].append((a, c))
                    conflicts[c].append((a, b))
    
    # 2. Helper to check if adding x to S is valid
    def can_add(x, S):
        for y, z in conflicts[x]:
            if y in S and z in S:
                return False
        return True
    
    # Check if adding y to (S \ removed_set) is valid
    def can_add_after_removal(y, S, removed_set):
        for u, v in conflicts[y]:
            if u in removed_set or v in removed_set:
                continue
            if u in S and v in S:
                return False
        return True

    # 3. Greedy Initialization
    def greedy_init(order):
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S

    # 4. Local Search with 1-swap and 2-opt moves
    def local_search(S, max_iters=1000):
        S = set(S)
        best_S = set(S)
        best_size = len(S)
        current_size = len(S)
        
        no_improve_count = 0
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try pure addition
            for y in elements:
                if y not in S and can_add(y, S):
                    S.add(y)
                    current_size += 1
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_S = set(S)
                    break
            
            if improved:
                no_improve_count = 0
                continue
            
            # Try 1-swap: remove x in S, add y not in S
            found_swap = False
            xs = list(S)
            random.shuffle(xs)
            ys = [y for y in elements if y not in S]
            random.shuffle(ys)
            
            for y in ys:
                if found_swap:
                    break
                for x in xs:
                    if time.time() > deadline:
                        return best_S
                    
                    if can_add_after_removal(y, S, {x}):
                        S.remove(x)
                        S.add(y)
                        improved = True
                        found_swap = True
                        break
            
            if improved:
                no_improve_count = 0
                continue
            
            # Try 2-opt: remove x1, x2 in S, add y1, y2 not in S
            # This can escape local optima unreachable by 1-swap
            found_2opt = False
            xs2 = list(S)
            random.shuffle(xs2)
            ys2 = [y for y in elements if y not in S]
            random.shuffle(ys2)
            
            # Limit the search space for 2-opt to keep it tractable
            num_x_to_try = min(len(xs2), 20)
            num_y_to_try = min(len(ys2), 20)
            
            for i in range(num_x_to_try):
                if found_2opt:
                    break
                x1 = xs2[i]
                for j in range(i + 1, num_x_to_try):
                    if found_2opt:
                        break
                    x2 = xs2[j]
                    removed = {x1, x2}
                    
                    for y1 in ys2[:num_y_to_try]:
                        if found_2opt:
                            break
                        # Check if y1 can be added after removing x1, x2
                        if not can_add_after_removal(y1, S, removed):
                            continue
                        
                        for y2 in ys2[:num_y_to_try]:
                            if y2 == y1:
                                continue
                            if found_2opt:
                                break
                            # Check if y2 can be added after removing x1, x2 and adding y1
                            # We need to check conflicts with S \ {x1,x2} U {y1}
                            # Simplified: check y2 against S \ {x1,x2} (ignoring y1 for speed, 
                            # then verify the full set)
                            if not can_add_after_removal(y2, S, removed):
                                continue
                            
                            # Verify the full move is valid
                            # New set = (S - {x1, x2}) U {y1, y2}
                            # Check all conflicts for y1 and y2 against the new set
                            valid = True
                            for u, v in conflicts[y1]:
                                if u == x1 or u == x2 or v == x1 or v == x2:
                                    continue
                                if u in S and v in S:
                                    valid = False
                                    break
                                if u == y2 or v == y2:
                                    # This would be a conflict between y1 and y2, but they're both added
                                    # Check if (y1, y2, w) forms a bad triple for some w in S\{x1,x2}
                                    pass
                            
                            if valid:
                                for u, v in conflicts[y2]:
                                    if u == x1 or u == x2 or v == x1 or v == x2:
                                        continue
                                    if u in S and v in S:
                                        valid = False
                                        break
                            
                            if valid:
                                S.remove(x1)
                                S.remove(x2)
                                S.add(y1)
                                S.add(y2)
                                improved = True
                                found_2opt = True
                                break
            
            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 30:
                    break
        
        return best_S

    best_result = set()
    
    # Multiple restarts with different initializations
    random.seed(42)
    
    # 1. Natural order
    S = greedy_init(elements)
    best_result = local_search(S, max_iters=500)
    
    # 2. Reverse order
    S = greedy_init(list(reversed(elements)))
    ls_result = local_search(S, max_iters=500)
    if len(ls_result) > len(best_result):
        best_result = ls_result
    
    # 3. Random orders
    for i in range(80):
        if time.time() > deadline:
            break
        
        order = elements[:]
        random.shuffle(order)
        S = greedy_init(order)
        
        improved_S = local_search(S, max_iters=300)
        if len(improved_S) > len(best_result):
            best_result = improved_S
    
    # Final polish on the best result with more iterations
    if time.time() < deadline - 5:
        final_S = local_search(best_result, max_iters=1000)
        if len(final_S) > len(best_result):
            best_result = final_S
    
    return list(best_result)
