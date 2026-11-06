import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    N = 256
    elements = list(range(1, N + 1))
    
    # Build conflict graph: conflict[a][b] = True if a+b is a perfect square
    # a and b can be equal (diagonal)
    # Perfect squares in [2, 512]
    squares = set()
    k = 2
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
    
    # conflict[a] = set of b values (b >= a) such that a + b is a square
    # Actually, let's make it symmetric: conflict[a][b] for all pairs
    # Use a list of sets for efficiency
    conflict = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            s = a + b
            if s in squares:
                conflict[a].add(b)
                if a != b:
                    conflict[b].add(a)
    
    def is_valid_solution(sol_set):
        """Verify a solution is valid."""
        sol_list = sorted(sol_set)
        for i in range(len(sol_list)):
            for j in range(i, len(sol_list)):
                if sol_list[i] + sol_list[j] in squares:
                    return False
        return True
    
    def count_conflicts_in_solution(sol_set, candidate):
        """Count how many elements in sol_set conflict with candidate."""
        cnt = 0
        for x in sol_set:
            if x in conflict[candidate]:
                cnt += 1
        return cnt
    
    def greedy_construct(order):
        """Greedy: process elements in given order, add if no conflict."""
        sol = set()
        for x in order:
            can_add = True
            for y in sol:
                if y in conflict[x]:
                    can_add = False
                    break
            if can_add:
                sol.add(x)
        return sol
    
    def local_search(sol_set, max_iter=5000):
        """Local search: try to add elements, then swap, then remove+add."""
        sol = set(sol_set)
        improved = True
        iters = 0
        while improved and iters < max_iter:
            improved = False
            iters += 1
            if time.time() > deadline - 5:
                break
            
            # Try to add any element not in sol
            added = False
            for x in elements:
                if x not in sol:
                    can_add = True
                    for y in sol:
                        if y in conflict[x]:
                            can_add = False
                            break
                    if can_add:
                        sol.add(x)
                        improved = True
                        added = True
                        break
            
            if added:
                continue
            
            # Try swap: remove one, add one
            sol_list = sorted(sol)
            for i in range(len(sol_list)):
                if time.time() > deadline - 5:
                    break
                x = sol_list[i]
                # Try adding each element not in sol
                for y in elements:
                    if y in sol:
                        continue
                    # Check if y conflicts with sol minus x
                    can_add_y = True
                    for z in sol:
                        if z == x:
                            continue
                        if z in conflict[y]:
                            can_add_y = False
                            break
                    if can_add_y:
                        sol.discard(x)
                        sol.add(y)
                        improved = True
                        break
                if improved:
                    break
            
            # Try remove one, add two
            if not improved:
                sol_list = sorted(sol)
                for i in range(len(sol_list)):
                    if time.time() > deadline - 5:
                        break
                    x = sol_list[i]
                    # Try adding two elements not in sol
                    for y in elements:
                        if y in sol:
                            continue
                        for z in elements:
                            if z in sol or z <= y:
                                continue
                            # Check y and z don't conflict with sol-minus-x, and y,z don't conflict
                            ok = True
                            for w in sol:
                                if w == x:
                                    continue
                                if w in conflict[y] or w in conflict[z]:
                                    ok = False
                                    break
                            if ok and y in conflict[z]:
                                ok = False
                            if ok:
                                sol.discard(x)
                                sol.add(y)
                                sol.add(z)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
    
        return sol
    
    best_sol = set()
    best_size = 0
    
    # Multiple restarts with different strategies
    restart = 0
    while time.time() < deadline - 10:
        restart += 1
        
        if restart == 1:
            # First: greedy by min degree (fewest conflicts first)
            degrees = [len(conflict[x]) for x in range(1, N + 1)]
            order = sorted(elements, key=lambda x: degrees[x])
            sol = greedy_construct(order)
        elif restart == 2:
            # Greedy by max degree
            degrees = [len(conflict[x]) for x in range(1, N + 1)]
            order = sorted(elements, key=lambda x: -degrees[x])
            sol = greedy_construct(order)
        elif restart == 3:
            # Random order
            order = elements[:]
            random.shuffle(order)
            sol = greedy_construct(order)
        elif restart == 4:
            # Greedy: prefer elements with fewer conflicts among remaining
            order = elements[:]
            random.shuffle(order)
            sol = greedy_construct(order)
        else:
            # Random
            order = elements[:]
            random.shuffle(order)
            sol = greedy_construct(order)
        
        # Local search
        sol = local_search(sol)
        
        if len(sol) > best_size:
            best_size = len(sol)
            best_sol = set(sol)
        
        # If we found a large solution, do more local search
        if len(sol) >= best_size and time.time() < deadline - 15:
            sol = local_search(sol, max_iter=20000)
            if len(sol) > best_size:
                best_size = len(sol)
                best_sol = set(sol)
    
    # Final verification
    if not is_valid_solution(best_sol):
        # Fallback: simple valid solution
        best_sol = set(range(1, 257, 3))
    
    return list(best_sol)
