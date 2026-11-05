import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 256
    numbers = list(range(1, n + 1))
    
    # Build conflict adjacency list
    conflict_adj = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                conflict_adj[a].add(b)
                conflict_adj[b].add(a)
    
    # Initial: all odds + greedy evens that don't conflict
    selected = set(x for x in numbers if x % 2 == 1)
    for e in range(2, n + 1, 2):
        if not (conflict_adj[e] & selected):
            selected.add(e)
    
    best_set = set(selected)
    best_size = len(selected)
    
    def run_ls(start):
        current = set(start)
        # icc[x] = count of x's conflict-neighbors currently in the set
        icc = [0] * (n + 1)
        for x in current:
            for nb in conflict_adj[x]:
                icc[nb] += 1
        
        for _ in range(2000):
            if time.time() - start_time > deadline - 3:
                break
            
            # Try to add any element with zero conflicts in set
            added = False
            for x in range(1, n + 1):
                if x not in current and icc[x] == 0:
                    current.add(x)
                    for nb in conflict_adj[x]:
                        icc[nb] += 1
                    added = True
                    break
            
            if added:
                continue
            
            # Try swap: remove one, add another
            swapped = False
            for rem in list(current):
                # Temporarily remove rem's contribution
                for nb in conflict_adj[rem]:
                    icc[nb] -= 1
                # Find an add candidate
                for add in range(1, n + 1):
                    if add not in current and add != rem and icc[add] == 0:
                        # Commit swap
                        current.discard(rem)
                        current.add(add)
                        for nb in conflict_adj[add]:
                            icc[nb] += 1
                        swapped = True
                        break
                if swapped:
                    break
                # Undo temporary removal
                for nb in conflict_adj[rem]:
                    icc[nb] += 1
            
            if not swapped:
                break
        
        return current
    
    # Multiple restarts with perturbation
    for restart in range(40):
        if time.time() - start_time > deadline - 3:
            break
        if restart == 0:
            start = set(best_set)
        else:
            start = set(best_set)
            k = random.randint(2, 5)
            for _ in range(k):
                if start:
                    start.discard(random.choice(list(start)))
            for _ in range(k):
                cands = [x for x in numbers if x not in start]
                if cands:
                    start.add(random.choice(cands))
        
        result = run_ls(start)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
    
    return list(best_set)
