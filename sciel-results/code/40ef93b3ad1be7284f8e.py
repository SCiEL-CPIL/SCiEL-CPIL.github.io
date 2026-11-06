import math
import random

def construct():
    n_max = 256
    
    def is_perfect_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n
    
    # Precompute conflict: conflict[i] = set of j in 1..n_max such that i+j is a perfect square
    adj = [set() for _ in range(n_max + 1)]
    for i in range(1, n_max + 1):
        for j in range(1, n_max + 1):
            if i + j >= 2 and is_perfect_square(i + j):
                adj[i].add(j)
    
    # Greedy MIS: min-degree
    def greedy_mis():
        current = set(range(1, n_max + 1))
        mis = []
        while current:
            min_deg = float('inf')
            best = None
            for v in current:
                deg = len(adj[v] & current)
                if deg < min_deg:
                    min_deg = deg
                    best = v
            if best is None:
                break
            mis.append(best)
            current -= (adj[best] | {best})
        return set(mis)
    
    # Local search to improve
    def local_search(initial_set, iterations=100000):
        s = set(initial_set)
        best_size = len(s)
        
        for _ in range(iterations):
            # Try to add a random element
            candidates = [i for i in range(1, n_max + 1) if i not in s]
            if candidates:
                # Try adding each candidate, see if it's valid
                added = False
                random.shuffle(candidates)
                for c in candidates:
                    if all(not is_perfect_square(c + x) for x in s):
                        s.add(c)
                        if len(s) > best_size:
                            best_size = len(s)
                        added = True
                        break
                if added:
                    continue
            
            # Try swapping: remove one element, add another
            if s:
                list_s = list(s)
                random.shuffle(list_s)
                swapped = False
                for rem in list_s:
                    s.remove(rem)
                    # Now try to add some element
                    candidates2 = [i for i in range(1, n_max + 1) if i not in s]
                    random.shuffle(candidates2)
                    for c in candidates2:
                        if all(not is_perfect_square(c + x) for x in s):
                            s.add(c)
                            if len(s) > best_size:
                                best_size = len(s)
                            swapped = True
                            break
                    if not swapped:
                        s.add(rem)  # revert
                    else:
                        break
                if swapped:
                    continue
            
            # Try removing one and adding two
            # ... this is getting complex
            
            # If no improvement, break or continue
            if not added and not swapped:
                # Try a different strategy
                pass
        
        return s
    
    # Get initial solution
    initial = greedy_mis()
    
    # Run local search
    random.seed(42)
    improved = local_search(initial, iterations=50000)
    
    # Verify
    def is_valid(s):
        for a in s:
            for b in s:
                if a <= b and is_perfect_square(a + b):
                    return False
        return True
    
    if not is_valid(improved):
        # Fallback
        result = []
        for i in range(1, n_max + 1):
            ok = True
            for j in result:
                if is_perfect_square(i + j):
                    ok = False
                    break
            if ok and not is_perfect_square(2 * i):
                result.append(i)
        return sorted(result)
    
    return sorted(improved)
