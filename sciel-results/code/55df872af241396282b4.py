import time
import random

def construct():
    start_time = time.time()
    deadline = 55.0
    
    def check_add(S, x):
        # Returns True if x can be added to S as the smallest element
        for b in S:
            den = b - x
            if den == 0: continue
            num = x * b
            if num % den != 0: continue
            c = num // den
            if c in S and c != b and c != x:
                return False
        return True

    def check_add_largest(S, x):
        # x is largest. Check if x forms a triple with any pair in S
        for a in S:
            den = x - a
            if den == 0: continue
            num = a * x
            if num % den != 0: continue
            c = num // den
            if c in S and c != a and c != x:
                return False
        return True

    def check_add_any(S, x):
        if not check_add(S, x):
            return False
        if not check_add_largest(S, x):
            return False
        return True

    def greedy_random():
        items = list(range(1, 181))
        random.shuffle(items)
        S = set()
        for x in items:
            if check_add_any(S, x):
                S.add(x)
        return S

    def greedy_backward():
        S = set()
        for x in range(180, 0, -1):
            if check_add(S, x):
                S.add(x)
        return S

    def greedy_forward_fixed():
        S = set()
        for x in range(1, 181):
            if check_add_largest(S, x):
                S.add(x)
        return S

    def local_search(S, max_iter=2000):
        """Improve S by trying to add/remove elements."""
        S = set(S)
        best = set(S)
        best_size = len(S)
        
        for iteration in range(max_iter):
            if time.time() - start_time > deadline - 3:
                break
            
            improved = False
            
            # Try adding any element not in S
            for x in range(1, 181):
                if x not in S and check_add_any(S, x):
                    S.add(x)
                    improved = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best = set(S)
            
            # If no additions possible, try removing one element to enable more
            if not improved:
                for x in list(S):
                    S2 = S - {x}
                    count_added = 0
                    for y in range(1, 181):
                        if y not in S2 and y != x and check_add_any(S2, y):
                            S2.add(y)
                            count_added += 1
                    if len(S2) > len(S):
                        S = S2
                        improved = True
                        if len(S) > best_size:
                            best_size = len(S)
                            best = set(S)
                        break
            
            if not improved:
                break
        
        return best

    best = greedy_backward()
    best_size = len(best)
    
    S2 = greedy_forward_fixed()
    if len(S2) > best_size:
        best = S2
        best_size = len(S2)

    # Random restarts with local search
    count = 0
    while time.time() - start_time < deadline - 8:
        S_r = greedy_random()
        S_improved = local_search(S_r, max_iter=500)
        
        if len(S_improved) > best_size:
            best = S_improved
            best_size = len(S_improved)
        
        count += 1
        if count > 1500: break

    # Final polish on the best solution
    if time.time() - start_time < deadline - 2:
        best_final = local_search(best, max_iter=3000)
        if len(best_final) > best_size:
            best = best_final
            best_size = len(best_final)

    return list(best)
