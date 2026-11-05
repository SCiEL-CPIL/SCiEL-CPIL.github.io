import random
import math

def construct():
    n = 256
    # 1. Generate initial greedy solution (smallest first)
    S = set()
    for x in range(1, n + 1):
        valid = True
        for a in S:
            for b in S:
                if a < b:
                    if 2 * b - a == x:
                        valid = False
                        break
            if not valid:
                break
        if valid:
            S.add(x)
    
    current_score = sum(1.0 / x for x in S)
    best_S = set(S)
    best_score = current_score
    
    # Precompute reciprocals
    recips = [0.0] * (n + 1)
    for i in range(1, n + 1):
        recips[i] = 1.0 / i
        
    def is_valid_swap(S, x, y):
        # Check if S' = S - {y} + {x} is valid.
        # Only APs involving x need to be checked.
        # y is removed.
        
        for a in S:
            if a == y:
                continue
            # a is in S'
            
            # Case 1: x, a, b (x is smallest)
            # b = 2a - x
            b = 2 * a - x
            if b > a and b in S and b != y:
                return False
                
            # Case 2: a, x, b (x is middle)
            # b = 2x - a
            b = 2 * x - a
            if b > x and b in S and b != y:
                return False
                
            # Case 3: a, b, x (x is largest)
            # b = (a + x) / 2
            if (a + x) % 2 == 0:
                b = (a + x) // 2
                if a < b < x and b in S and b != y:
                    return False
                    
        return True

    T = 1.0
    cooling = 0.999
    min_T = 1e-6
    steps = 50000
    
    for step in range(steps):
        if T < min_T:
            break
            
        # Pick random x not in S, y in S
        x = random.randint(1, n)
        
        if x in S:
            # Pick a new x not in S
            while x in S:
                x = random.randint(1, n)
                
            y_list = list(S)
            if not y_list:
                continue
            y = random.choice(y_list)
            
            if is_valid_swap(S, x, y):
                new_score = current_score - recips[y] + recips[x]
                delta = new_score - current_score
                if delta > 0 or random.random() < math.exp(-delta / T):
                    S.remove(y)
                    S.add(x)
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_S = set(S)
        else:
            # x not in S. Pick y in S.
            y_list = list(S)
            if not y_list:
                continue
            y = random.choice(y_list)
            
            if is_valid_swap(S, x, y):
                new_score = current_score - recips[y] + recips[x]
                delta = new_score - current_score
                if delta > 0 or random.random() < math.exp(-delta / T):
                    S.remove(y)
                    S.add(x)
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_S = set(S)
                        
        T *= cooling
        
    return sorted(best_S)
