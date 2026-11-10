import random
import time
import math

def is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

N = 256
squares = set()
for i in range(1, 23):
    squares.add(i*i)

conflict = [[False] * (N + 1) for _ in range(N + 1)]
for u in range(1, N + 1):
    for v in range(1, N + 1):
        if (u + v) in squares:
            conflict[u][v] = True

valid_nodes = [i for i in range(1, N + 1) if not conflict[i][i]]

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    best_set = set()
    best_size = 0
    
    # 1. Initial Greedy Randomized Search
    for _ in range(50):
        random.shuffle(valid_nodes)
        current = set()
        for num in valid_nodes:
            compatible = True
            for existing in current:
                if conflict[num][existing]:
                    compatible = False
                    break
            if compatible:
                current.add(num)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
            
    # 2. Local Search with Kicks
    current_best = set(best_set)
    
    while time.time() - start_time < time_limit:
        # Try to add any compatible node
        added = True
        while added:
            added = False
            for num in valid_nodes:
                if num in current_best:
                    continue
                compatible = True
                for existing in current_best:
                    if conflict[num][existing]:
                        compatible = False
                        break
                if compatible:
                    current_best.add(num)
                    added = True
                    break
        
        if len(current_best) > best_size:
            best_size = len(current_best)
            best_set = set(current_best)
            continue
        
        # If stuck, kick
        if len(current_best) >= 3:
            nodes_to_remove = random.sample(list(current_best), min(3, len(current_best)))
            for n in nodes_to_remove:
                current_best.remove(n)
            
            # Re-greedy
            added = True
            while added:
                added = False
                for num in valid_nodes:
                    if num in current_best:
                        continue
                    compatible = True
                    for existing in current_best:
                        if conflict[num][existing]:
                            compatible = False
                            break
                    if compatible:
                        current_best.add(num)
                        added = True
                        break
            
            if len(current_best) > best_size:
                best_size = len(current_best)
                best_set = set(current_best)

    return list(best_set)
