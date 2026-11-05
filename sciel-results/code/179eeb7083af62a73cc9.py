import time
import math
import random

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    N = 512
    
    # Precompute gcd table for O(1) coprime checks
    gcd_table = [[0]*N for _ in range(N)]
    for i in range(1, N):
        for j in range(i+1, N):
            gcd_table[i][j] = math.gcd(i, j)
    
    def get_score(s):
        return sum(1.0 / (N - a) for a in s)
    
    def greedy_init(order):
        chosen = []
        for a in order:
            if all(gcd_table[a][b] == 1 for b in chosen):
                chosen.append(a)
        return chosen
    
    # Multiple initialization strategies
    random.seed(42)
    
    # Strategy 1: Largest to smallest
    order1 = list(range(N-1, 0, -1))
    init1 = greedy_init(order1)
    
    # Strategy 2: Randomized (several tries)
    best_random_init = []
    best_random_score = 0
    for trial in range(5):
        order = list(range(N-1, 0, -1))
        random.shuffle(order)
        init = greedy_init(order)
        s = get_score(init)
        if s > best_random_score:
            best_random_score = s
            best_random_init = init
    
    # Strategy 3: Largest to smallest with slight random perturbation
    order3 = list(range(N-1, 0, -1))
    for i in range(len(order3)-1, 0, -1):
        if random.random() < 0.1:  # 10% chance to swap
            j = random.randint(0, i)
            order3[i], order3[j] = order3[j], order3[i]
    init3 = greedy_init(order3)
    
    candidates = [init1, best_random_init, init3]
    best_init = max(candidates, key=get_score)
    best_score = get_score(best_init)
    best_chosen = list(best_init)
    
    # Local search: 1-swap moves
    current = list(best_init)
    current_set = set(current)
    current_score = best_score
    
    while time.time() < deadline:
        improved = False
        for i in range(len(current)):
            if time.time() > deadline:
                break
            a = current[i]
            temp = [x for x in current if x != a]
            temp_set = current_set - {a}
            
            best_b = -1
            for b in range(N-1, a, -1):
                if b in temp_set:
                    continue
                if all(gcd_table[b][c] == 1 for c in temp):
                    best_b = b
                    break
            
            if best_b != -1:
                new_score = current_score - 1.0/(N-a) + 1.0/(N-best_b)
                if new_score > current_score + 1e-12:
                    current = temp
                    current.append(best_b)
                    current_set.remove(a)
                    current_set.add(best_b)
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_chosen = list(current)
                    break
        
        if not improved:
            break
    
    return best_chosen
