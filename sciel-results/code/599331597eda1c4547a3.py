import random
import math
import time

def construct():
    def score(S):
        # S is a set of ints
        # Use a boolean array for O(1) lookup
        present = [False] * 513
        for x in S:
            present[x] = True
            
        diffs = set()
        list_S = list(S)
        n = len(list_S)
        
        for i in range(n):
            a = list_S[i]
            for j in range(i+1, n):
                c = list_S[j]
                s = a + c
                if s % 2 == 0:
                    b = s // 2
                    if present[b]:
                        d = c - a
                        diffs.add(d)
        return len(diffs)

    def hill_climb(initial_S, max_batches=1000):
        current_S = set(initial_S)
        current_score = score(current_S)
        best_S = set(current_S)
        best_score = current_score
        
        for _ in range(max_batches):
            improved = False
            # Try a few random moves to find an improvement
            for _ in range(50):
                if random.random() < 0.5:
                    # Swap move: remove 2, add 2
                    items = list(current_S)
                    i = random.randint(0, 31)
                    j = random.randint(0, 31)
                    if i == j:
                        continue
                    x = items[i]
                    y = items[j]
                    
                    u = random.randint(1, 512)
                    v = random.randint(1, 512)
                    
                    if u in current_S or v in current_S or u == v:
                        continue
                        
                    new_S = set(current_S)
                    new_S.remove(x)
                    new_S.remove(y)
                    new_S.add(u)
                    new_S.add(v)
                else:
                    # Single move: remove 1, add 1
                    x = random.choice(list(current_S))
                    u = random.randint(1, 512)
                    
                    if u in current_S:
                        continue
                        
                    new_S = set(current_S)
                    new_S.remove(x)
                    new_S.add(u)
                
                new_score = score(new_S)
                
                if new_score > current_score:
                    current_S = new_S
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_S = set(current_S)
                    improved = True
                    break
                    
            if not improved:
                # Local optimum reached
                break
                
        return list(best_S), best_score

    start_time = time.time()
    time_limit = 110  # Leave buffer for 120s limit
    
    best_overall_S = []
    best_overall_score = 0
    
    # Run multiple restarts
    while time.time() - start_time < time_limit:
        # Random initialization
        init_S = set(random.sample(range(1, 513), 32))
        
        # Occasionally use a structured init (e.g., consecutive integers)
        # which has a known decent score, to provide a strong baseline
        if random.random() < 0.1:
            init_S = set(range(1, 33))
            
        S, sc = hill_climb(init_S, max_batches=500)
        
        if sc > best_overall_score:
            best_overall_score = sc
            best_overall_S = S
            
    return best_overall_S
