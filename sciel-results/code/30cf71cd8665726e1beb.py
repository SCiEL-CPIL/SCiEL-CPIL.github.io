import random
import math
import time

def construct():
    def score(S):
        present = set(S)
        list_S = list(S)
        n = len(list_S)
        diffs = set()
        
        for i in range(n):
            a = list_S[i]
            for j in range(i+1, n):
                c = list_S[j]
                s = a + c
                if s % 2 == 0:
                    b = s // 2
                    if b in present:
                        d = c - a
                        diffs.add(d)
        return len(diffs)

    def random_set():
        return set(random.sample(range(1, 513), 32))

    def mutate(S):
        S = set(S)
        # Pick a random element to replace
        x = random.choice(list(S))
        S.remove(x)
        # Pick a new random element
        y = random.randint(1, 512)
        while y in S:
            y = random.randint(1, 512)
        S.add(y)
        return S

    start_time = time.time()
    time_limit = 110
    
    # Initialize with a few random restarts to pick a good starting point
    best_S = set()
    best_score = 0
    
    for _ in range(5):
        S = random_set()
        sc = score(S)
        if sc > best_score:
            best_score = sc
            best_S = S

    current_S = set(best_S)
    current_score = best_score
    
    T = 5.0
    cooling_rate = 0.9995
    
    while time.time() - start_time < time_limit:
        new_S = mutate(current_S)
        new_score = score(new_S)
        
        delta = new_score - current_score
        
        if delta > 0 or (T > 0.01 and random.random() < math.exp(delta / T)):
            current_S = new_S
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_S = set(current_S)
        
        T *= cooling_rate
        if T < 0.01:
            # Reheat slightly or reset?
            # If stuck, maybe restart from best?
            # For simplicity, just keep T at floor
            T = 0.01

    return list(best_S)
