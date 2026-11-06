import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    def greedy(order):
        selected = []
        reachable = {0}
        for x in order:
            # Optimization: if reachable is large, set ops are slow.
            # But 200 is small.
            new = set()
            for s in reachable:
                ns = s + x
                if ns <= 200:
                    new.add(ns)
            if 200 not in new:
                selected.append(x)
                reachable |= new
        return selected

    # Initial best: Deterministic Greedy Descending
    best_set = greedy(range(100, 0, -1))
    best_size = len(best_set)
    
    # Randomized Search
    # Use a fixed seed for reproducibility in local dev, but random in eval?
    # The prompt implies we just return the candidate.
    
    trials = 0
    while time.time() < deadline:
        trials += 1
        # Strategy: Mix of random and perturbed descending
        r = random.random()
        if r < 0.3:
            # Pure random
            perm = list(range(1, 101))
            random.shuffle(perm)
            order = perm
        elif r < 0.6:
            # Descending with swaps
            order = list(range(100, 0, -1))
            for _ in range(5): # 5 random swaps
                i = random.randint(0, 98)
                j = random.randint(0, 98)
                order[i], order[j] = order[j], order[i]
        else:
            # Ascending? (Unlikely to be better, but explore)
            # Or "Randomized Greedy" where we pick from top K?
            order = list(range(100, 0, -1))
            # Randomly skip some? No, greedy decides.
            # Just shuffle the bottom half?
            bottom = order[50:]
            random.shuffle(bottom)
            order[50:] = bottom
        
        S = greedy(order)
        
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
            
    return best_set
