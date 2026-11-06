import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 55.0
    max_val = 4096
    
    best_len = 0
    best_S = []
    
    def run_greedy_and_search(seed):
        random.seed(seed)
        
        S = []
        S_set = set()
        used_sums = set()
        pair_sums = set()
        
        def can_add(x):
            if x in S_set:
                return False
            if 3 * x in used_sums:
                return False
            for s in S:
                if 2 * x + s in used_sums:
                    return False
            for p in pair_sums:
                if x + p in used_sums:
                    return False
            return True

        def add(x):
            S.append(x)
            S_set.add(x)
            
            used_sums.add(3 * x)
            
            old_S = S[:-1]
            for s in old_S:
                used_sums.add(2 * x + s)
                
            for i in range(len(old_S)):
                for j in range(i, len(old_S)):
                    used_sums.add(x + old_S[i] + old_S[j])
                    
            pair_sums.add(2 * x)
            for s in old_S:
                pair_sums.add(x + s)

        def remove(x):
            idx = S.index(x)
            S_rest = S[:idx] + S[idx+1:]
            
            S.pop(idx)
            S_set.remove(x)
            
            used_sums.remove(3 * x)
            
            for s in S_rest:
                used_sums.remove(2 * x + s)
                
            for i in range(len(S_rest)):
                for j in range(i, len(S_rest)):
                    used_sums.remove(x + S_rest[i] + S_rest[j])
                    
            pair_sums.remove(2 * x)
            for s in S_rest:
                pair_sums.remove(x + s)

        local_best_len = 0
        local_best_S = []
        
        # Initial greedy with randomization
        # We iterate 1..4096, but sometimes skip valid numbers
        skip_prob = 0.1
        for x in range(1, max_val + 1):
            if can_add(x):
                if random.random() > skip_prob:
                    add(x)
                    if len(S) > local_best_len:
                        local_best_len = len(S)
                        local_best_S = list(S)
        
        # Local search
        while time.time() - start_time < DEADLINE - 5.0: # Leave 5s for next restarts
            improved = False
            for x in range(1, max_val + 1):
                if can_add(x):
                    add(x)
                    improved = True
                    if len(S) > local_best_len:
                        local_best_len = len(S)
                        local_best_S = list(S)
            
            if not improved:
                if len(S) > 0:
                    for _ in range(20):
                        if time.time() - start_time > DEADLINE - 5.0:
                            break
                        idx = random.randint(0, len(S) - 1)
                        y = S[idx]
                        remove(y)
                        
                        swapped = False
                        for x in range(1, max_val + 1):
                            if x == y:
                                continue
                            if can_add(x):
                                add(x)
                                swapped = True
                                break
                        
                        if not swapped:
                            add(y) 
                            
        return local_best_len, local_best_S

    # Run multiple restarts
    # How many?
    # Each restart takes some time.
    # If greedy takes 1s, and search takes 10s, we can do 5 restarts.
    
    seed = 0
    while time.time() - start_time < DEADLINE - 10.0:
        l, S = run_greedy_and_search(seed)
        if l > best_len:
            best_len = l
            best_S = S
        seed += 1

    return best_S
