import time
import random

# EVOLVE-BLOCK-START
def construct():
    N = 1000
    max_sum = 2 * N
    sum_counts = [0] * (max_sum + 1)
    S = set()
    
    def try_add(x):
        if x in S:
            return False
        if sum_counts[2*x] + 1 > 2:
            return False
        for y in S:
            s = x + y
            if sum_counts[s] + 1 > 2:
                return False
        # Valid, apply
        sum_counts[2*x] += 1
        for y in S:
            sum_counts[x+y] += 1
        S.add(x)
        return True

    def try_remove(x):
        if x not in S:
            return
        S.remove(x)
        sum_counts[2*x] -= 1
        for y in S:
            sum_counts[x+y] -= 1

    # Greedy Initialization
    for x in range(1, N + 1):
        try_add(x)
        
    start_time = time.time()
    deadline = start_time + 115.0 # 115s to be safe
    
    best_size = len(S)
    best_S = set(S)
    
    p_add = 0.7
    p_swap = 0.2
    p_remove = 0.1
    
    while time.time() < deadline:
        r = random.random()
        
        if r < p_add:
            x = random.randint(1, N)
            if x not in S:
                if try_add(x):
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
        elif r < p_add + p_swap:
            # Swap
            if not S: continue
            u = random.choice(list(S))
            v = random.randint(1, N)
            if v not in S and v != u:
                # Remove u
                S.remove(u)
                sum_counts[2*u] -= 1
                for y in S:
                    sum_counts[u+y] -= 1
                
                # Check v
                valid = True
                if sum_counts[2*v] + 1 > 2:
                    valid = False
                else:
                    for y in S:
                        s = v + y
                        if sum_counts[s] + 1 > 2:
                            valid = False
                            break
                
                if valid:
                    # Add v
                    sum_counts[2*v] += 1
                    for y in S:
                        sum_counts[v+y] += 1
                    S.add(v)
                    
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                else:
                    # Revert u
                    S.add(u)
                    sum_counts[2*u] += 1
                    for y in S:
                        sum_counts[u+y] += 1
        else:
            # Remove
            if S:
                u = random.choice(list(S))
                try_remove(u)
                
    return list(best_S)
# EVOLVE-BLOCK-END
