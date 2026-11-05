import time
import random

N = 12
MAX_SUBSETS = 1 << N

def reset_length(a, b):
    """
    Computes the length of the shortest reset word for the automaton defined by a and b.
    Returns 0 if not synchronizing.
    """
    # BFS on subset automaton
    # State: subset represented as a bitmask
    # Start: all states (mask (1<<N) - 1)
    # Goal: any singleton (mask with exactly one bit set)
    
    start = (1 << N) - 1
    
    # dist[subset] = shortest distance from start to subset
    dist = [-1] * MAX_SUBSETS
    dist[start] = 0
    
    queue = [start]
    head = 0
    
    # Precompute transitions for single states for speed?
    # trans_a[state] = a[state]
    # trans_b[state] = b[state]
    
    while head < len(queue):
        s = queue[head]
        head += 1
        
        # Check if singleton
        if s & (s - 1) == 0: # power of 2
            return dist[s]
            
        d = dist[s]
        
        # Apply letter a
        # new_s_a = union of a[i] for all i in s
        new_s_a = 0
        temp = s
        while temp:
            # get lowest set bit
            lsb = temp & -temp
            i = lsb.bit_length() - 1
            new_s_a |= (1 << a[i])
            temp ^= lsb
        if dist[new_s_a] == -1:
            dist[new_s_a] = d + 1
            queue.append(new_s_a)
            
        # Apply letter b
        new_s_b = 0
        temp = s
        while temp:
            lsb = temp & -temp
            i = lsb.bit_length() - 1
            new_s_b |= (1 << b[i])
            temp ^= lsb
        if dist[new_s_b] == -1:
            dist[new_s_b] = d + 1
            queue.append(new_s_b)
            
    return 0

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0 # 50s safety margin
    
    # Initial automaton: Seed
    best_a = [(i + 1) % N for i in range(N)]
    best_b = [0] * N
    best_score = reset_length(best_a, best_b)
    
    # Local Search
    # We will mutate one transition at a time.
    # To improve speed, we can cache some computations, but N=12 is small.
    
    current_a = best_a[:]
    current_b = best_b[:]
    current_score = best_score
    
    stuck_count = 0
    max_stuck = 100 # Restart or random jump if stuck for 100 steps
    
    while time.time() < deadline:
        # Pick a random state and letter to mutate
        i = random.randint(0, N - 1)
        letter = random.choice([0, 1]) # 0 for a, 1 for b
        
        old_val = current_a[i] if letter == 0 else current_b[i]
        new_val = random.randint(0, N - 1)
        
        if old_val == new_val:
            continue
            
        if letter == 0:
            current_a[i] = new_val
        else:
            current_b[i] = new_val
            
        new_score = reset_length(current_a, current_b)
        
        if new_score > current_score:
            current_score = new_score
            stuck_count = 0
            if new_score > best_score:
                best_score = new_score
                best_a = current_a[:]
                best_b = current_b[:]
        else:
            # Revert if worse or equal (Hill Climbing)
            if letter == 0:
                current_a[i] = old_val
            else:
                current_b[i] = old_val
            stuck_count += 1
            
        if stuck_count > max_stuck:
            # Random restart from best
            current_a = best_a[:]
            current_b = best_b[:]
            current_score = best_score
            stuck_count = 0
            # Or make a few random mutations to escape local optimum
            for _ in range(5):
                i = random.randint(0, N - 1)
                letter = random.choice([0, 1])
                if letter == 0:
                    current_a[i] = random.randint(0, N - 1)
                else:
                    current_b[i] = random.randint(0, N - 1)
            current_score = reset_length(current_a, current_b)
            if current_score < best_score:
                 # Revert to best if random jump was bad
                 current_a = best_a[:]
                 current_b = best_b[:]
                 current_score = best_score

    return best_a, best_b
