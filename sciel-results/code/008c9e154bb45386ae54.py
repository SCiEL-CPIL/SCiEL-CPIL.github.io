import time
import random
from collections import deque

N = 12

def evaluate(automaton):
    """
    Computes the exact shortest reset word length for a given automaton.
    Returns -1 if the automaton is not synchronizing.
    automaton: list of two lists, each of length N.
    """
    full_set = (1 << N) - 1
    # BFS on the subset automaton
    # dist[subset] = length of shortest word reaching this subset from full_set
    dist = {full_set: 0}
    queue = deque([full_set])
    
    while queue:
        subset = queue.popleft()
        d = dist[subset]
        
        # Check if singleton
        if subset & (subset - 1) == 0:
            return d
            
        # Generate successors
        for letter in range(2):
            trans = automaton[letter]
            next_subset = 0
            # Iterate through bits in subset
            s = subset
            while s:
                # Get lowest set bit
                lsb = s & -s
                i = lsb.bit_length() - 1
                next_subset |= (1 << trans[i])
                s ^= lsb
            
            if next_subset not in dist:
                dist[next_subset] = d + 1
                queue.append(next_subset)
                
    return -1

def random_automaton():
    return [
        [random.randint(0, N-1) for _ in range(N)],
        [random.randint(0, N-1) for _ in range(N)]
    ]

def mutate_automaton(automaton):
    # Copy
    new_auto = [list(a) for a in automaton]
    # Pick a random letter and state
    letter = random.randint(0, 1)
    state = random.randint(0, N-1)
    # Change destination
    new_auto[letter][state] = random.randint(0, N-1)
    return new_auto

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # Initial candidates
    # 1. Halving map (known to synchronize, low score)
    auto_halving = [
        [(i + 1) % N for i in range(N)],
        [i // 2 for i in range(N)]
    ]
    
    # 2. Pakhomov-like (cycle + fold)
    auto_pakhomov = [
        [(i + 1) % N for i in range(N)],
        [0 if i == 0 else (i - 1 if i % 2 == 0 else i) for i in range(N)]
    ]
    
    best_score = -1
    best_auto = None
    
    # Evaluate initial candidates
    for auto in [auto_halving, auto_pakhomov]:
        score = evaluate(auto)
        if score > best_score:
            best_score = score
            best_auto = [list(a) for a in auto]
            
    current_auto = [list(a) for a in best_auto]
    
    # Local Search / Hill Climbing with Restarts
    while time.time() < deadline:
        # Generate a neighbor
        candidate = mutate_automaton(current_auto)
        
        # Evaluate
        score = evaluate(candidate)
        
        if score > best_score:
            best_score = score
            best_auto = candidate
            current_auto = candidate
            
            # If we hit the theoretical max, we can stop
            if best_score >= (N - 1) ** 2:
                break
        else:
            # If stuck, consider a random restart
            # Probability of restart increases if we haven't improved in a while?
            # Simple: 1% chance to restart
            if random.random() < 0.01:
                # Generate a new random starting point
                # To ensure it's synchronizing, we might need to try a few
                # Or just start from the best found so far's structure but with heavy mutation
                # For now, just random
                new_start = random_automaton()
                new_score = evaluate(new_start)
                if new_score > 0: # If it synchronizes
                    current_auto = new_start
                else:
                    # Keep current, maybe mutate heavily
                    pass
        # Small sleep to avoid CPU 100% if needed, but BFS is fast
        # No sleep needed, BFS is the bottleneck
        
    # Ensure we return a valid synchronizing automaton
    # If best_auto is None or invalid, fallback to halving
    if best_auto is None or evaluate(best_auto) < 0:
        return auto_halving
        
    return best_auto
