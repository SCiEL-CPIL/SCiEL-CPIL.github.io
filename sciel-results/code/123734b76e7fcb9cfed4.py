import random
import time
from collections import deque

N = 12

# Precompute bits for each mask to speed up BFS
BITS_FOR_MASK = []
for mask in range(1 << N):
    bits = []
    for i in range(N):
        if mask & (1 << i):
            bits.append(i)
    BITS_FOR_MASK.append(bits)

def reset_threshold(trans_a, trans_b):
    """
    Compute the exact shortest reset word length using BFS on the subset automaton.
    trans_a, trans_b: lists of length N, where trans_a[i] is the destination of state i under letter 'a'.
    Returns the length of the shortest reset word, or -1 if not synchronizing.
    """
    full_mask = (1 << N) - 1
    
    # Use a list as a queue for potential speedup over deque in tight loops
    visited = {full_mask: 0}
    queue = [full_mask]
    head = 0
    
    while head < len(queue):
        current = queue[head]
        head += 1
        dist = visited[current]
        
        if current & (current - 1) == 0:
            return dist
        
        bits = BITS_FOR_MASK[current]
        
        next_a = 0
        next_b = 0
        for i in bits:
            next_a |= (1 << trans_a[i])
            next_b |= (1 << trans_b[i])
        
        if next_a not in visited:
            visited[next_a] = dist + 1
            queue.append(next_a)
            
        if next_b not in visited:
            visited[next_b] = dist + 1
            queue.append(next_b)
    
    return -1

def construct():
    start_time = time.time()
    timeout = 55.0
    
    best_trans_a = None
    best_trans_b = None
    best_score = -1
    
    def evaluate(trans_a, trans_b):
        return reset_threshold(trans_a, trans_b)
    
    def first_improvement_hill_climb(trans_a, trans_b, max_iterations=100):
        """
        Perform first-improvement hill climbing.
        Try random order of changes. Accept first improvement.
        """
        current_a = list(trans_a)
        current_b = list(trans_b)
        current_score = evaluate(current_a, current_b)
        
        if current_score < 0:
            return current_a, current_b, current_score

        iterations = 0
        while iterations < max_iterations:
            iterations += 1
            improved = False
            
            # Shuffle the order of parameters to avoid bias
            # Create a list of all possible changes: (letter, index, new_val)
            # But generating all 144 changes every iteration is overhead.
            # Instead, pick a random letter, random index, random new_val?
            # That's random walk, not hill climb.
            
            # Better: Iterate through all changes in random order, stop at first improvement.
            # Generate indices 0..23 (12 for a, 12 for b) and shuffle
            indices = list(range(N * 2))
            random.shuffle(indices)
            
            for idx in indices:
                if idx < N:
                    # Change trans_a[idx]
                    i = idx
                    old_val = current_a[i]
                    # Try a random new value first? Or all?
                    # "First improvement" usually implies trying neighbors.
                    # If we try all N values, it's expensive.
                    # Let's try all values for this specific state.
                    best_delta = -1
                    best_val = old_val
                    for new_val in range(N):
                        if new_val == old_val:
                            continue
                        current_a[i] = new_val
                        new_score = evaluate(current_a, current_b)
                        if new_score > current_score:
                            # First improvement found
                            current_score = new_score
                            best_val = new_val
                            current_a[i] = new_val
                            improved = True
                            break
                        # Restore if not accepted? 
                        # We break if accepted. If not, we need to restore before next new_val?
                        # Yes, because we are iterating new_vals.
                        current_a[i] = old_val
                    
                    if improved:
                        break
                else:
                    # Change trans_b[idx - N]
                    i = idx - N
                    old_val = current_b[i]
                    best_delta = -1
                    best_val = old_val
                    for new_val in range(N):
                        if new_val == old_val:
                            continue
                        current_b[i] = new_val
                        new_score = evaluate(current_a, current_b)
                        if new_score > current_score:
                            # First improvement found
                            current_score = new_score
                            best_val = new_val
                            current_b[i] = new_val
                            improved = True
                            break
                        current_b[i] = old_val
                    
                    if improved:
                        break
            
            if not improved:
                # Local optimum reached
                break
        
        return current_a, current_b, current_score

    def random_automaton():
        ta = [random.randint(0, N - 1) for _ in range(N)]
        tb = [random.randint(0, N - 1) for _ in range(N)]
        return ta, tb

    def structured_automaton():
        # a is a cycle, b is random or constant
        ta = [(i + 1) % N for i in range(N)]
        # b is a "reseter" structure: mostly maps to 0, but one state maps to something else
        # This is a common pattern for synchronizing automata
        tb = [0] * N
        # Introduce some randomness
        for i in range(N):
            if random.random() < 0.2:
                tb[i] = random.randint(0, N - 1)
        return ta, tb

    def local_search_with_restarts(init_a, init_b, max_restarts=10):
        best_a = list(init_a)
        best_b = list(init_b)
        best_score = evaluate(best_a, best_b)
        
        current_a = list(init_a)
        current_b = list(init_b)
        
        for r in range(max_restarts):
            # Run first improvement hill climb
            new_a, new_b, score = first_improvement_hill_climb(current_a, current_b)
            
            if score > best_score:
                best_score = score
                best_a = list(new_a)
                best_b = list(new_b)
            
            # If we hit a local optimum or randomly, jump
            if score < 0 or random.random() < 0.7:
                # Random restart with structure
                if random.random() < 0.5:
                    current_a, current_b = random_automaton()
                else:
                    current_a, current_b = structured_automaton()
            else:
                # Mutation
                for _ in range(random.randint(1, 3)):
                    if random.random() < 0.5:
                        idx = random.randint(0, N-1)
                        current_a[idx] = random.randint(0, N-1)
                    else:
                        idx = random.randint(0, N-1)
                        current_b[idx] = random.randint(0, N-1)
        
        return best_a, best_b, best_score

    # Main loop
    while time.time() - start_time < timeout:
        # Mix of random and structured initializations
        if random.random() < 0.5:
            init_a, init_b = random_automaton()
        else:
            init_a, init_b = structured_automaton()
            
        best_a, best_b, score = local_search_with_restarts(init_a, init_b)
        
        if score > best_score:
            best_score = score
            best_trans_a = best_a
            best_trans_b = best_b

    if best_trans_a is None or best_score < 0:
        # Fallback: Known good automaton or simple cycle
        best_trans_a = [(i + 1) % N for i in range(N)]
        best_trans_b = [0] * N
        best_score = reset_threshold(best_trans_a, best_trans_b)
    
    return best_trans_a, best_trans_b

if __name__ == "__main__":
    ta, tb = construct()
    score = reset_threshold(ta, tb)
    print(f"Reset threshold: {score}")
    print(f"Trans_a: {ta}")
    print(f"Trans_b: {tb}")
