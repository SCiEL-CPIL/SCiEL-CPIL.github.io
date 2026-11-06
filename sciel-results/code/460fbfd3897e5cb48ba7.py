import random
import time

N = 12
FULL_MASK = (1 << N) - 1

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
    visited = [-1] * (1 << N)
    visited[FULL_MASK] = 0
    
    queue = [FULL_MASK]
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
        
        if visited[next_a] == -1:
            visited[next_a] = dist + 1
            queue.append(next_a)
            
        if visited[next_b] == -1:
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
    
    def hill_climb_first_improvement(trans_a, trans_b, max_iterations=100):
        """
        Perform first-improvement hill climbing.
        Try neighbors in random order, accept the first one that improves the score.
        This is much faster than best-improvement, allowing more restarts.
        """
        current_a = list(trans_a)
        current_b = list(trans_b)
        current_score = evaluate(current_a, current_b)
        
        if current_score < 0:
            return current_a, current_b, current_score

        for _ in range(max_iterations):
            improved = False
            # Generate a random order of neighbors
            # Neighbors are defined by (letter, index, new_val)
            # To avoid constructing a huge list, we can pick random indices and values
            # and check them one by one until an improvement is found.
            
            # We will try up to K random neighbors per iteration
            # If none improve, we stop (local minimum)
            # Or we can just loop until we find one or exhaust a limit.
            
            # Let's try a fixed number of random probes per step to ensure progress
            # If no improvement after K probes, break.
            max_probes = 20
            found = False
            for _ in range(max_probes):
                # Randomly select a letter
                letter = random.choice(['a', 'b'])
                # Randomly select an index
                i = random.randint(0, N - 1)
                # Randomly select a new value different from current
                old_val = current_a[i] if letter == 'a' else current_b[i]
                new_val = random.randint(0, N - 1)
                while new_val == old_val:
                    new_val = random.randint(0, N - 1)
                
                if letter == 'a':
                    current_a[i] = new_val
                    new_score = evaluate(current_a, current_b)
                    current_a[i] = old_val
                else:
                    current_b[i] = new_val
                    new_score = evaluate(current_a, current_b)
                    current_b[i] = old_val
                
                if new_score > current_score:
                    # Apply the change
                    if letter == 'a':
                        current_a[i] = new_val
                    else:
                        current_b[i] = new_val
                    current_score = new_score
                    improved = True
                    found = True
                    break
            
            if not found:
                break
        
        return current_a, current_b, current_score

    def random_automaton():
        ta = [random.randint(0, N - 1) for _ in range(N)]
        tb = [random.randint(0, N - 1) for _ in range(N)]
        return ta, tb

    def structured_automaton():
        # Based on Černý's construction:
        # 'a' is a cycle: 0->1, 1->2, ..., N-1->0
        # 'b' maps most states to 0, but one state to a different value to create a long reset path.
        ta = [(i + 1) % N for i in range(N)]
        
        # 'b' strategy:
        # Map N-1 states to 0.
        # Map 1 state (usually N-1 or 0) to 1 or some other state.
        # Let's randomize which state is the "exception" and where it maps.
        target = 0
        tb = [target] * N
        
        # Pick a random state to be the exception
        exc_idx = random.randint(0, N - 1)
        # Pick a random destination for the exception, preferably not 0 to avoid immediate reset
        # But it must be synchronizing.
        # If ta is a cycle, and tb maps all to 0 except one, it is synchronizing.
        # The length depends on the position of the exception.
        # To maximize, we want the exception to be "far" from the reset state in the cycle?
        # Actually, for Černý, the worst case is when the exception maps to 1 (or N-1).
        # Let's try a few variations.
        exc_dest = random.randint(1, N - 1)
        tb[exc_idx] = exc_dest
        
        # Add some noise to break the perfect structure and allow local search to find better variants
        # with probability 0.5, perturb 1-2 random entries
        if random.random() < 0.5:
            for _ in range(random.randint(1, 3)):
                idx = random.randint(0, N - 1)
                if random.random() < 0.5:
                    ta[idx] = random.randint(0, N - 1)
                else:
                    tb[idx] = random.randint(0, N - 1)
        
        return ta, tb

    # Main loop
    while time.time() - start_time < timeout:
        # Alternate between random and structured
        # Structured is more likely to be synchronizing and have a decent score
        if random.random() < 0.2:
            init_a, init_b = random_automaton()
        else:
            init_a, init_b = structured_automaton()
        
        # Perform hill climbing
        new_a, new_b, score = hill_climb_first_improvement(init_a, init_b)
        
        if score > best_score:
            best_score = score
            best_trans_a = new_a
            best_trans_b = new_b

    if best_trans_a is None or best_score < 0:
        # Fallback: Known good automaton
        best_trans_a = [(i + 1) % N for i in range(N)]
        best_trans_b = [0] * N
        best_trans_b[N-1] = 1 # Standard Černý
        best_score = reset_threshold(best_trans_a, best_trans_b)
    
    return best_trans_a, best_trans_b

if __name__ == "__main__":
    ta, tb = construct()
    score = reset_threshold(ta, tb)
    print(f"Reset threshold: {score}")
    print(f"Trans_a: {ta}")
    print(f"Trans_b: {tb}")
