import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    N = 256
    # A is a set of integers
    # Helper to check if adding x creates a 4-AP
    def is_valid_addition(x, A):
        # Check if x forms a 4-AP with elements in A
        # 1. x, x+d, x+2d, x+3d
        # 2. x-3d, x-2d, x-d, x
        # 3. x-2d, x-d, x, x+d
        # 4. x-d, x, x+d, x+2d
        
        # d can be at most (N-1)/3 for type 1, etc.
        # Just iterate d from 1 to N
        for d in range(1, N + 1):
            # Type 1: x, x+d, x+2d, x+3d
            if x + 3 * d <= N:
                if (x + d) in A and (x + 2 * d) in A and (x + 3 * d) in A:
                    return False
            # Type 2: x-3d, x-2d, x-d, x
            if x - 3 * d >= 1:
                if (x - 3 * d) in A and (x - 2 * d) in A and (x - d) in A:
                    return False
            # Type 3: x-2d, x-d, x, x+d
            if x - 2 * d >= 1 and x + d <= N:
                if (x - 2 * d) in A and (x - d) in A and (x + d) in A:
                    return False
            # Type 4: x-d, x, x+d, x+2d
            if x - d >= 1 and x + 2 * d <= N:
                if (x - d) in A and (x + d) in A and (x + 2 * d) in A:
                    return False
        return True

    # Helper to calculate change in 3-AP count when adding x
    def delta_score_add(x, A):
        delta = 0
        for d in range(1, N + 1):
            # x is end: x-2d, x-d, x
            if x - 2 * d >= 1:
                if (x - 2 * d) in A and (x - d) in A:
                    delta += 1
            # x is start: x, x+d, x+2d
            if x + 2 * d <= N:
                if (x + d) in A and (x + 2 * d) in A:
                    delta += 1
            # x is middle: x-d, x, x+d
            if x - d >= 1 and x + d <= N:
                if (x - d) in A and (x + d) in A:
                    delta += 1
        return delta

    # Helper to calculate change in 3-AP count when removing x
    def delta_score_remove(x, A):
        delta = 0
        for d in range(1, N + 1):
            # x is end: x-2d, x-d, x
            if x - 2 * d >= 1:
                if (x - 2 * d) in A and (x - d) in A:
                    delta += 1
            # x is start: x, x+d, x+2d
            if x + 2 * d <= N:
                if (x + d) in A and (x + 2 * d) in A:
                    delta += 1
            # x is middle: x-d, x, x+d
            if x - d >= 1 and x + d <= N:
                if (x - d) in A and (x + d) in A:
                    delta += 1
        return -delta # Removing decreases score by this amount

    # Calculate total score
    def calc_score(A):
        count = 0
        for d in range(1, N // 2 + 1):
            for a in range(1, N - 2 * d + 1):
                if a in A and (a + d) in A and (a + 2 * d) in A:
                    count += 1
        return count

    # Generate a random maximal 4-AP-free set
    def generate_random_maximal_set():
        A = set()
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for x in candidates:
            if is_valid_addition(x, A):
                A.add(x)
        return A

    # Initial sets
    best_set = set()
    best_score = -1
    
    # Try a few random initializations
    for _ in range(5):
        if time.time() > deadline: break
        A = generate_random_maximal_set()
        score = calc_score(A)
        if score > best_score:
            best_score = score
            best_set = A.copy()
            
    # Also try the seed's greedy set
    A_seed = set()
    for x in range(1, N + 1):
        if is_valid_addition(x, A_seed):
            A_seed.add(x)
    score_seed = calc_score(A_seed)
    if score_seed > best_score:
        best_score = score_seed
        best_set = A_seed.copy()

    # Simulated Annealing
    T = 10.0
    cooling_rate = 0.999
    A = best_set.copy()
    current_score = best_score
    
    while time.time() < deadline:
        # Pick a move
        if random.random() < 0.5:
            # Try to add
            if len(A) < N:
                # Pick a random candidate not in A
                # This is slow if A is large. 
                # Better: maintain a list of candidates? Or just pick random 1..N and check if in A.
                x = random.randint(1, N)
                if x not in A:
                    if is_valid_addition(x, A):
                        delta = delta_score_add(x, A)
                        # Accept if delta > 0 or with prob exp(delta/T)
                        if delta > 0 or random.random() < math.exp(delta / T):
                            A.add(x)
                            current_score += delta
                            if current_score > best_score:
                                best_score = current_score
                                best_set = A.copy()
        else:
            # Try to remove
            if len(A) > 0:
                x = random.choice(list(A))
                delta = delta_score_remove(x, A)
                # Removing x always keeps it 4-AP free
                if delta > 0 or random.random() < math.exp(delta / T):
                    A.remove(x)
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_set = A.copy()
                        
        T *= cooling_rate
        if T < 0.1:
            T = 0.1

    return sorted(best_set)

# EVOLVE-BLOCK-END
