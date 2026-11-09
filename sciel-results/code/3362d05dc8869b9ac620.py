import random
import time
import math
from collections import deque

N = 12
MAX_SETS = 1 << N

def compute_threshold(trans):
    """
    trans: list of two lists, each of length N.
    Returns the length of the shortest reset word, or float('inf') if not synchronizing.
    """
    # Precompute images of all subsets for both letters
    img0 = [0] * MAX_SETS
    img1 = [0] * MAX_SETS
    
    # Precompute destination masks
    dest0 = [0] * N
    dest1 = [0] * N
    for i in range(N):
        dest0[i] = 1 << trans[0][i]
        dest1[i] = 1 << trans[1][i]
        
    # Compute images for all subsets using DP
    for S in range(1, MAX_SETS):
        lsb = S & (-S)
        i = lsb.bit_length() - 1
        rest = S ^ lsb
        img0[S] = img0[rest] | dest0[i]
        img1[S] = img1[rest] | dest1[i]
        
    # BFS
    full_set = (1 << N) - 1
    visited = [False] * MAX_SETS
    visited[full_set] = True
    
    queue = deque()
    queue.append((full_set, 0))
    
    while queue:
        current_set, dist = queue.popleft()
        
        # Check if singleton
        if current_set & (current_set - 1) == 0:
            return dist
            
        # Expand
        next0 = img0[current_set]
        if not visited[next0]:
            visited[next0] = True
            queue.append((next0, dist + 1))
            
        next1 = img1[current_set]
        if not visited[next1]:
            visited[next1] = True
            queue.append((next1, dist + 1))
            
    return float('inf')

def construct():
    start_time = time.time()
    time_limit = 55  # Safety margin
    
    random.seed(42)
    
    best_trans = None
    best_threshold = 0
    
    # Helper to generate a random automaton
    def random_automaton():
        return [
            [random.randint(0, N-1) for _ in range(N)],
            [random.randint(0, N-1) for _ in range(N)]
        ]
    
    # Helper to generate a Pawlik-like automaton
    # One letter is a cycle, the other is a fold
    def pawlik_like_automaton():
        a = [(i + 1) % N for i in range(N)]
        # b is a random function, but biased towards "folds"
        # e.g., b(i) = i or b(i) = i-1
        b = [i for i in range(N)]
        # Randomly perturb b
        for _ in range(random.randint(1, 3)):
            idx = random.randint(0, N-1)
            b[idx] = random.randint(0, N-1)
        return [a, b]

    # Helper to generate a "cycle + random permutation" automaton
    def cycle_perm_automaton():
        a = [(i + 1) % N for i in range(N)]
        b = list(range(N))
        random.shuffle(b)
        return [a, b]
    
    # Helper to generate automata with known high-threshold structures
    def structured_automaton():
        # Start with a cycle
        a = [(i + 1) % N for i in range(N)]
        # Create b with specific folding patterns that tend to increase reset length
        b = [0] * N
        # Use a pattern that creates long transient behavior
        # b(i) = (i + k) % N for some k, with occasional perturbations
        k = random.randint(1, N-1)
        for i in range(N):
            b[i] = (i + k) % N
        # Add 1-2 perturbations to break symmetry
        for _ in range(random.randint(1, 2)):
            idx = random.randint(0, N-1)
            b[idx] = random.randint(0, N-1)
        return [a, b]
    
    # Helper to generate automata based on known high-threshold examples
    def known_high_threshold_automaton():
        # These are structures known to have high reset thresholds
        # Pattern 1: cycle with a "decrement" that has one exception
        if random.random() < 0.5:
            a = [(i + 1) % N for i in range(N)]
            b = [0] * N
            for i in range(N):
                b[i] = (i - 1) % N
            # One exception to break the perfect structure
            idx = random.randint(0, N-1)
            b[idx] = random.randint(0, N-1)
            return [a, b]
        else:
            # Pattern 2: two cycles with different lengths
            a = [0] * N
            b = [0] * N
            # First half cycles forward, second half cycles backward
            for i in range(N // 2):
                a[i] = (i + 1) % (N // 2)
                b[i] = (i - 1) % (N // 2)
            for i in range(N // 2, N):
                a[i] = (i - 1) % (N // 2) + N // 2
                b[i] = (i + 1) % (N // 2) + N // 2
            # Add perturbations
            for _ in range(random.randint(1, 3)):
                letter = random.randint(0, 1)
                idx = random.randint(0, N-1)
                if letter == 0:
                    a[idx] = random.randint(0, N-1)
                else:
                    b[idx] = random.randint(0, N-1)
            return [a, b]

    # Initial population - focus on structured automata
    candidates = []
    for _ in range(50):
        candidates.append(random_automaton())
    for _ in range(100):
        candidates.append(pawlik_like_automaton())
    for _ in range(50):
        candidates.append(cycle_perm_automaton())
    for _ in range(100):
        candidates.append(structured_automaton())
    for _ in range(50):
        candidates.append(known_high_threshold_automaton())
    
    # Evaluate initial population
    for trans in candidates:
        t = compute_threshold(trans)
        if t != float('inf') and t > best_threshold:
            best_threshold = t
            best_trans = [row[:] for row in trans]
            
    if best_trans is None:
        # Fallback to a simple known synchronizing automaton
        best_trans = [[(i + 1) % N for i in range(N)], [0] * N]
        best_threshold = compute_threshold(best_trans)

    # Improved Simulated Annealing / Hill Climbing
    current_trans = [row[:] for row in best_trans]
    current_threshold = best_threshold
    
    # Better annealing schedule
    temperature = 8.0
    cooling = 0.9995
    min_temperature = 0.5
    
    step = 0
    while time.time() - start_time < time_limit:
        step += 1
        
        # Pick a random transition to change
        q = random.randint(0, N - 1)
        a = random.randint(0, 1)
        old_dest = current_trans[a][q]
        new_dest = random.randint(0, N - 1)
        
        if new_dest == old_dest:
            continue
            
        current_trans[a][q] = new_dest
        
        new_threshold = compute_threshold(current_trans)
        
        if new_threshold == float('inf'):
            current_trans[a][q] = old_dest
            continue
            
        if new_threshold >= current_threshold:
            current_threshold = new_threshold
            if new_threshold > best_threshold:
                best_threshold = new_threshold
                best_trans = [row[:] for row in current_trans]
                # Reset temperature slightly when finding improvement
                temperature = max(temperature * 1.1, 5.0)
        else:
            delta = current_threshold - new_threshold
            if temperature > min_temperature:
                prob = math.exp(-delta / temperature)
                if random.random() < prob:
                    current_threshold = new_threshold
                else:
                    current_trans[a][q] = old_dest
            else:
                current_trans[a][q] = old_dest
                
        temperature = max(temperature * cooling, min_temperature)
        
        # Periodic restarts from best solution to escape local optima
        if step % 500 == 0:
            # With some probability, restart from the best solution
            if random.random() < 0.3:
                current_trans = [row[:] for row in best_trans]
                current_threshold = best_threshold
                temperature = max(temperature, 3.0)

    return best_trans
