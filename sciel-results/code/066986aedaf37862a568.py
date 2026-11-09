import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    target_size = 12
    max_val = 10000

    # 1. Precompute prime factors for all sums (2 to 20000)
    # This allows O(1) lookup of prime factors for any sum.
    sum_primes = [set() for _ in range(2 * max_val + 1)]
    for p in range(2, 2 * max_val + 1):
        # Check if p is prime
        is_prime = True
        for d in range(2, int(p**0.5) + 1):
            if p % d == 0:
                is_prime = False
                break
        if is_prime:
            for multiple in range(p, 2 * max_val + 1, p):
                sum_primes[multiple].add(p)

    def evaluate_set(A):
        """Calculate the union of prime factors of all pairwise sums."""
        all_primes = set()
        for i in range(target_size):
            for j in range(i + 1, target_size):
                s = A[i] + A[j]
                all_primes.update(sum_primes[s])
        return all_primes

    best_set = None
    best_score = float('inf')

    # --- Phase 1: Structured Initializations ---
    # We want to start close to the optimum. The optimum likely involves
    # small primes. Let's try to construct sets where sums are "smooth".
    
    # Strategy A: Arithmetic Progressions with smooth steps
    # If we pick a_i = r + i*d, sums are 2r + (i+j)d.
    # If d is highly composite (smooth), and r is small, sums might be smooth.
    
    # Strategy B: Multiples of a highly composite number
    # If a_i are multiples of M, sums are multiples of M.
    # Primes dividing sums are primes dividing (a_i/M + a_j/M) plus primes of M.
    # If we pick M to be product of small primes, we reduce the "new" primes.
    
    # Strategy C: Randomized Greedy Construction
    # Build set one by one, picking element that minimizes new primes introduced.
    
    def greedy_construct():
        """
        Construct a set greedily.
        Start with a random seed or small number.
        Add next number that minimizes the increase in distinct primes.
        """
        current_set = []
        current_primes = set()
        
        # Start with 1
        current_set.append(1)
        
        for _ in range(target_size - 1):
            best_candidate = None
            best_new_primes = None
            best_count = float('inf')
            
            # Sample candidates to save time
            # We can't check all 10000 * 11 = 110k candidates fully if it's slow,
            # but with precomputed sum_primes, it's fast.
            # Let's sample a subset of candidates from 1..10000
            # Focusing on small numbers might be better for smoothness?
            # Or random?
            
            candidates = random.sample(range(1, max_val + 1), 100)
            # Ensure distinctness
            candidates = [c for c in candidates if c not in current_set]
            
            for c in candidates:
                # Calculate new primes
                new_primes = set(current_primes)
                for x in current_set:
                    s = x + c
                    new_primes.update(sum_primes[s])
                
                count = len(new_primes)
                if count < best_count:
                    best_count = count
                    best_candidate = c
                    best_new_primes = new_primes
            
            if best_candidate is not None:
                current_set.append(best_candidate)
                current_primes = best_new_primes
            else:
                # Fallback: random
                c = random.randint(1, max_val)
                while c in current_set:
                    c = random.randint(1, max_val)
                current_set.append(c)
                for x in current_set[:-1]:
                    current_primes.update(sum_primes[x + c])
        
        return current_set

    # Generate initial candidates
    initial_sets = []
    
    # 1. Greedy
    for _ in range(5):
        initial_sets.append(greedy_construct())
        
    # 2. Multiples of highly composite numbers
    # 60 = 2^2 * 3 * 5
    # 120 = 2^3 * 3 * 5
    # 180 = 2^2 * 3^2 * 5
    # 240 = 2^4 * 3 * 5
    # 360 = 2^3 * 3^2 * 5
    # 720 = 2^4 * 3^2 * 5
    
    for M in [60, 120, 180, 240, 360, 720, 100, 50, 20]:
        if M * 12 <= max_val:
            # a_i = k_i * M where k_i are distinct small integers
            # Sums = (k_i + k_j) * M
            # Primes are primes of M union primes of (k_i + k_j)
            # To minimize primes of (k_i + k_j), pick k_i such that sums are smooth.
            # Simplest: k_i = 1, 2, ..., 12. Sums are 3..23. Primes: 2,3,5,7,11,13,17,19,23.
            # This adds many primes.
            # Better: pick k_i from a set where sums are smooth?
            # Let's just try k_i = 1..12 first.
            k_vals = list(range(1, 13))
            a_vals = [k * M for k in k_vals]
            if max(a_vals) <= max_val and len(set(a_vals)) == target_size:
                initial_sets.append(a_vals)
                
            # Try another set of k's: powers of 2? 1,2,4,8,16,32,64,128,256,512,1024,2048
            # Sums will be large, potentially prime.
            
            # Try k's that are smooth?
            # k = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 is standard.
            
    # 3. Random
    for _ in range(10):
        initial_sets.append(random.sample(range(1, max_val + 1), target_size))
        
    # 4. Small numbers
    initial_sets.append(list(range(1, 13)))
    initial_sets.append(list(range(2, 14)))

    # Evaluate all initial sets
    for A in initial_sets:
        if len(set(A)) != target_size or max(A) > max_val:
            continue
        primes = evaluate_set(A)
        score = len(primes)
        if score < best_score:
            best_score = score
            best_set = A[:]
            if best_score <= 4:
                return best_set

    # --- Phase 2: Local Search ---
    # Use the best initial set as starting point.
    # Move: Replace one element with a random new element.
    # Acceptance: Simulated Annealing or Hill Climbing with random restarts.
    
    current_set = best_set[:]
    current_score = best_score
    
    # Precompute counts for fast delta evaluation
    def get_counts(A):
        counts = {}
        active = set()
        for i in range(target_size):
            for j in range(i + 1, target_size):
                s = A[i] + A[j]
                for p in sum_primes[s]:
                    if p not in counts:
                        counts[p] = 0
                        active.add(p)
                    counts[p] += 1
        return counts, active

    counts, active = get_counts(current_set)
    
    no_improve_steps = 0
    max_no_improve = 2000
    temperature = 10.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        if no_improve_steps > max_no_improve:
            # Restart
            # Pick a random new set or one of the initial sets
            if initial_sets:
                current_set = random.choice(initial_sets).copy()
            else:
                current_set = random.sample(range(1, max_val + 1), target_size)
            
            if len(set(current_set)) != target_size:
                # Fix duplicates
                while len(set(current_set)) < target_size:
                    current_set.append(random.randint(1, max_val))
                current_set = current_set[:target_size]
            
            current_score = len(evaluate_set(current_set))
            if current_score < best_score:
                best_score = current_score
                best_set = current_set[:]
            
            counts, active = get_counts(current_set)
            no_improve_steps = 0
            temperature = 10.0
            continue
        
        # Pick index to change
        i = random.randint(0, target_size - 1)
        old_val = current_set[i]
        
        # Pick new value
        # Bias towards values that create smooth sums?
        # Just random for now, but maybe restrict range?
        new_val = random.randint(1, max_val)
        
        if new_val == old_val:
            continue
        
        if new_val in current_set and current_set.index(new_val) != i:
            # Duplicate check
            # If new_val is in set at j != i, it's invalid.
            continue
        
        # Calculate delta
        new_counts = counts.copy()
        new_active = set(active)
        
        # Remove old contributions
        for j in range(target_size):
            if i == j: continue
            s_old = old_val + current_set[j]
            for p in sum_primes[s_old]:
                new_counts[p] -= 1
                if new_counts[p] == 0:
                    new_active.discard(p)
        
        # Add new contributions
        for j in range(target_size):
            if i == j: continue
            s_new = new_val + current_set[j]
            for p in sum_primes[s_new]:
                if p not in new_counts:
                    new_counts[p] = 0
                    new_active.add(p)
                new_counts[p] += 1
        
        new_score = len(new_active)
        delta_score = new_score - current_score
        
        # Acceptance criterion
        accept = False
        if delta_score < 0:
            accept = True
        else:
            # Simulated annealing
            if temperature > 0.1:
                if random.random() < math.exp(-delta_score / temperature):
                    accept = True
        
        if accept:
            current_set[i] = new_val
            counts = new_counts
            active = new_active
            current_score = new_score
            no_improve_steps = 0
            
            if current_score < best_score:
                best_score = current_score
                best_set = current_set[:]
                if best_score <= 4:
                    return best_set
        else:
            no_improve_steps += 1
        
        temperature *= cooling_rate

    return best_set
