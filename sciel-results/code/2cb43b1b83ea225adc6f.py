import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit

    def score_set(A):
        """Calculate max(|A+A|, |A*A|) for a list of integers A."""
        sums = set()
        prods = set()
        # Optimization: use local variables for speed
        for a in A:
            for b in A:
                sums.add(a + b)
                prods.add(a * b)
        return max(len(sums), len(prods))

    def generate_hamming(n, primes):
        """Generate the first n numbers of the form prod(p_i^e_i) <= 10000."""
        # Use a priority queue or simple list generation
        # Given n=20, a simple iterative generation is fine
        # Start with 1
        candidates = [1]
        used = set([1])
        # This is a bit slow if not careful, but for n=20 it's trivial
        # Better: generate all smooth numbers up to 10000 and sort
        smooths = []
        # For 3-smooth: 2^i * 3^j
        # For 5-smooth: 2^i * 3^j * 5^k
        if primes == [2, 3]:
            for i in range(15):
                p2 = 2 ** i
                if p2 > 10000: break
                for j in range(15):
                    p3 = 3 ** j
                    val = p2 * p3
                    if val > 10000: break
                    smooths.append(val)
        elif primes == [2, 3, 5]:
            for i in range(15):
                p2 = 2 ** i
                if p2 > 10000: break
                for j in range(12):
                    p3 = 3 ** j
                    val23 = p2 * p3
                    if val23 > 10000: break
                    for k in range(10):
                        p5 = 5 ** k
                        val = val23 * p5
                        if val > 10000: break
                        smooths.append(val)
        else:
            return list(range(1, n+1))
            
        smooths = sorted(list(set(smooths)))
        return smooths[:n]

    def random_geometric(n, max_val=10000):
        """Generate a random set biased towards geometric progression."""
        A = set()
        # Pick a base and ratio
        base = random.randint(1, 5)
        ratio = random.uniform(1.1, 1.8)
        curr = base
        for _ in range(n):
            if curr > max_val:
                break
            A.add(int(curr))
            curr *= ratio
        # Fill remaining with randoms
        while len(A) < n:
            A.add(random.randint(1, max_val))
        return list(A)[:n]

    # 1. Prepare initial candidates
    candidates = []
    
    # Deterministic good seeds
    candidates.append(list(range(1, 21)))
    candidates.append(generate_hamming(20, [2, 3]))
    candidates.append(generate_hamming(20, [2, 3, 5]))
    
    # Add some random geometric sets
    for _ in range(10):
        candidates.append(random_geometric(20))
        
    # Add random uniform sets
    for _ in range(10):
        candidates.append(random.sample(range(1, 10001), 20))

    best_score = float('inf')
    best_A = None

    # 2. Local Search
    # We will iterate through candidates and improve them
    for A in candidates:
        if time.time() > deadline:
            break
            
        # Ensure validity
        A = list(set(A))
        if len(A) < 20:
            # Pad with randoms
            while len(A) < 20:
                A.append(random.randint(1, 10000))
            A = list(set(A))[:20]
            
        current_score = score_set(A)
        if current_score < best_score:
            best_score = current_score
            best_A = list(A)
            
        # Local Search Loop
        no_improve_count = 0
        max_no_improve = 1000  # Allow many tries
        
        while no_improve_count < max_no_improve and time.time() < deadline:
            improved = False
            
            # Try several random moves
            for _ in range(10):
                if time.time() > deadline:
                    break
                    
                i = random.randint(0, 19)
                old_val = A[i]
                
                # Generate new value
                r = random.random()
                if r < 0.4:
                    # Additive neighbor (small step)
                    step = random.choice([-3, -2, -1, 1, 2, 3])
                    new_val = old_val + step
                elif r < 0.7:
                    # Multiplicative neighbor
                    if old_val > 1:
                        # Divide by small factor
                        div = random.choice([2, 3, 4, 5])
                        new_val = old_val // div
                        if new_val < 1:
                            new_val = 1
                    else:
                        # Multiply
                        new_val = old_val * random.choice([2, 3, 4, 5])
                else:
                    # Random jump
                    new_val = random.randint(1, 10000)
                    
                if new_val < 1 or new_val > 10000:
                    continue
                if new_val in A and new_val != old_val:
                    continue
                
                A[i] = new_val
                new_score = score_set(A)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    no_improve_count = 0
                    if current_score < best_score:
                        best_score = current_score
                        best_A = list(A)
                else:
                    A[i] = old_val
                    no_improve_count += 1
                    
            if not improved:
                # If we couldn't improve in this batch, count it
                # The while loop condition checks no_improve_count
                # To avoid infinite loop on plateaus, we might want to accept equal scores occasionally?
                # Or just restart this candidate's local search from a mutation.
                # For now, let it run until no_improve_count > max.
                pass

    # If best_A is None (should not happen), fallback
    if best_A is None:
        return list(range(1, 21))
        
    return best_A
