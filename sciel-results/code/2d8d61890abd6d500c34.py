import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit
    
    # Primes in 2..40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Precompute prime factorization vectors
    # Vector length 12
    vectors = {}
    for n in range(2, 41):
        v = [0] * 12
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        vectors[n] = v
        
    # Candidate numbers
    candidates = list(range(2, 41))
    
    best_set = []
    best_size = 0
    
    # Greedy initialization
    current_set = []
    current_sums = {tuple([0]*12)} # Set of sum vectors
    
    # Sort candidates by "independence" (heuristic)
    # Primes first, then powers, then composites
    def score(n):
        v = vectors[n]
        # Prefer numbers with high max exponent or prime
        return -max(v) if max(v) > 1 else 0
        
    sorted_candidates = sorted(candidates, key=score)
    
    for n in sorted_candidates:
        if time.time() > deadline:
            break
        v = vectors[n]
        # Check if adding n creates a collision
        new_sums = set()
        for s in current_sums:
            new_s = tuple(s[i] + v[i] for i in range(12))
            if new_s in current_sums:
                break
            new_sums.add(new_s)
        else:
            current_set.append(n)
            current_sums.update(new_sums)
            
    best_set = list(current_set)
    best_size = len(current_set)
    
    # Local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best_set)):
            if time.time() > deadline:
                break
            # Try removing best_set[i]
            removed = best_set[i]
            temp_set = best_set[:i] + best_set[i+1:]
            temp_sums = set()
            # Reconstruct sums
            # This is expensive, so we might skip full reconstruction
            # Or just try adding other numbers
            
            # Try adding other numbers
            for j in range(40):
                if j + 2 in best_set:
                    continue
                if time.time() > deadline:
                    break
                n = j + 2
                v = vectors[n]
                # Check if n can be added to temp_set
                # This is also expensive
                
    return best_set
