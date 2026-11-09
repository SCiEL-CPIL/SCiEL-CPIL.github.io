import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    
    # Define candidate sets of primes
    # We want sets with sum(1/p) <= 1
    # Small primes are generally better for density
    candidate_sets = []
    
    # Set 1: Small primes
    candidate_sets.append([2, 3, 7])
    candidate_sets.append([2, 3, 11, 13])
    candidate_sets.append([2, 5, 7, 11])
    candidate_sets.append([3, 5, 7, 11, 13, 17, 19, 23])
    candidate_sets.append([2, 3, 7, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97])
    
    # Verify budget for candidates
    valid_sets = []
    for s in candidate_sets:
        total = sum(1.0/p for p in s)
        if total <= 1.0 + 1e-9:
            valid_sets.append(s)
    
    best_L = 0
    best_classes = []
    
    # Local search function
    def optimize(prime_set, max_L_search=2000):
        n_primes = len(prime_set)
        if n_primes == 0:
            return 0, []
        
        # Initialize residues randomly
        residues = [random.randint(0, p-1) for p in prime_set]
        
        def calculate_L(res):
            # Mark covered numbers up to max_L_search
            covered = [False] * (max_L_search + 1)
            for p, r in zip(prime_set, res):
                for i in range(r, max_L_search + 1, p):
                    if i >= 1:
                        covered[i] = True
            # Find first uncovered
            for i in range(1, max_L_search + 1):
                if not covered[i]:
                    return i - 1
            return max_L_search
        
        current_L = calculate_L(residues)
        improved = True
        while improved:
            improved = False
            for i in range(n_primes):
                p = prime_set[i]
                old_r = residues[i]
                # Try all other residues
                for new_r in range(p):
                    if new_r == old_r:
                        continue
                    residues[i] = new_r
                    new_L = calculate_L(residues)
                    if new_L > current_L:
                        current_L = new_L
                        improved = True
                    else:
                        residues[i] = old_r
                        # If we found a better one, we keep it and break? 
                        # No, we should try all to find the best local move?
                        # Or just accept first improvement?
                        # Accepting first improvement is faster.
                        # But let's try to find the best move for this prime.
                        # Actually, just accepting any improvement is standard hill climbing.
                        # Let's stick to first improvement for speed, or best improvement?
                        # Best improvement is better but slower.
                        # Given small p, best improvement is cheap.
                        pass
                # Reset to best found for this prime?
                # The loop above resets to old_r if not improved.
                # If improved, it keeps new_r.
                # But it might have found a better new_r later?
                # No, the loop iterates new_r. If it finds an improvement, it sets residues[i]=new_r and improved=True.
                # Then it continues to next new_r.
                # It doesn't break.
                # So it might overwrite a good residue with a slightly worse one?
                # No, it only updates if new_L > current_L.
                # So it keeps the best one found in this pass.
                # However, after the inner loop, residues[i] is the best one found.
                # This is correct.
        
        return current_L, [[p, r] for p, r in zip(prime_set, residues)]

    # Run optimization for each valid set
    # Use multiple restarts for each set
    num_restarts = 10
    
    for s in valid_sets:
        for _ in range(num_restarts):
            if time.time() > deadline:
                break
            
            L, classes = optimize(s)
            if L > best_L:
                best_L = L
                best_classes = classes
        if time.time() > deadline:
            break
    
    # If no valid set or L is 0, fallback to seed-like behavior?
    # But seed is valid.
    if not best_classes:
        # Fallback to seed
        primes_seed = [p for p in range(13, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
        return [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]
        
    return best_classes

# EVOLVE-BLOCK-START
# The code above is the construct function.
# We need to wrap it in the EVOLVE-BLOCK or just replace the block content.
# The prompt says "Keep the function names and signatures...".
# The current program has construct() inside EVOLVE-BLOCK.
# I will output the whole file.
# EVOLVE-BLOCK-END
