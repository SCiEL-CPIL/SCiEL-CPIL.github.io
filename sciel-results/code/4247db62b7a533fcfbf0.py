import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # 1. Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]

    primes = sieve(1000)
    
    # Precompute reciprocals as Fractions for exact budget checking
    # We need to track the sum of 1/p. 
    # To speed up comparisons, we can also use float sums for the "greedy" 
    # selection heuristic if needed, but exact fractions are required for the final validity.
    # However, 1/p sums for ~168 primes is small. We can maintain a Fraction sum.
    
    # We want to cover 1..L. Let's aim for a window significantly larger than 1000 
    # to allow the greedy algorithm to find a dense packing.
    # The maximum possible L is bounded by the sum of p's used? No, by the structure.
    # Let's set a search window W = 5000. 
    # If we cover 1..L, L is the answer.
    # We will track coverage for 1..W.
    
    W = 5000
    covered = [False] * (W + 1) # 1-indexed
    used_primes = set()
    current_sum = Fraction(0, 1)
    
    # Precompute the set of residues for each prime that are "valid" 
    # i.e., we only care about residues that cover at least one number in 1..W
    # Actually, for a prime p, any residue r covers numbers r, r+p, r+2p...
    # We need to check which r covers the most *uncovered* numbers.
    
    # Optimization: 
    # Instead of checking all p * all r (1000 * 1000 = 1M ops per step, too slow for many steps),
    # we can iterate through primes and for each prime, check all residues.
    # Number of steps: The sum of 1/p is 1. 
    # Sum_{p in P} 1/p <= 1.
    # If we pick small primes, we can pick many. 
    # Sum of reciprocals of primes up to 1000 is approx log(log(1000)) + M ~ 2.2.
    # So we can pick a subset. 
    # The greedy step picks the best (p, r).
    # Max number of classes? If we pick p=2, 1/2. p=3, 1/3. Sum 5/6. 
    # We can pick at most ~10-20 large primes or a few small ones.
    # Actually, the constraint is sum(1/p) <= 1.
    # The number of selected primes is limited. 
    # Worst case: we pick p=2 (0.5), p=3 (0.33), sum=0.83. 
    # We can pick a few more. 
    # The number of iterations is small (< 50).
    # In each iteration, we evaluate all unused primes and all residues.
    # Primes ~ 168. Residues ~ p.
    # Total work per iteration ~ sum_{p unused} p.
    # Sum of primes up to 1000 is approx 76127.
    # 50 iterations * 76127 ~ 3.8M operations. This is fast enough in Python?
    # 3.8M simple ops might take ~1-2 seconds. 
    # We have 110 seconds. This is very safe.
    
    # We need to efficiently count how many *uncovered* numbers a class (p, r) covers.
    # A class (p, r) covers {x in 1..W | x % p == r}.
    
    # We will maintain a list of uncovered numbers to speed up counting?
    # Or just iterate through 1..W for each candidate? 
    # 50 iterations * 168 primes * 5000 window = 42M checks. Might be tight in Python.
    
    # Better: 
    # For each prime p, we can maintain an array `counts[p]` of size p, 
    # where counts[p][r] is the number of uncovered numbers x in 1..W such that x % p == r.
    # When we add a class (p0, r0), we need to update counts for all other primes p.
    # This is expensive: for each new class, we update O(W) entries in the counts arrays?
    # No, we update the `covered` array.
    # Then for the next step, we need to recompute counts?
    # Recomputing counts for all primes takes O(W * num_primes) ~ 5000 * 168 = 840k ops.
    # 50 iterations * 840k = 42M ops. This is feasible in 110s.
    
    # Let's implement the recompute strategy. It's robust.
    
    best_classes = []
    
    while True:
        # Check time
        if time.time() > deadline - 5:
            break
            
        # Step 1: Compute counts of uncovered numbers for each residue of each unused prime
        # counts[p] is a list of length p. counts[p][r] = count of x in 1..W, not covered, x%p==r
        counts = {}
        for p in primes:
            if p in used_primes:
                continue
            c = [0] * p
            # Iterate through uncovered numbers? 
            # It's faster to iterate 1..W and increment? 
            # Or iterate 1..W, if not covered, p = x % p? No, we need counts for ALL p.
            # This would be O(W * num_unused_primes).
            # Let's just iterate x from 1 to W.
            # For each x, if not covered[x]:
            #    for p in unused_primes:
            #       counts[p][x % p] += 1
            # This is O(W * num_unused_primes). 
            # 5000 * 168 = 840k increments. 
            # 50 iterations = 42M increments. 
            # Python loops are slow. 42M might take ~10-20 seconds. 
            # We have 110s. This should work.
            
            # Optimization: Use a list of unused primes to avoid checking used ones.
            pass
            
        # Let's refine the counting loop to be as fast as possible.
        unused_primes = [p for p in primes if p not in used_primes]
        
        # Initialize counts
        counts = {p: [0]*p for p in unused_primes}
        
        # Count uncovered
        # Iterate x
        for x in range(1, W + 1):
            if not covered[x]:
                # For each unused prime, increment count for x % p
                for p in unused_primes:
                    counts[p][x % p] += 1
        
        # Step 2: Find the best (p, r)
        # We want to maximize the number of newly covered integers.
        # But we must respect the budget: current_sum + 1/p <= 1.
        best_p = None
        best_r = None
        best_gain = 0
        
        for p in unused_primes:
            # Check budget
            if current_sum + Fraction(1, p) > 1:
                continue
                
            # Find r with max count
            # max(counts[p])
            max_r = 0
            max_c = 0
            c_list = counts[p]
            for r in range(p):
                if c_list[r] > max_c:
                    max_c = c_list[r]
                    max_r = r
            
            if max_c > best_gain:
                best_gain = max_c
                best_p = p
                best_r = max_r
                
        if best_p is None:
            break
            
        # Step 3: Apply the choice
        used_primes.add(best_p)
        current_sum += Fraction(1, best_p)
        best_classes.append([best_p, best_r])
        
        # Update covered array
        # Mark all x in 1..W such that x % best_p == best_r as covered
        # Start from the smallest x >= 1 with x % best_p == best_r
        # x = best_r
        # If best_r == 0, x = best_p (since x>=1)
        # Actually, residues are 0..p-1. 
        # If r=0, numbers are p, 2p, ...
        # If r>0, numbers are r, r+p, ...
        
        start_x = best_r
        if start_x == 0:
            start_x = best_p
            
        # Iterate
        x = start_x
        while x <= W:
            if not covered[x]:
                covered[x] = True
            x += best_p
            
    # Determine the score L
    # L is the largest integer such that 1..L are all covered.
    L = 0
    for i in range(1, W + 1):
        if covered[i]:
            L = i
        else:
            break
            
    # The task asks for classes. 
    # We must ensure the sum of 1/p is exactly <= 1. 
    # Our greedy loop ensures this.
    
    return best_classes
