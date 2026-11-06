"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

def construct():
    start_time = time.time()
    deadline = 130.0 # Safety margin
    
    # Generate primes up to 1000
    primes = _sieve(1000)
    
    # We want to cover 1..L. Let's aim for a high L, e.g., 10000 or as much as possible.
    # The score is the largest L such that 1..L is covered.
    # We will use a greedy approach to maximize the prefix length covered.
    
    # Since we don't know L in advance, we can assume a large enough L for the bitset.
    # Max L is capped at 10^5.
    MAX_L = 100000
    
    # Bitset for coverage
    covered = 0
    
    # Current budget used (as Fraction for exactness, but float for speed in comparisons if needed)
    # The constraint is sum(1/p) <= 1.
    # Let's keep track of the sum of reciprocals.
    used_budget = Fraction(0, 1)
    max_budget = Fraction(1, 1)
    
    selected_classes = []
    
    # Precompute reciprocals as Fractions? Or just use floats for speed and check at the end?
    # The problem states "exact rationals" for the budget. 
    # Using Fraction for every comparison might be slow. 
    # However, the number of primes is small (~168). 
    # We can use float for the greedy selection heuristic (efficiency) and maintain a Fraction for the actual sum to ensure validity.
    
    # To speed up the "newly covered" calculation, we need to know which integers are currently uncovered.
    # An integer n is covered if there exists (p, r) in selected_classes such that n % p == r.
    # Instead of checking all selected classes for every candidate, we maintain a bitset of covered numbers.
    
    # Let's represent the coverage of 1..MAX_L as a Python int (bitstring).
    # Bit i represents if integer i+1 is covered? Or integer i?
    # Let's use 0-indexed for the bitset: bit i corresponds to integer i+1.
    # So integer 1 is bit 0, integer 2 is bit 1, etc.
    
    # Function to get the mask for a specific (p, r)
    # We need a mask where bit i is 1 if (i+1) % p == r.
    
    # Precomputing masks for all (p, r) pairs is too expensive (168 primes * avg p ~ 500 = ~84k masks of 100k bits each).
    # 84k * 100k bits = 8.4 * 10^9 bits = ~1 GB. Too much memory.
    
    # Alternative: Generate the mask on the fly?
    # Generating a 100k-bit mask takes time.
    # We can optimize by only considering integers up to a current "frontier" L_current.
    # But L_current grows.
    
    # Let's try a different representation:
    # For each prime p, the residues are r=0..p-1.
    # If we select (p, r), we cover all n such that n % p == r.
    
    # Greedy Step:
    # For each available prime p (not yet used), and each residue r in 0..p-1:
    #   Calculate the set of new integers covered in [1, MAX_L] by this class.
    #   new_mask = mask(p, r) & ~covered
    #   count = bit_count(new_mask)
    #   cost = 1/p
    #   efficiency = count / cost
    # Select the (p, r) with max efficiency, provided used_budget + 1/p <= 1.
    
    # To make this fast, we need to compute mask(p, r) efficiently.
    # We can precompute the base mask for each prime p?
    # mask_p is a list of p masks? No.
    
    # Let's precompute the bit positions for each residue class modulo p?
    # For a given p, the positions of r mod p are r, r+p, r+2p...
    # We can create a bitset for a specific (p, r) by iterating through those positions.
    
    # Since MAX_L is 100,000, and p is at most 1000, the number of set bits in a mask is roughly MAX_L/p.
    # For p=1000, ~100 bits. For p=2, 50,000 bits.
    # Constructing the mask by setting bits in a loop might be okay if we do it smartly.
    
    # However, doing this for every (p, r) at every step is O(P * R * L/p) which is O(P * L).
    # P ~ 168, L = 100,000. 168 * 100,000 = 1.68 * 10^7 operations per step.
    # If we do this for, say, 50 steps, that's ~10^9 operations. Might be too slow in Python.
    
    # Optimization:
    # We only need to evaluate candidates until we find a good one or time runs out.
    # Also, we can stop early if the budget is exhausted.
    
    # Let's refine the mask generation.
    # We can use a precomputed array of bit masks for small primes?
    # Or, we can approximate the gain.
    # The gain of (p, r) is the number of uncovered integers in its residue class.
    # Uncovered integers are those not in `covered`.
    
    # Let's try to implement the mask generation using bitwise operations on chunks?
    # Or simply:
    # def get_mask(p, r, limit):
    #     m = 0
    #     for i in range(r, limit, p):
    #         if i >= 1: # We cover 1..L, so index i-1
    #             m |= (1 << (i-1))
    #     return m
    
    # This loop is slow for small p.
    
    # Better approach:
    # Maintain a list of uncovered numbers? No, that's also large.
    
    # Let's stick to the bitset but optimize the mask creation.
    # We can precompute the masks for all primes and residues?
    # Memory: 168 primes. Sum of p for p<=1000 is approx 76127.
    # Total bits stored if we stored all masks: 76127 * 100000 bits = 7.6 * 10^9 bits = ~1GB. Still too much.
    
    # What if we only keep masks for the current "frontier"?
    # As we cover more numbers, the frontier moves.
    
    # Alternative Strategy:
    # Instead of global greedy, use a local search or simulated annealing on the residues of a fixed set of primes?
    # The seed program uses a fixed set of primes (all primes 13..1000) and fixes residues.
    # The budget sum(1/p) for primes 13..1000 is approx 0.92.
    # This leaves little room for optimization if we keep all primes.
    # Maybe we should select a SUBSET of primes?
    
    # Let's try:
    # 1. Select a subset of primes P' such that sum(1/p) <= 1.
    # 2. Assign residues to these primes to maximize coverage.
    
    # This is a two-stage problem.
    # Stage 1: Which primes?
    # Small primes cover more numbers per unit cost?
    # Cost is 1/p. Number of numbers covered by (p, r) in [1, L] is ~ L/p.
    # Efficiency (numbers per cost) is (L/p) / (1/p) = L.
    # So all primes have the same theoretical efficiency L!
    # However, overlaps reduce the effective gain.
    # Small primes have more overlap with each other?
    # Actually, small primes cover dense sets. Large primes cover sparse sets.
    # Sparse sets (large p) are less likely to overlap with existing coverage if the existing coverage is dense?
    # Or maybe they fill the gaps?
    
    # The seed uses primes 13..1000.
    # Let's try to use a greedy algorithm that picks the best (p, r) pair dynamically, allowing us to skip primes if they don't contribute enough new coverage.
    
    # To handle the speed issue of mask generation:
    # We can limit the "lookahead" L for the bitset to, say, 5000 initially, and expand if needed?
    # No, the score is the max L. We want to go as far as possible.
    
    # Let's try a different bitset trick.
    # We can store the coverage as a boolean array of size MAX_L+1.
    # Updating it for a new (p, r) involves iterating i=r, r+p, ... and setting covered[i]=True.
    # Checking "newly covered" count requires iterating again or maintaining a count.
    
    # If we use a boolean array:
    # - Update: O(L/p)
    # - Count new: O(L/p)
    # Total per candidate: O(L/p).
    # Sum over all candidates: Sum_{p in P} Sum_{r=0}^{p-1} O(L/p) = Sum_{p in P} p * O(L/p) = O(L * P).
    # This is the same complexity.
    
    # However, Python loops are slow.
    # We can use NumPy for the boolean array!
    # covered = np.zeros(MAX_L, dtype=bool)
    # For a candidate (p, r):
    #   idx = np.arange(r-1, MAX_L, p) # 0-indexed, so r-1 is the first index for value r (if r>=1)
    #   # Note: values are 1..L. Index 0 is value 1.
    #   # Value v is at index v-1.
    #   # v % p == r  => (idx+1) % p == r => idx % p == r-1 (if r>0) or p-1 (if r=0)
    #   # Let's stick to 1-based values for clarity in logic, but 0-based for array.
    #   # If r == 0, values are p, 2p, ... Indices are p-1, 2p-1, ...
    #   # If r > 0, values are r, r+p, ... Indices are r-1, r+p-1, ...
    
    #   if r == 0:
    #       start_idx = p - 1
    #   else:
    #       start_idx = r - 1
        
    #   if start_idx < MAX_L:
    #       indices = np.arange(start_idx, MAX_L, p)
    #       # Count how many are False in covered[indices]
    #       new_count = np.count_nonzero(~covered[indices])
    #       # If we select it:
    #       covered[indices] = True
    
    # This is vectorized! It should be much faster.
    
    MAX_L = 100000
    covered = np.zeros(MAX_L, dtype=bool)
    
    available_primes = primes[:]
    used_budget = Fraction(0, 1)
    selected = []
    
    # We need to be careful with time.
    # Each iteration of the greedy loop:
    # 1. For each p in available_primes:
    #    For each r in 0..p-1:
    #       Calculate new_count.
    # 2. Find best (p, r).
    # 3. Update covered and budget.
    # 4. Remove p from available_primes.
    
    # Complexity per step: O(P * L) vectorized ops.
    # P starts at 168, decreases.
    # Number of steps: at most 168.
    
    # Let's add a time check.
    
    while time.time() < deadline and available_primes:
        best_p = None
        best_r = None
        best_gain = -1
        best_cost = None
        
        # We can prune candidates if their cost alone exceeds remaining budget
        remaining_budget = max_budget - used_budget
        
        for p in available_primes:
            cost = Fraction(1, p)
            if cost > remaining_budget:
                continue
                
            # Iterate residues
            # We can vectorize the residue loop too?
            # For a fixed p, we want to find r that maximizes new_count.
            # new_count(r) = sum(1 for i in range(r-1, MAX_L, p) if not covered[i])
            
            # We can extract the columns corresponding to this prime's residue classes?
            # That's hard with a flat array.
            
            # Let's just loop r in Python, but use NumPy for the count.
            # p is small (max 1000), so loop is small.
            
            for r in range(p):
                if r == 0:
                    start_idx = p - 1
                else:
                    start_idx = r - 1
                    
                if start_idx >= MAX_L:
                    continue
                    
                indices = np.arange(start_idx, MAX_L, p)
                # Mask of uncovered
                # new_count = np.count_nonzero(~covered[indices])
                # This creates a temporary array. 
                # We can do: len(indices) - np.count_nonzero(covered[indices])
                count_covered = np.count_nonzero(covered[indices])
                total = len(indices)
                new_count = total - count_covered
                
                if new_count > best_gain:
                    best_gain = new_count
                    best_p = p
                    best_r = r
                    best_cost = cost
        
        if best_p is None:
            break
            
        # Select best_p, best_r
        selected.append([best_p, best_r])
        used_budget += best_cost
        
        # Update covered
        p = best_p
        r = best_r
        if r == 0:
            start_idx = p - 1
        else:
            start_idx = r - 1
            
        if start_idx < MAX_L:
            indices = np.arange(start_idx, MAX_L, p)
            covered[indices] = True
            
        # Remove best_p from available
        available_primes.remove(best_p)
        
    # Calculate final L
    # L is the largest integer such that 1..L are all covered.
    # Find the first False in covered.
    # If covered is all True up to some point, then False.
    
    # np.argmax(covered == False) gives the index of the first False.
    # If no False, argmax returns 0? No, it returns 0 if the first element is False.
    # If all True, it returns 0? 
    # Let's check: np.argmax([True, True, False]) -> 2.
    # np.argmax([True, True, True]) -> 0. 
    # So we need to check if all are True.
    
    if not np.all(covered):
        first_uncovered_idx = np.argmax(covered == False)
        # The integer is first_uncovered_idx + 1
        L = first_uncovered_idx
        # Wait. 
        # Index 0 is Integer 1.
        # If covered[0] is False, L=0.
        # If covered[0] is True, covered[1] is False, L=1.
        # argmax(covered==False) gives the index of the first False.
        # If index is k, then integers 1..k are covered?
        # No.
        # If covered[0]=True, covered[1]=False. argmax is 1. L should be 1.
        # So L = first_uncovered_idx.
        
        # Let's verify.
        # covered = [True, False, True]
        # covered==False = [False, True, False]
        # argmax = 1.
        # Integers covered: 1. Integer 2 is not. So L=1. Correct.
        
        # If covered = [False, True, True]
        # covered==False = [True, False, False]
        # argmax = 0.
        # Integer 1 is not covered. L=0. Correct.
        
        # If covered = [True, True, True]
        # covered==False = [False, False, False]
        # argmax = 0.
        # But all are covered. L should be MAX_L.
        
    else:
        L = MAX_L
        
    # Cap at 10^5
    if L > 100000:
        L = 100000
        
    return selected

# EVOLVE-BLOCK-END
