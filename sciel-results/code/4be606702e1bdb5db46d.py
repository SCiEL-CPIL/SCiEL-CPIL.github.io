"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import itertools

def _sieve_primes(limit):
    """Return list of primes up to limit."""
    if limit < 2:
        return []
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i, v in enumerate(is_prime) if v]

def construct():
    # Target: 12 distinct positive integers <= 10000
    # Objective: Minimize distinct prime factors of all pairwise sums.
    
    # Sums range from 1+2=3 to 9999+10000=19999.
    max_sum = 20000
    all_primes = _sieve_primes(max_sum)
    
    # We will try prime sets P in increasing order of size.
    # For each P, we check if a clique of size 12 exists in the compatibility graph.
    
    # Helper to check if a number is P-smooth
    def is_p_smooth(n, primes_set, primes_list):
        temp = n
        for p in primes_list:
            if p * p > temp:
                break
            if p in primes_set:
                while temp % p == 0:
                    temp //= p
            else:
                # If p is not in P, it doesn't help smoothness, 
                # but we must ensure no OTHER prime factors remain.
                # Actually, standard smoothness check:
                # Divide by all p in P. If remaining > 1, it must be prime and in P?
                # No, if remaining > 1 and is prime, it must be in P.
                # If remaining is composite, it has prime factors not in P (since we divided by all P primes <= sqrt(n)).
                pass
        # Correct logic:
        temp = n
        for p in primes_list:
            if p * p > temp:
                break
            if p in primes_set:
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            # temp is now a prime (or 1) that was not divided out.
            # It must be in primes_set to be smooth.
            if temp not in primes_set:
                return False
        return True

    # Precompute smoothness for a given prime set
    def compute_smooth_table(primes_tuple):
        primes_set = set(primes_tuple)
        primes_list = sorted(primes_tuple)
        smooth_table = [False] * (max_sum + 1)
        for s in range(2, max_sum + 1):
            smooth_table[s] = is_p_smooth(s, primes_set, primes_list)
        return smooth_table

    # Find a clique of size K in the graph defined by smooth_table
    # Vertices: 1..10000
    # Edge (u,v) if smooth_table[u+v] is True
    # Heuristic: Greedy with backtracking or simple greedy by degree.
    # Given the density might be low for small P, we might need to be careful.
    # However, for P={2,3,5}, density is higher.
    
    def find_clique(smooth_table, k=12, max_val=10000):
        # Optimization: Only consider vertices that have at least k-1 neighbors.
        # Precompute neighbors for each vertex? 10000 * 10000 is too big for memory/time in Python if done naively.
        
        # Alternative: Randomized Greedy / Local Search
        # Or: Structured Search
        # If P={2}, sums are powers of 2.
        # If P={2,3}, sums are 3-smooth.
        
        # Let's try a greedy approach with a limited candidate pool first.
        # Candidates: Numbers that are "likely" to work.
        # For small P, numbers that are multiples of the primes in P?
        # Not necessarily.
        
        # Let's use a simple backtracking with pruning.
        # To make it fast, we restrict the search space.
        # We can observe that if a+b is P-smooth, a and b must be "compatible".
        
        # Strategy:
        # 1. Generate all pairs (u, v) with u < v such that u+v is P-smooth.
        # 2. This is still O(N^2).
        
        # Let's try a different angle:
        # If we pick all numbers from a specific arithmetic progression?
        
        # Let's implement a simple greedy clique finder on the full set, but with a time limit or early exit.
        # Since we only need ONE clique, we can try:
        # 1. Pick the smallest number 1.
        # 2. Find all v such that 1+v is smooth.
        # 3. From those, pick the one with the most connections among the remaining.
        
        # Better:
        # Iterate through potential starting points.
        
        # Let's try to find a clique of size 12.
        # We will use a recursive backtracking with strong pruning.
        # To avoid O(N^2) precomputation of neighbors, we will compute compatibility on the fly
        # but we will restrict the candidate list to the first M numbers or numbers with high "smoothness potential".
        
        # Actually, for P={2,3,5}, the density is sufficient.
        # Let's just do a greedy selection:
        # 1. Start with an empty list `clique`.
        # 2. For each candidate `c` in 1..10000:
        #    If `c` is compatible with all in `clique`, add it.
        #    If len(clique) == 12, return clique.
        # This is O(10000 * 12 * 12) which is fast.
        # Does it work?
        # If we pick 1, then we need 11 numbers v such that 1+v is smooth.
        # If we pick 2, we need 2+v smooth.
        # The order matters.
        
        # Let's try multiple starting seeds or a smarter ordering.
        # Sort candidates by "degree" (number of compatible partners).
        # But calculating degree is O(N^2).
        
        # Heuristic:
        # Try starting with 1, 2, 3... and see if we can build a clique.
        # If not, try different sets of primes.
        
        # Let's implement the greedy:
        # For a given smooth_table:
        #   for start in range(1, 100): # Try small starts first
        #       clique = [start]
        #       for c in range(start + 1, 10001):
        #           if all(smooth_table[c + x] for x in clique):
        #               clique.append(c)
        #               if len(clique) == 12:
        #                   return clique
        #   return None
        
        # This might fail if the clique doesn't contain small numbers.
        # But usually, small numbers have more small sums which are more likely to be smooth for small P?
        # Actually, small sums are more likely to be smooth because they have fewer prime factors.
        # So starting with small numbers is a good heuristic.
        
        for start in range(1, 50):
            clique = [start]
            for c in range(start + 1, 10001):
                # Check compatibility with all in clique
                compatible = True
                for x in clique:
                    if not smooth_table[c + x]:
                        compatible = False
                        break
                if compatible:
                    clique.append(c)
                    if len(clique) == 12:
                        return clique
        return None

    # Try prime sets of increasing size
    # We need to generate combinations of primes.
    # Start with size 1, then 2, etc.
    
    # To limit computation, we only try the first few combinations of primes of each size.
    # The smallest primes are the most likely to form smooth numbers.
    
    # Size 1: {2}, {3}, {5}...
    # Size 2: {2,3}, {2,5}, {3,5}...
    
    for num_primes in range(1, 6):
        # Generate combinations of the first ~10 primes
        # This is a heuristic; we assume small primes are best.
        primes_to_consider = all_primes[:10]
        
        for combo in itertools.combinations(primes_to_consider, num_primes):
            smooth_table = compute_smooth_table(combo)
            clique = find_clique(smooth_table, 12, 10000)
            if clique:
                return clique

    # Fallback: Return a set that minimizes primes, even if not optimal.
    # Using even numbers gives factors {2, 3, 5, ...} depending on the sum.
    # e.g. 2+4=6 (2,3), 2+6=8 (2), 4+6=10 (2,5).
    # Union: {2, 3, 5}.
    # Let's return 12 even numbers.
    return list(range(2, 26, 2))

# EVOLVE-BLOCK-END
