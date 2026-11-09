"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 5000
_DEADLINE = time.time() + 320.0  # 330s wall-clock, 10s safety margin

# EVOLVE-BLOCK-START

def _primes(limit):
    """Return list of primes <= limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, ok in enumerate(sieve) if ok]

def construct():
    """
    Constructs the largest admissible tuple in [0, N].
    
    Strategy:
    The condition "for every prime p <= len(tuple), some residue class mod p is missed"
    is equivalent to saying there exists a residue r_p for each prime p <= k (k=len(tuple))
    such that no element in the tuple is congruent to r_p mod p.
    
    Let S be the tuple. S must be a subset of [0, N] \ (union of forbidden classes).
    To maximize |S|, we should include ALL numbers in [0, N] that are not in the 
    forbidden classes. Thus, the problem reduces to choosing residues r_p for each 
    prime p <= k to maximize the number of integers in [0, N] that survive the sieve
    defined by these forbidden classes.
    
    We use a greedy approach:
    1. Estimate the maximum possible k. Since the density of numbers surviving 
       k primes is roughly product(1 - 1/p), which decays slowly, but we are bounded 
       by N=5000. The maximum k is likely around 200-300? 
       Actually, let's just iterate k from high to low? Or start with k=0 and increase?
       
    Better approach: 
    The set of valid tuples for a fixed k is determined by the choice of residues 
    r_p for p <= k. The maximum size of the tuple for a fixed k is the number of 
    integers in [0, N] that are not congruent to r_p mod p for any p <= k.
    
    We want to maximize this count over all choices of residues.
    This is a combinatorial optimization. However, the choices for different primes 
    interact.
    
    Greedy Heuristic:
    Iterate primes in increasing order. For each prime p, choose the residue r_p 
    that removes the fewest *currently surviving* numbers.
    Why? Because we want to keep as many numbers as possible. If we pick a residue 
    r_p, we remove all numbers x in [0, N] such that x % p == r_p. If some of these 
    numbers are already removed by previous primes, they don't reduce our count further.
    So, for a given set of already-chosen residues for smaller primes, we should 
    choose r_p to minimize the number of *new* removals.
    
    This greedy strategy works well for sieve problems.
    
    Steps:
    1. Determine the maximum k to try. We can't know k in advance because k depends 
       on the final size. But we can guess an upper bound. 
       The maximum possible size is N+1 = 5001. So k <= 5001.
       Primes up to 5001.
       
    2. However, as k increases, we add more constraints (more primes to avoid), 
       so the maximum possible tuple size decreases. 
       Let M(k) be the maximum size of an admissible tuple of size exactly k? 
       No, the condition is "for every prime p <= len(tuple)". 
       So if we construct a set S of size m, it must satisfy the condition for all 
       p <= m.
       
       We can try to construct the set for a target size m. 
       If we succeed in creating a set of size m that is admissible for all p <= m, 
       then m is a valid score.
       
       We want the largest such m.
       
       Algorithm:
       Start with a high estimate of m, say 5000.
       Check if there exists an admissible set of size m.
       If yes, return it.
       If no, decrease m and try again.
       
       How to check if there exists an admissible set of size m?
       We need to find residues r_p for p <= m such that the number of integers 
       in [0, N] surviving the sieve is at least m.
       
       We can use the greedy residue selection to maximize the survivor count.
       Let C be the set of all integers in [0, N].
       For each prime p <= m:
           Choose r_p to minimize |C \ {x : x % p == r_p}|.
           Update C.
       
       If |C| >= m, then we can pick any m elements from C. This set will be 
       admissible for all p <= m because it is a subset of C, and C avoids r_p 
       for all p <= m.
       
       This greedy approach gives the *maximum possible* survivor count for a fixed 
       set of primes? Not necessarily globally optimal, but a strong heuristic.
       
       We can try to improve the greedy choice by looking ahead or using local search 
       on the residues, but let's start with the simple greedy.
       
       Optimization:
       We don't need to try all m from 5000 down to 0. We can binary search on m?
       The property "exists admissible set of size m" is not strictly monotonic in a 
       simple way? Actually, if there is an admissible set of size m, is there one 
       of size m-1? 
       If S is admissible of size m, then any subset S' of S of size m-1 is also 
       admissible for primes p <= m-1? 
       Yes, because if S avoids r_p for all p <= m, then S' also avoids r_p for 
       all p <= m-1. So the existence of a set of size m implies existence of a set 
       of size m-1.
       So we can binary search for the maximum m.
       
    3. Implementation details:
       - Precompute primes up to 5000.
       - Function `max_survivors(primes_list)`: 
         Returns the maximum number of survivors in [0, N] using greedy residue selection.
       - Binary search for the largest m such that max_survivors(primes <= m) >= m.
       
    4. Efficiency:
       - `max_survivors` involves iterating over primes and for each prime, checking 
         all residues to find the one that removes the fewest survivors.
         Naively, this is O(P * N * log N) or similar, which might be too slow for 
         many iterations in binary search.
         
       - We can optimize `max_survivors`:
         Maintain a boolean array `alive` of size N+1, initially all True.
         For each prime p:
             Count how many alive numbers fall into each residue class mod p.
             Let count[r] = number of x in [0, N] such that alive[x] and x % p == r.
             We want to choose r that minimizes count[r].
             Then, set alive[x] = False for all x such that x % p == r.
             
         This is O(N * number_of_primes) in the worst case, but with bitsets or 
         numpy, it can be fast.
         
         Let's use numpy for speed.
         
    5. Numpy implementation:
       - `alive` is a boolean numpy array of size N+1.
       - For a prime p:
         - We need to compute the count of True values in `alive` for each residue.
         - We can do this by taking `alive` and reshaping or using indexing.
         - `indices = np.arange(N+1)`
         - `residues = indices % p`
         - `alive_residues = alive[residues]` -> This is not quite right. 
           We need to sum `alive` where `indices % p == r`.
         - Better: `counts = np.bincount(residues, weights=alive, minlength=p)`
         - Find `r_min = np.argmin(counts)`
         - Update `alive`: `alive[residues == r_min] = False`
         
       This is O(N) per prime. With ~600 primes up to 5000, and N=5000, 
       total operations ~ 3 million, which is very fast in numpy.
       
    6. Binary search:
       - Low = 0, High = N+1.
       - For each mid, compute `max_survivors` for primes <= mid.
       - If `max_survivors` >= mid, then mid is feasible. Try larger.
       - Else, try smaller.
       
       We need to be careful: the set of primes depends on mid. 
       We can precompute the list of primes.
       
    7. Final construction:
       Once we find the max m, we run `max_survivors` for primes <= m to get the 
       final `alive` array.
       Then we pick the first m indices where `alive` is True.
       
    8. Edge cases:
       - What if greedy is not optimal? 
         The greedy choice of residues is not guaranteed to be globally optimal. 
         However, for sieve problems, it is often very good.
         To improve, we could try a few random restarts of the greedy process 
         (randomizing the order of primes or the tie-breaking in residue selection).
         
       - Let's add a simple local search or multiple greedy trials.
         Try 10 random permutations of the primes? Or just randomize the order of 
         checking residues when there's a tie?
         
         Actually, the order of primes matters. Processing small primes first is 
         usually good because they have more residue classes and thus more "flexibility" 
         in choosing a residue that overlaps with already-removed numbers? 
         Wait, small primes have FEWER residue classes (p is small). 
         Large primes have MORE residue classes.
         
         If we process large primes first, we have many choices for r_p. 
         We can choose r_p to align with already-removed numbers? 
         But initially, no numbers are removed.
         
         Let's stick to increasing order of primes, as it is standard for sieves.
         
    9. Let's implement the numpy-based greedy and binary search.
    """
    
    import numpy as np
    
    # Precompute primes up to N
    all_primes = _primes(N)
    prime_set = set(all_primes)
    
    # Function to compute max survivors for a given max prime
    def compute_max_survivors(max_prime):
        """
        Returns the maximum number of survivors in [0, N] using greedy residue selection
        for all primes <= max_prime.
        Also returns the final 'alive' array to construct the set.
        """
        alive = np.ones(N + 1, dtype=bool)
        indices = np.arange(N + 1)
        
        # Get primes <= max_prime
        primes_to_use = [p for p in all_primes if p <= max_prime]
        
        for p in primes_to_use:
            residues = indices % p
            # Count alive numbers in each residue class
            # weights=alive means we sum 1 for True, 0 for False
            counts = np.bincount(residues, weights=alive, minlength=p)
            
            # Find the residue class with the minimum count
            # Tie-breaking: choose the first one? Or random?
            # Let's choose the first one for determinism, or maybe the one that 
            # maximizes future flexibility? For now, first one.
            min_count = np.min(counts)
            # Find all residues with min count
            candidate_residues = np.where(counts == min_count)[0]
            # Choose one (e.g., the first)
            r_min = candidate_residues[0]
            
            # Update alive: set to False all indices with residue r_min
            mask = (residues == r_min)
            alive[mask] = False
            
            # Early termination: if alive count is 0, break
            if not np.any(alive):
                break
                
        return np.sum(alive), alive

    # Binary search for the maximum admissible size m
    # We know m <= N+1
    low = 0
    high = N + 1
    best_m = 0
    best_alive = np.ones(N + 1, dtype=bool)
    
    # We can optimize by not doing full binary search if we can estimate better.
    # But binary search is O(log N) calls to compute_max_survivors.
    # Each call is O(P * N) with numpy, which is fast.
    
    while low <= high:
        mid = (low + high) // 2
        survivors, alive_array = compute_max_survivors(mid)
        
        if survivors >= mid:
            best_m = mid
            best_alive = alive_array
            low = mid + 1
        else:
            high = mid - 1
            
    # Construct the final set
    if best_m == 0:
        return []
        
    # Get indices where alive is True
    result_indices = np.where(best_alive)[0]
    
    # We need exactly best_m elements. 
    # If we have more, we can pick any best_m of them.
    # If we have exactly best_m, we pick all.
    # The problem asks for "a" tuple, so any valid one is fine.
    # To be safe, if we have more than best_m, we just take the first best_m.
    # But wait, if we take a subset, is it still admissible?
    # Yes, any subset of an admissible set is admissible for the same primes.
    # And since the size is best_m, we only need to check primes <= best_m.
    # Our construction ensured that the full set of survivors avoids r_p for all p <= best_m.
    # So any subset also avoids them.
    
    final_set = result_indices[:best_m].tolist()
    
    return final_set

# EVOLVE-BLOCK-END
