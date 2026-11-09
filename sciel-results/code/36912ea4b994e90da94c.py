"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from random import Random

DEADLINE = time.time() + 85  # safety margin before 90s kill

def construct():
    n = 128
    nums = list(range(1, n + 1))
    
    # Precompute gcds
    gcds = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = gcd(nums[i], nums[j])
            gcds[i][j] = g
            gcds[j][i] = g
            
    # Build conflict graph for pairwise coprime constraint
    # If gcd(i, j) > 1, they cannot both be in a pairwise coprime set
    # This is a 2-uniform graph. We want Max Independent Set (MIS).
    # adj[i] is a set of indices j such that gcd(nums[i], nums[j]) > 1
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if gcds[i][j] > 1:
                adj[i].add(j)
                adj[j].add(i)
                
    # Tomita's algorithm for Maximum Independent Set
    # We use a recursive branch and bound with coloring heuristic.
    # For speed, we use bitmasks if n <= 64, but n=128 so we use sets/lists.
    # To make it faster, we will use a simplified greedy coloring + branching.
    
    # Since n=128 is small enough for a well-implemented B&B, but Python sets are slow,
    # let's try a different angle: 
    # The pairwise coprime constraint is very strong.
    # Let's just use a fast greedy + local search on the 2-uniform graph,
    # but the previous program already did that.
    
    # Let's try a different structural approach:
    # The condition is gcd(a,b)=gcd(a,c)=gcd(b,c).
    # If we pick all numbers that are prime powers of distinct primes, 
    # then gcd(p^a, q^b) = 1 for p != q.
    # So the set of all prime powers is pairwise coprime.
    # Let's find the largest set of prime powers in [1, 128].
    # 1 is coprime to everything.
    # Primes: 2, 3, 5, ..., 127
    # Prime powers: 4, 8, 9, 16, 25, 27, 32, 49, 64, 81, 100, 121, 125, ...
    # If we take all prime powers, any two have gcd 1.
    # So this set is valid.
    # Size: count of prime powers in [1, 128].
    
    prime_powers = set()
    for i in range(2, 129):
        # Check if i is a prime power
        # i = p^k
        # Simple check
        is_pp = False
        for p in range(2, 13): # primes up to 128
            if p * p > i:
                break
            if i % p == 0:
                # check if i is a power of p
                val = i
                while val % p == 0:
                    val //= p
                if val == 1:
                    is_pp = True
                    break
        if not is_pp:
            # check if i is prime
            is_prime = True
            for d in range(2, int(i**0.5) + 1):
                if i % d == 0:
                    is_prime = False
                    break
            if is_prime:
                is_pp = True
        if is_pp:
            prime_powers.add(i)
            
    # 1 is coprime to all
    candidate = set(prime_powers)
    candidate.add(1)
    
    # Let's see how big this is.
    # Primes <= 128: 31 primes
    # Prime powers:
    # 2: 4,8,16,32,64,128 (6)
    # 3: 9,27,81 (3)
    # 5: 25 (1)
    # 7: 49 (1)
    # 11: 121 (1)
    # 13: 169 > 128
    # So roughly 31 + 6 + 3 + 1 + 1 + 1 = 43?
    # Let's just compute it properly in code.
    
    # This set is valid.
    # Can we add more?
    # If we add a number like 6 (2*3), it shares gcd 2 with 2,4,8... and gcd 3 with 3,9...
    # So it conflicts with many.
    # But maybe we can swap some prime powers for composite numbers to get more?
    # Unlikely, as prime powers are "sparse" in terms of conflicts.
    
    # Let's try to improve this set using local search on the 2-uniform graph.
    # But wait, the pairwise coprime set is a subset of valid sets.
    # Are there valid sets that are NOT pairwise coprime?
    # Yes. e.g. {2, 4, 8}. gcd(2,4)=2, gcd(2,8)=2, gcd(4,8)=4. Not equal.
    # {2, 6, 10}. gcd(2,6)=2, gcd(2,10)=2, gcd(6,10)=2. Equal! Bad.
    # {2, 3, 5}. gcds 1,1,1. Good.
    # {4, 6, 9}. gcd(4,6)=2, gcd(4,9)=1, gcd(6,9)=3. Not equal. Good.
    # So {4, 6, 9} is a valid triple.
    # Is {4, 6, 9} pairwise coprime? No.
    # So the pairwise coprime constraint is too strong.
    
    # Let's go back to the 3-uniform hypergraph but use a faster search.
    # The previous program used SA. Let's try a different heuristic:
    # Maximize size by adding elements that have the fewest conflicts with the current set.
    
    # Precompute conflicts for 3-uniform hypergraph.
    # conflict[i][j] = set of k such that {i,j,k} is bad.
    # This is expensive to store (128^2 * 128 bits).
    # Instead, let's use a bitset for each pair? No, too big.
    # Let's just use the previous program's approach but with a better heuristic.
    
    # Actually, the previous program scored 1.75.
    # Let's try to find a larger set by using a different order.
    # Order by number of prime factors?
    
    # Let's try a simple greedy that picks the number with the fewest "bad pairs" 
    # with the current set.
    
    # For speed, I will implement a fast check for adding x to S.
    # _check_addition(S, x) checks if for all i, j in S, gcd(i,j) == gcd(i,x) == gcd(j,x).
    # This is O(|S|^2).
    
    # Let's try to optimize the previous program's local search.
    # The previous program used random restarts.
    
    # I will implement a deterministic greedy with a specific ordering,
    # then a local search that tries to swap elements to increase size.
    
    # Let's stick to the prime powers idea but relax it.
    # What if we include numbers that are products of two primes?
    # e.g. 6. It conflicts with 2,3,4,8,9,27...
    # It might be worth it if it allows adding many other numbers.
    
    # Let's try a random restart local search with a better move generator.
    # Moves: add, remove, swap.
    # But the previous program did this.
    
    # What if we use a different representation?
    # Represent the set as a bitmask (Python int).
    # Then we can use bit operations to check conflicts?
    # Precompute for each x, a bitmask of y such that gcd(x,y) > 1.
    # Then for a set S (bitmask), x can be added if for all i, j in S, 
    # gcd(i,j) == gcd(i,x) == gcd(j,x).
    # This is still hard to check with bitmasks alone.
    
    # Let's try a simpler approach:
    # 1. Start with all numbers.
    # 2. While there is a bad triple, remove one element from it.
    #    Choose the element that is in the most bad triples.
    # This is a hitting set heuristic.
    
    # Let's implement this.
    
    nums = list(range(1, 129))
    n = 128
    in_set = [True] * n
    
    # Precompute gcds
    gcds = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = gcd(nums[i], nums[j])
            gcds[i][j] = g
            gcds[j][i] = g
            
    def find_bad_triple():
        # Find any bad triple in the current set
        for i in range(n):
            if not in_set[i]: continue
            for j in range(i + 1, n):
                if not in_set[j]: continue
                g_ij = gcds[i][j]
                for k in range(j + 1, n):
                    if not in_set[k]: continue
                    if gcds[i][k] == g_ij and gcds[j][k] == g_ij:
                        return (i, j, k)
        return None

    # This is O(n^3) per check, which is 2M ops. Too slow if done many times.
    # We need to be smarter.
    
    # Let's go back to the previous program's strategy but with a key improvement:
    # Use a bitset for the set S.
    # And precompute for each pair (i,j), a bitset of k such that {i,j,k} is bad.
    # pair_bad[i][j] is a Python int (bitmask) of k.
    # Then to check if x can be added to S:
    # For each pair (i,j) in S, check if x is in pair_bad[i][j].
    # This is O(|S|^2) bit checks, which is fast.
    
    pair_bad = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g_ij = gcds[i][j]
            mask = 0
            for k in range(n):
                if k == i or k == j: continue
                if gcds[i][k] == g_ij and gcds[j][k] == g_ij:
                    mask |= (1 << k)
            pair_bad[i][j] = mask
            pair_bad[j][i] = mask
            
    def can_add(S_mask, x):
        # S_mask is a Python int representing the set
        # Iterate over pairs in S
        # To iterate over set bits in S_mask:
        # while S_mask:
        #   lsb = S_mask & -S_mask
        #   i = lsb.bit_length() - 1
        #   S_mask ^= lsb
        # This is O(|S|) to get elements, but we need pairs, so O(|S|^2).
        # Let's just use a list of indices for S for simplicity, as n=128 is small.
        pass

    # Let's use a list for S.
    S = []
    S_set = set()
    
    def can_add_list(S_list, x):
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                if pair_bad[S_list[i]][S_list[j]] & (1 << x):
                    return False
        return True

    # Greedy: add elements in order of fewest conflicts
    # Conflict count for x: number of pairs (i,j) in S such that x is in pair_bad[i][j]
    # This is expensive to compute for all x.
    
    # Let's just use a random restart local search with the fast check.
    
    best_S = []
    best_len = 0
    rng = Random(42)
    
    for restart in range(200):
        if time.time() > DEADLINE:
            break
            
        S = []
        S_set = set()
        order = list(range(n))
        rng.shuffle(order)
        
        for x in order:
            if can_add_list(S, x):
                S.append(x)
                S_set.add(x)
                
        # Local search: try to add random elements
        not_in = [x for x in range(n) if x not in S_set]
        for _ in range(1000):
            if time.time() > DEADLINE:
                break
            if not not_in:
                break
            x = rng.choice(not_in)
            if can_add_list(S, x):
                S.append(x)
                S_set.add(x)
                not_in.remove(x)
                
        if len(S) > best_len:
            best_len = len(S)
            best_S = S[:]
            
    # One more local search phase: try to swap elements
    # Remove a random element, try to add two random elements?
    # This is complex.
    
    # Let's just return the best found.
    if not best_S:
        return [1]
        
    result = sorted(nums[i] for i in best_S)
    return result
# EVOLVE-BLOCK-END
