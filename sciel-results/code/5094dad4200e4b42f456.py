"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START

def construct():
    """
    Find the longest interval [start, start+length-1] within [1, 1008]
    such that for all x, y in the interval, x*y % 1009 != 1.
    
    Strategy:
    1. Precompute modular inverses for all numbers 1..1008 modulo 1009.
    2. For each position i, determine the maximum valid right-boundary R[i]
       such that if i is in the interval, no j in the interval can be inv[i].
       Actually, simpler: 
       For an interval [s, e], it is valid iff for all x in [s, e], inv[x] is NOT in [s, e].
       
       We can find the longest valid interval by scanning all possible starts and expanding,
       or more efficiently, using a sliding window / two-pointer approach.
       
       Let's use a two-pointer approach:
       For a fixed left pointer L, we want the maximum R such that [L, R] is valid.
       As L increases, R can only stay or increase (monotonicity).
       
       How to check validity quickly?
       An interval [L, R] is invalid if there exists x in [L, R] such that inv[x] is in [L, R].
       
       Let's precompute for each x, its inverse inv[x].
       For a given L, as we extend R, we need to ensure that for all x in [L, R], inv[x] < L or inv[x] > R.
       
       Actually, let's think about the constraints differently.
       For a pair (x, y) with x*y ≡ 1 (mod 1009), they cannot both be in the interval.
       This includes x=y (self-inverse).
       
       The self-inverse elements modulo 1009 are solutions to x^2 ≡ 1 (mod 1009).
       1009 is prime. x^2 - 1 = (x-1)(x+1) ≡ 0 (mod 1009).
       So x ≡ 1 or x ≡ -1 ≡ 1008 (mod 1009).
       So 1 and 1008 are self-inverse. They cannot be in any valid interval.
       
       For other x, inv[x] != x. The pairs are disjoint (except for self-inverse).
       
       So the problem reduces to:
       Find the longest subarray of [1, 1008] that does not contain any "forbidden pair" {x, inv[x]}.
       Note that if x is in the interval, inv[x] must NOT be in the interval.
       
       This is equivalent to:
       For each i in [1, 1008], if inv[i] > i, then we have a pair (i, inv[i]).
       We cannot have both i and inv[i] in the interval.
       
       Let's define for each position i, a "constraint" that if i is in the interval, 
       then the interval cannot extend to include inv[i] (if inv[i] > i) or start before inv[i] (if inv[i] < i).
       
       Actually, let's use a different approach.
       Let's precompute an array `right_bound` where `right_bound[i]` is the maximum index j 
       such that the interval [i, j] is valid with respect to pairs where the smaller element is >= i.
       
       Wait, simpler:
       For each i, let's find the smallest j > i such that inv[j] < i. If such a j exists, then any interval 
       starting at i cannot extend past j-1? No, that's not right.
       
       Let's try the two-pointer method properly.
       Let L be the left end. We want to find the maximum R such that [L, R] is valid.
       [L, R] is valid iff for all x in [L, R], inv[x] is not in [L, R].
       
       Let's precompute `inv` for 1..1008.
       
       We can iterate L from 1 to 1008.
       Maintain R (initially L-1) and a data structure to track which "pairs" are "active" or violated.
       
       Actually, since the range is small (1008), we can afford an O(N^2) or even O(N^2 log N) solution, 
       but O(N) is possible.
       
       Let's try O(N) two-pointer:
       L = 1, R = 0
       best_len = 0, best_start = 1
       
       While L <= 1008:
           While R < 1008 and adding R+1 keeps [L, R+1] valid:
               R += 1
           If R >= L:
               len = R - L + 1
               If len > best_len:
                   best_len = len
                   best_start = L
           L += 1
       
       How to check if adding R+1 keeps [L, R+1] valid?
       We need to ensure that for all x in [L, R+1], inv[x] is not in [L, R+1].
       
       This check is O(N) in the worst case, leading to O(N^2) total. N=1008, so N^2 ≈ 10^6, which is fine.
       
       But we can do better.
       Let's precompute for each x, its inverse inv[x].
       For a fixed L, as R increases, the only new constraint is for x = R+1.
       We need inv[R+1] to not be in [L, R+1].
       Also, for all previous x in [L, R], we need inv[x] to not be in [L, R+1].
       Previously, we knew inv[x] was not in [L, R]. Now we must also ensure inv[x] != R+1.
       So we need to check if there exists x in [L, R] such that inv[x] == R+1.
       This is equivalent to checking if inv[R+1] is in [L, R].
       So the condition for [L, R+1] to be valid given [L, R] is valid is:
       1. inv[R+1] is not in [L, R+1] (i.e., inv[R+1] < L or inv[R+1] > R+1)
       2. There is no x in [L, R] such that inv[x] == R+1. But inv is a bijection, so this is exactly inv[R+1] in [L, R].
       
       Wait, if inv[R+1] is in [L, R], then the pair (R+1, inv[R+1]) is both in [L, R+1], which is invalid.
       So the condition is simply: inv[R+1] is not in [L, R+1].
       
       Let's verify.
       Suppose [L, R] is valid. We add R+1.
       The new pairs involving R+1 are (R+1, inv[R+1]).
       We need inv[R+1] not in [L, R+1].
       What about pairs (x, inv[x]) for x in [L, R]?
       We know inv[x] is not in [L, R].
       Could inv[x] be R+1?
       If inv[x] = R+1, then x = inv[R+1].
       So if inv[R+1] is in [L, R], then x = inv[R+1] is in [L, R], and inv[x] = R+1 is in [L, R+1].
       So the pair (x, R+1) is invalid.
       Thus, the condition "inv[R+1] not in [L, R+1]" is sufficient and necessary.
       
       So the algorithm is:
       L = 1
       R = 0
       best_len = 0
       best_start = 1
       
       for L in range(1, 1009):
           while R < 1008 and inv[R+1] not in [L, R+1]:
               R += 1
           if R >= L:
               len = R - L + 1
               if len > best_len:
                   best_len = len
                   best_start = L
       
       Return [best_start, best_len]
       
       Note: inv[R+1] not in [L, R+1] means inv[R+1] < L or inv[R+1] > R+1.
       
       Let's precompute inv[1..1008].
       1009 is prime. inv[x] = pow(x, 1007, 1009).
       
       Let's double check the self-inverse case.
       If x = 1, inv[1] = 1.
       Condition: inv[1] not in [L, 1]. 1 not in [L, 1] is false if L <= 1 <= 1.
       So if L=1, we cannot include 1. R will stay at 0.
       If L=2, we check inv[2]. inv[2] = 505 (since 2*505 = 1010 = 1 mod 1009).
       505 not in [2, 2] is true. So R can be at least 2.
       Then check inv[3]. ...
       
       This looks correct.
       
       Complexity: O(N) because R only increases.
    """
    N = 1008
    MOD = 1009
    
    # Precompute inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
    
    best_len = 0
    best_start = 1
    R = 0
    
    for L in range(1, N + 1):
        # Ensure R is at least L-1
        if R < L - 1:
            R = L - 1
            
        # Expand R as far as possible
        while R < N:
            next_r = R + 1
            inv_next = inv[next_r]
            # Check if inv_next is in [L, next_r]
            if inv_next < L or inv_next > next_r:
                R = next_r
            else:
                break
        
        if R >= L:
            current_len = R - L + 1
            if current_len > best_len:
                best_len = current_len
                best_start = L
        
        # If R == L-1, it means no valid interval starting at L (length 0)
        # But the problem says "nonempty interval". 
        # If best_len is 0, we might have a problem, but there must be a valid interval.
        # Actually, if L=1, R=0. 
        # If L=2, R can be at least 2 if inv[2] not in [2,2].
        
    return [best_start, best_len]

# EVOLVE-BLOCK-END
