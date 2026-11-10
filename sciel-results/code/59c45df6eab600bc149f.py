"""Search-based program: find (A,B) with 3 distinct ints in [1,128] maximizing |A+B| with pairwise-coprime distinct sums."""
import time
import random
from math import gcd
from itertools import combinations, product

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80  # safety margin under 90s wall-clock
    random.seed(42)

    def distinct_sums(A, B):
        return {a + b for a in A for b in B}

    def is_pairwise_coprime(vals):
        lst = list(vals)
        n = len(lst)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd(lst[i], lst[j]) != 1:
                    return False
        return True

    def score_of(A, B):
        s = distinct_sums(A, B)
        if len(s) < 2:
            return -1
        if not is_pairwise_coprime(s):
            return -1
        return len(s)

    # Theoretical max is 9 (3x3 sums all distinct and pairwise coprime)
    # Let's try to find such a pair, or as close as possible.
    
    # Strategy: 
    # 1. Try to find a set of 9 pairwise-coprime numbers that can be expressed as a+b for a in A, b in B where A,B have 3 elements each.
    # 2. If not found, maximize the number of distinct sums that are pairwise coprime.
    
    # Key insight: If we want 9 distinct sums, we need all a_i + b_j to be distinct.
    # This requires that the differences a_i - a_k are not equal to b_j - b_l for any i!=k, j!=l.
    # Also, all 9 sums must be pairwise coprime.
    
    # Let's try a structured approach first.
    # If A = {a1, a2, a3} and B = {b1, b2, b3}, the sums are:
    # a1+b1, a1+b2, a1+b3
    # a2+b1, a2+b2, a2+b3
    # a3+b1, a3+b2, a3+b3
    
    # For all 9 to be pairwise coprime, we need to avoid shared prime factors.
    # Let's try small primes and construct sums as primes or products of distinct primes.
    
    # Let's try a heuristic search with pruning.
    
    best_score = -1
    best_pair = ([1, 7, 13], [4, 10, 16])
    
    # First, try to find a valid pair with 9 distinct pairwise-coprime sums.
    # This is a hard constraint. Let's try random search with some structure.
    
    # Precompute gcd table for speed? No, just use math.gcd.
    
    # Let's try a targeted search.
    # We want 9 numbers in [2, 256] (since min sum is 1+1=2, max is 128+128=256)
    # that are pairwise coprime and can be arranged in a 3x3 grid of sums.
    
    # Let's generate candidate sets of 9 pairwise-coprime numbers first, then check if they can be a sumset.
    # But that's expensive. Let's do random search with scoring.
    
    # Random search with restarts
    for iteration in range(500000):
        if time.time() > deadline:
            break
        
        # Generate random A and B
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        
        s = distinct_sums(A, B)
        if len(s) < 9:
            continue
        
        # Check if pairwise coprime
        if is_pairwise_coprime(s):
            best_score = 9
            best_pair = (A, B)
            break
    
    # If we didn't find 9, let's try to find the best we can with a more structured approach.
    # Actually, let's try to be smarter. Let's fix A to be small and search B.
    
    if best_score < 9:
        # Try structured search
        # Let A be small: {1, p, q} where p,q are primes
        # Then sums are: 1+b1, 1+b2, 1+b3, p+b1, p+b2, p+b3, q+b1, q+b2, q+b3
        # We need all 9 to be pairwise coprime.
        
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        
        for p in small_primes:
            if time.time() > deadline:
                break
            for q in small_primes:
                if q <= p:
                    continue
                A = [1, p, q]
                if time.time() > deadline:
                    break
                # Now search for B = {b1, b2, b3}
                # We need:
                # 1+b1, 1+b2, 1+b3, p+b1, p+b2, p+b3, q+b1, q+b2, q+b3 all pairwise coprime
                # Let's try random B
                for _ in range(10000):
                    if time.time() > deadline:
                        break
                    B = random.sample(range(1, 129), 3)
                    s = distinct_sums(A, B)
                    if len(s) == 9 and is_pairwise_coprime(s):
                        best_score = 9
                        best_pair = (A, B)
                        break
                if best_score == 9:
                    break
            if best_score == 9:
                break
    
    # If still not found, let's do a broader random search
    if best_score < 9:
        for iteration in range(1000000):
            if time.time() > deadline:
                break
            A = random.sample(range(1, 129), 3)
            B = random.sample(range(1, 129), 3)
            s = distinct_sums(A, B)
            if len(s) < 9:
                continue
            if is_pairwise_coprime(s):
                best_score = 9
                best_pair = (A, B)
                break
    
    # If we still haven't found 9, let's try to maximize the number of pairwise-coprime distinct sums
    # by trying to find a valid pair with as many as possible.
    if best_score < 9:
        # Let's try to find the best score achievable
        # We know the seed gives 9 distinct sums? Let's check.
        # A = [1, 7, 13], B = [4, 10, 16]
        # Sums: 1+4=5, 1+10=11, 1+16=17, 7+4=11, 7+10=17, 7+16=23, 13+4=17, 13+10=23, 13+16=29
        # Distinct: {5, 11, 17, 23, 29}
        # Are they pairwise coprime? 
        # gcd(5,11)=1, gcd(5,17)=1, gcd(5,23)=1, gcd(5,29)=1
        # gcd(11,17)=1, gcd(11,23)=1, gcd(11,29)=1
        # gcd(17,23)=1, gcd(17,29)=1
        # gcd(23,29)=1
        # Yes! So the seed has 5 distinct pairwise-coprime sums.
        # Score is 5.
        
        # Let's verify:
        A_seed = [1, 7, 13]
        B_seed = [4, 10, 16]
        s_seed = distinct_sums(A_seed, B_seed)
        print(f"Seed sums: {s_seed}, size: {len(s_seed)}, pairwise coprime: {is_pairwise_coprime(s_seed)}")
        
        best_score = max(best_score, score_of(A_seed, B_seed))
        if best_score == score_of(A_seed, B_seed):
            best_pair = (A_seed, B_seed)
        
        # Now let's try to beat the seed
        # We want more than 5 distinct pairwise-coprime sums.
        # Let's do a focused search to find pairs with 6, 7, 8, or 9 distinct pairwise-coprime sums.
        
        for target in [9, 8, 7, 6]:
            if time.time() > deadline:
                break
            found = False
            for iteration in range(200000):
                if time.time() > deadline:
                    break
                A = random.sample(range(1, 129), 3)
                B = random.sample(range(1, 129), 3)
                s = distinct_sums(A, B)
                if len(s) < target:
                    continue
                if is_pairwise_coprime(s):
                    if len(s) > best_score:
                        best_score = len(s)
                        best_pair = (A, B)
                    if len(s) == target:
                        found = True
                        break
            if found:
                break
    
    # Final check
    final_A, final_B = best_pair
    final_s = distinct_sums(final_A, final_B)
    final_score = len(final_s) if is_pairwise_coprime(final_s) else -1
    
    if final_score < 0:
        # Fallback to seed
        final_A, final_B = [1, 7, 13], [4, 10, 16]
    
    return [final_A, final_B]
# EVOLVE-BLOCK-END
