import math
from itertools import combinations

def construct():
    # Precompute primes up to 256
    is_prime = [True] * 257
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(256**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, 257, i):
                is_prime[j] = False
    
    # Candidate B sets to try. 
    # The directive suggests {2, 3, 5}. 
    # We can try a few small prime sets to increase chances of finding a valid A.
    candidate_Bs = [
        [2, 3, 5],
        [2, 3, 7],
        [2, 3, 11],
        [3, 5, 7],
        [2, 5, 7]
    ]
    
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = 5
    
    # Function to check if a pair (A, B) is valid and return its score
    def get_score(A, B):
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        distinct_sums = list(sums)
        n = len(distinct_sums)
        # Check pairwise coprime
        for i in range(n):
            for j in range(i+1, n):
                if math.gcd(distinct_sums[i], distinct_sums[j]) != 1:
                    return 0
        return n

    # Check the seed first
    s = get_score(best_A, best_B)
    if s > best_score:
        best_score = s

    # Try to find a score of 9 by fixing B and searching A for all-prime sums
    for B in candidate_Bs:
        found = False
        # Iterate over all possible A sets
        # A = [a1, a2, a3] with 1 <= a1 < a2 < a3 <= 128
        for a1 in range(1, 126):
            for a2 in range(a1 + 1, 127):
                for a3 in range(a2 + 1, 129):
                    A = [a1, a2, a3]
                    # Check if all sums are prime
                    all_prime = True
                    for a in A:
                        for b in B:
                            if not is_prime[a + b]:
                                all_prime = False
                                break
                        if not all_prime:
                            break
                    if all_prime:
                        # If all sums are distinct primes, they are pairwise coprime
                        # Check distinctness just in case, though primes > 1 are distinct if values are distinct
                        # Wait, if a1+b1 == a2+b2, they are not distinct. 
                        # The problem requires DISTINCT values in A+B.
                        # If sums are not distinct, cardinality < 9.
                        # But if all are prime and distinct, score is 9.
                        sums = []
                        for a in A:
                            for b in B:
                                sums.append(a + b)
                        if len(set(sums)) == 9:
                            best_A = A
                            best_B = B
                            best_score = 9
                            found = True
                            break
                if found:
                    break
            if found:
                break
        if found:
            break
            
    # If we didn't find 9, we might want to do a random search for 6,7,8?
    # The current program does random search. The directive says "keep this program's approach and improve it" OR "design a new program... built on this approach".
    # The "approach" in the second part of the directive is the fixed-B prime search.
    # If the fixed-B search fails, we should probably fall back to a more robust search or keep the random search but optimized.
    # However, finding 9 is the goal. If we fail to find 9 with small B, maybe larger B or different A?
    # Let's add a fallback random search similar to the original but faster, to ensure we at least beat the seed if the structured search fails.
    
    if best_score < 9:
        import random
        # Random search for better scores (6, 7, 8)
        # We can bias towards primes in sums? 
        # Let's just do a quick random search to see if we can find > 5.
        for _ in range(200000):
            A = list(random.sample(range(1, 129), 3))
            B = list(random.sample(range(1, 129), 3))
            s = get_score(A, B)
            if s > best_score:
                best_score = s
                best_A = A
                best_B = B
                if best_score == 9:
                    break

    return [best_A, best_B]
