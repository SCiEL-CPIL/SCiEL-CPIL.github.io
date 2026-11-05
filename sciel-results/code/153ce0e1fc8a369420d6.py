import itertools
import random
import time

def is_sidon(s):
    sums = set()
    for i in range(len(s)):
        for j in range(i, len(s)):
            total = s[i] + s[j]
            if total in sums:
                return False
            sums.add(total)
    return True

def max_sidon_size(s):
    # Exhaustive search for max Sidon subset
    n = len(s)
    for k in range(n, 3, -1):
        for subset in itertools.combinations(s, k):
            if is_sidon(subset):
                return k
    return 3

def construct():
    # We want to minimize the maximum Sidon subset size.
    # The theoretical minimum is 4? No, for n=12, can we get 4?
    # A Sidon set of size 5 requires at least 5*4/2 = 10 distinct sums (if no diagonals) 
    # With diagonals, number of sums is k(k+1)/2.
    # For k=5, 15 sums.
    # For k=4, 10 sums.
    
    # Let's try to construct a set where every subset of size 5 has a collision.
    # This is equivalent to saying the ambient set has no Sidon subset of size 5.
    
    # One strategy: Use arithmetic progressions or clusters.
    # If we take 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 (range 12), 
    # what is the max Sidon size?
    # 0, 1, 3, 4, 9? 
    # 0+1=1, 0+3=3, 0+4=4, 0+9=9
    # 1+3=4 (collision with 0+4)
    # So {0,1,3,4} is not Sidon.
    
    # Let's use a pre-computed or heuristic search to find a set with max Sidon size 4.
    # If we can't find one, we settle for 5.
    
    # Heuristic: Random search with local optimization
    # Start with a random set of 12 distinct integers in [0, 128]
    # Check max_sidon_size.
    # If it's 4, return it.
    # If it's 5, try to improve by swapping elements.
    
    # Since the search space is small, we can do a quick random search.
    # But we must be fast (< 60s).
    
    # Let's try a specific construction:
    # Two clusters of 6?
    # {0, 1, 2, 3, 4, 5} U {60, 61, 62, 63, 64, 65}
    # Max Sidon size?
    # From first cluster: max Sidon size of {0..5} is 3? (0,1,3? 0+1=1, 0+3=3, 1+3=4. 0+1, 0+3, 1+3, 0+0, 1+1, 3+3. Sums: 0,1,3,4,2,6. All distinct? 0,1,2,3,4,6. Yes. So {0,1,3} is Sidon. Size 3.)
    # Can we get 4 from {0..5}? 
    # 0,1,2,4? 0+1=1, 0+2=2, 0+4=4, 1+2=3, 1+4=5, 2+4=6. Sums: 0,1,2,3,4,5,6,1,4,8? 
    # 0+0=0, 0+1=1, 0+2=2, 0+4=4
    # 1+1=2 (collision with 0+2)
    # So no.
    
    # What about mixing?
    # If we pick 3 from first and 2 from second?
    # {0,1,3} U {60, 62}
    # Sums within first: 0,1,3,4,2,6
    # Sums within second: 120, 122, 124
    # Cross sums: 60,61,63, 62,63,65
    # 63 appears twice (1+62, 3+60). Collision.
    
    # Let's just run a quick random search in construct() to find a set with max_sidon_size 4.
    # If not found quickly, return a set with max_sidon_size 5.
    
    # Known good set for max Sidon size 4?
    # Let's try: 0, 1, 2, 4, 8, 16, 32, 64, 5, 10, 20, 40
    # This is sparse, likely has large Sidon subsets.
    
    # We want DENSE sets.
    # Let's try to find one via random search.
    
    best_set = list(range(12))
    best_size = max_sidon_size(best_set)
    
    # Quick random search
    start_time = time.time()
    timeout = 55.0  # Leave some buffer
    
    for _ in range(10000):
        if time.time() - start_time > timeout:
            break
        
        # Generate random set
        s = random.sample(range(129), 12)
        size = max_sidon_size(s)
        
        if size < best_size:
            best_set = s
            best_size = size
            if best_size == 4:
                break
    
    # If we found 4, great. If not, maybe try local search on the best 5?
    # But for now, just return the best found.
    
    return best_set
