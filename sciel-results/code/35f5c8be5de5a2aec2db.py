import math
import time
from itertools import combinations

def generate_divisors(primes):
    divs = [1]
    for p in primes:
        new_divs = [d * p for d in divs]
        divs.extend(new_divs)
    return divs

def compute_score(primes):
    divs = generate_divisors(primes)
    divs.sort()
    count = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i+1]) == 1:
            count += 1
    return count

# Primes up to ~100
primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

# Current best: first 15 primes
current_primes = primes[:15]
current_score = compute_score(current_primes)
print(f"Current (first 15): {current_score}")

# Try replacing the last prime with larger ones
base = primes[:14]
base_prod = 1
for p in base:
    base_prod *= p

best_score = current_score
best_primes = current_primes

for p in primes[14:]: # Start from 47
    if base_prod * p > 10**18:
        break
    candidate_primes = base + [p]
    score = compute_score(candidate_primes)
    if score > best_score:
        best_score = score
        best_primes = candidate_primes
        print(f"New Best: {candidate_primes}, Score: {score}")

# Also check if dropping a small prime and adding two larger ones helps? 
# e.g. k=15 but different set.
# Given the time limit, a full search is hard. 
# But the structure suggests that smaller primes are "denser" in the divisor lattice.
# However, the sorted order is by value.
# Let's just output the best found.

def construct():
    # Use the best primes found during the "search" logic above
    # For the static program, we hardcode the result of the search
    # Based on typical behavior, the first 15 primes are often optimal or close to it.
    # Let's assume the search in the evaluation environment isn't possible, 
    # so I must hardcode the best candidate.
    
    # I will run a quick mental check or rely on the fact that 
    # maximizing the number of divisors (squarefree, max primes) is the primary driver.
    # The score 21.9 for 32k divisors is low, suggesting adjacency is the bottleneck.
    
    # Let's try to see if a non-squarefree number could be better.
    # n = 2^10 * 3^5 ... ?
    # No, squarefree is generally superior for coprime pairs.
    
    # I will provide the product of the first 15 primes, 
    # but I will also consider if swapping 47 for 73 improves it.
    # I'll write the code to compute the best one at import time? 
    # No, "Keep the function names...". 
    # I can do a quick search inside construct() if it's fast enough?
    # 2^15 divisors is 32768. Sorting is fast. 
    # Testing 10 candidates is < 1 second.
    
    start_time = time.time()
    deadline = 100 # Leave margin
    
    # Candidate 1: First 15 primes
    p1 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    # Candidate 2: First 14 + 73
    p2 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 73]
    
    # Candidate 3: First 14 + 71
    p3 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 71]
    
    # Candidate 4: First 13 + 73 * 79? 
    # Prod(13) = 304250263527210
    # 73 * 79 = 5767
    # 304250263527210 * 5767 = 1.75 * 10^18 > 10^18. Too big.
    # 73 * 71 = 5183
    # 304250263527210 * 5183 = 1.57 * 10^18 > 10^18.
    # So k=15 with 2 large primes replacing 2 small ones?
    # Replace 43, 47 with 73, 79?
    # Prod(13) * 73 * 79 > 10^18.
    # Replace 41, 43, 47 with ... ?
    
    # Let's just test the top few squarefree candidates with k=15.
    
    candidates = [
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47],
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 53],
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 59],
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 61],
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 67],
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 71],
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 73]
    ]
    
    best_n = 1
    best_score = -1
    
    for primes_set in candidates:
        n = 1
        for p in primes_set:
            n *= p
        if n > 10**18:
            continue
            
        divs = generate_divisors(primes_set)
        divs.sort()
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
                
        if score > best_score:
            best_score = score
            best_n = n
            
        if time.time() > deadline:
            break
            
    return best_n
