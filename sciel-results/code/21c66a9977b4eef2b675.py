"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from itertools import combinations

def construct():
    """
    Find a maximum-cardinality subset of [1, 255] containing 255,
    such that gcd(a, b) > 1 for all distinct pairs a, b in the set.
    
    255 = 3 * 5 * 17.
    Any element must share a factor with 255 (divisible by 3, 5, or 17).
    Additionally, any two elements must share a common factor.
    
    Strategy:
    - The set of all multiples of 3 in [1, 255] is a valid clique (size 85).
    - The set of all multiples of 5 in [1, 255] is a valid clique (size 51).
    - The set of all multiples of 17 in [1, 255] is a valid clique (size 15).
    
    We can try to combine elements from different prime families if they share other factors.
    However, a simpler and very strong heuristic is to start with the largest single-prime
    family (multiples of 3) and try to add elements from other families that are compatible
    with ALL elements already in the set.
    
    But wait: if we have multiples of 3 and multiples of 5, a multiple of 3 (like 3) and a
    multiple of 5 (like 5) have gcd 1. So we cannot simply mix them.
    
    A better approach:
    The condition is that the set is a clique in the graph where edges connect numbers with gcd > 1.
    
    Let's analyze the structure.
    255 is in the set.
    Let S be the set.
    For any x in S, gcd(x, 255) > 1 => x is divisible by 3, 5, or 17.
    
    Consider the subsets:
    A = {x in [1,255] : 3|x}
    B = {x in [1,255] : 5|x}
    C = {x in [1,255] : 17|x}
    
    S must be a subset of A union B union C.
    
    If S contains an element from A\B\C (divisible by 3 only, not 5 or 17) and an element from B\A\C (divisible by 5 only, not 3 or 17), their gcd is 1. Thus, we cannot have such a pair.
    
    This suggests that the optimal set is likely contained entirely within one of the "prime power" or "prime" multiples, or a specific intersection.
    
    Actually, let's look at the clique property more closely.
    If we pick all multiples of 3, size is floor(255/3) = 85.
    If we pick all multiples of 5, size is floor(255/5) = 51.
    If we pick all multiples of 17, size is floor(255/17) = 15.
    
    Can we do better than 85?
    Suppose we have a set S.
    If S contains a number not divisible by 3, it must be divisible by 5 or 17.
    Let x in S not divisible by 3. Then x is divisible by 5 or 17.
    If x is divisible by 5, then for any y in S, gcd(x,y)>1.
    If y is divisible by 3 but not 5, and x is divisible by 5 but not 3, then gcd(x,y) might be 1 unless they share another factor.
    
    Let's try a greedy clique algorithm or a simple exhaustive search over the "types".
    
    Types of numbers in [1,255] relative to 3,5,17:
    We can categorize by their prime factors among {3,5,17} and others.
    
    However, a known result for this specific problem (max clique in gcd graph) often yields the set of all multiples of the smallest prime factor of n. Here n=255, smallest prime factor is 3.
    
    Let's verify if we can add any non-multiple-of-3 to the set of multiples of 3.
    Let S0 = {x : 3|x}. |S0| = 85.
    Can we add x not divisible by 3?
    x must be divisible by 5 or 17.
    If x is divisible by 5, consider y=3 (which is in S0). gcd(x,3) must be >1. Since 3 is prime, 3 must divide x. But we assumed x is not divisible by 3. Contradiction.
    So we cannot add any x not divisible by 3 to the full set of multiples of 3.
    
    What if we remove some multiples of 3 to add more non-multiples of 3?
    Let S be a valid set.
    Let S3 = S ∩ {x : 3|x}.
    Let S5 = S ∩ {x : 5|x, 3∤x}.
    Let S17 = S ∩ {x : 17|x, 3∤x, 5∤x}.
    Let S517 = S ∩ {x : 5|x, 17|x, 3∤x}.
    
    Any element in S5 must have gcd > 1 with every element in S3.
    Take x in S5. x is divisible by 5, not 3.
    Take y in S3. y is divisible by 3.
    gcd(x,y) > 1.
    Since 3∤x, the common factor must not be 3.
    So x and y must share a prime factor p ≠ 3.
    This must hold for ALL y in S3.
    This implies that every y in S3 must be divisible by some prime factor of x (other than 3).
    This is a very strong constraint.
    
    For example, if x=5, then every y in S3 must be divisible by 5.
    If S3 contains 3, then gcd(5,3)=1, invalid.
    So if we include 5, we cannot include 3.
    
    Let's try to construct a set larger than 85.
    Suppose we take all multiples of 15 (divisible by 3 and 5). Size = floor(255/15) = 17.
    We can add multiples of 3 not divisible by 5?
    Let S = {x : 15|x} ∪ {y : 3|y, 5∤y}.
    Check compatibility:
    Take x=15, y=3. gcd(15,3)=3>1. OK.
    Take x=30, y=3. gcd(30,3)=3>1. OK.
    Take x=15, y=9. gcd(15,9)=3>1. OK.
    It seems any multiple of 3 is compatible with any multiple of 15.
    Wait, the set {y : 3|y} already includes all multiples of 15.
    So S = {y : 3|y} is just the multiples of 3.
    
    Let's try mixing 3 and 5 more carefully.
    Let S contain numbers divisible by 3 and numbers divisible by 5.
    If x in S is div by 5 not 3, and y in S is div by 3 not 5, they must share a factor p≠3,5.
    
    Consider the set of all even numbers? 255 is odd, so 255 cannot be in a set of all even numbers.
    
    Let's look at the multiples of 3 again.
    Is it possible to have a set larger than 85?
    The total number of integers in [1,255] divisible by 3, 5, or 17 is:
    |A ∪ B ∪ C| = |A| + |B| + |C| - |A∩B| - |A∩C| - |B∩C| + |A∩B∩C|
    |A| = 85
    |B| = 51
    |C| = 15
    |A∩B| (div by 15) = 17
    |A∩C| (div by 51) = 5
    |B∩C| (div by 85) = 3
    |A∩B∩C| (div by 255) = 1
    
    Union size = 85 + 51 + 15 - 17 - 5 - 3 + 1 = 127.
    
    So the maximum possible size is at most 127.
    
    Let's try a heuristic search.
    We want to maximize |S|.
    
    Start with S = {255}.
    Add vertices greedily?
    
    Alternatively, note that if we take all multiples of 3, we get 85.
    Can we replace some multiples of 3 with non-multiples of 3 to gain more?
    
    Let's try the set of all multiples of 15 (17 elements) plus all multiples of 3 not divisible by 5?
    That's just all multiples of 3.
    
    What if we take all multiples of 5 (51) and all multiples of 3 that are also multiples of 5?
    That's just all multiples of 5.
    
    Let's try a mixed set.
    Let S be composed of:
    1. Multiples of 3 that are NOT divisible by 5 or 17?
    2. Multiples of 5 that are NOT divisible by 3 or 17?
    3. Multiples of 17 that are NOT divisible by 3 or 5?
    
    If we have x in group 1 (div 3, not 5,17) and y in group 2 (div 5, not 3,17), gcd(x,y) must be >1.
    Since they don't share 3 or 5, they must share another prime p.
    This requires x and y to both be divisible by p.
    
    This looks like a complex constraint satisfaction problem.
    
    However, there is a simple construction:
    Take all multiples of 3. Size 85.
    
    Let's check if we can do better with a different base.
    What about all multiples of 15? Size 17.
    Add all multiples of 3? Already included.
    
    Let's try to add to the multiples of 3.
    We showed we can't add any non-multiple of 3 if we keep 3 in the set.
    What if we remove 3?
    If we remove 3, can we add 5?
    If S contains 5, then every element in S must be divisible by 5 or share another factor with 5 (which is 5).
    So if 5 is in S, every element must be divisible by 5.
    Then S is a subset of multiples of 5. Max size 51.
    This is smaller than 85.
    
    So, if we include any prime p in S, and p is the only common factor for some pair, it restricts the set heavily.
    
    Specifically, if S contains a number x such that its prime factors are a subset of {3}, i.e., x is a power of 3, then every other element in S must be divisible by 3.
    The powers of 3 in [1,255] are 3, 9, 27, 81.
    If any of these is in S, S must be a subset of multiples of 3.
    Max size in multiples of 3 is 85.
    
    If we exclude all pure powers of 3, can we do better?
    Suppose S contains no number that is a power of 3.
    Then every number in S has a prime factor other than 3, or is 1 (but 1 is coprime to everything, so 1 cannot be in S unless S={1}, but 255 must be in S, so 1 is out).
    
    Let's assume the optimal set is indeed the multiples of 3.
    
    Let's double check with a small search for a better set.
    We can formulate this as:
    Maximize sum(x_i)
    Subject to:
    x_255 = 1
    x_i * x_j <= 1 if gcd(i,j)=1 for i!=j
    
    This is a quadratic integer program.
    
    Given the time limit and the simplicity of the "multiples of smallest prime" heuristic, and the fact that 85 is quite large, I will bet on the multiples of 3.
    
    Let's quickly check if a set based on 15 is better?
    Multiples of 15: 1, 15, 30, ..., 255. (17 elements)
    Can we add multiples of 3 not divisible by 5?
    Let S = {x : 15|x} ∪ {y : 3|y, 5∤y}.
    This is exactly {x : 3|x}.
    
    What if we add multiples of 5 not divisible by 3?
    Let S = {x : 15|x} ∪ {y : 5|y, 3∤y}.
    This is exactly {x : 5|x}. Size 51.
    
    What if we mix?
    S = {x : 15|x} ∪ {y : 3|y, 5∤y} ∪ {z : 5|z, 3∤z}.
    We need gcd(y, z) > 1 for all y in part 2, z in part 3.
    y is div by 3, not 5.
    z is div by 5, not 3.
    gcd(y,z) must be a prime p ≠ 3,5.
    So y and z must share a factor p.
    
    If we pick y=3, then z must be div by 3. But z is not div by 3. Contradiction.
    So we cannot include 3.
    
    If we pick y=9, z must be div by 3. Contradiction.
    
    It seems any y in part 2 that is a power of 3 forces z to be div by 3.
    If we exclude all powers of 3 from part 2, say we only take y=21 (3*7).
    Then z must be div by 7.
    So z must be a multiple of 35 (5*7).
    
    Let's try:
    S = {x : 15|x} ∪ {y : 3|y, 5∤y, 7|y} ∪ {z : 5|z, 3∤z, 7|z}
    Part 1: multiples of 15.
    Part 2: multiples of 21, not 5.
    Part 3: multiples of 35, not 3.
    
    Check compatibility:
    Part 1 vs Part 2: 15 and 21. gcd=3. OK.
    Part 1 vs Part 3: 15 and 35. gcd=5. OK.
    Part 2 vs Part 3: 21 and 35. gcd=7. OK.
    
    Let's count sizes.
    Part 1: floor(255/15) = 17.
    Part 2: multiples of 21 in [1,255] not div by 5.
    Multiples of 21: 21, 42, 63, 84, 105, 126, 147, 168, 189, 210, 231. (11 numbers)
    Remove those div by 5: 105, 210. (2 numbers)
    Remaining: 9 numbers.
    Part 3: multiples of 35 in [1,255] not div by 3.
    Multiples of 35: 35, 70, 105, 140, 175, 210, 245. (7 numbers)
    Remove those div by 3: 105, 210. (2 numbers)
    Remaining: 5 numbers.
    
    Total size = 17 + 9 + 5 = 31.
    This is much smaller than 85.
    
    It seems hard to beat 85.
    
    Let's try one more:
    S = {x : 3|x} is 85.
    
    Is there any set larger than 85?
    The number of elements in [1,255] is 255.
    The number of elements coprime to 255 is phi(255) = 255 * (1-1/3) * (1-1/5) * (1-1/17) = 255 * 2/3 * 4/5 * 16/17 = 15 * 4 * 16 / 5 = 3 * 4 * 16 = 192? No.
    255 * 2/3 = 170.
    170 * 4/5 = 136.
    136 * 16/17 = 8 * 16 = 128.
    So 128 numbers are coprime to 255.
    255 - 128 = 127 numbers share a factor with 255.
    
    So the maximum possible size is 127.
    
    But we established that mixing different prime factors is very restrictive.
    
    I will stick with the multiples of 3.
    
    Let's verify the seed program's output.
    `list(range(5, 256, 5))` returns multiples of 5. Size 51.
    My proposed solution returns multiples of 3. Size 85.
    85 > 51.
    
    Score calculation:
    Seed score: 51/51 = 1.0? No, score is value/seed.
    Wait, the prompt says "The evaluator measures the raw value (higher is better). The search score is value/seed".
    If seed returns 51 elements, and I return 85, score = 85/51 ≈ 1.66.
    
    Let's check the observed changes.
    One child returned `list(range(3, 256, 3))` and got score 1.666...
    85/51 = 1.666...
    This confirms that returning multiples of 3 is an improvement.
    
    Can we do better than 85?
    Let's try to add some numbers to the multiples of 3 by removing some.
    We need to remove all numbers in S3 that are "incompatible" with the new numbers.
    
    If we add a number x not div by 3, x must be div by 5 or 17.
    Let x be div by 5.
    Then every y in S must be div by 5 or share a factor with x.
    If x=5, every y must be div by 5.
    Then S is subset of multiples of 5. Size 51.
    
    If x=25, every y must be div by 5.
    
    If x=35 (5*7), every y must be div by 5 or 7.
    S must be subset of {y : 5|y or 7|y}.
    Also 255 is in S. 255 is div by 3,5,17.
    If y is div by 3 but not 5 or 7, then gcd(y, 35) might be 1.
    e.g., y=3. gcd(3,35)=1.
    So y=3 cannot be in S.
    We must remove all y in S3 that are not div by 5 or 7.
    
    Let's calculate the size of S = {y : (5|y or 7|y) and gcd(y,35)>1 is trivial} ∩ {y : gcd(y,255)>1}.
    Actually, if S is a clique, and 35 is in S, then all y in S must have gcd(y,35)>1.
    So S ⊆ {y : 5|y or 7|y}.
    Also 255 is in S.
    
    Let's maximize |S| subject to S ⊆ {y : 5|y or 7|y} and S is a clique.
    The set {y : 5|y or 7|y} is not necessarily a clique.
    e.g., 5 and 7 are in the set, but gcd(5,7)=1.
    So we cannot have both 5 and 7.
    
    If we have 5, we cannot have 7.
    So S ⊆ {y : 5|y} or S ⊆ {y : 7|y} or S ⊆ {y : 35|y} plus some mixed?
    If we have 5 in S, then no 7-only numbers.
    So S ⊆ {y : 5|y}.
    Max size 51.
    
    If we don't have 5 or 7 as pure primes, but have composites?
    This is getting complicated. The multiples of 3 seem robust.
    
    I will return the multiples of 3.
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # The set of all multiples of 3 in [1, 255]
    # This is a valid clique because any two multiples of 3 share the factor 3.
    # It contains 255.
    # Size is 85.
    
    candidate = list(range(3, 256, 3))
    
    # Verify validity
    if not candidate:
        return [255]
    if 255 not in candidate:
        candidate.append(255)
        candidate.sort()
        
    # Check gcd condition
    for i in range(len(candidate)):
        for j in range(i+1, len(candidate)):
            if math.gcd(candidate[i], candidate[j]) == 1:
                # Fallback if something goes wrong (should not happen)
                return [255]
                
    return candidate
# EVOLVE-BLOCK-END
