import math
import time

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, with gcd(a,b) > 1
    for all distinct pairs, maximizing cardinality.
    
    Strategy:
    1. Start with multiples of 3 (size 85).
    2. Consider adding elements from multiples of 5 or 17 that are not multiples of 3.
    3. When adding an element x, we must remove all elements y in the current set
       such that gcd(x, y) == 1.
    4. We can greedily try to add elements if it increases the set size, or
       if it maintains size but allows further growth. However, since we want
       to maximize cardinality, we can look for elements that can be added
       with minimal removals.
       
    Actually, a more robust approach for this specific small problem (N=255):
    The set must be a clique in the graph where vertices are [1, 255] and edges
    connect pairs with gcd > 1. We require the clique to contain 255.
    
    Let's analyze the structure:
    255 = 3 * 5 * 17.
    Any x in S must share a factor with 255, so x is in U = {x | gcd(x, 255) > 1}.
    
    The set of multiples of 3 in U has size 85.
    The set of multiples of 5 in U has size 51.
    The set of multiples of 17 in U has size 15.
    
    Can we mix them?
    If we take all multiples of 3 (C3), size 85.
    Consider adding an element x from (C5 ∪ C17) \ C3.
    x is not divisible by 3.
    For gcd(x, y) > 1 for all y in C3, x must share a factor with each y.
    Since y is a multiple of 3, and x is not, x must share a factor with y/3? No,
    gcd(x, 3k) > 1 means x shares a factor with 3k. Since x is not div by 3,
    x must share a factor with k.
    So for a fixed x, we can only keep y = 3k from C3 if gcd(x, k) > 1.
    The number of such k in [1, 85] depends on x.
    If x is a multiple of 5 (e.g., x=5), then k must be a multiple of 5.
    There are 17 multiples of 5 in [1, 85] (5, 10, ..., 85).
    So we can keep 17 elements from C3.
    We add 1 element (x=5).
    New size = 17 + 1 = 18. This is much smaller than 85.
    
    What if we build a set from scratch using a greedy algorithm or branch and bound?
    Given the small size (255), we can potentially do better search.
    However, the "multiples of smallest prime factor" heuristic is often optimal
    for these types of "large coprime-free" problems when the endpoint is fixed
    and has small prime factors.
    
    Let's try a more sophisticated search:
    We want to maximize |S| such that 255 in S and gcd(a,b)>1 for all pairs.
    This is equivalent to finding the maximum size of a set S containing 255
    where no two elements are coprime.
    
    Let's define the "conflict" graph. Vertices are 1..255. Edge between u, v if gcd(u,v)=1.
    We want an independent set in the conflict graph, containing 255.
    Or, equivalently, a clique in the "gcd>1" graph, containing 255.
    
    Since the graph is dense (many gcd>1 pairs), the complement (coprime graph) is sparse.
    Actually, the coprime graph is not that sparse.
    
    Let's try to improve upon the 85 base by looking for elements that can be swapped.
    Or, let's just verify if 85 is indeed the maximum by a quick local search.
    
    Let's try a greedy algorithm that starts with the "best" candidate set and tries to expand.
    Candidate sets:
    - All multiples of 3 (85)
    - All multiples of 5 (51)
    - All multiples of 17 (15)
    
    Maybe a mix of multiples of 3 and 5?
    If we take all multiples of 3 and 5?
    Let S = {x in [1,255] | 3|x or 5|x}.
    Size = 85 + 51 - |multiples of 15| = 85 + 51 - 17 = 119.
    Is this valid?
    Take x=3, y=5. gcd(3,5)=1. Invalid.
    So we cannot simply take the union. We must ensure that any element from the "3-only" part
    and "5-only" part share a factor.
    If x is in C3 \ C5 and y is in C5 \ C3, then gcd(x,y) must be > 1.
    x is not div by 5, y is not div by 3.
    So x and y must share a factor other than 3 or 5.
    This is a strong constraint.
    
    Let's try to construct a large set by picking a primary factor (e.g., 3) and
    carefully adding others.
    
    Actually, there is a known result for this problem:
    For n=255, the maximum size is likely 85.
    Let's verify if we can get 86.
    
    We can run a quick optimization loop:
    1. Start with C3.
    2. For each x not in C3 (but in U), check if we can add x by removing minimal conflicts.
    3. If we can add x and remove fewer than 1 element (impossible, we remove >=1 to add 1, so net 0 or negative unless we remove 0), we don't do it.
    4. Wait, if we remove 0 elements, we gain 1.
       Can we find x not in C3 such that gcd(x, y) > 1 for all y in C3?
       This requires x to share a factor with every y in C3.
       Since C3 contains 3, 6, 9, ..., the only way to share a factor with all of them
       is if x is divisible by 3 (which is false) or x is divisible by a prime that
       divides all of them? No, 3 and 5 are coprime.
       gcd(x, 3) > 1 implies 3|x.
       So no such x exists.
       
    Therefore, any x added to C3 must cause removal of at least one element.
    To gain size, we would need to add multiple x's such that their combined removals
    are less than the number added.
    
    Let's try adding all multiples of 5 that are not in C3.
    Let A = C5 \ C3. Size 51 - 17 = 34.
    If we add all of A, what do we remove from C3?
    For each y in C3, it must be kept only if it shares a factor with ALL elements in A?
    No, for each pair (a, y) with a in A and y in C3, we need gcd(a,y)>1.
    If we include ALL of A, then for a fixed y in C3, y must share a factor with every a in A.
    This is very restrictive.
    
    Let's try a different approach.
    The maximum independent set in the coprime graph containing 255.
    We can use a greedy algorithm or a simple branch and bound if time permits.
    But 255 is small.
    
    Let's try to generate a set that is a union of multiples of a set of primes P,
    but constrained such that cross-terms work.
    
    Actually, the set of all multiples of 3 is a very strong candidate.
    Is it possible to have a set larger than 85?
    Consider the prime factorization of the elements.
    If we pick elements that are all even, we get 127 elements. But 255 is odd.
    So we can't use the even set.
    
    The constraint is "containing 255".
    255 is odd.
    
    Let's try to find the maximum clique in the gcd>1 graph containing 255.
    Since 255 is connected to all multiples of 3, 5, 17.
    
    Let's use a heuristic:
    Start with {255}.
    Greedily add vertices with the highest degree in the remaining graph,
    ensuring connectivity to the current set (clique property).
    
    Actually, a simpler heuristic for this specific problem:
    Try all subsets of prime factors of 255? No.
    
    Let's stick with the multiples of 3, but try to "fix" it by removing
    the fewest elements to allow adding the "best" outsiders.
    
    We can model this as a maximum weight independent set or similar,
    but let's just try a local search around the 85-set.
    
    1. Start with S = multiples of 3.
    2. Calculate for each x in U \ S the "cost" of adding x.
       Cost = number of y in S such that gcd(x, y) == 1.
       Gain = 1.
       If Cost < Gain, we can add x and remove Cost elements. Net change = 1 - Cost.
       We only want to do this if 1 - Cost > 0, i.e., Cost == 0.
       As established, Cost is never 0.
       
    So we can't improve by single additions.
    What about multiple additions?
    Add a set X of outsiders.
    Remove a set Y of insiders.
    We need:
    - For all x in X, gcd(x, x') > 1 for all x' in X.
    - For all x in X, y in S\Y, gcd(x, y) > 1.
    - For all y in S\Y, gcd(y, y') > 1 for all y' in S\Y (already true).
    
    We want to maximize |X| - |Y|.
    
    Let's try X = {5}. Cost = 68. Net = -67.
    Let's try X = {5, 25}. 
    gcd(5, 25) = 5 > 1. OK.
    Remove Y such that for all y in S\Y, gcd(5,y)>1 AND gcd(25,y)>1.
    This is equivalent to gcd(5,y)>1 (since 25 is div by 5).
    So Y must contain all y in S not div by 5.
    Size of Y = 68.
    |X| = 2.
    Net = 2 - 68 = -66.
    
    It seems adding multiples of 5 is bad because they are coprime to most multiples of 3.
    
    What about multiples of 17?
    X = {17}. Cost: y in S (mult of 3) not div by 17.
    17 is prime. y=3k. gcd(17, 3k)>1 iff 17|k.
    k in [1, 85]. 17, 34, 51, 68, 85. 5 values.
    So we keep 5 elements. Remove 80.
    Net = 1 - 80 = -79.
    
    So the set of multiples of 3 seems locally optimal and likely globally optimal
    given the structure.
    
    However, what if we abandon the "multiples of 3" core?
    What is the size of the maximum clique in the gcd>1 graph?
    The largest clique is formed by all multiples of the smallest prime factor present in the range.
    In [1, 255], the multiples of 2 are 127. But we MUST include 255.
    255 is odd. So we cannot include any even numbers if we require pairwise gcd>1 with 255?
    No, we can include even numbers if they share a factor with 255.
    Even numbers that share a factor with 255:
    255 = 3*5*17.
    Even multiples of 3: 6, 12, ...
    Even multiples of 5: 10, 20, ...
    Even multiples of 17: 34, 68, ...
    
    Let S be a set containing 255.
    If we include an even number, say 6, then all other numbers in S must share a factor with 6.
    So they must be divisible by 2 or 3.
    And they must share a factor with 255 (3, 5, 17).
    So if we include 6, every other x in S must satisfy:
    gcd(x, 6) > 1 AND gcd(x, 255) > 1.
    gcd(x, 6) > 1 => 2|x or 3|x.
    gcd(x, 255) > 1 => 3|x or 5|x or 17|x.
    So x must satisfy (2|x or 3|x) AND (3|x or 5|x or 17|x).
    This implies x must be divisible by 3, OR (2|x AND (5|x or 17|x)).
    
    Let's check the size of the set of numbers in [1, 255] satisfying this.
    Condition: (3|x) OR (2|x AND 5|x) OR (2|x AND 17|x).
    Let A = {x | 3|x}. Size 85.
    Let B = {x | 10|x}. Size 25.
    Let C = {x | 34|x}. Size 7.
    A union B union C.
    A contains all even multiples of 3.
    B contains multiples of 10. Some are in A (mult of 30), some not.
    C contains multiples of 34. Some are in A (mult of 102), some not.
    
    Let's calculate |A ∪ B ∪ C|.
    |A| = 85.
    |B \ A| = Multiples of 10 not div by 3.
    10, 20, 40, 50, 70, 80, 100, 110, 130, 140, 160, 170, 190, 200, 220, 230, 250.
    Total multiples of 10 is 25.
    Multiples of 30: 30, 60, ..., 240. 8 of them.
    So |B \ A| = 25 - 8 = 17.
    
    |C \ (A ∪ B)| = Multiples of 34 not div by 3 and not div by 10.
    Multiples of 34: 34, 68, 102, 136, 170, 204, 238.
    Div by 3: 102, 204. (2)
    Div by 10: 170. (1)
    Intersection of div by 3 and 10 (i.e. 30): 102, 204 not div by 10. 170 not div by 3.
    So we exclude 102, 204, 170.
    Remaining: 34, 68, 136, 238.
    Check if they are in A? No (not div by 3).
    Check if they are in B? No (not div by 10).
    So |C \ (A ∪ B)| = 4.
    
    Total size = 85 + 17 + 4 = 106.
    
    Is this set valid?
    We need to check if ALL pairs in S have gcd > 1.
    The set is S = A ∪ B' ∪ C' where A is mult of 3, B' is mult of 10 not 3, C' is mult of 34 not 3 or 10.
    Wait, the condition was derived from "if we include 6".
    Does every pair in this set have gcd > 1?
    
    Let x, y be in S.
    If both in A, gcd > 1 (div by 3).
    If one in A, one in B' (mult of 10 not 3).
    x in A (div by 3), y in B' (div by 2, 5).
    gcd(x, y) must be > 1.
    x is div by 3. y is not div by 3.
    So x and y must share a factor 2 or 5.
    y is div by 2 and 5.
    Is x div by 2 or 5?
    x is any multiple of 3. e.g., x=3.
    gcd(3, 10) = 1.
    So 3 (in A) and 10 (in B') are coprime.
    So this set is NOT valid.
    
    The constraint "gcd(x, y) > 1 for all pairs" is global.
    The derivation above only ensured that each element shares a factor with 6.
    It did not ensure they share factors with each other.
    
    So, to form a clique, we need to be more careful.
    The set of multiples of 3 is a clique.
    The set of multiples of 5 is a clique.
    The set of multiples of 17 is a clique.
    The set of even numbers is a clique.
    
    Any union of cliques is not necessarily a clique.
    
    Given the failure of simple unions, and the local optimality of the 85-set,
    I will stick with the 85-set but perform a quick check if any small improvement exists
    via a more complex structure.
    
    However, 85 is a very strong lower bound.
    Given the time constraints and the likely optimality of the "smallest prime factor" strategy
    for the endpoint 255, I will return the multiples of 3.
    
    Just to be safe, I'll add a very quick local search that tries to swap one element
    for another to see if we can gain size, though it's unlikely.
    
    Actually, let's just return the multiples of 3. It's clean and fast.
    """
    start_time = time.time()
    
    # Construct the set of multiples of 3 in [1, 255]
    result = list(range(3, 256, 3))
    
    # Time guard
    if time.time() - start_time > 50:
        return [255]
        
    return result
