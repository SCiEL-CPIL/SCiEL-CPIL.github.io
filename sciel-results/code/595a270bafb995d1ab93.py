"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from random import Random

DEADLINE = time.time() + 80  # safety margin before 90s kill

def _precompute_conflicts():
    """
    Precompute a conflict graph where nodes are integers 1..128.
    An edge (u, v) exists if there is a third integer w such that
    gcd(u,v), gcd(u,w), gcd(v,w) are all equal.
    Actually, the condition is: no three DISTINCT elements a,b,c have
    gcd(a,b) == gcd(a,c) == gcd(b,c).
    
    So, we need to avoid any triple {a,b,c} such that g_ab = g_ac = g_bc.
    Let's define a "bad triple" as such a set.
    We want to find the largest subset S such that no bad triple is fully contained in S.
    This is equivalent to finding the Maximum Independent Set in a 3-uniform hypergraph,
    which is hard. However, we can reformulate it.
    
    If we fix a value g, let S_g be the set of numbers divisible by g.
    If a, b, c are all in S_g, then g divides gcd(a,b), gcd(a,c), gcd(b,c).
    But the gcds must be EXACTLY equal to each other.
    
    Let's think differently.
    For a set S, the condition fails if there exist distinct a,b,c in S such that
    gcd(a,b) = gcd(a,c) = gcd(b,c) = g.
    This implies a, b, c are all multiples of g.
    Let a=g*a', b=g*b', c=g*c'.
    Then gcd(a',b') = gcd(a',c') = gcd(b',c') = 1.
    So, the condition fails if there exists a g and three distinct multiples of g
    whose quotients are pairwise coprime.
    
    This is still complex. Let's stick to the graph formulation for MIS.
    We can define a conflict between two elements x, y if they are "dangerous" together.
    But the constraint is on triples.
    
    Alternative approach:
    We can use a greedy algorithm with a strong pruning heuristic based on the structure.
    Or, we can use a SAT solver / ILP solver if available, but we only have stdlib + numpy/scipy.
    
    Let's try a different structural insight.
    If we choose only numbers that are prime powers, say p^k.
    gcd(p^i, q^j) = 1 if p != q.
    If p = q, gcd(p^i, p^j) = p^min(i,j).
    Consider three numbers: p^a, p^b, p^c.
    gcds are p^min(a,b), p^min(a,c), p^min(b,c).
    These are equal iff min(a,b) = min(a,c) = min(b,c).
    This implies a=b=c, which is not allowed (distinct elements).
    Wait, if a < b and a < c, then min(a,b)=a, min(a,c)=a, min(b,c)=min(b,c).
    For them to be equal, we need a = min(b,c). So a <= b and a <= c and a = min(b,c).
    This means a is the smallest, and b,c >= a.
    If a < b and a < c, then min(b,c) >= a. If min(b,c) > a, then gcds are a, a, >a -> not equal.
    If min(b,c) = a, then b=a or c=a, not distinct.
    So, three distinct prime powers of the SAME prime never form a bad triple?
    Let's check: 2, 4, 8.
    gcd(2,4)=2, gcd(2,8)=2, gcd(4,8)=4. Not equal.
    4, 8, 16.
    gcd(4,8)=4, gcd(4,16)=4, gcd(8,16)=8. Not equal.
    It seems prime powers of the same prime are safe.
    
    What about prime powers of DIFFERENT primes?
    p^a, q^b, r^c with distinct primes.
    gcds are all 1. So they form a bad triple!
    So we cannot have three numbers that are pairwise coprime.
    This means we can have at most 2 numbers from any set of pairwise coprime numbers.
    
    This suggests a strategy:
    Group numbers by their "squarefree kernel" or something similar?
    
    Let's look at the inspiration 1: "precomputed conflict-graph + Maximum Independent Set (MIS) via greedy coloring (Tomita)".
    How to build the conflict graph for a 3-uniform constraint?
    Usually, MIS is for 2-uniform (edges).
    If we have a 3-uniform hypergraph, we can't directly use standard MIS.
    However, we can reduce it.
    If a triple {a,b,c} is bad, we cannot include all three.
    This is a 3-SAT like constraint.
    
    Maybe we can define a conflict graph where x and y are connected if they "force" a conflict with some z?
    No, that's not transitive.
    
    Let's try a different approach: **Greedy with Lookahead / Backtracking**.
    Since N=128 is small, we can use a recursive backtracking with strong pruning.
    Pruning: If we have a set S, and a candidate x, check if adding x creates a bad triple with any pair in S.
    This is O(|S|^2).
    We can order the elements to maximize the chance of success.
    
    Key Insight from Literature (Ruzsa-Szemeredi type):
    The problem is related to finding large sets with no 3-term arithmetic progressions in the gcd sense? No.
    
    Let's look at the structure of bad triples again.
    gcd(a,b) = gcd(a,c) = gcd(b,c) = g.
    This means a,b,c are multiples of g, and a/g, b/g, c/g are pairwise coprime.
    
    Strategy:
    1. Iterate over possible g.
    2. For a fixed g, let M_g = {x/g : x is multiple of g, x in [1,128]}.
    3. We cannot pick 3 numbers from M_g that are pairwise coprime.
    4. This implies that for each g, the subset of M_g we pick must not contain 3 pairwise coprime numbers.
    5. By the Erdos-Ko-Rado type or simple combinatorics, if we pick numbers from M_g, they must share a common factor?
       If we pick a set A from M_g, and no 3 are pairwise coprime, does it mean all share a factor?
       Not necessarily. E.g., 2, 3, 6. gcd(2,3)=1, gcd(2,6)=2, gcd(3,6)=3. Not a bad triple.
       Wait, the condition is that the three pairwise gcds are EQUAL.
       In 2,3,6: 1, 2, 3. Not equal.
       So 2,3,6 is a VALID triple.
       
    So the constraint is specific: the three gcds must be IDENTICAL.
    
    Let's re-evaluate "Prime Powers".
    If we pick all prime powers, do we have any bad triples?
    Bad triple requires gcd(a,b)=gcd(a,c)=gcd(b,c)=g.
    If a,b,c are prime powers.
    Case 1: Distinct primes. gcds are 1,1,1. BAD TRIPLE.
    So we cannot have three numbers with distinct prime bases.
    This means we can have at most 2 numbers with distinct prime bases.
    So we can pick at most 2 "odd" prime bases? No, we can pick many powers of 2, many powers of 3, etc.
    But we can only pick at most 1 number from the "rest"?
    If we pick 2^a, 3^b, 5^c. gcds are 1. Bad.
    So we can pick at most 2 numbers that are not powers of the same prime?
    Actually, if we pick 2^a, 2^b, 3^c.
    gcd(2^a, 2^b) = 2^min.
    gcd(2^a, 3^c) = 1.
    gcd(2^b, 3^c) = 1.
    For these to be equal, we need 2^min = 1. So min=0. Impossible for distinct powers > 1.
    So {2^a, 2^b, 3^c} is VALID if a,b >= 1.
    
    So, we can have many powers of 2, many powers of 3, etc.
    But we cannot have three numbers from three DIFFERENT prime bases.
    So, we can pick all powers of 2, all powers of 3, and at most one power of 5?
    If we pick 2^a, 3^b, 5^c. Bad (gcds 1,1,1).
    So we can pick all powers of 2, all powers of 3, and NO powers of 5?
    Or pick all powers of 2, and at most 2 powers of other primes?
    If we pick 2^a, 3^b, 5^c, 7^d.
    3^b, 5^c, 7^d is a bad triple (gcds 1,1,1).
    So we can pick at most 2 numbers from the set of "non-2-powers".
    And those 2 must not form a bad triple with any 2-power?
    2^a, 3^b, 5^c. Gcds: 2^k, 1, 1. Not equal unless 2^k=1.
    So 2^a, 3^b, 5^c is valid.
    
    So the strategy "All prime powers" fails because of the distinct primes issue.
    We need to restrict the number of distinct prime bases.
    
    Revised Strategy:
    1. Pick all numbers of the form 2^k (k>=0, 2^k <= 128).
    2. Pick all numbers of the form 3^k (k>=0, 3^k <= 128).
    3. Can we pick anything else?
       If we pick x, it has a prime factor p != 2,3.
       If we pick x, y, z all with prime factors not in {2,3}, and they are pairwise coprime, bad.
       If they share factors, maybe okay.
       
    Let's try a computational approach with the Tomita MIS idea, but adapted.
    Since the constraint is "no 3 pairwise coprime numbers in any g-multiple set", it's complex.
    
    Let's go back to the code. The current code scores 1.875.
    The inspiration suggests MIS.
    How to map to MIS?
    If we define a graph where x and y are connected if they are "incompatible" with some z?
    No.
    
    Let's try a simple but effective heuristic:
    **Greedy by "Conflict Degree"**.
    Precompute for each x, the number of pairs (y,z) such that {x,y,z} is a bad triple.
    Sort x by this count (ascending).
    Greedily add x if valid.
    
    This is O(N^3) to precompute, which is 128^3 ~ 2M. Very fast.
    Then greedy is O(N^3) worst case, but with pruning, much faster.
    
    Let's implement this.
    
    1. Precompute bad triples.
       For each triple (a,b,c) with a<b<c:
         if gcd(a,b)==gcd(a,c)==gcd(b,c): mark as bad.
    2. For each x, count how many bad triples contain x.
    3. Sort x by count (lowest first).
    4. Greedy add.
    
    This should be significantly better than random greedy.
    """
    N = 128
    numbers = list(range(1, N+1))
    
    # Precompute gcds for all pairs
    gcds = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            gcds[i][j] = gcd(i, j)
    
    # Precompute bad triples and conflict counts
    # A triple (a,b,c) is bad if gcds[a][b] == gcds[a][c] == gcds[b][c]
    conflict_count = [0]*(N+1)
    # We can also store the actual conflicts for fast checking later, 
    # but for greedy, we just need to check validity on the fly.
    # However, knowing the count helps ordering.
    
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            g_ab = gcds[a][b]
            for c in range(b+1, N+1):
                if gcds[a][c] == g_ab and gcds[b][c] == g_ab:
                    conflict_count[a] += 1
                    conflict_count[b] += 1
                    conflict_count[c] += 1
    
    # Sort numbers by conflict count (ascending)
    ordered_numbers = sorted(numbers, key=lambda x: conflict_count[x])
    
    return ordered_numbers, gcds

def _is_valid_addition(S, x, gcds):
    """Check if adding x to set S keeps the property."""
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(i+1, n):
            b = S[j]
            g_ab = gcds[a][b]
            g_ax = gcds[a][x]
            g_bx = gcds[b][x]
            if g_ab == g_ax == g_bx:
                return False
    return True

def _greedy_forward(order, gcds):
    """Greedy: add elements in given order if valid."""
    S = []
    for x in order:
        if _is_valid_addition(S, x, gcds):
            S.append(x)
    return S

def construct():
    if time.time() > DEADLINE:
        return [1]
        
    ordered_numbers, gcds = _precompute_conflicts()
    
    # Strategy 1: Greedy by conflict count (ascending)
    S1 = _greedy_forward(ordered_numbers, gcds)
    best = S1
    best_len = len(S1)
    
    # Strategy 2: Greedy by conflict count (descending) - maybe less conflicts first is better,
    # but sometimes high conflict nodes are "hubs" and removing them allows more others?
    # No, we are adding. Low conflict nodes are "safer".
    
    # Strategy 3: Randomized restarts with the ordered list, but shuffling ties?
    # Or just different random permutations as backup.
    rng = Random(42)
    
    # Try a few random permutations to see if we beat the heuristic
    for i in range(50):
        if time.time() > DEADLINE:
            break
        order = list(range(1, 129))
        rng.shuffle(order)
        S = _greedy_forward(order, gcds)
        if len(S) > best_len:
            best = S
            best_len = len(S)
            
    # Local search: try to add more
    not_in = [x for x in range(1, 129) if x not in best]
    for _ in range(100):
        if time.time() > DEADLINE:
            break
        if not not_in:
            break
        x = rng.choice(not_in)
        if _is_valid_addition(best, x, gcds):
            best.append(x)
            not_in.remove(x)
            
    if not best:
        return [1]
        
    return sorted(best)
# EVOLVE-BLOCK-END
