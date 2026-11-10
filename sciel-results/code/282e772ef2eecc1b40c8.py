"""Improved solver for: no selected integer divides the sum of any 3 distinct others."""
import time
import random
from itertools import combinations

N = 192
MAX_CARD = 48

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin under 90s wall-clock limit

    # Precompute conflict information.
    # For each candidate value a (1..192), we want to know which sets S of other values
    # cause a conflict: a divides sum of some 3 distinct elements of S.
    # We'll build a graph where edge (u,v) means u and v cannot both be in the set?
    # Actually the constraint is: for selected set S, for every a in S, for every triple
    # {x,y,z} in S\{a}, a does NOT divide x+y+z.
    # This is a 3-uniform hypergraph constraint, not a simple graph.
    
    # Strategy: greedy with multi-start + local search.
    # Precompute for each pair of values, and use fast validity checks.
    
    # Precompute: for each a in 1..192, and for each pair (x,y) in 1..192, x!=y,
    # determine which z values would make a divide x+y+z.
    # Actually, let's think differently.
    
    # For a given set S and candidate a not in S, adding a is valid iff:
    # 1. For every triple {x,y,z} in S, a does not divide x+y+z.
    # 2. For every pair {x,y} in S and a itself... wait, a is not in the summands.
    #    The constraint is: a must not divide the sum of any 3 DISTINCT OTHER selected integers.
    #    So when we add a to S, we need:
    #    - For all triples in S: a ∤ sum(triple)
    #    - For all pairs {x,y} in S: a ∤ (a+x+y) ... no wait.
    #    Let me re-read: "No selected integer may divide the sum of any k=3 DISTINCT OTHER selected integers."
    #    So for a in S, we look at triples from S\{a}. When we add a to existing S:
    #    - Check: does a divide sum of any triple from S? (triples of other elements)
    #    - Also: for each existing element b in S, we need to check if b divides sum of any triple from S∪{a}\{b}.
    #      The new triples involving a are: {a, x, y} for x,y in S.
    #      So for each b in S, check if b divides a+x+y for any pair {x,y} in S.
    
    # This is expensive to check naively. Let's use a different approach.
    
    # Multi-start greedy with randomized ordering, then local search to improve.
    
    # Precompute all triple sums for values 1..192? That's C(192,3) ≈ 1.1M triples.
    # For each triple, record its sum. Then for a candidate a, a conflicts with triple T
    # if a divides sum(T).
    
    # Alternative: for each a, precompute the set of triple-sums mod a that equal 0.
    # When checking if a can be added to S, we need no triple from S has sum divisible by a.
    
    # Let's precompute for each a (1..192), a set of "bad" triples? Too memory heavy.
    
    # Better: for each a, precompute for each pair (x,y), the set of z such that a | x+y+z.
    # Then when checking if a can join S: for each pair (x,y) in S, check if any z in S\{x,y}
    # satisfies a | x+y+z.
    
    # Even better: for each a and each pair (x,y), compute r = (-x-y) mod a. Then a divides
    # x+y+z iff z ≡ r (mod a). So we just need to check if S contains a z ≡ r (mod a) with z≠x,y.
    
    # Precompute: for a in 1..192, for x in 1..192, for y in 1..192, x!=y:
    #   r[a][x][y] = (-x-y) % a
    # But that's 192^3 ≈ 7M entries. Too much.
    
    # Instead, compute on the fly but cache per a.
    
    # Let's use a bitset approach for efficiency.
    # For each a, maintain a bitset of which values are "present" in S.
    # For checking a new candidate a against S:
    #   For each pair (x,y) in S: need to check if exists z in S, z!=x,y, with z ≡ (-x-y) mod a.
    
    # This is still O(|S|^2) per check. With |S| up to ~40, that's ~1600 checks per addition,
    # times 192 candidates = ~300K operations per greedy pass. Multi-start with 50 starts = 15M. Fine.
    
    # But we also need to check existing elements b in S against new triples involving a.
    # For each b in S, for each pair (x,y) in S: check if b | a+x+y.
    # That's another O(|S|^2) per b, so O(|S|^3) total. With |S|=40, that's 64K per addition.
    # Times 192 = 12M per pass. With 50 starts = 600M. Too slow in Python.
    
    # Let me optimize: when adding a to S, the new triples are {a,x,y} for x,y in S.
    # For each existing b in S, we need: b ∤ a+x+y for all pairs x,y in S.
    # Equivalently: for each b in S, for all x,y in S, (a+x+y) % b != 0.
    # We can precompute for each b, the set of pairs (x,y) where x+y ≡ -a (mod b).
    
    # Actually, let me just implement it carefully and use early termination.
    
    def is_valid_addition(a, S):
        """Check if a can be added to set S (as a set of values)."""
        S_set = set(S)
        # Check 1: a does not divide sum of any triple in S
        for x, y, z in combinations(S, 3):
            if (x + y + z) % a == 0:
                return False
        # Check 2: for each b in S, b does not divide a+x+y for any pair x,y in S
        for b in S:
            for x, y in combinations(S, 2):
                if (a + x + y) % b == 0:
                    return False
        return True
    
    def is_valid_set(S):
        """Check if S is a valid set."""
        S_list = list(S)
        for a in S_list:
            others = [v for v in S_list if v != a]
            for x, y, z in combinations(others, 3):
                if (x + y + z) % a == 0:
                    return False
        return True
    
    # Multi-start greedy
    best_set = []
    best_size = 0
    
    all_values = list(range(1, N + 1))
    
    # Try several random orderings
    rng = random.Random(42)
    
    num_starts = 200
    for start_idx in range(num_starts):
        if time.time() > deadline:
            break
        
        # Randomize ordering
        order = all_values[:]
        rng.shuffle(order)
        
        S = []
        S_set = set()
        
        for a in order:
            if time.time() > deadline:
                break
            if len(S) >= MAX_CARD:
                break
            if is_valid_addition(a, S):
                S.append(a)
                S_set.add(a)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
        
        # Local search: try to swap elements to improve
        # Try removing one and adding two, or just adding if possible
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            S_set = set(S)
            # Try adding any remaining element
            for a in all_values:
                if a in S_set:
                    continue
                if is_valid_addition(a, S):
                    S.append(a)
                    S_set.add(a)
                    improved = True
                    if len(S) >= MAX_CARD:
                        break
                    if time.time() > deadline:
                        break
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
        
        if len(S) >= MAX_CARD:
            break
    
    return best_set

# EVOLVE-BLOCK-END
