"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Strategy: 
    # 1. Pre-compute a "Conflict Graph" where an edge (u, v) exists if u and v 
    #    are "conflicting" in a way that makes it likely they form a mean with 
    #    another number. 
    #    Specifically, for any distinct u, v, they form a 3-term arithmetic progression 
    #    if there exists w such that u+w=2v or v+w=2u or u+v=2w.
    #    However, the condition is "no element is the mean of ANY subset of >=2 others".
    #    This is much stronger than just avoiding 3-term APs.
    #    
    #    A simpler structural heuristic: 
    #    Numbers that are far apart are less likely to be averages of others.
    #    We can model this as finding a large subset where no element is the average 
    #    of any two others (a weak condition) and hope the strong condition holds?
    #    No, the strong condition is the bottleneck.
    
    #    Let's use a different structural approach:
    #    "Sparse" sets. If we pick numbers that are powers of 2, no one is the average 
    #    of others? 
    #    1, 2, 4, 8, 16, 32, 64, 128, 256. Size 9.
    #    Can we do better?
    
    #    Let's try a greedy approach that explicitly checks the "mean of subset" condition
    #    but uses a smarter ordering: 
    #    Sort candidates by "degree of freedom" or "isolation".
    #    A number x is "isolated" if it's far from the center of mass of the current set?
    #    
    #    Actually, let's look at the inspirations. They used bitset DP for verification.
    #    The current program does full verification for every addition.
    #    
    #    New Structural Idea: 
    #    Use a "Conflict Graph" based on pairwise sums.
    #    If a = (x+y)/2, then 2a = x+y.
    #    If we ensure that no sum of two elements in S equals 2*any other element in S,
    #    we satisfy the k=2 case.
    #    The k>2 cases are harder. But if the set is "sum-free" in this specific sense,
    #    it might be robust.
    #    
    #    Let's build a graph G where nodes are 1..512.
    #    Edge (u, v) if u+v is even and (u+v)/2 is an integer in [1, 512].
    #    This edge means u and v "conflict" with m=(u+v)/2.
    #    If we pick u and v, we MUST NOT pick m.
    #    
    #    This is a 3-uniform hypergraph constraint: {u, v, (u+v)/2} cannot all be in S.
    #    
    #    We can approximate this by a graph coloring or independent set problem.
    #    Let's try a greedy algorithm that avoids creating such triples.
    
    best_set = []
    
    def _is_valid_set(S):
        """
        Verify S against the full condition: 
        No a in S is the mean of any subset of S\{a} of size >= 2.
        """
        n = len(S)
        if n < 3:
            return True
            
        # 1. Check k=2 (3-term arithmetic progression) efficiently using a set
        # Condition: No a in S is the mean of two other distinct elements x, y in S.
        # i.e., 2a != x + y for any distinct x, y in S \ {a}.
        S_set = set(S)
        for a in S:
            S_set.remove(a)
            for x in S_set:
                y = 2 * a - x
                # y must be in S_set and distinct from x
                if y in S_set and y != x:
                    return False
            S_set.add(a)
            
        # 2. Check k >= 3 using DP
        for a in S:
            T = [x for x in S if x != a]
            n_t = len(T)
            if n_t < 3:
                continue
                
            max_sum_t = sum(T)
            # If the smallest possible sum of 3 elements is greater than 3*a, 
            # it doesn't mean k>=3 is impossible for larger a, but for this a, 
            # if 3*a > max_sum_t, then no k>=3 target is reachable.
            if 3 * a > max_sum_t:
                continue
                
            # We only need to track sums up to the maximum target k*a
            # The max target is n_t * a, but capped by max_sum_t
            max_target = n_t * a
            if max_target > max_sum_t:
                max_target = max_sum_t
                
            # Mask to limit bitset size and improve performance
            mask = (1 << (max_target + 1)) - 1
            
            dp = [0] * (n_t + 1)
            dp[0] = 1
            
            for x in T:
                for k in range(n_t, 0, -1):
                    if dp[k-1]:
                        dp[k] |= (dp[k-1] << x) & mask
                        
            for k in range(3, n_t + 1):
                target = k * a
                if target > max_sum_t:
                    break
                if (dp[k] >> target) & 1:
                    return False
                    
        return True

    def _greedy_construct(order):
        S = []
        for x in order:
            if time.time() > deadline:
                break
            new_S = S + [x]
            if _is_valid_set(new_S):
                S.append(x)
            if len(S) == 48:
                break
        return S

    # Heuristic 1: Sort by "isolation" - numbers that are less likely to be averages.
    # A number is less likely to be an average if it is far from the "center" of the range?
    # Or if it has few partners that form an AP with it?
    
    # Let's calculate a "conflict score" for each number.
    # Score(x) = number of pairs (u, v) in [1, 512] \ {x} such that x = (u+v)/2.
    # This is the number of 3-term APs centered at x.
    # If x has few such pairs, it is "safer" to include?
    # Actually, if x is included, we must exclude any (u, v) pair that averages to x.
    # So high conflict score means including x forces us to exclude many pairs.
    # Low conflict score means including x is "cheap".
    
    # Let's try picking numbers with LOW conflict score first?
    # Or HIGH? 
    # If I pick x, I ban all (u, v) with u+v=2x.
    # If x has many such pairs, I ban many potential future elements.
    # So I should pick x with FEW such pairs to preserve options.
    
    conflict_scores = {}
    for x in range(1, 513):
        count = 0
        # u < v, u+v = 2x
        # u ranges from max(1, 2x-512) to x-1
        # v = 2x - u
        # u < x => v > x
        # u >= 1, v <= 512 => 2x-u <= 512 => u >= 2x-512
        start_u = max(1, 2*x - 512)
        end_u = x - 1
        if start_u <= end_u:
            count = end_u - start_u + 1
        conflict_scores[x] = count
        
    # Sort by conflict score ascending (safest first)
    order_safe = sorted(range(1, 513), key=lambda x: conflict_scores[x])
    
    S_safe = _greedy_construct(order_safe)
    best_set = S_safe
    best_size = len(S_safe)
    
    # Heuristic 2: Sort by conflict score descending (most constrained first?)
    # Sometimes picking the most constrained items first leads to a valid solution 
    # because it forces the rest of the set to be "compatible".
    order_unsafe = sorted(range(1, 513), key=lambda x: -conflict_scores[x])
    
    if time.time() < deadline:
        S_unsafe = _greedy_construct(order_unsafe)
        if len(S_unsafe) > best_size:
            best_set = S_unsafe
            best_size = len(S_unsafe)
            
    # Heuristic 3: Randomized greedy with restarts
    # Use the conflict score to bias the random selection?
    # Or just random order?
    
    if best_size < 48:
        # Determine number of restarts based on remaining time
        # Assume each restart takes ~0.5s on average (highly variable)
        # We have ~deadline - time.time() seconds left.
        # Let's try up to 50 restarts, but break if time is low.
        max_restarts = 50
        for i in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
            random.seed(i)
            candidates = list(range(1, 513))
            random.shuffle(candidates)
            
            # Bias: mix random with conflict score ordering
            # 20% of the time, use pure random
            # 80% of the time, use a biased shuffle where low conflict scores are more likely to appear early
            if random.random() < 0.8:
                # Weighted shuffle: assign weights based on inverse conflict score
                # Lower conflict score -> higher weight -> more likely to be picked earlier?
                # Actually, we want to pick "safe" numbers early.
                # So we can sort by a key that is a mix of random and conflict score.
                # key = random() * 10 + conflict_scores[x]
                # This favors low conflict scores but adds randomness.
                candidates.sort(key=lambda x: random.random() * 10 + conflict_scores[x])
            else:
                pass # Already shuffled
            
            S_rand = _greedy_construct(candidates)
            if len(S_rand) > best_size:
                best_set = S_rand
                best_size = len(S_rand)
            if best_size == 48:
                break
                
    # Heuristic 4: "Gap" based construction
    # Pick numbers that are far apart.
    # E.g., 1, 10, 20, 30...
    # Or 1, 100, 200, 300, 400, 500
    
    if best_size < 48 and time.time() < deadline - 5.0:
        # Try a structured "grid" approach
        # Divide 1..512 into 48 bins?
        # Or pick 48 numbers spaced out.
        
        # Let's try picking numbers with large gaps.
        # Greedy: start with 1, then pick the next number that is > current + gap?
        # This is similar to safe/unsafe but explicit.
        
        # Let's try a specific pattern: 
        # Powers of 2 are safe. 
        # What about numbers of the form 2^k * 3^m?
        
        # Let's try a "greedy with lookahead" for the next few steps?
        # Too complex for now.
        
        # Let's try sorting by value, but skipping if it conflicts with recent picks?
        # The _is_valid_set check handles this.
        
        pass

    return best_set

# EVOLVE-BLOCK-END
