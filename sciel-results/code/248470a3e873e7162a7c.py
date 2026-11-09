import time
import random
from itertools import combinations

N = 192
MAX_SIZE = 48
K = 3

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_size = 0
    best_set = []
    
    # Precompute conflict hyperedges:
    # A hyperedge is a frozenset of 4 distinct integers {a, b, c, d} such that
    # a divides (b + c + d).
    # Note: The condition is "No selected integer may divide the sum of any k=3 DISTINCT OTHER selected integers."
    # So if we select {a, b, c, d}, we must NOT have (a | (b+c+d)) OR (b | (a+c+d)) OR (c | (a+b+d)) OR (d | (a+b+c)).
    # Thus, the hyperedge exists if ANY of these conditions is true.
    
    # To make the independent set problem work, a hyperedge connects the 4 vertices if they form a "bad" configuration.
    # We want a set of vertices with NO hyperedge fully contained in it.
    
    # Generating all hyperedges explicitly might be memory heavy (192 choose 4 is ~50M, but we only care about valid divisibility).
    # However, we can compute "conflict count" on the fly or precompute a compact representation.
    
    # Alternative: Precompute for each number x, the list of pairs (y, z) such that x | (y+z+w) is not directly possible with just pairs.
    # Let's stick to the on-the-fly check but optimize the "degree" calculation for the heuristic.
    
    # Actually, for the Minimum Degree Heuristic in a Hypergraph:
    # Degree of a vertex v = number of hyperedges containing v.
    # When we select a vertex v, we "remove" it and all hyperedges containing it.
    # We want to select vertices such that no hyperedge is fully selected.
    # This is equivalent to finding a set S such that for every hyperedge H, H is not a subset of S.
    # This is the Independent Set problem in a hypergraph.
    
    # Heuristic: At each step, pick the vertex v that is contained in the FEWEST hyperedges that are still "active" (i.e., not yet broken by previous selections).
    # A hyperedge is "broken" if at least one of its vertices has already been selected? No, that's for Vertex Cover.
    # For Independent Set: We select vertices. A hyperedge is "violated" if ALL its vertices are selected.
    # So, we can select a vertex v if it doesn't complete any hyperedge.
    # The "risk" of selecting v is the number of hyperedges H containing v where the other |H|-1 vertices are ALREADY selected.
    # If that count is 0, v is safe.
    # We want to pick a safe vertex that minimizes future conflicts.
    # A good proxy for "future conflicts" is the number of hyperedges containing v that are still "intact" (i.e., none of their vertices have been selected yet).
    # If we pick v from a set of safe vertices, we should pick the one that "breaks" the most potential future hyperedges? Or the one that is in the fewest?
    # Standard Min-Degree for Independent Set in graphs: Pick vertex with min degree.
    # Here, let's define the "active degree" of v as the number of hyperedges H containing v such that H \ {v} is disjoint from the current selected set S.
    # If we pick v, we break all these hyperedges (they can never be completed again because v is now in S, but wait... if v is in S, the hyperedge is NOT completed unless ALL are in S. By adding v, we make it HARDER to complete? No, we make it EASIER to complete if the rest are already there.
    
    # Let's re-evaluate.
    # Condition: No hyperedge H is a subset of S.
    # If we add v to S, we create a violation if there exists H containing v such that H \ {v} is already in S.
    # So, a vertex v is "safe" if there is NO hyperedge H containing v such that H \ {v} subseteq S.
    # Among safe vertices, which one to pick?
    # Picking v will "consume" v. It doesn't break hyperedges, it just adds a vertex.
    # However, picking v might make other vertices UNSAFE in the future?
    # Actually, adding vertices to S only makes it harder to satisfy the condition (more potential for a full hyperedge to be formed).
    # So, we want to pick vertices that are "least likely" to be part of a completed hyperedge in the future.
    # A vertex v is "dangerous" if it is in many hyperedges H where the other vertices are also "likely to be picked".
    # Since we don't know the future, a standard heuristic is to pick the vertex with the MINIMUM "potential to cause conflicts".
    # Let's define the score of a safe vertex v as the number of hyperedges H containing v.
    # We pick the safe vertex with the MINIMUM score. This is the Min-Degree heuristic.
    
    # Precomputation:
    # 1. List of all hyperedges.
    # 2. For each vertex, list of hyperedges it belongs to.
    
    # Given N=192, number of hyperedges could be large.
    # Let's estimate. For each a, how many (b,c,d) such that a | (b+c+d)?
    # Sum is up to 3*191 = 573.
    # For a=1, ALL triples work. That's C(191, 3) ~ 1.1M hyperedges just for a=1?
    # Wait, the hyperedge is {a,b,c,d}. If a=1, then {1,b,c,d} is a hyperedge for ANY b,c,d.
    # So 1 is in ~1.1M hyperedges.
    # Storing all hyperedges explicitly is too much memory (1.1M * 4 ints * 4 bytes ~ 17MB, actually okay, but iterating is slow).
    
    # Let's optimize the check.
    # Instead of storing all hyperedges, we can compute the "safety" and "degree" on the fly using the structure.
    
    # Safety Check for v given S:
    # v is unsafe if exists b,c,d in S such that v | (b+c+d).
    # This is the same check as in the original code's `_check_addition` part 1.
    
    # Degree Heuristic for v:
    # We want to minimize the number of "active" hyperedges containing v.
    # An active hyperedge is one that is not yet broken.
    # Wait, in Independent Set, we don't "break" hyperedges by selecting vertices. We "complete" them.
    # Selecting a vertex v adds it to the set.
    # The "risk" of v is how many hyperedges containing v have their OTHER vertices already in S? No, that would make it unsafe.
    # The "risk" for the FUTURE is: how many hyperedges containing v have their other vertices NOT in S yet, but are "likely" to be picked?
    # This is hard to estimate.
    
    # Let's try a different heuristic:
    # Pick the vertex v that minimizes the number of "potential partners".
    # Or, simply: Randomized Greedy with a "safety margin" score.
    
    # Given the complexity, let's stick to a robust Randomized Greedy that uses the "Minimum Conflict Potential" heuristic.
    # For each safe candidate v, calculate a "penalty" score.
    # Penalty = Number of pairs (b,c) in S such that there exists d in [1, N] \ S such that v | (b+c+d) AND b | (v+c+d) ... this is getting too complex.
    
    # Let's use a simpler structural change:
    # The original code uses `combinations` which is slow.
    # Let's use a NumPy-based vectorized check for safety, and a heuristic based on "divisibility density".
    
    # Actually, the "Minimum Degree" in the context of Hypergraph Independent Set often refers to:
    # Pick the vertex that appears in the fewest hyperedges.
    # Since we can't store all hyperedges, we can approximate the "degree" of v by the number of (b,c,d) such that v | (b+c+d).
    # This is a static property of v.
    # Let's precompute this static "conflict potential" for each v.
    
    # static_degree[v] = number of triples (b,c,d) from [1, N] \ {v} such that v | (b+c+d).
    # We prefer vertices with LOW static_degree.
    
    # Let's precompute this.
    
    def compute_static_degrees():
        degrees = [0] * (N + 1)
        # For each v, count how many triples sum to a multiple of v.
        # This is O(N^4) if done naively. 192^4 is ~1.3B. Too slow.
        
        # Optimization:
        # For a fixed v, we need count of (b,c,d) such that b+c+d = k*v.
        # This is a convolution problem.
        # Let A be the indicator array of available numbers (initially all 1s except v).
        # We need the coefficient of x^{k*v} in A(x)^3.
        # We can use FFT? Or just simple loops with pruning?
        
        # Given N=192, max sum is 576.
        # For each v, possible sums are v, 2v, ..., up to 576.
        # We can compute the number of triples summing to S for all S in 1..576.
        # This can be done for ALL v at once?
        # Let T[S] be the number of triples (b,c,d) from [1,N] summing to S.
        # Then static_degree[v] = sum_{k} T[k*v].
        # But we must exclude v from b,c,d.
        # T[S] includes triples with v.
        # Let T_all[S] be triples from [1,N].
        # Let T_no_v[S] be triples from [1,N]\{v}.
        # T_no_v[S] = T_all[S] - (triples containing v that sum to S).
        # Triples containing v: (v, b, c) such that v+b+c=S => b+c = S-v.
        # Let P[S-v] be the number of pairs (b,c) from [1,N]\{v} such that b+c = S-v.
        # So T_no_v[S] = T_all[S] - P_no_v[S-v].
        
        # Step 1: Compute T_all[S] for S in 3..3N.
        # Step 2: For each v, compute P_no_v[sum] for sum in 2..2N.
        # Step 3: static_degree[v] = sum_{k} T_no_v[k*v].
        
        # Computing T_all:
        # T_all[S] = sum_{b=1..N} sum_{c=1..N} [b+c <= S-1] * [S-b-c in 1..N]
        # This is O(N^2) per S? No, O(N^3) total if we do it naively.
        # We can use convolution.
        # A[x] = 1 if 1<=x<=N else 0.
        # T_all is the 3-fold convolution of A.
        # We can use numpy FFT.
        
        import numpy as np
        
        A = np.zeros(N + 1)
        A[1:N+1] = 1.0
        
        # Convolve A with A to get pairs
        P_all = np.convolve(A, A, mode='full')
        # P_all[s] is number of pairs (b,c) in [1,N]^2 summing to s.
        
        # Convolve P_all with A to get triples
        T_all = np.convolve(P_all, A, mode='full')
        # T_all[s] is number of triples (b,c,d) in [1,N]^3 summing to s.
        
        # Now, for each v, we need to subtract triples containing v.
        # Triples containing v: (v, b, c) with b,c in [1,N]\{v}.
        # Sum = v + b + c.
        # Let P_no_v[s] be number of pairs (b,c) in [1,N]\{v}^2 summing to s.
        # Then number of triples containing v summing to S is P_no_v[S-v].
        
        # We need to compute P_no_v for each v.
        # P_no_v[s] = P_all[s] - (pairs containing v that sum to s).
        # Pairs containing v: (v, c) or (c, v) or (v,v).
        # Since b,c are distinct from v? The problem says "distinct OTHER selected integers".
        # In the static degree calculation, we are counting potential conflicts.
        # The condition is "distinct other". So b, c, d must be distinct from v.
        # Also b, c, d must be distinct from each other?
        # "sum of any k=3 DISTINCT OTHER selected integers".
        # Yes, b, c, d must be distinct.
        
        # My convolution T_all counts (b,c,d) not necessarily distinct.
        # This is a problem. The convolution counts (1,1,1) as one triple, but we need distinct.
        # However, for large N, the number of non-distinct triples is small.
        # Or we can correct for it.
        # Number of triples with at least one equal element:
        # 1. All three equal: (x,x,x). Sum 3x. Count 1 for each x.
        # 2. Two equal: (x,x,y) or (x,y,x) or (y,x,x). Sum 2x+y.
        # 
        # Let's stick to the on-the-fly check for safety, but use the static degree as a heuristic for ORDERING.
        # The static degree doesn't need to be exact, just a good proxy for "how many conflicts this number has".
        # Numbers that divide many sums are "bad".
        # Small numbers divide many sums. Large numbers divide few.
        # So, static degree is likely inversely proportional to v.
        # So, we should pick LARGE numbers first?
        # The original code tried descending order and got 1.0.
        # The inspiration says "Greedy removal from densest valid-likely pool (144-192) ... found 48".
        
        # So, a strategy of picking from the high end is good.
        # Let's combine:
        # 1. Start with the set of high numbers (e.g., 144-192).
        # 2. Use a "Conflict Avoidance" heuristic to pick the "safest" number from this pool.
        # 3. If the pool is exhausted or we reach 48, stop.
        # 4. If we need more, expand the pool to lower numbers.
        
        pass
    # I will implement a hybrid:
    # 1. Precompute a "conflict score" for each number based on divisibility frequency.
    # 2. Use a greedy algorithm that picks the number with the LOWEST conflict score among those that are currently SAFE.
    # 3. If no safe number is found in the current "tier", move to the next tier.
    
    # Precompute conflict scores (approximate)
    # Score[v] = number of multiples of v in the range [3, 3N].
    # This is a proxy for how many sums can be divisible by v.
    scores = [0] * (N + 1)
    for v in range(1, N + 1):
        scores[v] = 3 * N // v  # Approx count of multiples
    
    # Sort numbers by score (ascending, so low score = good)
    # But we also want to consider the "safety" at runtime.
    
    best_set = []
    
    # Strategy: Iterative Deepening / Tiered Greedy
    # Tiers: [192..144], [143..96], [95..48], [47..1]
    # Start with highest tier.
    
    tiers = [
        list(range(192, 143, -1)),
        list(range(143, 95, -1)),
        list(range(95, 47, -1)),
        list(range(47, 0, -1))
    ]
    
    S = []
    S_set = set()
    
    def is_safe(x, S_list):
        # Check if x can be added to S_list
        # 1. x does not divide sum of any 3 distinct in S_list
        # 2. No a in S_list divides x + b + c for any b,c in S_list \ {a}
        
        n = len(S_list)
        if n < 3:
            # Part 1 is trivially true
            pass
        else:
            # Check 1: x | (a+b+c)
            # Use vectorized sum check if possible, but n is small (<=48)
            # O(n^3) is 48^3 = 110k ops. Fast enough.
            for a, b, c in combinations(S_list, 3):
                if (a + b + c) % x == 0:
                    return False
        
        if n < 2:
            pass
        else:
            # Check 2: a | (x+b+c)
            for a in S_list:
                # Need b, c in S_list \ {a}, b != c
                # (x + b + c) % a != 0
                # Iterate b
                for b in S_list:
                    if b == a: continue
                    for c in S_list:
                        if c == a or c == b: continue
                        if (x + b + c) % a == 0:
                            return False
        return True

    # Greedy with Heuristic
    # At each step, from the current tier's remaining candidates, pick the one with lowest score that is safe.
    # If none, move to next tier.
    
    tier_idx = 0
    candidates = []
    
    while len(S) < MAX_SIZE and tier_idx < len(tiers):
        if not candidates:
            candidates = tiers[tier_idx]
        
        # Filter candidates to those not in S
        candidates = [x for x in candidates if x not in S_set]
        
        if not candidates:
            tier_idx += 1
            continue
            
        # Find safe candidates
        safe_candidates = []
        for x in candidates:
            if is_safe(x, S):
                safe_candidates.append(x)
                
        if not safe_candidates:
            # No safe candidate in this tier. Move to next.
            # But wait, maybe a lower number would be safe?
            # Yes, so we move to next tier.
            tier_idx += 1
            candidates = []
            continue
        
        # Pick the one with lowest conflict score
        safe_candidates.sort(key=lambda x: scores[x])
        x = safe_candidates[0]
        
        S.append(x)
        S_set.add(x)
        
        # Remove x from candidates (already filtered, but for next iteration)
        # The loop restarts, so candidates will be re-filtered.
        
    if len(S) > len(best_set):
        best_set = S
        
    # Local Search / Swapping to improve
    # Try to replace elements with others to allow more additions?
    # Or just try to add more if we didn't reach 48.
    
    # If we have less than 48, try adding from any remaining numbers
    remaining = [x for x in range(1, 193) if x not in S_set]
    remaining.sort(key=lambda x: scores[x])
    
    for x in remaining:
        if len(S) >= MAX_SIZE: break
        if is_safe(x, S):
            S.append(x)
            S_set.add(x)
            
    if len(S) > len(best_set):
        best_set = S
        
    return best_set

# To ensure we don't timeout, we add a simple time check in the main loop if it gets slow.
# The is_safe function is O(N^3) ~ 100k ops. 48 steps * 100k = 5M ops. Very fast.
# The bottleneck is the Python loop overhead.
# 48 * 48^3 is fine.

# Let's add a randomized restart if the deterministic one fails to reach 48?
# The prompt says "Greedy removal from densest valid-likely pool ... found 48".
# So the deterministic tiered greedy should work.

# Final check: Ensure the function signature is correct.
def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Precompute scores
    scores = [0] * (N + 1)
    for v in range(1, N + 1):
        scores[v] = 3 * N // v
        
    tiers = [
        list(range(192, 143, -1)),
        list(range(143, 95, -1)),
        list(range(95, 47, -1)),
        list(range(47, 0, -1))
    ]
    
    best_set = []
    
    # Run deterministic greedy
    S = []
    S_set = set()
    tier_idx = 0
    
    def is_safe(x, S_list):
        n = len(S_list)
        if n >= 3:
            for a, b, c in combinations(S_list, 3):
                if (a + b + c) % x == 0:
                    return False
        if n >= 2:
            for a in S_list:
                for b in S_list:
                    if b == a: continue
                    for c in S_list:
                        if c == a or c == b: continue
                        if (x + b + c) % a == 0:
                            return False
        return True

    while len(S) < MAX_SIZE and tier_idx < len(tiers) and time.time() < deadline:
        candidates = [x for x in tiers[tier_idx] if x not in S_set]
        if not candidates:
            tier_idx += 1
            continue
            
        safe_candidates = []
        for x in candidates:
            if is_safe(x, S):
                safe_candidates.append(x)
                
        if not safe_candidates:
            tier_idx += 1
            continue
        
        safe_candidates.sort(key=lambda x: scores[x])
        x = safe_candidates[0]
        S.append(x)
        S_set.add(x)
        
    # Fill remaining if possible
    if len(S) < MAX_SIZE:
        remaining = [x for x in range(1, 193) if x not in S_set]
        remaining.sort(key=lambda x: scores[x])
        for x in remaining:
            if len(S) >= MAX_SIZE: break
            if time.time() > deadline: break
            if is_safe(x, S):
                S.append(x)
                S_set.add(x)
                
    if len(S) > len(best_set):
        best_set = S
        
    # If we didn't reach 48, try a few random restarts with the same logic but shuffled tiers?
    # No, the deterministic one should be strong.
    
    return best_set
