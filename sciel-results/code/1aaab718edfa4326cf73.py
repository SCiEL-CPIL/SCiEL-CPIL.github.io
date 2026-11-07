# EVOLVE-BLOCK-START
from math import isqrt, gcd
import time

def construct():
    N = 256
    deadline = time.time() + 55  # safety margin
    
    # Sieve to get primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # For each prime, the largest prime power <= N
    prime_power = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_power[p] = power
    
    # Primes that MUST be alone: p > N/2 (can't pair with 2), or more generally
    # A prime p must be alone if for every other prime q, p*q > N
    # Since smallest other prime is 2, p must be alone if p*2 > N, i.e., p > 128
    # But also, if p is grouped with q, we need p*q <= N
    # Let's identify which primes can potentially be grouped
    
    # For a group of primes, the minimum product is the product of all primes in the group
    # (each to power 1). If that minimum product > N, the group is infeasible.
    
    # Strategy: 
    # 1. Large primes (p > 128) must be alone
    # 2. For remaining primes, find optimal partition into groups where each group's 
    #    minimum product (product of primes) <= N, and for each group, compute the 
    #    best number (max product with exponents >= 1, total <= N)
    
    # Let's separate primes into "large" (must be alone) and "small" (can potentially group)
    # A prime p can be in a group with another prime q iff p*q <= N (minimum product of group of 2)
    # For a group of k primes, minimum product is product of all k primes
    
    # Primes > N//2 must be alone (can't even pair with 2)
    # Actually p > N/2 means p*2 > N, so p can't be in any group of size >= 2
    large_primes = [p for p in primes if p * 2 > N]
    small_primes = [p for p in primes if p * 2 <= N]
    
    # For small_primes, we need to find optimal grouping
    # This is a partition problem. Let's use a search approach.
    
    # First, let's compute for any subset of small_primes, the best number we can form
    # But the number of subsets is too large. Let's be smarter.
    
    # Key observation: if we group primes {p1, p2, ..., pk}, the best number is:
    # Start with base = p1 * p2 * ... * pk
    # Then we can multiply by additional factors of any pi while staying <= N
    # This is like: maximize base * (p1^a1 * p2^a2 * ... * pk^ak) <= N, ai >= 0
    # Which is: maximize product where we start with each pi to power 1
    
    def best_number_for_group(group_primes):
        """Find the largest number <= N using exactly the primes in group_primes, each to power >= 1."""
        if not group_primes:
            return 1
        base = 1
        for p in group_primes:
            base *= p
        if base > N:
            return 0  # infeasible
        
        # Now try to add more powers
        best = base
        # DFS: try multiplying by each prime in the group
        def dfs(current, idx):
            nonlocal best
            if current > best:
                best = current
            for p in group_primes:
                new_val = current * p
                if new_val <= N:
                    dfs(new_val, idx)
                # Prune: if current * p > N for all p, stop
            # Optimization: if current * min(group_primes) > N, no further improvement possible
            if current * min(group_primes) > N:
                return
        
        dfs(base, 0)
        return best
    
    # Now we need to partition small_primes into groups to maximize total sum
    # This is a set partitioning problem - NP-hard in general, but small_primes is small
    
    # How many small primes are there? Primes <= 128: 
    # 2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127
    # That's 31 primes. Too many for exhaustive partition search.
    
    # But most of these can't actually be grouped! 
    # For two primes p, q to be in the same group, p*q <= 256
    # 127 * 2 = 254 <= 256, so 127 can group with 2
    # 113 * 2 = 226 <= 256, so 113 can group with 2
    # 109 * 2 = 218 <= 256
    # 107 * 2 = 214 <= 256
    # 103 * 2 = 206 <= 256
    # 101 * 2 = 202 <= 256
    # 97 * 2 = 194 <= 256
    # 89 * 2 = 178 <= 256
    # 83 * 2 = 166 <= 256
    # 79 * 2 = 158 <= 256
    # 73 * 2 = 146 <= 256
    # 71 * 2 = 142 <= 256
    # 67 * 2 = 134 <= 256
    # 61 * 2 = 122 <= 256
    # 59 * 2 = 118 <= 256
    # 53 * 2 = 106 <= 256
    # 47 * 2 = 94 <= 256
    # 43 * 2 = 86 <= 256
    # 41 * 2 = 82 <= 256
    # 37 * 2 = 74 <= 256
    # 31 * 2 = 62 <= 256
    # 29 * 2 = 58 <= 256
    # 23 * 2 = 46 <= 256
    # 19 * 2 = 38 <= 256
    # 17 * 2 = 34 <= 256
    # 13 * 2 = 26 <= 256
    # 11 * 2 = 22 <= 256
    # 7 * 2 = 14 <= 256
    # 5 * 2 = 10 <= 256
    # 3 * 2 = 6 <= 256
    
    # But can THREE primes be grouped? p*q*r <= 256
    # 2*3*5 = 30, 2*3*7=42, 2*3*11=66, 2*3*13=78, 2*3*17=102, 2*3*19=114, 2*3*23=138, 2*3*29=174, 2*3*31=186, 2*3*37=222, 2*3*41=246
    # 2*3*43 = 258 > 256
    # 2*5*7 = 70, 2*5*11=110, 2*5*13=130, 2*5*17=170, 2*5*19=190, 2*5*23=230
    # 2*5*29 = 290 > 256
    # 2*7*11 = 154, 2*7*13=182, 2*7*17=238
    # 2*7*19 = 266 > 256
    # 2*11*13 = 286 > 256
    # 3*5*7 = 105, 3*5*11=165, 3*5*13=195, 3*5*17=255
    # 3*5*19 = 285 > 256
    # 3*7*11 = 231
    # 3*7*13 = 273 > 256
    # 5*7*11 = 385 > 256
    
    # So groups of 3 are possible for very small primes.
    # Groups of 4: 2*3*5*7 = 210 <= 256
    # 2*3*5*11 = 330 > 256
    # So only {2,3,5,7} can form a group of 4.
    
    # Groups of 5: 2*3*5*7*11 = 2310 > 256. No groups of 5.
    
    # So the structure is:
    # - Most primes are alone
    # - Small primes can form groups of size 2, 3, or 4
    # - Only {2,3,5,7} can form a group of 4
    
    # Let me think about this differently. The key question is: for each prime, 
    # is it better to use it alone (as p^k) or combine it with other primes?
    
    # For prime p alone: best is p^k where p^k <= 256
    # For prime p in a group: the number is some composite using p and other primes
    
    # Let's just do a smart search. Since the problem is small, let me try:
    # 1. Start with all primes as prime powers (current solution)
    # 2. Try to merge pairs of small primes and see if it improves the sum
    # 3. Try to merge triples
    # 4. Try the quadruple {2,3,5,7}
    
    # Current solution sum:
    current_result = [1] + [prime_power[p] for p in primes]
    current_sum = sum(current_result)
    
    # For a pair (p, q) with p < q, both small:
    # Alone: prime_power[p] + prime_power[q]
    # Together: best_number_for_group([p, q])
    # If together > alone, it's an improvement
    
    # Let's find all beneficial merges
    
    # First, let's compute best_number_for_group for all feasible pairs
    feasible_pairs = []
    for i in range(len(small_primes)):
        for j in range(i + 1, len(small_primes)):
            p, q = small_primes[i], small_primes[j]
            if p * q <= N:
                feasible_pairs.append((p, q))
    
    # For each feasible pair, compute the gain
    pair_gains = {}
    for p, q in feasible_pairs:
        together = best_number_for_group([p, q])
        alone = prime_power[p] + prime_power[q]
        gain = together - alone
        if gain > 0:
            pair_gains[(p, q)] = gain
    
    # Now we need to select a set of non-overlapping pairs (matching) to maximize total gain
    # This is a maximum weight matching problem on the graph of small primes
    
    # Build adjacency for matching
    # Use greedy or DP for maximum weight matching
    
    # Since the graph is small, let's use a simple approach:
    # Try all possible matchings? No, too many.
    # Use a greedy approach: sort pairs by gain, pick non-conflicting ones
    
    # Actually, let's use a proper maximum weight matching via DP on subsets
    # But 31 primes is too many for 2^31 subsets.
    
    # Let's use a greedy approach with local search
    # Or better: since the structure is limited, let's think about it
    
    # Key insight: if we merge p and q into one number, we lose the ability to merge 
    # p with r or q with s. So this is a matching problem.
    
    # Let me try a different approach: use a maximum weight matching algorithm
    # For small graphs, we can use DP or just try all combinations
    
    # Actually, let me think about which pairs actually have positive gain:
    # p alone: p^k <= 256 (largest)
    # q alone: q^k <= 256 (largest)
    # together: max number using p and q, each to power >= 1, <= 256
    
    # For p=2, q=3:
    # 2 alone: 128, 3 alone: 81, sum = 209
    # together: max 2^a * 3^b <= 256, a,b >= 1
    # 2*3=6, 2*9=18, 2*27=54, 2*81=162, 4*3=12, 4*9=36, 4*27=108, 4*81=324>256
    # 8*3=24, 8*9=72, 8*27=216, 16*3=48, 16*9=144, 16*27=432>256
    # 32*3=96, 32*9=288>256, 64*3=192, 64*9=576>256
    # 128*3=384>256
    # Max is 216 (8*27) or 192 (64*3) or 162 (2*81)
    # 216 > 209, so gain = 7
    
    # For p=2, q=5:
    # 2 alone: 128, 5 alone: 125, sum = 253
    # together: max 2^a * 5^b <= 256, a,b >= 1
    # 2*5=10, 2*25=50, 2*125=250, 4*5=20, 4*25=100, 4*125=500>256
    # 8*5=40, 8*25=200, 16*5=80, 16*25=400>256, 32*5=160, 64*5=320>256
    # 128*5=640>256
    # Max is 250 (2*125) or 200 (8*25) or 160 (32*5)
    # 250 < 253, so no gain
    
    # For p=2, q=7:
    # 2 alone: 128, 7 alone: 49, sum = 177
    # together: max 2^a * 7^b <= 256, a,b >= 1
    # 2*7=14, 2*49=98, 2*343>256, 4*7=28, 4*49=196, 8*7=56, 8*49=392>256
    # 16*7=112, 32*7=224, 64*7=448>256, 128*7=896>256
    # Max is 224 (32*7) or 196 (4*49) or 112 (16*7)
    # 224 > 177, gain = 47
    
    # So there are beneficial pairs! Let me compute all of them properly.
    
    # Let me restructure: compute all pair gains, then find maximum weight matching
    
    # For efficiency, let's limit to primes where pairing could help
    # A pair (p,q) helps if best_number([p,q]) > prime_power[p] + prime_power[q]
    
    # Let me compute this for all feasible pairs
    
    # Then find max weight matching. For a graph with ~30 nodes, we can use
    # a simple DP or just enumerate matchings for the relevant subgraph.
    
    # Actually, let me just do a greedy matching: sort by gain descending, 
    # pick pairs that don't conflict.
    
    # But greedy matching isn't optimal. Let me think of a better approach.
    
    # Since the number of "beneficial" pairs is likely small, let me first identify them,
    # then do an exact search on the subgraph of beneficial pairs.
    
    beneficial_pairs = sorted(pair_gains.items(), key=lambda x: -x[1])
    
    # Now find maximum weight matching on the graph formed by beneficial pairs
    # Use a simple backtracking search since the graph is small
    
    used = set()
    best_matching = []
    best_gain = 0
    
    def search_matching(pairs, idx, current_used, current_gain, current_matching):
        nonlocal best_matching, best_gain
        if current_gain > best_gain:
            best_gain = current_gain
            best_matching = current_matching[:]
        
        if idx >= len(pairs):
            return
        
        # Pruning: even if we take all remaining pairs, can we beat best?
        remaining_max = sum(g for _, g in pairs[idx:])
        if current_gain + remaining_max <= best_gain:
            return
        
        # Option 1: skip this pair
        search_matching(pairs, idx + 1, current_used, current_gain, current_matching)
        
        # Option 2: take this pair if both primes are unused
        (p, q), gain = pairs[idx]
        if p not in current_used and q not in current_used:
            new_used = current_used | {p, q}
            search_matching(pairs, idx + 1, new_used, current_gain + gain, current_matching + [(p, q)])
    
    if beneficial_pairs:
        search_matching(beneficial_pairs, 0, set(), 0, [])
    
    # Apply the matching
    result = [1]
    matched_primes = set()
    for p, q in best_matching:
        num = best_number_for_group([p, q])
        result.append(num)
        matched_primes.add(p)
        matched_primes.add(q)
    
    for p in primes:
        if p not in matched_primes:
            result.append(prime_power[p])
    
    # Now try to improve by considering triples
    # A triple (p,q,r) is beneficial if best_number([p,q,r]) > sum of the best for each (considering they might be in pairs)
    # This gets complex. Let me try a local search approach.
    
    # Actually, let me also try the quadruple {2,3,5,7}
    # best_number([2,3,5,7]): max 2^a*3^b*5^c*7^d <= 256, all >= 1
    # 2*3*5*7 = 210
    # 2*3*5*7*2 = 420 > 256
    # So the only option is 210 itself
    # Alone: 128 + 81 + 125 + 49 = 383
    # Together: 210
    # 210 < 383, so the quadruple is NOT beneficial
    
    # What about triples?
    # {2,3,5}: best = max 2^a*3^b*5^c <= 256, all >= 1
    # 2*3*5=30, 2*3*25=150, 2*9*5=90, 2*27*5=270>256, 4*3*5=60, 4*9*5=180, 8*3*5=120, 8*9*5=360>256
    # 16*3*5=240, 32*3*5=480>256, 2*3*125=750>256
    # Max: 240 (16*3*5) or 180 (4*9*5) or 150 (2*3*25)
    # 240
    # Alone: 128+81+125 = 334
    # 240 < 334, not beneficial
    
    # {2,3,7}: max 2^a*3^b*7^c <= 256
    # 2*3*7=42, 2*9*7=126, 2*27*7=378>256, 4*3*7=84, 4*9*7=252, 8*3*7=168, 16*3*7=336>256
    # 2*3*49=294>256, 4*3*49=588>256
    # Max: 252 (4*9*7) or 168 (8*3*7) or 126 (2*9*7)
    # 252
    # Alone: 128+81+49 = 258
    # 252 < 258, not beneficial
    
    # {2,5,7}: max 2^a*5^b*7^c <= 256
    # 2*5*7=70, 2*25*7=350>256, 4*5*7=140, 8*5*7=280>256, 2*5*49=490>256
    # Max: 140 (4*5*7) or 70
    # 140
    # Alone: 128+125+49 = 302
    # 140 < 302, not beneficial
    
    # So triples don't seem beneficial. Let me verify with code.
    
    # Let me also try: what if we don't use a prime at all? 
    # No, using a prime always adds at least p to the sum, so we always want to use it.
    
    # Let me verify the result
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                # Fallback to original
                return [int(x) for x in [1] + [prime_power[p] for p in primes]]
    
    # Check all values are in range and distinct
    if len(set(result)) != len(result):
        return [int(x) for x in [1] + [prime_power[p] for p in primes]]
    if any(x < 1 or x > N for x in result):
        return [int(x) for x in [1] + [prime_power[p] for p in primes]]
    
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
