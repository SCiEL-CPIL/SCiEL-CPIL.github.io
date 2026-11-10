"""
Bipartite graph approach: find a 3x3 biclique where all 9 sums are prime.
"""
import time
from math import gcd
from itertools import combinations, permutations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # Safety margin

    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def distinct_sums(A, B):
        return {a + b for a in A for b in B}

    def is_pairwise_coprime(vals):
        lst = list(vals)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if gcd(lst[i], lst[j]) != 1:
                    return False
        return True

    def score_of(A, B):
        s = distinct_sums(A, B)
        if len(s) < 2:
            return -1
        if not is_pairwise_coprime(s):
            return -1
        return len(s)

    # Fallback initial best
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score_of(best_A, best_B)
    if best_score < 0:
        best_score = 0

    # Precompute primes up to 256
    primes = [p for p in range(2, 257) if is_prime(p)]
    prime_set = set(primes)

    # Build bipartite graph: left nodes = A candidates [1,128], right nodes = B candidates [1,128]
    # Edge (a, b) exists if a+b is prime
    # We want to find a 3x3 biclique: 3 A-nodes and 3 B-nodes where all 9 edges exist
    # and all 9 sums are distinct primes

    # For each A candidate, compute which B candidates connect to it
    A_range = list(range(1, 129))
    B_range = list(range(1, 129))

    # adj_a[a] = set of b such that a+b is prime
    adj_a = {}
    for a in A_range:
        adj_a[a] = set()
        for b in B_range:
            if (a + b) in prime_set:
                adj_a[a].add(b)

    # Compute degree for each A node
    degree_a = {a: len(adj_a[a]) for a in A_range}

    # Sort A candidates by degree (highest first) for greedy heuristic
    A_sorted = sorted(A_range, key=lambda a: -degree_a[a])

    # Strategy: 
    # 1. Pick first A node (highest degree)
    # 2. From its neighbors, pick 2nd A node that maximizes common neighbors with 1st
    # 3. From common neighbors of 1st and 2nd, pick 3rd A node that maximizes common neighbors
    # 4. The common neighbors of all 3 A nodes are candidate B nodes
    # 5. Pick 3 B nodes from common neighbors such that all 9 sums are distinct

    # Also try the dual: pick B nodes first

    def try_find_biclique(A_order, B_order, adj_from_A_to_B):
        """Try to find a 3x3 biclique using greedy heuristic."""
        if len(A_order) < 3 or len(B_order) < 3:
            return None
        
        # Pick first A node
        a1 = A_order[0]
        common = adj_from_A_to_B[a1]
        
        if len(common) < 3:
            return None
        
        # Pick second A node: maximize |common ∩ adj[a2]|
        best_a2 = None
        best_common2 = None
        best_count = 0
        for a2 in A_order[1:]:
            if a2 == a1:
                continue
            c2 = common & adj_from_A_to_B[a2]
            cnt = len(c2)
            if cnt > best_count:
                best_count = cnt
                best_a2 = a2
                best_common2 = c2
            if best_count >= 3:
                # Early exit if we already have enough
                break
        
        if best_a2 is None or best_count < 3:
            return None
        
        # Pick third A node: maximize |common2 ∩ adj[a3]|
        best_a3 = None
        best_common3 = None
        best_count3 = 0
        for a3 in A_order[2:]:
            if a3 in (a1, best_a2):
                continue
            c3 = best_common2 & adj_from_A_to_B[a3]
            cnt = len(c3)
            if cnt > best_count3:
                best_count3 = cnt
                best_a3 = a3
                best_common3 = c3
            if best_count3 >= 3:
                break
        
        if best_a3 is None or best_count3 < 3:
            return None
        
        # Now best_common3 has at least 3 B candidates
        # We need to pick 3 B nodes such that all 9 sums are distinct
        B_candidates = list(best_common3)
        
        if len(B_candidates) < 3:
            return None
        
        # Try all combinations of 3 B nodes
        for b1, b2, b3 in combinations(B_candidates, 3):
            A_sel = [a1, best_a2, best_a3]
            B_sel = [b1, b2, b3]
            
            # Check all 9 sums are distinct
            sums = set()
            valid = True
            for a in A_sel:
                for b in B_sel:
                    s = a + b
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            
            if valid:
                # Verify pairwise coprime (should be true since all are prime and distinct)
                if is_pairwise_coprime(sums):
                    return [A_sel, B_sel]
        
        return None

    # Try with A-sorted order
    result = try_find_biclique(A_sorted, B_range, adj_a)
    if result:
        A, B = result
        s = score_of(A, B)
        if s >= 9:
            return [A, B]
        elif s > best_score:
            best_score = s
            best_A, best_B = A[:], B[:]

    # If greedy didn't work, try a more systematic approach
    # Iterate over pairs of A nodes, compute common B neighbors, then find third A node
    
    start_time = time.time()
    
    # Precompute for each pair of A nodes, the set of common B neighbors
    # This is expensive, so we'll do it on the fly with pruning
    
    # Try all pairs of A nodes from the top-degree list
    A_top = A_sorted[:50]  # Limit to top 50 for speed
    
    for i in range(len(A_top)):
        if time.time() > deadline - 10:
            break
        a1 = A_top[i]
        for j in range(i + 1, len(A_top)):
            a2 = A_top[j]
            common_b = adj_a[a1] & adj_a[a2]
            if len(common_b) < 3:
                continue
            
            # Try to find a3 such that adj_a[a3] & common_b has at least 3 elements
            for k in range(len(A_top)):
                if time.time() > deadline - 5:
                    break
                a3 = A_top[k]
                if a3 in (a1, a2):
                    continue
                common_b3 = common_b & adj_a[a3]
                if len(common_b3) < 3:
                    continue
                
                # Try to pick 3 B nodes from common_b3
                B_cands = list(common_b3)
                if len(B_cands) < 3:
                    continue
                
                # Limit the number of combinations to try
                max_combos = min(len(B_cands), 20)
                B_cands = B_cands[:max_combos]
                
                for b1, b2, b3 in combinations(B_cands, 3):
                    A_sel = [a1, a2, a3]
                    B_sel = [b1, b2, b3]
                    
                    sums = set()
                    valid = True
                    for a in A_sel:
                        for b in B_sel:
                            s = a + b
                            if s in sums:
                                valid = False
                                break
                            sums.add(s)
                        if not valid:
                            break
                    
                    if valid and is_pairwise_coprime(sums):
                        s = score_of(A_sel, B_sel)
                        if s >= 9:
                            return [A_sel, B_sel]
                        elif s > best_score:
                            best_score = s
                            best_A, best_B = A_sel[:], B_sel[:]
    
    # If still no 9, try the structured prime grid approach as fallback
    # (same as before but with time guard)
    prime_list = primes
    offset_sets = {p: set() for p in primes}
    for p in primes:
        for q in primes:
            if q < p:
                offset_sets[p].add(p - q)
    
    for i1 in range(len(prime_list)):
        if time.time() > deadline - 3:
            break
        p11 = prime_list[i1]
        for i2 in range(i1 + 1, len(prime_list)):
            p21 = prime_list[i2]
            for i3 in range(i2 + 1, len(prime_list)):
                p31 = prime_list[i3]
                
                sets = [offset_sets[p11], offset_sets[p21], offset_sets[p31]]
                sets.sort(key=len)
                
                common_offsets = sets[0] & sets[1]
                if not common_offsets:
                    continue
                common_offsets = common_offsets & sets[2]
                if not common_offsets:
                    continue
                
                offsets_list = list(common_offsets)
                
                for d12 in offsets_list:
                    if time.time() > deadline - 2:
                        break
                    for d13 in offsets_list:
                        if d12 == d13:
                            continue
                        
                        p12 = p11 - d12
                        p22 = p21 - d12
                        p32 = p31 - d12
                        p13 = p11 - d13
                        p23 = p21 - d13
                        p33 = p31 - d13
                        
                        if not all(p in prime_set for p in [p12, p22, p32, p13, p23, p33]):
                            continue
                        
                        all_primes = [p11, p21, p31, p12, p22, p32, p13, p23, p33]
                        if len(set(all_primes)) != 9:
                            continue
                        
                        lower = max(1, p11 - 128, p21 - 128, p31 - 128, 1 + d12, 1 + d13)
                        upper = min(128, p11 - 1, p21 - 1, p31 - 1, 128 + d12, 128 + d13)
                        
                        if lower <= upper:
                            b1 = lower
                            A = [p11 - b1, p21 - b1, p31 - b1]
                            B = [b1, b1 - d12, b1 - d13]
                            
                            if len(set(A)) == 3 and len(set(B)) == 3:
                                if all(1 <= x <= 128 for x in A) and all(1 <= x <= 128 for x in B):
                                    s = score_of(A, B)
                                    if s == 9:
                                        return [A, B]
                                    elif s > best_score:
                                        best_score = s
                                        best_A, best_B = A[:], B[:]
    
    return [best_A, best_B]

# EVOLVE-BLOCK-END
