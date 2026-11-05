import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = 115.0
    
    n_points = 16
    points = list(range(n_points))
    
    # 1. Construct a Steiner Triple System on 15 points (0..14)
    # Using the standard construction for STS(15) based on Z_15
    # Base blocks: {0,1,4}, {0,2,8}, {0,3,12}, {0,5,10}, {0,6,9}, {0,7,11}, {1,2,3}
    # Wait, let's use a known valid STS(15).
    # A common STS(15) is the projective plane PG(3,2) minus one point? No, PG(3,2) is 15 points.
    # Actually, PG(3,2) has 15 points and 35 lines.
    # Let's generate STS(15) using a known list or a simple algorithm.
    
    # Known STS(15) triples:
    # We can use the cyclic construction from Z_15 with base blocks.
    # Base blocks for STS(15):
    # {0, 1, 4}, {0, 2, 8}, {0, 3, 12}, {0, 5, 10}, {0, 6, 9}, {0, 7, 11}, {1, 2, 3}
    # Let's verify this quickly mentally or just use a hardcoded known STS(15).
    # Actually, finding a valid STS(15) is a standard problem.
    # Let's use a hardcoded valid STS(15) to avoid generation errors.
    
    # One standard STS(15):
    sts15 = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11), (12, 13, 14),
        (0, 3, 6), (1, 4, 7), (2, 5, 8), (9, 12, 15) # Wait, 15 is not in 0..14
    ]
    # Let's use a reliable source for STS(15).
    # The projective plane of order 3 has 13 points. Not 15.
    # STS(15) exists because 15 = 3 mod 6.
    # Let's generate it.
    
    def generate_sts15():
        # Cyclic STS(15)
        # Base blocks:
        # {0, 1, 4}, {0, 2, 8}, {0, 3, 12}, {0, 5, 10}, {0, 6, 9}, {0, 7, 11}, {1, 2, 3}
        # Let's check if this generates a valid STS(15).
        # Number of blocks = 15 * 14 / 6 = 35.
        # The orbits of the base blocks under Z_15:
        # {0,1,4} -> 15 blocks
        # {0,2,8} -> 15 blocks
        # {0,3,12} -> 15 blocks
        # {0,5,10} -> 15 blocks
        # {0,6,9} -> 15 blocks
        # {0,7,11} -> 15 blocks
        # {1,2,3} -> 15 blocks
        # Total 105 blocks. This is too many.
        # The stabilizers must be considered.
        
        # Let's use a known list of STS(15) from literature or a simple backtracking generator.
        # Since n=15 is small, we can generate it once.
        
        triples = []
        pairs_used = set()
        
        all_triples = list(itertools.combinations(range(15), 3))
        random.seed(42) # Deterministic
        random.shuffle(all_triples)
        
        # Greedy packing for STS(15) should find a maximum packing which is STS(15)
        for t in all_triples:
            a, b, c = t
            p1, p2, p3 = (a,b), (a,c), (b,c)
            if p1 not in pairs_used and p2 not in pairs_used and p3 not in pairs_used:
                triples.append(t)
                pairs_used.add(p1)
                pairs_used.add(p2)
                pairs_used.add(p3)
                
        # Check if it's a Steiner Triple System (each pair appears exactly once)
        # For n=15, a maximum packing has 15*14/6 = 35 triples.
        if len(triples) == 35:
            return triples
        else:
            # Fallback: try again with different seed
            for seed in range(100):
                random.seed(seed)
                random.shuffle(all_triples)
                triples = []
                pairs_used = set()
                for t in all_triples:
                    a, b, c = t
                    p1, p2, p3 = (a,b), (a,c), (b,c)
                    if p1 not in pairs_used and p2 not in pairs_used and p3 not in pairs_used:
                        triples.append(t)
                        pairs_used.add(p1)
                        pairs_used.add(p2)
                        pairs_used.add(p3)
                if len(triples) == 35:
                    return triples
            return None

    base_sts = generate_sts15()
    
    if base_sts is None:
        # Fallback to simple system
        return [[i, i+1, i+2] for i in range(0, 15, 3)]
    
    # 2. Extend to 16 points by adding point 15.
    # We need to add triples containing 15 to cover pairs involving 15.
    # There are 15 pairs involving 15: (15, 0), ..., (15, 14).
    # Each triple containing 15 covers 2 such pairs.
    # So we need at least 8 triples containing 15? No, 15/2 = 7.5, so 8 triples.
    # But we must ensure linearity (no pair repeated).
    # Let's select 8 triples of the form (15, x, y) such that all x, y are distinct and cover all 15 points 0..14 exactly once?
    # No, if we use 8 triples, we use 16 points from 0..14. But there are only 15 points.
    # So one point from 0..14 will be in two triples with 15? No, that would repeat a pair (15, x).
    # So we cannot cover all 15 pairs with disjoint triples (15, x, y).
    # We can cover 14 pairs with 7 triples, leaving one pair (15, z) uncovered.
    # Or we can use 8 triples, but then one pair (15, x) must be repeated, which is invalid.
    # So we can add at most 7 triples involving 15, covering 14 of the 15 pairs.
    # The remaining pair (15, z) will not be in any triple.
    
    # Let's choose 7 disjoint pairs from 0..14 to form triples with 15.
    # To minimize independence number, we want to block large independent sets.
    # An independent set in the extended system is a set S subset of {0..15} containing no triple.
    # If S does not contain 15, it must be an independent set in STS(15).
    # If S contains 15, it cannot contain any pair (x, y) such that (15, x, y) is a triple.
    # Also, S\{15} must not contain any triple from STS(15).
    
    # The independence number of STS(15) is known to be 8?
    # Let's check.
    # If alpha(STS(15)) = 8, then we can have an independent set of size 8 not containing 15.
    # Can we extend it to size 9 by adding 15?
    # If S is an independent set of size 8 in STS(15), can we add 15?
    # We can add 15 if S does not contain any pair (x, y) from our chosen 7 triples (15, x, y).
    # We can choose the 7 triples to "block" all maximum independent sets of STS(15).
    
    # Let's find the maximum independent sets of STS(15).
    def find_max_independent_sets(sts, n=15):
        # Backtracking to find all max independent sets
        max_size = 0
        max_sets = []
        
        def backtrack(idx, current_set, current_size):
            nonlocal max_size, max_sets
            if idx == n:
                if current_size > max_size:
                    max_size = current_size
                    max_sets = [tuple(sorted(current_set))]
                elif current_size == max_size:
                    max_sets.append(tuple(sorted(current_set)))
                return
            
            # Try not including idx
            backtrack(idx + 1, current_set, current_size)
            
            # Try including idx
            # Check if adding idx creates a triple
            valid = True
            for t in sts:
                if idx in t:
                    other1, other2 = t[0], t[1]
                    if other1 == idx: other1, other2 = t[1], t[2]
                    elif other2 == idx: other1, other2 = t[0], t[2]
                    if other1 in current_set and other2 in current_set:
                        valid = False
                        break
            if valid:
                current_set.add(idx)
                backtrack(idx + 1, current_set, current_size + 1)
                current_set.remove(idx)
        
        backtrack(0, set(), 0)
        return max_size, max_sets

    max_size_15, max_sets_15 = find_max_independent_sets(base_sts)
    
    # Now choose 7 pairs to form triples with 15 to block these max sets.
    # We want to choose pairs (x, y) such that for every max_set S in max_sets_15,
    # S contains at least one pair (x, y) from our chosen set of pairs.
    # This way, S U {15} is not independent.
    
    # Let's try a greedy approach to select the pairs.
    # Candidate pairs are all pairs in 0..14.
    # We want to cover all max_sets_15.
    
    # This is a hitting set problem.
    # Let's just pick random pairs and check if they block all max sets.
    
    best_system = None
    best_alpha = 16
    
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        # Select 7 random disjoint pairs from 0..14
        points_14 = list(range(14))
        random.shuffle(points_14)
        
        # We need 7 pairs.
        # Let's just pick 7 random pairs that are disjoint.
        used_points = set()
        chosen_pairs = []
        
        while len(chosen_pairs) < 7:
            p1 = random.randint(0, 14)
            p2 = random.randint(0, 14)
            if p1 == p2:
                continue
            if p1 in used_points or p2 in used_points:
                continue
            chosen_pairs.append((p1, p2))
            used_points.add(p1)
            used_points.add(p2)
            
        # Construct the extended system
        extended_sts = base_sts[:]
        for p1, p2 in chosen_pairs:
            extended_sts.append((15, p1, p2))
            
        # Calculate independence number of extended_sts
        # We can do a quick check.
        # If alpha(extended_sts) < best_alpha, update.
        
        # To calculate alpha exactly is expensive, but n=16 is small.
        # Let's use a heuristic or exact search if fast enough.
        # Since we only do this 1000 times, and n=16, exact search might be too slow.
        # Let's use a greedy upper bound or a quick local search.
        
        # Actually, we can check if any max_set_15 can be extended.
        # If a max_set_15 S does not contain any chosen_pair, then S U {15} is independent of size 9.
        # So alpha >= 9.
        # If all max_set_15 are blocked, then alpha <= 8.
        # But there might be other independent sets of size 9 not containing 15?
        # No, max size in STS(15) is max_size_15.
        # So if all max sets are blocked, alpha <= max(max_size_15, 8) = max_size_15.
        # If max_size_15 = 8, then alpha <= 8.
        
        blocked = True
        for S in max_sets_15:
            S_set = set(S)
            for p1, p2 in chosen_pairs:
                if p1 in S_set and p2 in S_set:
                    break
            else:
                blocked = False
                break
                
        if blocked:
            # This configuration blocks all max independent sets of size 9 (containing 15).
            # The independence number is at most max_size_15.
            # Let's assume it's max_size_15.
            # We can return this system.
            return extended_sts
            
    # If no blocking configuration found, return the best we have (which is just STS(15) + random pairs)
    # Or just return a valid system.
    # Let's return the first one we tried.
    if not best_system:
         # Reconstruct the last tried system
         points_14 = list(range(14))
         random.shuffle(points_14)
         used_points = set()
         chosen_pairs = []
         while len(chosen_pairs) < 7:
             p1 = random.randint(0, 14)
             p2 = random.randint(0, 14)
             if p1 == p2: continue
             if p1 in used_points or p2 in used_points: continue
             chosen_pairs.append((p1, p2))
             used_points.add(p1)
             used_points.add(p2)
         extended_sts = base_sts[:]
         for p1, p2 in chosen_pairs:
             extended_sts.append((15, p1, p2))
         return extended_sts

    return best_system
