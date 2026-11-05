import time
import random
import itertools
from collections import defaultdict

def construct():
    """
    Constructs a linear triple system on 16 points with a small independence number.
    Strategy: 
    1. Start with a known STS(15) on points 0..14.
    2. Add point 15.
    3. Use a randomized search to add triples involving point 15 (and potentially others)
       to minimize the independence number.
    4. The goal is to break all large independent sets.
    
    We use a greedy approach with backtracking or local search to find a set of triples
    that minimizes the maximum independent set size.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    n = 16
    points = list(range(n))
    
    # Helper: Check if a set of triples is valid (linear)
    def is_linear(triples):
        pair_set = set()
        for t in triples:
            for i in range(3):
                for j in range(i+1, 3):
                    pair = (min(t[i], t[j]), max(t[i], t[j]))
                    if pair in pair_set:
                        return False
                    pair_set.add(pair)
        return True
    
    # Helper: Compute independence number (max set with no triple)
    def independence_number(triples):
        # Use a simple heuristic or exact for small n?
        # n=16 is small enough for some exact methods, but brute force 2^16 is 65536.
        # Let's just check all subsets? No, that's slow if called many times.
        # Better: Use a greedy upper bound or a quick SAT-like check?
        # For now, let's use a simple recursive branch and bound or just a heuristic.
        # Actually, let's implement a quick check for independence number <= k.
        # Or just return the size of the largest independent set found by a greedy heuristic?
        # The task says "score is the largest point set containing no triple".
        # We need the exact value to compare.
        
        # Since n=16, we can represent subsets as bitmasks.
        # Precompute for each triple the mask of points it contains.
        # A subset S is independent if for all triples T, (S & T) != T (i.e., T is not a subset of S).
        # Equivalently, S is independent if it does not contain any triple.
        
        # To find the max independent set, we can iterate over all masks? 2^16 = 65536.
        # That is fast enough.
        
        triple_masks = []
        for t in triples:
            mask = 0
            for p in t:
                mask |= (1 << p)
            triple_masks.append(mask)
        
        max_size = 0
        # Iterate over all subsets? 
        # Actually, we want the largest S such that no triple is contained in S.
        # This is the complement of the covering problem.
        
        # Let's just iterate all masks from 0 to 2^16 - 1.
        # If a mask contains a triple, it's not independent.
        # We want the max popcount of a mask that does NOT contain any triple.
        
        # Optimization: If a mask is not independent, any superset is also not independent.
        # So we can use a boolean array `independent` and propagate?
        # Or simply check each mask. 65536 * (num_triples) operations.
        # Num triples is at most C(16,3)/... roughly 56-80.
        # 65536 * 80 = 5 million ops. Very fast in Python if optimized? Maybe a bit slow.
        
        # Let's try a different approach: 
        # Start with all points. Remove points to break triples?
        # Or, simply, since we are searching for a *good* system, we can evaluate
        # the independence number by finding the max clique in the complement graph?
        # No, independence number in hypergraph.
        
        # Let's stick to the bitmask check but optimize it.
        # Precompute a set of triple masks.
        # For a candidate set S (mask), check if any triple mask is a subset of S.
        
        # Actually, a faster way: 
        # Let F be the family of triples.
        # S is independent iff S contains no element of F.
        
        # Let's just do the 65536 loop. It's the safest bet for correctness.
        
        # To speed it up, we can precompute the list of triple masks.
        # And for each mask, check if it contains any triple mask.
        
        # Even better: 
        # We can iterate masks in decreasing order of popcount. 
        # The first one that is independent is the answer.
        
        # Generate masks in decreasing order of popcount?
        # Or just iterate all and keep track of max.
        
        # Let's try:
        # independent = [True] * (1 << n)
        # independent[0] = True
        # For each triple t, mark all supersets of t as False?
        # That's expensive.
        
        # Let's just do the direct check.
        # But wait, we only call this a few times during search.
        
        max_ind = 0
        for mask in range(1 << n):
            # Quick check: does this mask contain any triple?
            is_ind = True
            for tm in triple_masks:
                if (mask & tm) == tm:
                    is_ind = False
                    break
            if is_ind:
                sz = bin(mask).count('1')
                if sz > max_ind:
                    max_ind = sz
        
        return max_ind
    
    # Helper: Try to add a triple to the system and return new system if valid
    def try_add(triples, new_triple):
        # Check linearity
        for t in triples:
            inter = set(t) & set(new_triple)
            if len(inter) > 1:
                return None
        return triples + [new_triple]
    
    # Initial system: STS(15) on points 0..14
    # Standard STS(15) construction (e.g., from projective plane PG(2,2) extended? No, PG(2,2) is 7 points.
    # STS(15) exists. Let's use a known one or generate one?
    # Let's use a simple greedy construction for STS(15) first?
    # Or hardcode a known STS(15).
    
    # A known STS(15) (Steiner Triple System of order 15):
    # Points 0..14
    sts15 = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 9, 12], [1, 10, 13], [2, 11, 14],
        [3, 9, 13], [4, 10, 14], [5, 11, 12],
        [6, 9, 14], [7, 10, 12], [8, 11, 13],
        [3, 7, 11], [4, 8, 12], [5, 6, 10],
        [6, 7, 13], [8, 9, 10], [0, 4, 8],
        [1, 5, 9], [2, 6, 10], [0, 5, 7],
        [1, 6, 8], [2, 7, 9], [3, 8, 10],
        [4, 7, 11], [5, 8, 14], [0, 11, 14],
        [1, 12, 14], [2, 13, 11], [3, 12, 10],
        [4, 13, 9], [5, 12, 13], [6, 13, 8],
        [7, 14, 12], [0, 10, 13], [1, 9, 11],
        [2, 8, 12], [3, 11, 14], [4, 6, 15] # Wait, 15 is not in 0..14.
    ]
    # I need a valid STS(15). Let me generate one or find a correct list.
    # Actually, let's just use a random greedy construction for the base system to avoid hardcoding errors.
    
    def generate_sts(n):
        # Greedy random construction
        triples = []
        pairs = set()
        points = list(range(n))
        all_pairs = list(itertools.combinations(points, 2))
        random.shuffle(all_pairs)
        
        for p1, p2 in all_pairs:
            # Find a point p3 such that (p1, p3) and (p2, p3) are not in pairs
            # and (p1, p2) is not in pairs (it is, since we're processing it)
            candidates = [p3 for p3 in points if p3 != p1 and p3 != p2]
            random.shuffle(candidates)
            for p3 in candidates:
                if (min(p1, p3), max(p1, p3)) not in pairs and (min(p2, p3), max(p2, p3)) not in pairs:
                    triples.append([p1, p2, p3])
                    pairs.add((min(p1, p2), max(p1, p2)))
                    pairs.add((min(p1, p3), max(p1, p3)))
                    pairs.add((min(p2, p3), max(p2, p3)))
                    break
        
        return triples
    
    # Generate a base system on 0..14
    base_triples = generate_sts(15)
    
    # Now we have a linear system on 0..14.
    # We want to add triples involving 15 to reduce the independence number.
    # Initially, the independence number might be high.
    # Let's calculate the current independence number.
    # But we can just run a search to add triples.
    
    best_system = None
    best_score = 100  # Max possible is 16
    
    # We will perform a random search.
    # We start with base_triples.
    # We try to add triples containing 15.
    # Also, we might try to swap triples? No, let's keep it simple.
    
    # Let's define a search that tries to add k triples involving 15.
    # The number of triples we can add is limited by linearity.
    
    for attempt in range(50):
        if time.time() - start_time > deadline:
            break
            
        # Start with base STS(15)
        current_triples = list(base_triples)
        
        # Try to add triples involving 15
        # Candidate triples: [15, a, b] where a, b in 0..14
        # We need to ensure linearity.
        
        # Greedy addition
        candidates = []
        for a in range(15):
            for b in range(a+1, 15):
                candidates.append([15, a, b])
        
        random.shuffle(candidates)
        
        for t in candidates:
            # Check if we can add t
            can_add = True
            for existing in current_triples:
                inter = set(existing) & set(t)
                if len(inter) > 1:
                    can_add = False
                    break
            if can_add:
                current_triples.append(t)
        
        # Evaluate this system
        score = independence_number(current_triples)
        
        if score < best_score:
            best_score = score
            best_system = current_triples
            # If we find a very good score, we might want to continue searching for even better?
            # But the task is to minimize.
            
        # Try a different base STS(15) or different random order?
        # Let's regenerate the base STS(15) for the next attempt.
        base_triples = generate_sts(15)
        
        # Also, try adding triples not involving 15? 
        # The base STS(15) already covers all pairs in 0..14.
        # So any new triple must involve 15.
        # Because if a triple [a,b,c] with a,b,c in 0..14 is added, 
        # it shares pairs with existing triples?
        # In a STS(15), every pair is in exactly one triple.
        # So [a,b,c] shares pair (a,b) with some existing triple.
        # So we cannot add any triple entirely within 0..14.
        # Thus, all added triples must involve 15.
        
        # So the search is purely about which triples [15, a, b] to add.
        
    if best_system is None:
        # Fallback
        return base_triples
        
    return best_system

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
