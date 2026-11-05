import time
import random
from collections import defaultdict

def construct():
    """
    Constructs a linear triple system on 16 points with a small independence number.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    n = 16
    points = list(range(n))
    
    # Helper: Compute independence number using bitmasks
    # We want the largest S such that no triple is contained in S.
    # This is equivalent to finding the maximum size of a set S where for every triple t, |S ∩ t| < 3.
    # Since n=16, 2^16 = 65536 masks.
    
    def calc_independence(triples):
        if not triples:
            return n
        
        # Convert triples to bitmasks
        triple_masks = []
        for t in triples:
            mask = 0
            for p in t:
                mask |= (1 << p)
            triple_masks.append(mask)
        
        # Optimize: Check masks in decreasing order of popcount?
        # Or just iterate all?
        # 65536 iterations * num_triples checks.
        # To speed up, we can pre-check: if a mask is not independent, its supersets are not.
        # But simple iteration is likely fast enough for a few hundred calls.
        
        # Let's use a trick: 
        # S is independent iff it does NOT contain any triple.
        # Let's iterate mask from 0 to 2^16-1.
        # But we want max popcount.
        
        # Faster approach:
        # For each point subset, it's independent if it's not a superset of any triple.
        
        max_ind = 0
        # Precompute popcount for all masks?
        # popcount = [0] * (1 << n)
        # for i in range(1, 1 << n): popcount[i] = popcount[i >> 1] + (i & 1)
        
        # Let's just use bin(mask).count('1') as it's fast in C.
        
        for mask in range(1 << n):
            # Optimization: If bin(mask).count('1') <= max_ind, skip? 
            # No, because we might find a larger one later? 
            # Actually, if we iterate in a way that larger masks come first, we can break.
            # But generating masks by popcount is complex.
            
            # Simple check:
            if bin(mask).count('1') <= max_ind:
                continue
                
            is_ind = True
            for tm in triple_masks:
                if (mask & tm) == tm:
                    is_ind = False
                    break
            
            if is_ind:
                max_ind = bin(mask).count('1')
                
        return max_ind

    # Helper: Check linearity
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

    # Strategy:
    # 1. Generate a STS(15) on points 0..14.
    # 2. Add point 15.
    # 3. Add triples involving 15 to minimize independence number.
    
    def generate_sts15():
        """
        Greedy random construction of STS(15) on points 0..14.
        """
        triples = []
        pairs = set()
        pts = list(range(15))
        all_pairs = []
        for i in range(15):
            for j in range(i+1, 15):
                all_pairs.append((i, j))
        
        random.shuffle(all_pairs)
        
        for p1, p2 in all_pairs:
            # Find p3
            candidates = [p for p in pts if p != p1 and p != p2]
            random.shuffle(candidates)
            for p3 in candidates:
                pair_13 = (min(p1, p3), max(p1, p3))
                pair_23 = (min(p2, p3), max(p2, p3))
                if pair_13 not in pairs and pair_23 not in pairs:
                    triples.append([p1, p2, p3])
                    pairs.add((p1, p2))
                    pairs.add(pair_13)
                    pairs.add(pair_23)
                    break
        
        return triples

    best_system = None
    best_score = 17 # Max is 16, so 17 is safe initial
    
    # Pre-compute all candidate triples involving 15
    candidates_15 = []
    for a in range(15):
        for b in range(a+1, 15):
            candidates_15.append([15, a, b])
    
    # Main search loop
    num_attempts = 0
    while time.time() - start_time < deadline:
        num_attempts += 1
        
        # Generate a random STS(15)
        base_triples = generate_sts15()
        
        # We want to add triples from candidates_15.
        # We can't add all, due to linearity.
        # A triple [15, a, b] shares point 15 with all other [15, x, y].
        # It shares a with [15, a, y] and b with [15, x, b].
        # So we are picking a matching in the graph of points 0..14?
        # No, [15, a, b] uses points a and b.
        # Two triples [15, a, b] and [15, c, d] are linear if {a,b} ∩ {c,d} is empty.
        # So we are looking for a matching in the complete graph K_15?
        # Actually, we can pick a set of pairs {a,b} such that no point appears in more than one pair.
        # Because if point 'a' is in [15, a, b] and [15, a, c], they share 15 and a -> not linear.
        # So we must pick a matching (set of disjoint edges) on points 0..14.
        
        # Max size of matching in K_15 is 7 (14 points used).
        # So we can add at most 7 triples involving 15.
        
        # Let's generate random matchings.
        available = list(range(15))
        random.shuffle(available)
        
        current_triples = list(base_triples)
        
        # Greedily add disjoint pairs
        i = 0
        while i < len(available) - 1:
            a = available[i]
            b = available[i+1]
            current_triples.append([15, a, b])
            i += 2
        
        # Evaluate
        score = calc_independence(current_triples)
        
        if score < best_score:
            best_score = score
            best_system = list(current_triples)
            
            # If we find a very good score, we might want to try local improvements?
            # But for now, let's just keep sampling.
            # We can also try to remove some triples? No, removing increases independence number.
            # We want to ADD triples.
            
            # Try to find a better matching?
            # The structure of the matching matters.
            # Random matchings might not be optimal.
            
    if best_system is None:
        # Fallback: Just STS(15) + one triple [15, 0, 1]
        return generate_sts15() + [[15, 0, 1]]
        
    return best_system

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
