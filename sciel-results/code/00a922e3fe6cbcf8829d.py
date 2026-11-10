import random
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a linear triple system on 16 points minimizing the independence number.
    Uses Simulated Annealing to search for a dense linear hypergraph.
    """
    start_time = time.time()
    deadline = 120.0  # Safety margin for 150s limit
    points = 16
    
    # Helper: Check if a set of points is independent (contains no triple)
    def is_independent(subset, triples):
        # Fast check: if any triple is subset of 'subset'
        # Since triples are small, we can iterate
        # Optimization: Pre-compute triples as sets for O(1) lookup? 
        # No, we need to check if ANY triple is inside subset.
        # Better: Iterate triples and check if all 3 points are in subset.
        subset_set = set(subset)
        for t in triples:
            if t[0] in subset_set and t[1] in subset_set and t[2] in subset_set:
                return False
        return True

    # Helper: Calculate a lower bound on independence number using Greedy MIS
    # This is fast. If greedy finds size k, alpha >= k.
    # We want to MINIMIZE alpha, so we want to MINIMIZE the greedy result?
    # Actually, we want the TRUE alpha to be small.
    # If greedy finds a large independent set, alpha is large.
    # So we want a configuration where even the greedy algorithm struggles to find large independent sets?
    # No, greedy is a lower bound. 
    # Strategy: Minimize the size of the Maximal Independent Set found by Greedy.
    # If we can make the greedy MIS small, the true alpha is likely small.
    def greedy_mis_size(triples):
        # Randomly order points? Or fixed order?
        # Fixed order is deterministic and fast.
        # Better: Try a few random orderings to get a tighter lower bound?
        # For speed, let's use a fixed order 0..15 for the heuristic score.
        # Actually, a "bad" greedy order might miss large independent sets.
        # We want the heuristic to be as close to true alpha as possible from below.
        # Let's use a randomized greedy with a few iterations to estimate alpha.
        # But that's expensive. 
        # Let's stick to a single pass with random shuffling for diversity in evaluation.
        
        # To be safe and fast, let's just check if there exists an independent set of size 6 or 7?
        # That's a search problem itself.
        
        # Let's use a simple heuristic:
        # Score = - (size of largest independent set found by greedy random restarts)
        
        best_mis = 0
        # 5 random restarts for MIS estimation
        for _ in range(5):
            rem = list(range(points))
            random.shuffle(rem)
            mis = []
            # Greedy: pick point if it doesn't complete a triple with existing MIS?
            # No, MIS means NO triple is fully contained in MIS.
            # So we can add a point v if adding it to current MIS doesn't create a triple.
            # Wait, if current MIS is I, and we add v, new MIS is I U {v}.
            # This is valid iff no triple is subset of I U {v}.
            # Since I was independent, no triple is subset of I.
            # So we only need to check triples containing v and two points in I.
            for v in rem:
                valid = True
                for t in triples:
                    if v in t:
                        # Check if the other 2 points of t are in mis
                        other1, other2 = t[0], t[1]
                        if other1 == v: other1, other2 = t[1], t[2]
                        if other2 == v: other1, other2 = t[0], t[1]
                        if other1 in mis and other2 in mis:
                            valid = False
                            break
                if valid:
                    mis.append(v)
            if len(mis) > best_mis:
                best_mis = len(mis)
        return best_mis

    # Initial State: Random valid linear hypergraph
    # Start with empty, add random triples until no more can be added or max time?
    # Better: Start with a known good structure or random fill.
    
    def random_valid_triple(used_pairs):
        # Try to find a triple {a,b,c} such that pairs (a,b), (b,c), (a,c) are not in used_pairs
        # And {a,b,c} is not already a triple (implied by pair uniqueness in linear hypergraph? 
        # No, distinct triples can share 1 point. Pairs must be unique across ALL triples for linearity?
        # Yes: "any two triples share at most one point" <=> "no pair of points appears in more than one triple".
        
        # Generate random permutations of 0..15
        # Pick 3 distinct points
        while True:
            a, b, c = random.sample(range(points), 3)
            if (min(a,b), max(a,b)) in used_pairs: continue
            if (min(b,c), max(b,c)) in used_pairs: continue
            if (min(a,c), max(a,c)) in used_pairs: continue
            return (a,b,c)

    # Initialize
    triples = []
    used_pairs = set()
    
    # Fill with random triples
    max_attempts = 1000
    attempts = 0
    while attempts < max_attempts:
        t = random_valid_triple(used_pairs)
        if t is None:
            break
        triples.append(t)
        a,b,c = t
        used_pairs.add((min(a,b), max(a,b)))
        used_pairs.add((min(b,c), max(b,c)))
        used_pairs.add((min(a,c), max(a,c)))
        attempts += 1
        
    # Simulated Annealing
    # State: list of triples
    # Objective: Minimize alpha (approximated by greedy MIS)
    
    best_alpha = greedy_mis_size(triples)
    best_triples = triples[:]
    
    current_alpha = best_alpha
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    while time.time() < deadline:
        # Move
        # 1. Add a random valid triple
        # 2. Remove a random triple
        # 3. Swap a random triple for another?
        
        move_type = random.random()
        new_triples = triples[:]
        
        if move_type < 0.7:
            # Add
            t = random_valid_triple(used_pairs)
            if t is not None:
                new_triples.append(t)
                # Update used_pairs logic is tricky in-place. 
                # Let's just validate the whole thing or maintain state carefully.
                # For simplicity and correctness, let's reconstruct used_pairs for the move?
                # That's O(E). E is small (< 20). So it's fine.
                
                # Check if t is valid w.r.t new_triples
                # It is by construction of random_valid_triple using old used_pairs?
                # No, we need to check against new_triples.
                # Let's just use a helper to check linearity.
                
                def is_linear(trip_list):
                    pairs = set()
                    for t in trip_list:
                        a,b,c = t
                        p1 = (min(a,b), max(a,b))
                        p2 = (min(b,c), max(b,c))
                        p3 = (min(a,c), max(a,c))
                        if p1 in pairs or p2 in pairs or p3 in pairs:
                            return False
                        pairs.add(p1); pairs.add(p2); pairs.add(p3)
                    return True
                
                if not is_linear(new_triples):
                    continue
                    
        elif move_type < 0.9:
            # Remove
            if triples:
                idx = random.randint(0, len(new_triples)-1)
                new_triples.pop(idx)
        else:
            # Swap: Remove one, try to add another
            if triples:
                idx = random.randint(0, len(new_triples)-1)
                removed = new_triples.pop(idx)
                # Try to add a random valid triple
                # Reconstruct used_pairs from new_triples
                temp_used = set()
                for t in new_triples:
                    a,b,c = t
                    temp_used.add((min(a,b), max(a,b)))
                    temp_used.add((min(b,c), max(b,c)))
                    temp_used.add((min(a,c), max(a,c)))
                
                t = random_valid_triple(temp_used)
                if t is not None:
                    new_triples.append(t)
                else:
                    # Restore
                    new_triples.insert(idx, removed)
                    continue

        # Evaluate
        new_alpha = greedy_mis_size(new_triples)
        
        # Acceptance
        delta = new_alpha - current_alpha # We want to minimize, so delta < 0 is good
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            triples = new_triples
            current_alpha = new_alpha
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_triples = triples[:]
        
        T *= cooling
        if T < T_min:
            T = T_min
            # Optional: restart with randomization? 
            # Let's just keep going.
            
    return best_triples
# EVOLVE-BLOCK-END
