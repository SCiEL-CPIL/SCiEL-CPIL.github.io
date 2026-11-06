import time
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    
    # 1. Construct a verified STS(15)
    # Cyclic STS(15) with base blocks {0, 1, 4}, {0, 2, 7}, {0, 3, 9}
    # Wait, let's verify these base blocks generate a linear STS.
    # A known set of base blocks for STS(15) in Z_15:
    # {0, 1, 4}, {0, 2, 7}, {0, 3, 9}
    # Let's check linearity of the generated system.
    
    sts15 = set()
    base_blocks = [(0, 1, 4), (0, 2, 7), (0, 3, 9)]
    for bb in base_blocks:
        for d in range(15):
            block = tuple(sorted([(bb[0] + d) % 15, (bb[1] + d) % 15, (bb[2] + d) % 15]))
            sts15.add(block)
            
    sts15_list = [list(t) for t in sorted(sts15)]
    
    # Verify linearity
    def is_linear(triples):
        for i in range(len(triples)):
            for j in range(i + 1, len(triples)):
                common = set(triples[i]) & set(triples[j])
                if len(common) > 1:
                    return False
        return True

    if not is_linear(sts15_list):
        # Fallback: Use a different known STS(15) if the cyclic one fails (it shouldn't)
        # Or just use a simple valid linear system, though independence number will be worse.
        # For now, assume the cyclic one works or use a hardcoded valid one if it fails.
        # Let's try another set of base blocks if needed.
        # Actually, the standard cyclic STS(15) uses difference family.
        # Let's just hardcode a verified STS(15) to be safe.
        sts15_list = [
            [0, 1, 4], [0, 2, 7], [0, 3, 9], [0, 5, 11], [0, 6, 14],
            [1, 2, 12], [1, 3, 5], [1, 6, 10], [1, 7, 13], [1, 8, 11],
            [2, 3, 11], [2, 4, 8], [2, 6, 13], [2, 9, 12], [2, 10, 14],
            [3, 4, 10], [3, 6, 8], [3, 7, 12], [3, 13, 14], [3, 5, 9],
            [4, 5, 7], [4, 6, 12], [4, 9, 13], [4, 11, 14], [4, 8, 10],
            [5, 6, 9], [5, 8, 13], [5, 10, 14], [5, 7, 12],
            [6, 7, 8], [6, 11, 13], [6, 10, 14],
            [7, 9, 10], [7, 11, 12],
            [8, 9, 11], [8, 10, 14],
            [9, 13, 14],
            [10, 11, 12],
            [11, 13, 14],
            [12, 13, 14]
        ]
        # This hardcoded list might not be linear either. 
        # Let's stick to the cyclic generation but verify it.
        # If cyclic fails, we are in trouble. 
        # Let's re-verify the cyclic construction.
        # Base blocks for STS(15) in Z_15:
        # It is known that {0,1,4}, {0,2,7}, {0,3,9} do NOT form an STS(15) because 
        # the number of blocks is 3 * 15 = 45, but STS(15) needs 35.
        # The orbit size is 15 only if stabilizer is trivial.
        # For {0,1,4}, stabilizer is {0}. Orbit size 15.
        # For {0,2,7}, stabilizer is {0}. Orbit size 15.
        # For {0,3,9}, stabilizer is {0}. Orbit size 15.
        # Total 45 blocks. This is NOT an STS.
        # We need a difference family that sums to correct counts.
        
        # Correct cyclic STS(15) base blocks:
        # We need 35 blocks.
        # Orbits of size 15: 2 orbits? 30 blocks. Plus 5 blocks of size 1? No.
        # Actually, STS(15) can be constructed.
        # Let's use a verified STS(15) from literature.
        # One standard STS(15) is derived from the projective plane of order 2? No, that's STS(7).
        # STS(15) is the Witt design? No.
        
        # Let's use a simple greedy construction for STS(15) if cyclic is hard, 
        # but we need it to be linear.
        # Actually, the prompt says "Linear triple systems... any two triples share at most one point".
        # It does NOT require it to be a Steiner Triple System (which requires every pair to be in exactly one triple).
        # It only requires LINEARITY.
        # So we can use ANY linear 3-uniform hypergraph on 15 points.
        # To minimize independence number, we want it to be "dense" in terms of covering independent sets.
        # A Steiner Triple System is a good candidate because it is "regular" and "dense".
        # But if we can't easily get a valid STS(15), we can use a random linear system or a greedy one.
        
        # Let's try to build a linear system on 15 points that has low independence number.
        # Greedy: Add triples that don't share >1 point.
        # To minimize alpha, we should add triples that "hit" potential independent sets.
        
        pass # This block is for fallback if cyclic fails. We'll replace the logic below.

    # Let's assume the cyclic construction in the original code was flawed.
    # I will implement a robust greedy linear system construction for 15 points.
    # Then add point 15.
    
    # Step 1: Construct a linear system on 15 points (0-14)
    # We want to minimize the independence number of this subsystem.
    # Greedy heuristic: Pick triples randomly, keep if linear.
    # To be better, pick triples that cover pairs frequently.
    
    points_15 = list(range(15))
    triples_15 = []
    pair_count = {}
    for p1 in points_15:
        for p2 in points_15:
            if p1 < p2:
                pair_count[(p1, p2)] = 0
                
    # We want to add as many triples as possible while maintaining linearity.
    # Max number of triples in a linear 3-uniform hypergraph on 15 points?
    # Each triple uses 3 pairs. Total pairs = 105.
    # If we could use all pairs, 35 triples. This is the STS(15) bound.
    # Let's try to reach 35 triples (STS) using a backtracking/greedy approach.
    
    import random
    random.seed(42)
    
    all_pairs = list(combinations(points_15, 2))
    random.shuffle(all_pairs)
    
    # Greedy STS construction
    # For each pair, if not covered, try to find a third point to form a linear triple.
    # This is not guaranteed to produce an STS, but a large linear system.
    
    used_pairs = set()
    for p1, p2 in all_pairs:
        if (p1, p2) in used_pairs:
            continue
        # Find a third point p3 such that (p1, p3) and (p2, p3) are not used 
        # and the triple is linear with existing ones.
        candidates = [p for p in points_15 if p != p1 and p != p2]
        random.shuffle(candidates)
        for p3 in candidates:
            new_triple = tuple(sorted([p1, p2, p3]))
            new_pairs = [(min(a,b), max(a,b)) for a,b in [(p1,p2), (p1,p3), (p2,p3)]]
            
            # Check if pairs are available
            if any(p in used_pairs for p in new_pairs):
                continue
                
            # Check linearity with existing triples
            ok = True
            for t in triples_15:
                if len(set(new_triple) & set(t)) > 1:
                    ok = False
                    break
            
            if ok:
                triples_15.append(list(new_triple))
                for p in new_pairs:
                    used_pairs.add(p)
                break
                
    # This greedy approach might not reach 35 triples.
    # Let's check how many we got.
    # print(len(triples_15))
    
    # Now add point 15.
    # We want to add triples [15, u, v] where {u, v} is a pair in 0-14.
    # Constraint: All such triples must have disjoint pairs {u, v}.
    # We can add at most 7 such triples.
    
    # To minimize independence number, we should choose pairs {u, v} that are "critical".
    # A pair {u, v} is critical if it appears in many large independent sets of the current system.
    # Or simpler: Just pick 7 random disjoint pairs from the available pairs?
    # No, we should pick pairs that are NOT in the current system?
    # Wait, if {u, v} is already in a triple in triples_15, then {u, v} is "covered".
    # Does that matter?
    # The independence number of the whole system is the max size of a set S of points 
    # such that no triple is fully contained in S.
    
    # Let's just add 7 disjoint pairs.
    # To be strategic, we can try to pick pairs that "break" the largest independent sets.
    # For now, let's just pick 7 random disjoint pairs.
    # We can improve this by checking the independence number and swapping pairs if it decreases.
    
    def get_independence_number(triples):
        n = 16
        # Check from size 8 down
        for k in range(8, 0, -1):
            for subset in combinations(range(n), k):
                s = set(subset)
                has_triple = False
                for t in triples:
                    if all(p in s for p in t):
                        has_triple = True
                        break
                if not has_triple:
                    return k
        return 0

    # Select 7 disjoint pairs for point 15
    available_pairs = list(combinations(points_15, 2))
    random.shuffle(available_pairs)
    
    selected_pairs = []
    used_points_for_15 = set()
    
    for p1, p2 in available_pairs:
        if p1 in used_points_for_15 or p2 in used_points_for_15:
            continue
        selected_pairs.append((p1, p2))
        used_points_for_15.add(p1)
        used_points_for_15.add(p2)
        if len(selected_pairs) == 7:
            break
            
    triples_15_with_15 = [[15, p1, p2] for p1, p2 in selected_pairs]
    
    all_triples = triples_15 + triples_15_with_15
    
    # Verify linearity
    if not is_linear(all_triples):
        # Should not happen if we picked disjoint pairs for 15 and linear system for 15
        # But let's be safe
        valid = []
        for t in all_triples:
            ok = True
            for v in valid:
                if len(set(t) & set(v)) > 1:
                    ok = False
                    break
            if ok:
                valid.append(t)
        all_triples = valid
        
    # Local Search: Try to swap one of the selected pairs for 15 with another to improve independence number
    current_indep = get_independence_number(all_triples)
    
    # If we have time, try to optimize
    if time.time() - start_time < DEADLINE - 10:
        # Try swapping each selected pair with a random other pair
        for i in range(len(selected_pairs)):
            if time.time() - start_time > DEADLINE - 5:
                break
            # Remove current pair i
            p1_old, p2_old = selected_pairs[i]
            # Try to replace with a random pair not using points already in other selected pairs
            other_points = set()
            for j, (p1, p2) in enumerate(selected_pairs):
                if j != i:
                    other_points.add(p1)
                    other_points.add(p2)
                    
            candidates = [(a, b) for a in points_15 if a not in other_points for b in points_15 if b not in other_points and a < b]
            if not candidates:
                continue
            random.shuffle(candidates)
            
            for p1_new, p2_new in candidates:
                # Check if this pair creates a conflict with existing triples_15?
                # No, triples_15 don't involve 15.
                # But we must ensure linearity.
                # New triple is [15, p1_new, p2_new].
                # It shares 15 with other triples_15_with_15.
                # It shares p1_new, p2_new with triples_15.
                # If {p1_new, p2_new} is a pair in some triple in triples_15, 
                # does that violate linearity?
                # Triple T1 = [15, p1_new, p2_new].
                # Triple T2 = [p1_new, p2_new, x] in triples_15.
                # Intersection is {p1_new, p2_new}. Size 2. VIOLATION.
                # So we MUST choose pairs {u, v} that are NOT in any triple of triples_15.
                
                # Let's check if (p1_new, p2_new) is in used_pairs
                if (min(p1_new, p2_new), max(p1_new, p2_new)) in used_pairs:
                    continue
                    
                # If valid, try it
                new_selected_pairs = selected_pairs[:]
                new_selected_pairs[i] = (p1_new, p2_new)
                new_triples_15_with_15 = [[15, p1, p2] for p1, p2 in new_selected_pairs]
                new_all_triples = triples_15 + new_triples_15_with_15
                
                if is_linear(new_all_triples):
                    new_indep = get_independence_number(new_all_triples)
                    if new_indep < current_indep:
                        current_indep = new_indep
                        selected_pairs = new_selected_pairs
                        all_triples = new_all_triples
                        break # Restart swap search from beginning? Or just continue?
                        # Let's break inner loop and restart outer to pick up new state
                        break
                        
    return all_triples
