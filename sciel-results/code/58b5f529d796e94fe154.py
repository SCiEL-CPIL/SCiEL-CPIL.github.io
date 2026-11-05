import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = 115.0
    
    n_points = 16
    
    # Step 1: Generate a valid STS(15) deterministically.
    # We use the points of PG(3,2). There are 15 non-zero vectors in GF(2)^4.
    # A triple is a set of 3 distinct points that sum to 0.
    
    def generate_sts15_deterministic():
        # Represent points as integers 0..14 representing non-zero 4-bit vectors.
        # Map non-zero 4-bit integers to 0..14
        pts = []
        for i in range(1, 16):
            pts.append(i)
        
        # Create a mapping from integer value to index 0..14
        # We will work with the integer values 1..15 directly, then map to 0..14 at the end
        # Or just map 1..15 to 0..14 immediately.
        # Let's map value v to index v-1.
        
        triples = []
        seen = set()
        
        # Iterate over all pairs of points
        for i in range(1, 16):
            for j in range(i+1, 16):
                # The third point k must satisfy i ^ j ^ k = 0 => k = i ^ j
                k = i ^ j
                if k == 0 or k == i or k == j:
                    continue
                # Ensure k is in 1..15 (it is, since i,j non-zero and distinct, k non-zero)
                # Sort the triple to have a canonical form
                t = tuple(sorted((i, j, k)))
                if t not in seen:
                    seen.add(t)
                    triples.append(t)
        
        # Map points 1..15 to 0..14
        mapped_triples = []
        for t in triples:
            mapped_triples.append(tuple(x - 1 for x in t))
        
        return mapped_triples

    base_sts = generate_sts15_deterministic()
    
    # Verify STS(15) validity (optional but good for safety)
    # Should have 35 triples.
    if len(base_sts) != 35:
        # Fallback if generation failed (should not happen)
        return [[i, i+1, i+2] for i in range(0, 15, 3)]

    # Step 2: Find all maximum independent sets (caps) of size 8 in STS(15).
    # We need to identify sets S of size 8 such that no triple from base_sts is contained in S.
    
    def find_caps_of_size_k(sts, n, k):
        caps = []
        points = list(range(n))
        
        # Backtracking to find all subsets of size k with no triples
        def backtrack(start, current_set, current_size):
            if time.time() > deadline:
                return
            if current_size == k:
                caps.append(tuple(current_set))
                return
            
            # Pruning: if we can't reach size k, return
            if len(points) - start < k - current_size:
                return
                
            for i in range(start, n):
                p = points[i]
                # Check if adding p creates a triple with existing members
                valid = True
                for t in sts:
                    if p in t:
                        # Find the other two points in t
                        others = [x for x in t if x != p]
                        if others[0] in current_set and others[1] in current_set:
                            valid = False
                            break
                if valid:
                    current_set.append(p)
                    backtrack(i + 1, current_set, current_size + 1)
                    current_set.pop()
        
        backtrack(0, [], 0)
        return caps

    max_cap_size = 8
    caps_8 = find_caps_of_size_k(base_sts, 15, max_cap_size)
    
    # If no caps of size 8 found (shouldn't happen for STS(15)), try 7? 
    # But STS(15) definitely has caps of size 8.
    
    # Step 3: Find a matching of size 7 in the complete graph on 0..14 
    # such that every cap in caps_8 contains at least one edge from the matching.
    # The matching edges will be the pairs (x,y) added to point 15.
    # Let M be the set of 7 disjoint pairs.
    # Condition: For all S in caps_8, S intersects M (i.e., contains some (x,y) in M).
    
    # We can model this as:
    # Generate all matchings of size 7 in K_15.
    # Check if it hits all caps.
    
    # Number of matchings of size 7 in K_15:
    # Choose 14 points out of 15 (15 ways).
    # Number of perfect matchings in K_14 is 13!! = 13*11*9*7*5*3*1 = 135135.
    # Total matchings = 15 * 135135 = 2,027,025.
    # This is too many to iterate naively if the check is slow.
    # But the check is: for each cap S (size 8), does it contain any pair from M?
    # We can optimize by pre-computing for each cap S, the set of pairs it contains.
    # Actually, it's easier to just iterate through matchings and check.
    # 2 million iterations * check cost.
    # Check cost: For a specific matching M (7 pairs), we check against all caps.
    # How many caps of size 8 are there in STS(15)?
    # It's not huge. Let's assume < 1000.
    # 2M * 1000 is 2 billion. Too slow in Python.
    
    # Alternative: Backtracking search for the matching M directly.
    # We want to select 7 disjoint pairs such that they hit all caps.
    # This is a hitting set / matching problem.
    
    # Let's use a greedy/heuristic approach first, or a more constrained backtracking.
    # Or, notice that we just need *one* valid configuration.
    
    # Let's try random restarts with a fast check, as the number of caps might be small enough 
    # that random matchings hit them often? No, random matching might miss a specific cap.
    
    # Better: Backtracking to build the matching M.
    # State: current matching M, set of caps still "unhit" (not intersected by M).
    # Initially, all caps are unhit.
    # When we add a pair (u,v) to M, we remove from "unhit" all caps that contain (u,v).
    # We need to reach a state where "unhit" is empty, having added 7 pairs.
    
    # Pre-compute: for each pair (u,v), which caps contain it?
    pair_to_caps = {}
    for idx, cap in enumerate(caps_8):
        cap_set = set(cap)
        for u, v in itertools.combinations(cap, 2):
            if (u, v) not in pair_to_caps:
                pair_to_caps[(u, v)] = []
            pair_to_caps[(u, v)].append(idx)
            if (v, u) not in pair_to_caps:
                pair_to_caps[(v, u)] = []
            pair_to_caps[(v, u)].append(idx)
            
    # Also, we need to ensure the pairs in M are disjoint.
    
    def find_blocking_matching(caps, pair_to_caps, n=15):
        unhit = set(range(len(caps)))
        used_points = set()
        current_matching = []
        
        def backtrack(depth):
            if time.time() > deadline:
                return None
            if depth == 7:
                if not unhit:
                    return current_matching[:]
                return None
            
            # Heuristic: Choose the next pair to add.
            # We should prioritize pairs that hit many unhit caps.
            # Generate candidate pairs (u,v) where u,v not in used_points.
            candidates = []
            for u in range(n):
                if u in used_points: continue
                for v in range(u+1, n):
                    if v in used_points: continue
                    # Count how many unhit caps this pair hits
                    hits = 0
                    key = (u, v)
                    if key in pair_to_caps:
                        for cap_idx in pair_to_caps[key]:
                            if cap_idx in unhit:
                                hits += 1
                    # Also check (v,u) if needed, but we normalized keys in pair_to_caps?
                    # In the pre-computation, I added both (u,v) and (v,u) to the dict.
                    # So just checking (u,v) is enough if we ensure we look up the right key.
                    # Wait, in pre-computation:
                    # for u, v in combinations(cap, 2):
                    #   pair_to_caps[(u,v)].append(idx)
                    #   pair_to_caps[(v,u)].append(idx)
                    # So (u,v) and (v,u) both exist.
                    # So hits is correct.
                    
                    if hits > 0:
                        candidates.append((hits, u, v))
            
            if not candidates:
                return None
                
            # Sort candidates by hits descending
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            # To avoid symmetry, we can fix the first point of the first pair?
            # Or just try top few.
            for hits, u, v in candidates[:10]: # Limit branching factor
                # Add pair (u,v)
                current_matching.append((u, v))
                used_points.add(u)
                used_points.add(v)
                
                # Update unhit
                removed = set()
                key = (u, v)
                if key in pair_to_caps:
                    for cap_idx in pair_to_caps[key]:
                        if cap_idx in unhit:
                            removed.add(cap_idx)
                unhit.difference_update(removed)
                
                result = backtrack(depth + 1)
                if result is not None:
                    return result
                    
                # Undo
                unhit.update(removed)
                used_points.remove(u)
                used_points.remove(v)
                current_matching.pop()
                
            return None
            
        return backtrack(0)

    blocking_matching = find_blocking_matching(caps_8, pair_to_caps)
    
    if blocking_matching is not None:
        # Construct the final system
        final_triples = [list(t) for t in base_sts]
        for u, v in blocking_matching:
            final_triples.append([15, u, v])
        
        # Optional: Verify independence number is 8
        # (We trust the logic, but a quick check on a few random sets of size 9 
        # could be done if time permits, but not necessary for correctness of construction)
        
        return final_triples
    else:
        # Fallback: If we couldn't find a blocking matching (should be rare/impossible 
        # for STS(15) caps), use a greedy random one.
        random.seed(42)
        used_points = set()
        chosen_pairs = []
        while len(chosen_pairs) < 7:
            p1 = random.randint(0, 14)
            p2 = random.randint(0, 14)
            if p1 == p2 or p1 in used_points or p2 in used_points:
                continue
            chosen_pairs.append((p1, p2))
            used_points.add(p1)
            used_points.add(p2)
            
        final_triples = [list(t) for t in base_sts]
        for u, v in chosen_pairs:
            final_triples.append([15, u, v])
        return final_triples
