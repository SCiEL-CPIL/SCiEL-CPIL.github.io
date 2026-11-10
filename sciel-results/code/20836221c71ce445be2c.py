import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin for 120s limit
    
    # 1. Greedy Construction
    # We use a set of uncovered triples for fast lookup
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    # Helper to compute gain of adding x to partial block
    def get_gain(x, current_block):
        gain = 0
        # We need to check all (t-1) subsets of current_block
        # and see if adding x creates an uncovered triple
        for rest in itertools.combinations(current_block, t - 1):
            # Create the triple
            tri = tuple(sorted(rest + (x,)))
            if tri in uncovered:
                gain += 1
        return gain

    while uncovered and time.time() < deadline:
        # Pick the first uncovered triple (arbitrary, but deterministic is fine for seed)
        # To add diversity, we could pick randomly, but let's stick to deterministic first
        # Actually, randomizing the start can help escape local minima.
        # Let's try a few random restarts if time permits? 
        # For now, one greedy run.
        
        start_triple = next(iter(uncovered))
        block = list(start_triple)
        
        while len(block) < k and time.time() < deadline:
            # Find best x
            best_x = None
            best_gain = -1
            # Optimization: Only consider x not in block
            candidates = [x for x in range(v) if x not in block]
            
            # Sort candidates by potential gain? 
            # Computing gain for all is O(k * C(k, t-1) * v)
            # For v=24, k=7, t=3: C(6,2)=15. 15*24 = 360 ops per step. Very fast.
            
            for x in candidates:
                g = get_gain(x, block)
                if g > best_gain:
                    best_gain = g
                    best_x = x
                elif g == best_gain and best_x is not None:
                    # Tie breaking: random or fixed? Fixed for reproducibility in seed, 
                    # but random might help. Let's use random for better search.
                    if random.random() < 0.5:
                        best_x = x

            if best_x is None:
                break # Should not happen
            
            block.append(best_x)
            
        block.sort()
        blocks.append(block)
        
        # Update uncovered set
        for tri in itertools.combinations(block, t):
            uncovered.discard(tri)
            
    # If we ran out of time during construction, we might have an invalid covering.
    # We must ensure validity. If time ran out, we should finish the construction?
    # The greedy construction is fast. 76 blocks * 7 points * 360 ops ~ 200k ops. 
    # It will finish in < 0.1s. So construction is safe.
    
    # 2. Local Search / Optimization
    # We try to remove or replace blocks to reduce the count.
    
    # Data structure for coverage counts
    # Map each triple to the number of blocks covering it
    # Since v=24, t=3, total triples = C(24,3) = 2024.
    # We can use a dict or a list of lists.
    
    # Let's use a dictionary for simplicity, or a 3D array? 
    # A list of size 2024 is fine. We need a mapping from triple to index.
    
    triple_to_idx = {}
    all_triples = list(itertools.combinations(range(v), t))
    for i, tri in enumerate(all_triples):
        triple_to_idx[tri] = i
        
    coverage = [0] * len(all_triples)
    
    # Initialize coverage
    for b in blocks:
        for tri in itertools.combinations(b, t):
            coverage[triple_to_idx[tri]] += 1
            
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate through blocks in reverse to handle removals safely?
        # Or just iterate and mark for removal/replacement.
        
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
                
            B = blocks[i]
            
            # Find essential triples: covered only by B
            essential_triples = []
            for tri in itertools.combinations(B, t):
                if coverage[triple_to_idx[tri]] == 1:
                    essential_triples.append(tri)
                    
            if not essential_triples:
                # Can remove B
                # Update coverage
                for tri in itertools.combinations(B, t):
                    coverage[triple_to_idx[tri]] -= 1
                blocks.pop(i)
                improved = True
                # Restart loop to re-evaluate other blocks
                break 
                
            else:
                # Try to replace B
                # Check if union of points in essential_triples has size <= k
                union_points = set()
                for tri in essential_triples:
                    union_points.update(tri)
                    
                if len(union_points) <= k:
                    # We can form a new block containing all essential points
                    new_block_set = set(union_points)
                    # Fill remaining spots
                    remaining_spots = k - len(new_block_set)
                    
                    # Find best points to add
                    # Gain: how many currently covered triples (with count > 1) or uncovered?
                    # Actually, we want to maximize the total coverage provided by the new block.
                    # Since essential triples MUST be covered, and they are in new_block, 
                    # we just need to maximize coverage of OTHER triples.
                    
                    best_new_block = None
                    best_score = -1
                    
                    # Generate candidates for the remaining spots
                    # This is a combinatorial problem. C(v - len(new_block_set), remaining_spots)
                    # v=24, len(new_block_set) could be up to 7.
                    # If len=3, C(21, 4) = 5985.
                    # If len=4, C(20, 3) = 1140.
                    # If len=5, C(19, 2) = 171.
                    # If len=6, C(18, 1) = 18.
                    # If len=7, C(17, 0) = 1.
                    
                    # 5985 candidates * cost of scoring.
                    # Scoring: For each candidate block, count how many triples it covers 
                    # that are NOT in essential_triples? 
                    # Actually, we just need to see if the new block is "better".
                    # A simple metric: Does it cover more total triples than the old block?
                    # Old block covers C(7,3)=35 triples.
                    # New block covers C(7,3)=35 triples.
                    # The essential ones are covered by both.
                    # We want to cover as many *other* triples as possible that are 
                    # currently covered by B (and thus would lose coverage) or 
                    # simply maximize the "redundancy" of other blocks?
                    
                    # Better metric: 
                    # If we replace B with B_new:
                    # Triples in B \ B_new will have their coverage reduced by 1.
                    # Triples in B_new \ B will have their coverage increased by 1.
                    # We want to ensure no triple drops to 0.
                    # Essential triples are in both, so they stay at 1.
                    # Other triples in B: if coverage was 1, they would drop to 0 -> INVALID.
                    # So, we can only replace B with B_new if for all T in B \ B_new, coverage[T] > 1.
                    
                    valid_replacements = []
                    
                    # Iterate all possible completions
                    available_points = [x for x in range(v) if x not in new_block_set]
                    
                    for combo in itertools.combinations(available_points, remaining_spots):
                        candidate_block = list(new_block_set) + list(combo)
                        candidate_block.sort()
                        
                        # Check validity: 
                        # All triples in B that are NOT in candidate_block must have coverage > 1
                        # (because B is being removed, so their coverage drops by 1)
                        
                        is_valid = True
                        b_set = set(B)
                        c_set = set(candidate_block)
                        
                        for tri in itertools.combinations(B, t):
                            if tri not in c_set: # Wait, tri is a tuple of points. 
                                # Check if tri is a subset of candidate_block
                                if not all(p in c_set for p in tri):
                                    # This triple is in B but not in candidate_block
                                    if coverage[triple_to_idx[tri]] <= 1:
                                        is_valid = False
                                        break
                                        
                        if is_valid:
                            # Score: How many new triples does it cover?
                            # Or just count total coverage?
                            # Let's score by number of triples in candidate_block that are 
                            # NOT in B. This adds new coverage.
                            score = 0
                            for tri in itertools.combinations(candidate_block, t):
                                if not all(p in b_set for p in tri):
                                    score += 1
                                    
                            if score > best_score:
                                best_score = score
                                best_new_block = candidate_block
                                
                    if best_new_block is not None and best_new_block != B:
                        # Perform replacement
                        # Update coverage for old B
                        for tri in itertools.combinations(B, t):
                            coverage[triple_to_idx[tri]] -= 1
                        # Update coverage for new block
                        for tri in itertools.combinations(best_new_block, t):
                            coverage[triple_to_idx[tri]] += 1
                        blocks[i] = best_new_block
                        improved = True
                        # Restart to re-evaluate
                        break
                        
    return blocks
