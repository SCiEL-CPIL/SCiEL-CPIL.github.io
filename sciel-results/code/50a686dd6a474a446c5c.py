import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    # 150s kill limit. Use 145s as deadline to be safe (5s margin for return/eval overhead).
    deadline = start_time + 145.0

    # Precompute all t-subsets
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}

    def get_id(triplet_tuple):
        return triplet_to_id[triplet_tuple]

    # Precompute triplet IDs for each block candidate? No, we construct blocks dynamically.
    # But we can precompute the triplet IDs for any k-subset? Too many (C(24,7) ~ 346k).
    # Instead, we compute on the fly.

    # Precompute pairs for each point? No, just compute gain on the fly for speed in small v.
    # However, for gain calculation: gain(p) = number of uncovered triplets formed by p and pairs in current block.
    # For t=3, this is sum over pairs (a,b) in block of 1 if {a,b,p} is uncovered.

    def greedy_construct(v, k, t, seed=0, bias=0.0):
        """
        Greedy construction.
        bias: 0.0 means pure greedy (max gain). 1.0 means random.
        Intermediate values mix the two.
        """
        rng = random.Random(seed)
        uncovered = set(range(n_triplets))
        blocks = []
        
        # To speed up gain calculation, we can precompute pairs in the current block
        # and check membership in uncovered set.
        
        while uncovered:
            # Pick a starting triplet
            # Strategy: Pick a random uncovered triplet, or one that is "hard"
            # For now, random is good for diversity.
            start_triplet_id = rng.choice(list(uncovered))
            start_triplet = all_triplets[start_triplet_id]
            
            block = list(start_triplet)
            block_set = set(block)
            
            while len(block) < k:
                # Calculate gain for each candidate point
                best_point = -1
                max_gain = -1
                second_best_gain = -1
                second_best_point = -1
                
                # Pairs in current block (for t=3)
                pairs_in_block = list(itertools.combinations(block, t - 1))
                
                # Get candidates
                # Optimization: Instead of iterating all v points, we can restrict
                # to points that appear in uncovered triplets containing pairs from block?
                # That's complex. v=24 is small, so iterating all is fine.
                
                gains = {}
                for p in range(v):
                    if p in block_set:
                        continue
                    gain = 0
                    for pair in pairs_in_block:
                        triplet = tuple(sorted(pair + (p,)))
                        tid = get_id(triplet)
                        if tid in uncovered:
                            gain += 1
                    gains[p] = gain
                    if gain > max_gain:
                        second_best_gain = max_gain
                        second_best_point = best_point
                        max_gain = gain
                        best_point = p
                    elif gain > second_best_gain:
                        second_best_gain = gain
                        second_best_point = p

                if max_gain == 0:
                    # No gain, pick random
                    candidates = [p for p in range(v) if p not in block_set]
                    best_point = rng.choice(candidates)
                else:
                    # Mix greedy and random based on bias
                    # If bias is 0, always pick best.
                    # If bias is 1, always pick random from top candidates?
                    # Let's use a softmax-like selection or simple probability.
                    # P(best) = 1 - bias, P(second) = bias * 0.5, P(random) = bias * 0.5
                    r = rng.random()
                    if r < (1.0 - bias):
                        pass # keep best_point
                    elif second_best_point != -1 and r < (1.0 - bias) + bias * 0.5:
                        best_point = second_best_point
                    else:
                        candidates = [p for p in range(v) if p not in block_set]
                        best_point = rng.choice(candidates)
                
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for triplet in itertools.combinations(block, t):
                tid = get_id(triplet)
                uncovered.discard(tid)
        
        return blocks

    # 1. Initial Construction
    # We want to find a good starting point.
    # Use a mix of pure greedy and slightly randomized greedy.
    
    best_blocks = None
    best_count = float('inf')
    
    # Phase 1: Pure Greedy with different seeds (deterministic logic, random start)
    # This is fast.
    for seed in range(20):
        if time.time() > deadline - 45:  # Leave 45s for optimization
            break
        blocks = greedy_construct(v, k, t, seed, bias=0.0)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    # Phase 2: Randomized Greedy (bias > 0) to escape local optima
    # This is slower per run but might find better structures.
    for seed in range(15):
        if time.time() > deadline - 25:  # Leave 25s for optimization
            break
        # Use higher bias for more diversity
        bias = 0.1 + 0.04 * seed
        blocks = greedy_construct(v, k, t, seed, bias=bias)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        best_blocks = greedy_construct(v, k, t, 0, bias=0.0)

    # 2. Local Search Optimization
    count = np.zeros(n_triplets, dtype=np.int32)
    
    def update_counts(blocks, delta=1):
        for block in blocks:
            for triplet in itertools.combinations(block, t):
                tid = get_id(triplet)
                count[tid] += delta

    update_counts(best_blocks, 1)
    current_blocks = list(best_blocks)
    
    def is_redundant(block):
        for triplet in itertools.combinations(block, t):
            if count[get_id(triplet)] == 1:
                return False
        return True

    def try_remove_redundant(blocks):
        removed = False
        new_blocks = []
        for b in blocks:
            if is_redundant(b):
                update_counts([b], -1)
                removed = True
            else:
                new_blocks.append(b)
        return new_blocks, removed

    def try_swap_2_for_1(blocks):
        n = len(blocks)
        if n < 2:
            return blocks, False
            
        # To avoid O(n^2) explosion if n is large, we can limit checks or randomize.
        # But n is ~76, so n^2 is ~5700. Each check is fast.
        # Let's iterate.
        
        # We want to find a pair (i, j) such that replacing them with a single block b'
        # is valid.
        # Condition: All triplets uniquely covered by b_i or b_j must be covered by b'.
        
        # Optimization: Precompute the set of triplet IDs for each block?
        # Let's do it on the fly but efficiently.
        
        for i in range(n):
            if time.time() > deadline - 5:
                return blocks, False
            b1 = blocks[i]
            ids_b1 = set()
            for triplet in itertools.combinations(b1, t):
                ids_b1.add(get_id(triplet))
                
            for j in range(i + 1, n):
                if time.time() > deadline - 5:
                    return blocks, False
                b2 = blocks[j]
                ids_b2 = set()
                for triplet in itertools.combinations(b2, t):
                    ids_b2.add(get_id(triplet))
                
                # Union of IDs
                # We need to check IDs in ids_b1 | ids_b2
                # For each tid in union:
                #   if count[tid] == (1 if tid in ids_b1 else 0) + (1 if tid in ids_b2 else 0)
                #   then this tid is "critical" for this pair.
                
                # If no critical tids, we can remove both.
                # If critical tids exist, can we form a block covering them?
                
                critical_tids = []
                # Iterate through the smaller set? No, we need union.
                # Just iterate through ids_b1 and ids_b2.
                
                # To speed up, we can just check if count[tid] == 1 or 2 appropriately.
                
                # Let's build a set of critical points
                critical_points = set()
                is_valid_swap = True
                
                # Check b1's triplets
                for tid in ids_b1:
                    c = count[tid]
                    covered_by_b1 = 1
                    covered_by_b2 = 1 if tid in ids_b2 else 0
                    if c == covered_by_b1 + covered_by_b2:
                        # This triplet is only covered by b1 and/or b2
                        # It must be covered by the new block
                        critical_points.update(all_triplets[tid])
                        
                # Check b2's triplets (only those not in b1 to avoid double work? 
                # No, if tid is in both, c == 2. If c==2, it's critical. 
                # If tid is only in b2, c == 1. If c==1, it's critical.
                # So we must check all in b2 too.
                
                for tid in ids_b2:
                    if tid in ids_b1:
                        continue # Already checked
                    c = count[tid]
                    covered_by_b1 = 0
                    covered_by_b2 = 1
                    if c == covered_by_b1 + covered_by_b2:
                        critical_points.update(all_triplets[tid])
                
                if not critical_points:
                    # Both blocks are redundant together
                    update_counts([b1, b2], -1)
                    return [b for idx, b in enumerate(blocks) if idx != i and idx != j], True
                
                if len(critical_points) <= k:
                    # We can form a block containing all critical points
                    new_block = list(critical_points)
                    # Fill with random points
                    # To make it deterministic-ish for reproducibility within the swap, 
                    # we can use a fixed seed or just random.
                    # Since we are in local search, random fill is okay.
                    while len(new_block) < k:
                        p = random.randint(0, v - 1)
                        if p not in new_block:
                            new_block.append(p)
                    new_block.sort()
                    
                    # Verify new block covers all critical points (it does by construction)
                    # And doesn't break anything else?
                    # The new block covers a superset of critical points? No, exactly critical points + fillers.
                    # Does it cover any other triplets? Yes, but that's fine (increases count).
                    # The condition for validity is that ALL previously covered triplets remain covered.
                    # We checked that the "critical" ones (those with count dropping to 0) are in new_block.
                    # Triplets with count > critical threshold remain covered by other blocks.
                    # So this swap is valid.
                    
                    update_counts([b1, b2], -1)
                    update_counts([new_block], 1)
                    
                    new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                    new_blocks.append(new_block)
                    return new_blocks, True

        return blocks, False

    def try_swap_1_for_1(blocks):
        n = len(blocks)
        if n < 1:
            return blocks, False

        # For each block, check if we can replace it with a better one that allows further reduction
        # Or simply, check if we can find a block that covers all unique triplets of block i 
        # and potentially allows removing another block?
        # This is harder. Let's stick to 2-for-1 and removal for now, 
        # but add a simple "rebuild" step if stuck?
        # Actually, the inspiration had a try_swap_1_for_1. Let's implement a simple version:
        # Try to replace block i with a new block that covers the same unique triplets 
        # but maybe overlaps differently?
        # Given time constraints, let's rely on 2-for-1 and removal.
        # However, we can add a "random restart" of local search if stuck?
        # No, let's just ensure the main loop is robust.

        return blocks, False

    # Main Optimization Loop
    progress_made = True
    # Alternate between removal and swapping?
    # Or just loop until no progress.
    
    while progress_made and time.time() < deadline - 2:
        progress_made = False
        
        # 1. Remove redundants
        current_blocks, removed = try_remove_redundant(current_blocks)
        if removed:
            progress_made = True
            
        # 2. Swap 2 for 1
        if not progress_made:
            current_blocks, swapped = try_swap_2_for_1(current_blocks)
            if swapped:
                progress_made = True
                # After a swap, we might create new redundants, so loop continues

    # Final pass to ensure no redundants remain (just in case)
    current_blocks, _ = try_remove_redundant(current_blocks)

    return current_blocks

# Test/Entry
if __name__ == "__main__":
    # Test with small parameters to ensure no crash
    # blocks = construct(8, 4, 2)
    # print(len(blocks))
    
    # Run for the actual task parameters if needed by evaluator
    # The evaluator calls construct(24, 7, 3)
    pass
