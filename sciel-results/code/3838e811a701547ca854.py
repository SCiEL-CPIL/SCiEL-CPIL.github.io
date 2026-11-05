import itertools
import random
import time
import heapq
import math

def construct(v, k, t):
    # Parameters for the specific task are v=24, k=7, t=3
    # But we keep it general for robustness if possible, though optimized for this case.
    
    all_subs = list(itertools.combinations(range(v), t))
    sub_to_idx = {s: i for i, s in enumerate(all_subs)}
    n_subs = len(all_subs)
    
    # Precompute the list of t-subsets for any given k-subset (block)
    # To speed up, we can precompute the triples for a block if we generate many, 
    # but blocks are dynamic. We'll use a helper.
    
    def get_block_triples(block):
        return list(itertools.combinations(block, t))

    def get_block_triple_indices(block):
        return [sub_to_idx[tr] for tr in get_block_triples(block)]

    # --- Phase 1: Greedy Construction ---
    # We use a "Most Constrained First" heuristic.
    # For each uncovered triple, we know how many blocks *could* cover it?
    # Actually, any block containing the triple covers it.
    # The number of such blocks is C(v-t, k-t). This is constant for all triples!
    # So "Most Constrained" doesn't help initially.
    
    # Better Greedy: "Maximizing Marginal Gain".
    # Pick the triple that is currently "hardest" to cover? 
    # Or just standard Greedy: Pick a block that covers the maximum number of currently uncovered triples.
    
    # Since C(24,7) is large, we can't iterate all blocks.
    # We construct a block around a specific uncovered triple.
    
    def greedy_construct(seed):
        random.seed(seed)
        uncovered = set(range(n_subs))
        blocks = []
        
        # To speed up, we track counts of how many times a triple is covered.
        # But for greedy, we just need to know if it's covered.
        
        # We will repeatedly pick an uncovered triple, and build a block around it.
        # To diversify, we pick a random uncovered triple.
        
        while uncovered:
            # Pick a random uncovered triple
            triple_idx = random.choice(list(uncovered))
            triple = all_subs[triple_idx]
            
            # Build a block containing this triple
            # Start with the triple
            block = list(triple)
            
            # Greedily add points to maximize coverage of uncovered triples
            while len(block) < k:
                best_x = -1
                max_gain = -1
                
                # Candidates: points not in block
                for x in range(v):
                    if x in block:
                        continue
                    
                    # Gain = number of uncovered triples formed by adding x to block
                    # Triples formed: (t-1) elements from block + x
                    gain = 0
                    for rest in itertools.combinations(block, t-1):
                        new_triple = tuple(sorted(rest + (x,)))
                        idx = sub_to_idx[new_triple]
                        if idx in uncovered:
                            gain += 1
                    
                    if gain > max_gain:
                        max_gain = gain
                        best_x = x
                        
                if best_x == -1:
                    break # Should not happen
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for tr in get_block_triples(block):
                idx = sub_to_idx[tr]
                uncovered.discard(idx)
                
        return blocks

    # --- Phase 2: Local Search (Simulated Annealing) ---
    
    def local_search(initial_blocks, time_limit):
        current_blocks = initial_blocks[:]
        current_count = len(current_blocks)
        
        # Maintain coverage counts
        cover_counts = [0] * n_subs
        for b in current_blocks:
            for tr in get_block_triples(b):
                cover_counts[sub_to_idx[tr]] += 1
        
        # Verify validity
        if any(c == 0 for c in cover_counts):
            return None, float('inf') # Invalid
            
        best_blocks = current_blocks[:]
        best_count = current_count
        
        start_time = time.time()
        
        # Simulated Annealing Parameters
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.95
        
        # We need to be able to generate candidate blocks for swaps.
        # A candidate block should cover critical triples (count==1) of the block being replaced.
        
        while time.time() - start_time < time_limit:
            
            # --- Move 1: Try to Remove a Redundant Block ---
            # Find a block that is entirely covered by others (all its triples have count > 1)
            removed = False
            for i in range(len(current_blocks)):
                b = current_blocks[i]
                redundant = True
                for tr in get_block_triples(b):
                    if cover_counts[sub_to_idx[tr]] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove it
                    for tr in get_block_triples(b):
                        cover_counts[sub_to_idx[tr]] -= 1
                    current_blocks.pop(i)
                    current_count -= 1
                    removed = True
                    break
                    
            if removed:
                if current_count < best_count:
                    best_count = current_count
                    best_blocks = current_blocks[:]
                continue # Restart loop to check for more removals
            
            # --- Move 2: Try a Swap (Replace a block) ---
            # Pick a random block to replace
            if not current_blocks:
                break
                
            i = random.randint(0, len(current_blocks) - 1)
            b_old = current_blocks[i]
            
            # Identify critical triples in b_old (those with count == 1)
            # If we remove b_old, these triples become uncovered.
            # The new block MUST cover all critical triples.
            
            critical_triples = []
            for tr in get_block_triples(b_old):
                if cover_counts[sub_to_idx[tr]] == 1:
                    critical_triples.append(tr)
            
            # If no critical triples, b_old is redundant? 
            # Wait, if no critical triples, all triples in b_old have count > 1.
            # Then b_old is redundant. Why didn't Move 1 catch it?
            # Move 1 checks if *all* triples in b have count > 1.
            # If critical_triples is empty, then all have count > 1. So it IS redundant.
            # So if we reach here, critical_triples should not be empty.
            
            if not critical_triples:
                # This is a redundant block. Remove it.
                for tr in get_block_triples(b_old):
                    cover_counts[sub_to_idx[tr]] -= 1
                current_blocks.pop(i)
                current_count -= 1
                if current_count < best_count:
                    best_count = current_count
                    best_blocks = current_blocks[:]
                continue
            
            # We need to find a new block b_new such that:
            # 1. It contains all points in the critical triples? No, it must CONTAIN the critical triples.
            #    i.e., for each tr in critical_triples, tr is a subset of b_new.
            # 2. It is a k-subset of {0..v-1}.
            
            # Intersection of all critical triples? No, union of points in critical triples must be in b_new.
            # Let U be the set of points appearing in any critical triple.
            # b_new must contain U.
            
            U = set()
            for tr in critical_triples:
                U.update(tr)
                
            if len(U) > k:
                # Impossible to cover all critical triples with one block
                # We might need to add a block first, then remove? 
                # Or skip this swap.
                # For now, skip.
                continue
            
            # We need to pick k - len(U) additional points from the remaining v - len(U) points.
            # To maximize the chance of making other blocks redundant, we want b_new to cover
            # triples that are currently heavily covered (count > 1), so that if we remove b_old,
            # we don't leave gaps, and potentially we can remove *other* blocks later?
            # Actually, the goal of the swap is to keep the count the same (1 in, 1 out) 
            # but change the coverage profile to enable future removals.
            
            # Heuristic: Pick random points for the rest of the block.
            # To be smarter, we could pick points that appear in many "over-covered" triples.
            
            available_points = [p for p in range(v) if p not in U]
            remaining_slots = k - len(U)
            
            # Sample a few candidate new blocks
            found_new_block = False
            for _ in range(20): # Try 20 random completions
                if remaining_slots > len(available_points):
                    break
                extra_points = random.sample(available_points, remaining_slots)
                b_new = sorted(list(U) + extra_points)
                
                # Check if b_new is valid (distinct elements, size k) - it is by construction.
                # Check if b_new covers all critical triples.
                # Since U is the union of points in critical triples, and b_new contains U,
                # does b_new contain the critical triples?
                # Yes, if a triple is a subset of U, and U is subset of b_new, then triple is subset of b_new.
                
                # Calculate the "gain" or "loss" of swapping b_old with b_new.
                # We don't change the count, but we change coverage.
                # We want to increase the "redundancy" of the system.
                # Metric: Sum of (count - 1) for all triples? Or number of triples with count == 1?
                # Let's use: Number of triples with count == 1 (critical). We want to minimize this.
                
                # Simulate the swap
                # 1. Remove b_old
                # 2. Add b_new
                
                # To avoid modifying state permanently if we reject, we simulate.
                # But for SA, we accept based on energy.
                
                # Energy = -current_count + penalty * (number of uncovered triples)
                # Here we maintain coverage, so penalty is 0 if valid.
                # We want to minimize count. 
                # Since count doesn't change, we use a secondary objective:
                # Maximize the "slack" or "redundancy".
                # Let's define Energy = - (sum of cover_counts) ? No.
                # Let's define Energy = number of triples with count == 1.
                # We want to MINIMIZE this.
                
                # Calculate new critical count
                # We need to update cover_counts virtually.
                # This is expensive if done naively for every candidate.
                
                # Let's just do the swap and see if it allows a removal in the next step?
                # Or use a simpler heuristic: 
                # Accept the swap if the new block covers more "over-covered" triples than the old one?
                
                # Let's just perform the swap and let the "Remove" step in the next iteration clean up.
                # This is a "random walk" in the space of coverings.
                
                # Perform Swap
                for tr in get_block_triples(b_old):
                    cover_counts[sub_to_idx[tr]] -= 1
                for tr in get_block_triples(b_new):
                    cover_counts[sub_to_idx[tr]] += 1
                    
                current_blocks[i] = b_new
                found_new_block = True
                break
            
            if not found_new_block:
                # Try to add a block? 
                # If we can't swap, maybe we can add a block that covers critical triples of MULTIPLE blocks?
                # That increases count. We only do this if we expect to remove 2+ blocks later.
                # For simplicity, we skip adding blocks in this basic SA.
                pass
            
            # Cooling
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
        return best_blocks, best_count

    # --- Main Execution ---
    
    start_time = time.time()
    total_time_limit = 110 # Leave 10s for safety
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple iterations with different random seeds
    seed = 42
    while time.time() - start_time < total_time_limit:
        # Greedy
        blocks = greedy_construct(seed)
        
        # Local Search
        ls_blocks, ls_count = local_search(blocks, time_limit=total_time_limit - (time.time() - start_time))
        
        if ls_blocks is not None and ls_count < best_count:
            best_count = ls_count
            best_blocks = ls_blocks[:]
            
        seed += 1
        
        # If we are getting close to the lower bound or time is running out, break
        if best_count <= 70: # Lower bound is 69
            break
            
    if best_blocks is None:
        # Fallback: return the last greedy result
        blocks = greedy_construct(seed)
        return blocks
        
    return best_blocks
