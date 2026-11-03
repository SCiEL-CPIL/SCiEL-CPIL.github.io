import itertools
import random
import heapq
import time

def construct(v, k, t):
    points = range(v)
    all_triples = list(itertools.combinations(points, t))
    triple_to_index = {trip: i for i, trip in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Pre-compute all blocks and their triple lists
    all_blocks = []
    block_to_triples = []
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    for block in itertools.combinations(points, k):
        block_idx = len(all_blocks)
        all_blocks.append(block)
        trips = []
        for trip in itertools.combinations(block, t):
            idx = triple_to_index[trip]
            trips.append(idx)
            triple_to_blocks[idx].append(block_idx)
        block_to_triples.append(trips)
        
    num_blocks_total = len(all_blocks)
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple restarts
    num_restarts = 10
    time_limit = 110.0
    start_time = time.time()
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        random.seed(restart)
        
        uncovered = set(range(num_triples))
        blocks_in_use = []
        block_in_use_set = set()
        
        # Greedy Phase
        while uncovered:
            # Pick the triple with the fewest available covering blocks (Most Constrained)
            # To avoid O(N_uncovered * Deg) cost, we can approximate or just pick random if it's too slow.
            # For v=24, k=7, t=3, num_triples=2024.
            # Let's try MCV.
            min_deg = float('inf')
            target_idx = -1
            # Optimization: Sample a few random uncovered triples to find a low-degree one quickly?
            # Or just iterate. 2024 * (avg deg ~ 6000) is too slow if done every step.
            # Let's stick to Random Target but optimize the Block Selection.
            
            # Actually, a better heuristic: Pick a random uncovered triple.
            target_idx = random.choice(list(uncovered))
            
            # Find best block covering target_idx
            best_block_idx = -1
            max_covered = -1
            
            candidates = triple_to_blocks[target_idx]
            # Filter candidates that are already in use? No, we can reuse blocks? 
            # No, blocks must be distinct in the final list? "blocks must be distinct". 
            # But in greedy set cover, we don't pick the same block twice.
            
            for b_idx in candidates:
                if b_idx in block_in_use_set:
                    continue
                covered = 0
                for t_idx in block_to_triples[b_idx]:
                    if t_idx in uncovered:
                        covered += 1
                if covered > max_covered:
                    max_covered = covered
                    best_block_idx = b_idx
                    if covered == len(block_to_triples[b_idx]):
                        break
            
            if best_block_idx != -1:
                blocks_in_use.append(best_block_idx)
                block_in_use_set.add(best_block_idx)
                for t_idx in block_to_triples[best_block_idx]:
                    uncovered.discard(t_idx)
        
        # Local Search Phase
        # 1. Remove redundant blocks
        # 2. Swap blocks to enable further removals
        
        def get_critical_triples(block_idx, blocks_in_use, block_in_use_set):
            """Return list of triples covered ONLY by block_idx in the current set."""
            critical = []
            for t_idx in block_to_triples[block_idx]:
                # Check if any other block in use covers t_idx
                is_covered_by_other = False
                for b_idx in blocks_in_use:
                    if b_idx == block_idx:
                        continue
                    if t_idx in block_to_triples[b_idx]: # This is O(35) check
                        is_covered_by_other = True
                        break
                if not is_covered_by_other:
                    critical.append(t_idx)
            return critical

        # This local search is O(N_blocks * N_blocks * C(k,t)^2) which might be slow.
        # Let's use a faster approximation for critical triples using counts.
        
        triple_count = [0] * num_triples
        for b_idx in blocks_in_use:
            for t_idx in block_to_triples[b_idx]:
                triple_count[t_idx] += 1
        
        changed = True
        while changed:
            changed = False
            new_blocks_in_use = []
            for b_idx in blocks_in_use:
                # Check if essential
                essential = False
                for t_idx in block_to_triples[b_idx]:
                    if triple_count[t_idx] == 1:
                        essential = True
                        break
                if not essential:
                    # Remove
                    for t_idx in block_to_triples[b_idx]:
                        triple_count[t_idx] -= 1
                    changed = True
                else:
                    new_blocks_in_use.append(b_idx)
            blocks_in_use = new_blocks_in_use
            block_in_use_set = set(blocks_in_use)
            
            # Swap Phase: Try to replace a block with another to reduce "rigidity"
            # We look for a block B that has critical triples, and see if there's a B' 
            # that covers all critical triples of B. If so, swap B -> B'.
            # This keeps count same but might allow B' to be redundant later or allow 
            # removal of other blocks.
            
            # To keep it fast, we only do this if we have time.
            if time.time() - start_time > time_limit - 5:
                break
                
            for i in range(len(blocks_in_use)):
                b_idx = blocks_in_use[i]
                # Get critical triples
                criticals = []
                for t_idx in block_to_triples[b_idx]:
                    if triple_count[t_idx] == 1:
                        criticals.append(t_idx)
                
                if not criticals:
                    continue # Should have been removed in previous pass, but just in case
                
                # Look for a replacement block B' not in use that covers all criticals
                # We can restrict search to blocks that cover at least one critical triple
                # and check if they cover ALL.
                
                # Gather candidate replacement blocks from the first critical triple
                candidates = triple_to_blocks[criticals[0]]
                
                found_swap = False
                for b_new in candidates:
                    if b_new in block_in_use_set:
                        continue
                    
                    # Check if b_new covers all criticals
                    can_replace = True
                    for t_crit in criticals:
                        if t_crit not in block_to_triples[b_new]:
                            can_replace = False
                            break
                    if can_replace:
                        # Perform swap
                        # Update counts for old block
                        for t_idx in block_to_triples[b_idx]:
                            triple_count[t_idx] -= 1
                        # Update counts for new block
                        for t_idx in block_to_triples[b_new]:
                            triple_count[t_idx] += 1
                        
                        blocks_in_use[i] = b_new
                        block_in_use_set.discard(b_idx)
                        block_in_use_set.add(b_new)
                        
                        found_swap = True
                        changed = True
                        break
                
                if found_swap:
                    # Restart the scan from the beginning as counts changed
                    break
        
        if len(blocks_in_use) < best_count:
            best_count = len(blocks_in_use)
            best_blocks = [list(all_blocks[b]) for b in blocks_in_use]
            
    return best_blocks
