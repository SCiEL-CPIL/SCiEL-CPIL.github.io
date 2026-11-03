import itertools
import random
import heapq
import numpy as np

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
        
    num_blocks = len(all_blocks)
    
    # Convert block_to_triples to numpy array for fast summing
    # Shape: (num_blocks, C(k, t))
    # C(7,3) = 35
    block_triples_np = np.array(block_to_triples, dtype=np.int32)
    
    best_blocks = None
    best_count = float('inf')
    
    num_restarts = 30 # Increase restarts, each is fast now
    
    for restart in range(num_restarts):
        random.seed(restart)
        
        # Greedy Phase
        uncovered = np.ones(num_triples, dtype=bool)
        blocks = []
        
        # Get a list of indices to pick from randomly
        # We can just pick a random index and check if it's uncovered?
        # Or maintain a list. Since num_triples is ~2000, maintaining a list is fine.
        # But random.choice on a list of 2000 is fast.
        
        while uncovered.any():
            # Pick a random uncovered triple
            # To avoid O(N) scan for a random uncovered, we can just pick random indices until we hit an uncovered one?
            # Probability of hitting uncovered is high initially.
            # Later, it might be low.
            # Better: Maintain a list of uncovered indices.
            
            # For simplicity and speed, let's just pick a random index from 0..num_triples-1
            # and if it's covered, retry.
            # In the worst case (last few triples), this might be slow.
            # Let's use a list of uncovered indices.
            
            # Optimization: Pre-compute a list of uncovered indices?
            # Updating it is O(N).
            
            # Let's stick to the simple retry method, but limit retries?
            # Or just pick from the set.
            
            # Actually, `uncovered` is a numpy bool array.
            # `np.where(uncovered)[0]` gives all uncovered indices.
            # This is O(N) but in C.
            # Then `random.choice` on that list.
            
            unc_idx = np.where(uncovered)[0]
            if len(unc_idx) == 0:
                break
            target_idx = random.choice(unc_idx)
            
            # Find best block covering target_idx
            best_block_idx = -1
            max_covered = -1
            
            # Iterate candidates
            for b_idx in triple_to_blocks[target_idx]:
                # Count covered triples
                # block_triples_np[b_idx] is an array of triple indices
                # uncovered is a bool array
                # Sum of uncovered at those indices
                covered = np.sum(uncovered[block_triples_np[b_idx]])
                
                if covered > max_covered:
                    max_covered = covered
                    best_block_idx = b_idx
                    # Early exit if we cover all triples in this block?
                    # len(block_triples_np[b_idx]) is 35.
                    if covered == 35:
                        break
            
            if best_block_idx != -1:
                blocks.append(best_block_idx)
                # Update uncovered
                uncovered[block_triples_np[best_block_idx]] = False
        
        # Local Search: Add-Prune
        # Convert blocks to a set for O(1) lookup
        current_set = set(blocks)
        
        # Prune function
        def prune(block_list):
            # Count coverage
            triple_count = np.zeros(num_triples, dtype=np.int32)
            for b_idx in block_list:
                triple_count[block_triples_np[b_idx]] += 1
            
            new_blocks = []
            for b_idx in block_list:
                # Check if any triple is covered only by this block
                # If min coverage of its triples is > 1, it's redundant
                # Actually, if ALL its triples have count > 1, it's redundant.
                # We need to check if there exists a triple with count == 1.
                # np.any(triple_count[block_triples_np[b_idx]] == 1)
                if not np.any(triple_count[block_triples_np[b_idx]] == 1):
                    # Redundant, remove
                    triple_count[block_triples_np[b_idx]] -= 1
                else:
                    new_blocks.append(b_idx)
            
            return new_blocks

        # Initial prune
        blocks = prune(blocks)
        
        # Add-Prune Loop
        # Try to add random blocks and prune
        # Number of iterations depends on time. Let's do 500 iterations.
        for _ in range(500):
            # Pick a random block not in current_set
            # We can just pick a random index and check
            b_idx = random.randint(0, num_blocks - 1)
            if b_idx in current_set:
                continue
            
            # Add it
            new_blocks = blocks + [b_idx]
            current_set.add(b_idx)
            
            # Prune
            pruned_blocks = prune(new_blocks)
            
            if len(pruned_blocks) < len(blocks):
                # Improvement found
                blocks = pruned_blocks
                current_set = set(blocks)
            elif len(pruned_blocks) == len(blocks):
                # Same size, maybe different structure
                # Accept with some probability to explore?
                # Or just accept if it's different?
                # Let's accept 20% of the time to escape local minima
                if random.random() < 0.2:
                    blocks = pruned_blocks
                    current_set = set(blocks)
            else:
                # Revert
                current_set.discard(b_idx)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(all_blocks[b]) for b in blocks]
            
    return best_blocks
