import itertools
import random

def construct(v, k, t):
    points = range(v)
    all_triples = list(itertools.combinations(points, t))
    
    best_blocks = None
    best_count = float('inf')
    
    # Number of restarts to find a better local minimum
    num_restarts = 50
    
    for _ in range(num_restarts):
        # Shuffle triples to break symmetry and find different paths
        triples_copy = all_triples.copy()
        random.shuffle(triples_copy)
        
        uncovered = set(triples_copy)
        blocks = []
        
        while uncovered:
            # Pick a triple to cover. 
            # Strategy: Pick the triple that is in the fewest "candidate" blocks? 
            # Or just the first one? 
            # Picking the first one in the shuffled list is effectively random.
            target_triple = next(iter(uncovered))
            
            # Find all blocks of size k containing target_triple
            # target_triple has t elements. We need k-t more from the remaining v-t points.
            remaining_points = [p for p in points if p not in target_triple]
            candidates = []
            
            # Generate combinations of size k-t from remaining_points
            for comb in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(tuple(target_triple) + comb))
                candidates.append(block)
            
            # Evaluate each candidate by how many uncovered triples it covers
            best_candidate = None
            max_covered = -1
            
            for block in candidates:
                # Count how many triples in this block are in 'uncovered'
                # A block of size k contains C(k, t) triples
                covered_count = 0
                for trip in itertools.combinations(block, t):
                    if trip in uncovered:
                        covered_count += 1
                
                if covered_count > max_covered:
                    max_covered = covered_count
                    best_candidate = block
                    # Early exit if we cover all triples in the block? 
                    # Max possible is C(k,t). If we hit that, it's perfect.
                    if covered_count == len(list(itertools.combinations(block, t))):
                        break
            
            if best_candidate is not None:
                blocks.append(list(best_candidate))
                # Remove covered triples from uncovered
                for trip in itertools.combinations(best_candidate, t):
                    uncovered.discard(trip)
        
        # Refinement: Remove redundant blocks
        # We iterate until no more blocks can be removed
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for i, block in enumerate(blocks):
                # Check if block is redundant
                # A block is redundant if all its triples are covered by other blocks
                is_redundant = True
                for trip in itertools.combinations(block, t):
                    # Check if trip is covered by any block other than current one
                    # Optimization: Precompute coverage counts? 
                    # For now, simple check. 
                    # To be faster, we can maintain a count of how many blocks cover each triple.
                    pass # Placeholder for optimization logic below
                
                # The simple check above is O(N_blocks * C(k,t)). 
                # Let's use a coverage count array for efficiency in the refinement loop.
                
            # Let's implement the refinement with counts properly.
            # This part needs to be robust.
            
            # Re-implementing refinement with counts
            # Create a count of how many times each triple is covered
            triple_count = {}
            for b in blocks:
                for trip in itertools.combinations(b, t):
                    triple_count[trip] = triple_count.get(trip, 0) + 1
            
            # Now check for redundancy
            redundant_indices = []
            for i, block in enumerate(blocks):
                # If any triple in this block has count 1, the block is essential
                # If all triples in this block have count > 1, the block is redundant
                essential = False
                for trip in itertools.combinations(block, t):
                    if triple_count[trip] == 1:
                        essential = True
                        break
                if not essential:
                    redundant_indices.append(i)
            
            if redundant_indices:
                changed = True
                # Remove redundant blocks
                # It's safer to remove one by one and update counts, 
                # but removing all identified at once might be unsafe if they overlap in essentiality?
                # Actually, if a block is redundant, removing it doesn't uncover any triple.
                # So we can remove all identified redundant blocks at once.
                blocks = [b for i, b in enumerate(blocks) if i not in set(redundant_indices)]
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [b[:] for b in blocks] # Deep copy
            
    return best_blocks
