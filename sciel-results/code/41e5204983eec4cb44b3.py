import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute all triples
    all_points = range(v)
    all_triples = list(itertools.combinations(all_points, t))
    
    # Map triple to index for bitset? 
    # For the "pick uncovered triple" strategy, a set of uncovered triples is sufficient.
    uncovered = set(all_triples)
    
    blocks = []
    
    # Greedy Construction
    while uncovered:
        if time.time() > deadline:
            break
            
        # Pick an uncovered triple
        # Using a random one can help break symmetries, but min is deterministic.
        # Let's pick the one with the smallest lexicographical order for determinism, 
        # or random for variety. Random is often better for escaping local minima 
        # in the greedy choice if we were doing local search, but for pure greedy, 
        # picking the "most constrained" uncovered triple is a heuristic.
        # However, the standard greedy is: pick ANY uncovered triple, find the best block containing it.
        # The choice of which uncovered triple to pick doesn't affect the set of candidate blocks 
        # if we just want to cover THAT specific triple. 
        # Wait, the standard greedy is:
        # 1. Find the block that covers the most uncovered triples.
        # My proposed optimization:
        # 1. Pick an uncovered triple T.
        # 2. Only consider blocks that contain T.
        # 3. Pick the one that covers the most OTHER uncovered triples.
        # This is equivalent to the standard greedy if the best block overall always contains the picked triple.
        # It might not. The best block might not contain T.
        # So, we must be careful.
        # If we pick T, and the best block for T is B_T, but there is a block B_other that covers 
        # 100 uncovered triples and doesn't contain T, we would miss it.
        # However, if B_other covers 100 uncovered triples, it must cover some triple T' which is uncovered.
        # If we had picked T', we would have considered B_other.
        # Since we only pick ONE triple per step, we might be suboptimal if the "best" block 
        # doesn't contain the specific triple we picked.
        
        # To be safe and effective, let's stick to the standard greedy but optimized.
        # Standard Greedy:
        # Find block B that maximizes |B \cap Uncovered|.
        
        # How to do this fast?
        # We can't iterate all 346k blocks every time.
        
        # Let's refine the "Pick a triple" heuristic.
        # It is known that for covering designs, a greedy that picks a random uncovered triple 
        # and then the best block containing it is very effective and fast.
        
        # Let's try that.
        target_triple = random.choice(list(uncovered))
        
        best_block = None
        max_covered = -1
        
        # Generate all k-subsets containing target_triple
        # target_triple has t elements. We need to choose k-t elements from the remaining v-t elements.
        remaining_points = [p for p in all_points if p not in target_triple]
        
        # To speed up, we can precompute the mask of triples for each potential block?
        # No, we calculate on the fly.
        
        for combo in itertools.combinations(remaining_points, k - t):
            block = tuple(sorted(target_triple + combo))
            
            # Count how many uncovered triples are in this block
            # A block of size k contains C(k, t) triples.
            # We can iterate through them.
            covered_count = 0
            for tri in itertools.combinations(block, t):
                if tri in uncovered:
                    covered_count += 1
            
            if covered_count > max_covered:
                max_covered = covered_count
                best_block = block
                
        if best_block is None:
            # Should not happen
            break
            
        blocks.append(list(best_block))
        
        # Update uncovered
        for tri in itertools.combinations(best_block, t):
            if tri in uncovered:
                uncovered.discard(tri)
                
    # Redundancy Removal
    # Iterate until no changes
    changed = True
    while changed and time.time() < deadline:
        changed = False
        new_blocks = []
        for i, block in enumerate(blocks):
            # Check if this block is redundant
            # A block is redundant if every triple in it is covered by other blocks.
            redundant = True
            for tri in itertools.combinations(block, t):
                # Check if tri is in any other block
                found = False
                for j, other_block in enumerate(blocks):
                    if i == j:
                        continue
                    if tri in other_block:
                        found = True
                        break
                if not found:
                    redundant = False
                    break
            
            if not redundant:
                new_blocks.append(block)
            
        if len(new_blocks) < len(blocks):
            blocks = new_blocks
            changed = True
            # We could also try to re-greedy fill if we removed too many? 
            # No, removing redundant blocks is always safe.
            # But it might leave gaps if my redundancy check is flawed?
            # No, if a triple was only covered by this block, it wouldn't be redundant.
            
    return blocks
