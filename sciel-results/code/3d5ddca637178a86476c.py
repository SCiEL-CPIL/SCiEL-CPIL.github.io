import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute the list of triples for each possible block? 
    # No, we will generate them on the fly for candidates.
    # But we need to know which triples a block covers quickly.
    # We can precompute a mapping from a block (tuple) to its triple indices.
    # However, there are C(24,7) blocks. Storing them is okay if we don't store the reverse map (triple->blocks) which is huge.
    # Actually, storing block->triples is ~346k * 35 ints. That's ~12M ints. ~50MB. Acceptable.
    # But the previous failure was likely due to the *reverse* map or the time to build it in pure Python loops with defaultdict.
    # Let's try to build block_triples more efficiently or just generate on the fly.
    # Generating on the fly: for a given block, iterate itertools.combinations(block, t) and map to idx.
    # 35 lookups per block evaluation.
    
    # We will NOT precompute all blocks. We will generate candidates on the fly.
    
    def get_block_triples(block):
        """Returns list of triple indices covered by this block."""
        return [triple_to_idx[triple] for triple in itertools.combinations(block, t)]

    def random_blocks_containing(triple):
        """Yields random blocks containing the given triple."""
        # triple is a tuple of t elements
        # We need to choose k-t elements from the remaining v-t elements
        remaining = [x for x in range(v) if x not in triple]
        # To speed up, we can pre-compute the list of all combinations of size k-t from range(v) 
        # and filter? No, that's C(24,4)=10626.
        # Better: just sample from the remaining set.
        # For greedy, we want to evaluate *all* candidates covering a specific triple to find the best gain.
        # Number of candidates = C(v-t, k-t) = C(24-3, 7-3) = C(21, 4) = 5985.
        # This is small. We can iterate all of them.
        
        # Pre-compute the list of all subsets of size k-t from the remaining points?
        # The "remaining" set changes per triple.
        # So we must generate them.
        pass

    def get_candidates_for_triple(triple):
        """Generate all blocks containing the given triple."""
        triple_set = set(triple)
        remaining = [x for x in range(v) if x not in triple_set]
        # We need to choose k-t elements from remaining
        for add in itertools.combinations(remaining, k - t):
            yield tuple(sorted(triple + add))

    def greedy_cover(seed):
        """Run greedy set cover with on-the-fly candidate generation."""
        random.seed(seed)
        np.random.seed(seed)
        
        # uncovered_mask[i] is True if triple i is not yet covered
        uncovered_mask = np.ones(n_triples, dtype=bool)
        selected_blocks = []
        
        while np.any(uncovered_mask):
            if time.time() > deadline:
                break
                
            # Pick a random uncovered triple
            uncovered_indices = np.where(uncovered_mask)[0]
            target_idx = random.choice(uncovered_indices)
            target_triple = triples[target_idx]
            
            best_block = None
            best_gain = -1
            
            # Iterate through all blocks containing target_triple
            # There are C(v-t, k-t) such blocks.
            for block in get_candidates_for_triple(target_triple):
                # Calculate gain: number of uncovered triples in this block
                block_triples = get_block_triples(block)
                gain = np.sum(uncovered_mask[block_triples])
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                elif gain == best_gain and random.random() < 0.1:
                    # Occasional random tie-break for diversity
                    best_block = block
                    best_gain = gain
            
            if best_block is None:
                break
                
            selected_blocks.append(best_block)
            
            # Update uncovered mask
            block_triples = get_block_triples(best_block)
            uncovered_mask[block_triples] = False
            
        return selected_blocks

    def prune_blocks(blocks):
        """Remove redundant blocks."""
        if not blocks:
            return blocks
        
        # Map block tuple to its index in the list for quick reference?
        # We work with block tuples directly.
        remaining = list(blocks)
        
        # Coverage count for each triple
        cover_count = np.zeros(n_triples, dtype=int)
        for block in remaining:
            cover_count[get_block_triples(block)] += 1
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_remaining = []
            for block in remaining:
                if time.time() > deadline:
                    new_remaining.append(block)
                    continue
                
                b_triples = get_block_triples(block)
                # If any triple covered by this block has count 1, we cannot remove it
                if np.any(cover_count[b_triples] == 1):
                    new_remaining.append(block)
                else:
                    # Remove it
                    cover_count[b_triples] -= 1
                    changed = True
            
            remaining = new_remaining
            
        return remaining

    def local_search(blocks):
        """Try to reduce block count by swapping."""
        if len(blocks) <= 1:
            return blocks
            
        remaining = list(blocks)
        
        # Coverage counts
        cover_count = np.zeros(n_triples, dtype=int)
        for block in remaining:
            cover_count[get_block_triples(block)] += 1
            
        improved = True
        max_iters = 500
        iters = 0
        
        while improved and time.time() < deadline - 5 and iters < max_iters:
            improved = False
            iters += 1
            
            # Try to remove each block
            for i in range(len(remaining)):
                if time.time() > deadline - 5:
                    break
                
                block = remaining[i]
                b_triples = get_block_triples(block)
                
                # Find lost triples (count becomes 0)
                lost = []
                for t_idx in b_triples:
                    if cover_count[t_idx] == 1:
                        lost.append(t_idx)
                    
                if not lost:
                    # Can remove
                    cover_count[b_triples] -= 1
                    remaining.pop(i)
                    improved = True
                    break
                
                # Try to find a replacement block covering all lost triples
                # Candidate must cover the first lost triple
                first_lost_triple = triples[lost[0]]
                
                found = False
                for candidate in get_candidates_for_triple(first_lost_triple):
                    # Check if candidate is already in remaining
                    # This check is O(N). N is ~80. OK.
                    if candidate in remaining:
                        continue
                    
                    c_triples = get_block_triples(candidate)
                    c_set = set(c_triples)
                    
                    # Check if candidate covers all lost triples
                    if all(t in c_set for t in lost):
                        # Swap
                        cover_count[b_triples] -= 1
                        cover_count[c_triples] += 1
                        remaining[i] = candidate
                        improved = True
                        found = True
                        break
                        
                if found:
                    break
                    
        return remaining

    # Main Execution
    best_blocks = None
    best_size = float('inf')
    
    def update_best(blocks):
        nonlocal best_blocks, best_size
        if blocks:
            pruned = prune_blocks(blocks)
            # Run local search
            local = local_search(pruned)
            
            if len(local) < best_size:
                best_size = len(local)
                best_blocks = local

    # Run multiple greedy restarts
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        seed = i * 1000 + 42
        blocks = greedy_cover(seed)
        if blocks:
            update_best(blocks)
            
    if best_blocks is None:
        # Fallback: return a valid cover if possible, else empty
        # Greedy should have worked. If not, return a small dummy to avoid crash, 
        # though it will be invalid.
        return []

    return [list(b) for b in best_blocks]
