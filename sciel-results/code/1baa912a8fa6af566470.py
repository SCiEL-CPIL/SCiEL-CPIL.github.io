"""Improved covering design program: Fast randomized greedy + Greedy Set Cover Reversal reduction."""
import itertools
import time
import random
import sys
from collections import Counter, defaultdict

def construct(v, k, t):
    start_time = time.time()
    # Hard deadline: 140s to leave 10s safety margin for the evaluator's 150s limit
    deadline = start_time + 140.0
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    all_tsubsets_set = set(all_tsubsets)
    
    # Map each t-subset to the list of all k-blocks that contain it.
    # This is expensive memory-wise but C(24,3)=2024 and C(24,7)=346104 is manageable.
    # Actually, storing the list of blocks for each triple is better for fast lookup.
    # triple_to_blocks: dict[tuple, list[frozenset]]
    triple_to_blocks = defaultdict(list)
    
    # We don't want to generate all 346k blocks upfront and iterate all their triples (too slow).
    # Instead, for each triple, generate all blocks containing it.
    # Number of blocks per triple: C(v-t, k-t) = C(21, 4) = 5985.
    # Total entries: 2024 * 5985 ~ 12 million. This might be too slow to build in Python.
    
    # Alternative: On-the-fly generation with caching or random sampling.
    # Let's use a helper to generate random blocks containing a specific triple.
    
    def get_random_blocks_containing(triple, num_samples, rng):
        """
        Generate up to `num_samples` distinct random k-subsets containing `triple`.
        """
        triple_set = set(triple)
        other_points = [p for p in range(v) if p not in triple_set]
        blocks = set()
        # We need to pick k-t points from other_points
        # If num_samples is small, just sample combinations
        # To ensure distinctness, we use a set
        while len(blocks) < num_samples:
            # Randomly select k-t points
            combo = rng.sample(other_points, k - t)
            block = frozenset(triple_set.union(combo))
            blocks.add(block)
            # If we can't get enough unique blocks (unlikely with C(21,4)=5985), break
            if len(blocks) > 200: break 
        return list(blocks)

    def compute_gain(block, uncovered_set):
        """
        Compute how many uncovered triples are covered by this block.
        """
        gain = 0
        for tri in itertools.combinations(block, t):
            if tri in uncovered_set:
                gain += 1
        return gain

    def fast_greedy_cover(seed, num_samples=10):
        """
        Randomized greedy covering.
        For each uncovered triple, sample `num_samples` random blocks containing it,
        and pick the one with the highest gain.
        """
        rng = random.Random(seed)
        uncovered = set(all_tsubsets_set)
        blocks = []
        
        while uncovered and time.time() < deadline - 60.0:  # Leave 60s for reduction
            # Pick a random uncovered triple to prioritize
            pivot_triple = rng.choice(tuple(uncovered))
            
            # Generate candidate blocks
            candidates = get_random_blocks_containing(pivot_triple, num_samples, rng)
            
            best_block = None
            best_gain = -1
            
            for block in candidates:
                gain = compute_gain(block, uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block is not None:
                blocks.append(best_block)
                # Remove covered triples
                for tri in itertools.combinations(best_block, t):
                    uncovered.discard(tri)
            
            # Optional: Randomly check if we can stop early? No, must cover all.
        
        # If we exited due to time, we might have uncovered triples.
        # We must finish the covering to be valid.
        # Do a strict greedy finish for the remaining uncovered triples.
        if uncovered:
            while uncovered:
                pivot_triple = min(uncovered) # Deterministic for finish
                candidates = get_random_blocks_containing(pivot_triple, 20, rng)
                best_block = None
                best_gain = -1
                for block in candidates:
                    gain = compute_gain(block, uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_block = block
                if best_block is not None:
                    blocks.append(best_block)
                    for tri in itertools.combinations(best_block, t):
                        uncovered.discard(tri)
        
        return [list(b) for b in blocks]

    def reduce_covering(blocks, time_limit):
        """
        Greedy Set Cover Reversal.
        1. Count coverage for each triple.
        2. Try to remove each block.
        3. If removing block i leaves some triples uncovered, find the single best
           block (not in the current set) that covers the most of those uncovered triples.
           If such a block exists and is different from block i, and the total size doesn't increase
           (or if we are just looking for redundancy, we only keep it if size decreases).
           
           Actually, standard "Reverse Delete" works like this:
           - Start with a valid cover.
           - For each block B in the cover:
               - Remove B.
               - If the cover is still valid, keep it removed.
               - If not valid, let U be the set of uncovered triples.
               - Find the block B_new that covers the most elements of U.
               - If B_new exists and B_new != B:
                   - Add B_new.
                   - Now the cover is valid.
                   - We have effectively replaced B with B_new.
               - Else (no better block):
                   - Add B back.
           
           This process can sometimes reduce the number of blocks if B was redundant,
           or replace B with a "denser" block B_new that helps subsequent removals.
        """
        if not blocks:
            return blocks
        
        blocks_set = set(frozenset(b) for b in blocks)
        
        # Precompute triples for each block for fast coverage counting
        block_triples = [set(itertools.combinations(b, t)) for b in blocks]
        
        # Coverage count
        triple_count = Counter()
        for bt in block_triples:
            triple_count.update(bt)
        
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for i in range(len(blocks)):
                if time.time() > time_limit:
                    break
                
                # Try removing block i
                bi = blocks[i]
                ti = block_triples[i]
                
                # Check if any triple in ti is covered only by bi
                is_redundant = True
                uncovered_by_removal = set()
                for tri in ti:
                    if triple_count[tri] <= 1:
                        is_redundant = False
                        uncovered_by_removal.add(tri)
                
                if is_redundant:
                    # Block i is completely redundant. Remove it.
                    for tri in ti:
                        triple_count[tri] -= 1
                    blocks.pop(i)
                    block_triples.pop(i)
                    blocks_set.remove(frozenset(bi))
                    changed = True
                    break # Restart loop as indices changed
                else:
                    # Not redundant. Can we replace it with a better block?
                    # Find a block that covers the most of `uncovered_by_removal`
                    # We restrict search to blocks containing the pivot triple of uncovered_by_removal?
                    # Or just sample random blocks.
                    
                    # Pick a pivot from uncovered_by_removal to search for replacements
                    pivot = next(iter(uncovered_by_removal))
                    # Generate candidates
                    rng_local = random.Random(hash(pivot) + i)
                    candidates = get_random_blocks_containing(pivot, 5, rng_local)
                    
                    best_replace_block = None
                    best_gain = 0
                    
                    for cand in candidates:
                        # Skip if already in set
                        if frozenset(cand) in blocks_set:
                            continue
                        # Gain is how many of uncovered_by_removal it covers
                        gain = len(uncovered_by_removal.intersection(cand))
                        if gain > best_gain:
                            best_gain = gain
                            best_replace_block = cand
                    
                    if best_replace_block is not None and best_gain > 0:
                        # Replace block i with best_replace_block
                        # Update counts
                        for tri in ti:
                            triple_count[tri] -= 1
                        
                        blocks.pop(i)
                        block_triples.pop(i)
                        blocks_set.remove(frozenset(bi))
                        
                        new_block = list(best_replace_block)
                        new_triples = set(itertools.combinations(new_block, t))
                        
                        blocks.append(new_block)
                        block_triples.append(new_triples)
                        blocks_set.add(frozenset(new_block))
                        
                        for tri in new_triples:
                            triple_count[tri] += 1
                        
                        changed = True
                        break # Restart loop
        
        return blocks

    # Main logic
    best_blocks = None
    best_len = float('inf')
    
    # Phase 1: Generate initial candidates using fast greedy
    # We try a few seeds to get diverse starts
    seeds = [42, 123, 456, 789]
    candidates = []
    
    for seed in seeds:
        if time.time() > deadline - 60.0:
            break
        blocks = fast_greedy_cover(seed, num_samples=15)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        candidates.append(blocks)
    
    if not candidates:
        # Fallback: one strict greedy
        blocks = fast_greedy_cover(42, num_samples=10)
        candidates = [blocks]
        best_blocks = blocks
        best_len = len(blocks)

    # Phase 2: Reduce the best candidates
    # We spend the remaining time on reduction
    # Sort candidates by length to prioritize the best ones for reduction
    candidates.sort(key=lambda x: len(x))
    
    for i, blocks in enumerate(candidates):
        # Time budget per reduction: share remaining time among top candidates
        # Or just try the best one first, then next if time permits.
        time_left = deadline - time.time()
        if time_left < 5.0:
            break
        
        # Allocate up to 80% of remaining time to the first candidate, rest to others
        # Simple heuristic: Try the best one for (time_left - 10s)
        alloc_time = time_left - 10.0
        if alloc_time < 5.0:
            alloc_time = 5.0
            
        reduced = reduce_covering(blocks, start_time + alloc_time)
        
        if len(reduced) < best_len:
            best_len = len(reduced)
            best_blocks = reduced
            
        # If we have time, try the next candidate
        if time.time() < deadline - 5.0:
            # Reset blocks from candidate for next iteration?
            # No, reduce_covering modifies in place. We need to copy.
            pass 

    return best_blocks
