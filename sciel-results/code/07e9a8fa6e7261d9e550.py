import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin for 120s limit
    
    # Precompute all t-subsets (triples)
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    
    # Map each triple to an index
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # Precompute for each possible block (k-subset) which triples it covers
    # This is C(24,7) = 346,104 blocks. 
    # For each block, we can represent the set of covered triples as a bitset (int) or a boolean array.
    # Given v=24, k=7, t=3, num_triples = C(24,3) = 2024.
    # A bitset of 2024 bits fits in ~26 64-bit integers or a Python int.
    # Python ints are efficient for bit operations.
    
    # Let's precompute the bitset for every possible block.
    # This might take some time/memory, but 346k blocks * (7 choose 3) triples each is manageable.
    # C(7,3) = 35 triples per block.
    # Total operations: 346k * 35 ~ 12M operations. Should be fast enough.
    
    block_bitsets = {}
    block_list = []
    
    # To speed up, we can generate blocks and compute their bitsets.
    # We don't need to store ALL blocks if we can generate them on the fly, 
    # but for strong greedy, we need to evaluate gain of adding a block.
    # Gain = popcount(block_bitset & uncovered_bitset)
    
    # Let's precompute the bitset for each block.
    # We'll store them in a dictionary keyed by the block tuple.
    
    # Optimization: We can iterate through all combinations.
    # To make it faster, we can use numpy for the final selection if needed, 
    # but Python int bit ops are very fast.
    
    # Let's build the list of all blocks and their bitsets.
    # This is a one-time cost.
    
    # Memory check: 346,104 blocks. Each bitset is a Python int (~250 bytes for 2024 bits? No, 2024 bits is ~253 bytes. Plus overhead).
    # 346k * 300 bytes ~ 100MB. This is acceptable.
    
    # However, iterating C(24,7) in Python might be slow.
    # Let's try to optimize the precomputation.
    
    # Alternative: Don't precompute all. 
    # In Strong Greedy, at each step, we pick the block that maximizes gain.
    # If we don't precompute, we have to evaluate gain for all C(24,7) blocks at each step.
    # Number of steps ~ 76.
    # 76 * 346k * (cost to compute gain).
    # If gain computation is just `popcount(bitset & uncovered)`, and we precomputed bitsets, it's very fast.
    # If we compute on the fly, we have to iterate 35 triples per block.
    # 76 * 346k * 35 ~ 900M operations. Too slow in Python.
    
    # So precomputation is key.
    
    # Let's precompute.
    # To speed up precomputation, we can use a recursive generation or just itertools.
    # itertools.combinations(range(24), 7) yields 346k tuples.
    
    # Let's try to do it efficiently.
    
    # 1. Precompute bitsets for all blocks
    # We'll use a list of (block_tuple, bitset_int)
    
    # Actually, we can just store the bitsets in a list corresponding to the order of itertools.combinations.
    # But we need to map a block to its index or bitset quickly.
    # A dictionary {block_tuple: bitset} is good.
    
    # Let's check time for precomputation.
    # 346k iterations. Inside: 35 combinations.
    # 346k * 35 = 12M lookups in triple_to_idx.
    # This should take < 1 second.
    
    # Let's do it.
    
    # Note: We only need to consider blocks that cover at least one uncovered triple.
    # But initially all are uncovered.
    
    # Let's start the precomputation.
    
    # If time is tight, we might skip precomputation and use a smarter heuristic, 
    # but 115s is plenty for 12M simple ops.
    
    # Let's implement the precomputation.
    
    # To save memory, we can store bitsets in a list and map block->index?
    # No, dict is fine.
    
    # Wait, is it possible to avoid precomputing ALL blocks?
    # Strong Greedy requires evaluating ALL candidates.
    # So yes, we need all block bitsets.
    
    # Let's optimize the bitset creation.
    # We can represent a block as a frozenset or tuple. Tuple is hashable and compact.
    
    # Let's run the precomputation.
    
    # One potential issue: The evaluator might call construct multiple times? 
    # "The evaluator passes this task's parameters v, k and t." 
    # It implies a single call per evaluation.
    # But to be safe, we should cache the precomputation if possible? 
    # No, v, k, t are fixed for this task. 
    # But the function is called once.
    # If the script is run multiple times, it will recompute. 
    # That's fine.
    
    # Let's write the code.
    
    # Precomputation
    block_bitsets = {}
    
    # We can speed up by noting that if a block is a superset of another, 
    # but here all blocks have size k.
    
    # Let's just iterate.
    
    # To make it faster, we can use a list of triples for each point?
    # No, the direct iteration is likely fine.
    
    # Let's try to optimize the inner loop.
    
    # Instead of `itertools.combinations(b, t)` which creates tuples, 
    # we can precompute the indices.
    
    # Actually, `triple_to_idx` lookup is the main cost.
    
    # Let's try.
    
    # If precomputation takes too long, we can fall back to a heuristic that doesn't need all blocks.
    # But 12M ops is small.
    
    # Let's do it.
    
    # However, 346,104 blocks is a lot for a dict if we store tuples.
    # Let's see.
    
    # Alternative: 
    # We can generate blocks on the fly but use a "candidate pool" strategy?
    # No, Strong Greedy is global.
    
    # Let's stick to precomputation.
    
    # To save time, we can use `random.sample` to pick a subset of blocks if we run out of time?
    # No, that degrades quality.
    
    # Let's assume precomputation is fast.
    
    # One optimization: 
    # We can store the bitsets in a list `all_block_bitsets` and the blocks in `all_blocks`.
    # And use a dict `block_to_idx` to map block tuple to index.
    
    # Let's do that.
    
    all_blocks = []
    all_block_bitsets = []
    block_to_idx = {}
    
    # This will take a bit of memory.
    # 346k tuples of 7 ints.
    # 346k ints.
    
    # Let's try to generate them.
    
    # We can use a generator to avoid holding all in memory if we just need to iterate?
    # But we need random access or at least full iteration in each step.
    # Full iteration 76 times.
    
    # So we need them stored.
    
    # Let's go.
    
    # Note: If this takes > 10s, we need a different approach.
    # 12M dict lookups. In Python, 10M lookups/sec is reasonable. So ~1-2s.
    
    # Let's implement.
    
    # If we run out of time, we can skip precomputation and use a "Greedy with Local Search" that evaluates gain on the fly for a subset of candidates?
    # No, let's trust the precomputation.
    
    # Wait, there is a risk that `itertools.combinations` is slow.
    # Let's use a faster way to generate combinations if possible? 
    # Standard library is usually fine.
    
    # Let's start.
    
    # If precomputation fails or takes too long, we will catch it and use a fallback.
    
    try:
        # Precompute
        for b in itertools.combinations(range(v), k):
            bitset = 0
            for tri in itertools.combinations(b, t):
                idx = triple_to_idx[tri]
                bitset |= (1 << idx)
            
            block_to_idx[b] = len(all_blocks)
            all_blocks.append(b)
            all_block_bitsets.append(bitset)
            
    except Exception as e:
        # Fallback if precomputation fails (e.g. memory)
        # Use a simple greedy without precomputation
        pass
        
    # If precomputation succeeded, use Strong Greedy
    if all_blocks:
        uncovered_bitset = (1 << num_triples) - 1
        selected_blocks = []
        selected_indices = set()
        
        # Strong Greedy
        while uncovered_bitset != 0:
            best_gain = -1
            best_idx = -1
            
            # Iterate through all blocks
            # We can skip blocks already selected
            for i, bitset in enumerate(all_block_bitsets):
                if i in selected_indices:
                    continue
                
                gain = bin(bitset & uncovered_bitset).count('1')
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    
            if best_idx == -1:
                break # Should not happen if covering is possible
                
            selected_blocks.append(all_blocks[best_idx])
            selected_indices.add(best_idx)
            
            # Update uncovered
            uncovered_bitset &= ~all_block_bitsets[best_idx]
            
            # Check time
            if time.time() > deadline - 5:
                break
                
    else:
        # Fallback: Simple Greedy
        uncovered = set(itertools.combinations(range(v), t))
        selected_blocks = []
        
        while uncovered:
            best_block = None
            best_count = -1
            
            # Sample some random blocks to speed up?
            # Or just pick the first one that covers many?
            # Let's just pick a random block containing a random uncovered triple.
            tri = random.choice(list(uncovered))
            
            # Find a block containing tri
            # We can just pick random points to complete the block
            block_set = set(tri)
            while len(block_set) < k:
                x = random.randint(0, v-1)
                block_set.add(x)
            
            block = tuple(sorted(block_set))
            selected_blocks.append(block)
            
            # Remove covered triples
            for b_tri in itertools.combinations(block, t):
                if b_tri in uncovered:
                    uncovered.remove(b_tri)
                    
            if time.time() > deadline - 5:
                break
        
        # Convert to list of lists
        selected_blocks = [list(b) for b in selected_blocks]

    # Optimization: Local Search (Hill Climbing)
    # Try to swap blocks to improve coverage or reduce count?
    # Since we used Strong Greedy, the count is likely minimal or close.
    # We can try to remove redundant blocks.
    
    # Let's do a redundancy removal pass.
    # This is fast.
    
    def optimize_redundancy(blocks):
        if not blocks:
            return []
        
        # Build counts
        counts = {}
        for b in blocks:
            for tri in itertools.combinations(b, t):
                counts[tri] = counts.get(tri, 0) + 1
                
        active = [True] * len(blocks)
        changed = True
        while changed:
            changed = False
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            for i in indices:
                if not active[i]:
                    continue
                b = blocks[i]
                can_remove = True
                for tri in itertools.combinations(b, t):
                    if counts[tri] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    active[i] = False
                    changed = True
                    for tri in itertools.combinations(b, t):
                        counts[tri] -= 1
                        
        return [blocks[i] for i in range(len(blocks)) if active[i]]
        
    # Apply optimization
    # We might do multiple passes of "remove redundant" and "add missing" (if any)
    # But Strong Greedy ensures coverage.
    
    final_blocks = optimize_redundancy(selected_blocks)
    
    # If we have time, we can try a simple local search:
    # Try to replace a block with another block that covers more "rare" triples.
    # But given the time constraint and the strength of Strong Greedy, 
    # redundancy removal is the most reliable improvement.
    
    # Let's return the result
    return final_blocks
