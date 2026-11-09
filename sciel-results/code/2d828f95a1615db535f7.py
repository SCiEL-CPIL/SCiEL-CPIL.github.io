"""Covering design C(24,7,3) - improved greedy with local search."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def construct(v, k, t):
    deadline = time.time() + 140.0  # safety margin before 150s kill
    rng = random.Random(42)
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # For each pair (since t=3), precompute which triples contain that pair
    # This helps with fast gain computation
    # For t=3: a triple (a,b,c) contains pairs (a,b), (a,c), (b,c)
    
    def greedy_cover(seed_offset=0):
        """Randomized greedy set cover with reverse indexing."""
        rng_local = random.Random(42 + seed_offset)
        uncovered = set(range(n_tsubs))
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                return None
            
            # Pick a random uncovered triple to start the block
            start_triple = all_tsubs[rng_local.choice(sorted(uncovered))]
            block = list(start_triple)
            
            while len(block) < k:
                if time.time() > deadline:
                    return None
                
                # Compute gain for each candidate point
                best_gain = -1
                best_point = None
                # Precompute which (t-1)-subsets of current block exist
                block_set = set(block)
                
                # For each candidate point not in block, count how many uncovered triples it would complete
                gains = {}
                for x in range(v):
                    if x in block_set:
                        continue
                    # Count uncovered triples that contain x and (t-1) elements from block
                    g = 0
                    for rest in itertools.combinations(block, t - 1):
                        triple = tuple(sorted(rest + (x,)))
                        idx = tsub_to_idx[triple]
                        if idx in uncovered:
                            g += 1
                    gains[x] = g
                
                # Pick point with max gain (break ties randomly)
                max_g = max(gains.values())
                candidates = [x for x, g in gains.items() if g == max_g]
                best_point = rng_local.choice(candidates)
                block.append(best_point)
            
            block.sort()
            blocks.append(tuple(block))
            
            # Update uncovered
            for triple in itertools.combinations(block, t):
                idx = tsub_to_idx[triple]
                uncovered.discard(idx)
        
        return [list(b) for b in blocks]
    
    def local_search_remove(blocks, max_iterations=50):
        """Try to remove blocks that are redundant (all their triples covered by other blocks)."""
        if not blocks:
            return blocks
        
        blocks = [list(b) for b in blocks]
        n_blocks = len(blocks)
        
        # Count coverage: for each triple, how many blocks cover it
        triple_count = {}
        for b in blocks:
            for triple in itertools.combinations(b, t):
                triple_count[triple] = triple_count.get(triple, 0) + 1
        
        removed = 0
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            progress = False
            for i in range(len(blocks)):
                # Check if block i is redundant
                redundant = True
                for triple in itertools.combinations(blocks[i], t):
                    if triple_count.get(triple, 0) <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove block i
                    for triple in itertools.combinations(blocks[i], t):
                        triple_count[triple] -= 1
                    blocks.pop(i)
                    removed += 1
                    progress = True
                    break
            
            if not progress:
                break
        
        return blocks
    
    def local_search_swap(blocks, max_iterations=200):
        """Try to replace blocks with ones that cover more uncovered triples."""
        if not blocks:
            return blocks
        
        blocks = [list(b) for b in blocks]
        
        # Build coverage map
        triple_blocks = {}
        for i, b in enumerate(blocks):
            for triple in itertools.combinations(b, t):
                if triple not in triple_blocks:
                    triple_blocks[triple] = []
                triple_blocks[triple].append(i)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Find a block that could potentially be improved
            # Try replacing one point in a block to cover more currently-uncovered triples
            best_improvement = 0
            best_swap = None
            
            for i, b in enumerate(blocks):
                for pos in range(len(b)):
                    old_point = b[pos]
                    # Try each alternative point
                    for new_point in range(v):
                        if new_point == old_point:
                            continue
                        if new_point in b:
                            continue
                        
                        new_block = b[:pos] + [new_point] + b[pos+1:]
                        new_block_set = set(new_block)
                        
                        # Count new triples covered that weren't before
                        old_triples = set(itertools.combinations(b, t))
                        new_triples = set(itertools.combinations(new_block, t))
                        
                        gained = new_triples - old_triples
                        lost = old_triples - new_triples
                        
                        # Only accept if we gain more than we lose
                        # And we don't lose any triple that's only covered by this block
                        safe_loss = 0
                        for triple in lost:
                            covering = triple_blocks.get(triple, [])
                            if len(covering) <= 1:
                                safe_loss += 1
                        
                        net_gain = len(gained) - safe_loss
                        if net_gain > best_improvement:
                            best_improvement = net_gain
                            best_swap = (i, pos, old_point, new_point)
            
            if best_swap is None:
                break
            
            i, pos, old_point, new_point = best_swap
            blocks[i][pos] = new_point
            blocks[i].sort()
            
            # Update triple_blocks
            old_triples = set(itertools.combinations(blocks[i], t))
            for triple in old_triples:
                if i in triple_blocks.get(triple, []):
                    triple_blocks[triple].remove(i)
                if not triple_blocks[triple]:
                    del triple_blocks[triple]
            
            new_triples = set(itertools.combinations(blocks[i], t))
            for triple in new_triples:
                if triple not in triple_blocks:
                    triple_blocks[triple] = []
                triple_blocks[triple].append(i)
        
        return blocks
    
    # Main search loop
    best_blocks = None
    best_count = float('inf')
    
    # Try multiple greedy runs with different seeds
    n_trials = 0
    while time.time() < deadline - 5.0:
        blocks = greedy_cover(n_trials)
        if blocks is None:
            break
        
        # Apply local search
        blocks = local_search_remove(blocks, max_iterations=30)
        blocks = local_search_swap(blocks, max_iterations=50)
        blocks = local_search_remove(blocks, max_iterations=20)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
        
        n_trials += 1
        
        # If we've found a good solution, try more refinements
        if best_count <= 76 and n_trials >= 5:
            break
    
    if best_blocks is None:
        # Fallback: simple greedy
        blocks = greedy_cover(0)
        if blocks is None:
            # Very basic fallback
            uncovered = set(itertools.combinations(range(v), t))
            blocks = []
            while uncovered:
                block = list(min(uncovered))
                while len(block) < k:
                    best_gain = -1
                    best_point = None
                    for x in range(v):
                        if x in block:
                            continue
                        g = sum(1 for rest in itertools.combinations(block, t-1)
                                if tuple(sorted(rest + (x,))) in uncovered)
                        if g > best_gain:
                            best_gain = g
                            best_point = x
                    block.append(best_point)
                block.sort()
                blocks.append(block)
                uncovered.difference_update(itertools.combinations(block, t))
        return blocks
    
    return best_blocks
# EVOLVE-BLOCK-END
