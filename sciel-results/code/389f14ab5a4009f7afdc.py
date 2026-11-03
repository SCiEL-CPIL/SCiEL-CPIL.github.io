"""Covering design C(24,7,3) with fast greedy + restarts + local search."""
import itertools
import random
import time
from collections import defaultdict


def _build_block_triples_map(v, k):
    """Precompute for each k-subset which t-subsets it covers (for t=3)."""
    # We'll use a dictionary mapping frozenset -> list of triples
    # But for speed, we'll work with sorted tuples
    pass


def _greedy_cover(v, k, t, uncovered_triples, block_triple_cache, rng, deadline):
    """
    Greedy covering algorithm.
    uncovered_triples: set of frozensets of size t
    block_triple_cache: dict mapping frozenset(block) -> frozenset of triples it covers
    """
    blocks = []
    uncovered = set(uncovered_triples)
    
    while uncovered:
        if time.time() > deadline:
            return blocks  # Return what we have (may be incomplete)
        
        # Pick the first uncovered triple (deterministic within a restart)
        # Use a random choice for diversity
        target_triple = next(iter(uncovered))
        
        # Build a block greedily
        block = list(target_triple)
        block_set = frozenset(block)
        
        while len(block) < k:
            best_point = None
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Compute how many new uncovered triples this point would cover
                # For each (t-1)-subset of current block, check if (t-1-subset + x) is uncovered
                new_triples = 0
                for rest in itertools.combinations(block, t - 1):
                    triple = frozenset(rest + (x,))
                    if triple in uncovered:
                        new_triples += 1
                if new_triples > best_gain:
                    best_gain = new_triples
                    best_point = x
            
            if best_point is None:
                break
            block.append(best_point)
            block_set.add(best_point)
        
        if len(block) < k:
            break
            
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for triple in itertools.combinations(block, t):
            uncovered.discard(frozenset(triple))
    
    return blocks


def _local_improve(blocks, v, k, t, deadline, rng, max_iterations=500):
    """
    Try to reduce block count by merging or removing redundant blocks.
    Also try swapping points to improve coverage density.
    """
    if not blocks:
        return blocks
    
    blocks = [list(b) for b in blocks]
    n = len(blocks)
    
    # Build coverage map: for each triple, which blocks cover it
    triple_to_blocks = defaultdict(set)
    for i, block in enumerate(blocks):
        for triple in itertools.combinations(block, t):
            triple_to_blocks[frozenset(triple)].add(i)
    
    improved = True
    iteration = 0
    while improved and iteration < max_iterations:
        if time.time() > deadline:
            break
        improved = False
        iteration += 1
        
        # Try to remove a block if all its triples are covered by other blocks
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            removable = True
            for triple in itertools.combinations(blocks[i], t):
                fs = frozenset(triple)
                covering = triple_to_blocks.get(fs, set())
                if i not in covering or len(covering) == 1:
                    removable = False
                    break
            if removable:
                # Remove block i
                for triple in itertools.combinations(blocks[i], t):
                    triple_to_blocks[frozenset(triple)].discard(i)
                del blocks[i]
                # Re-index
                for j in range(len(blocks)):
                    for triple in itertools.combinations(blocks[j], t):
                        fs = frozenset(triple)
                        triple_to_blocks[fs] = {x - 1 if x > i else x for x in triple_to_blocks[fs]}
                improved = True
                break
    
    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5 second safety margin
    
    all_triples = list(itertools.combinations(range(v), t))
    all_triples_fs = [frozenset(tr) for tr in all_triples]
    
    best_blocks = None
    best_count = float('inf')
    
    rng = random.Random(42)
    
    # Multiple random restarts with different strategies
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        uncovered = set(all_triples_fs)
        
        # Randomize the order of processing uncovered triples
        # We'll do a randomized greedy
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                break
            
            # Pick a random uncovered triple as seed
            target = next(iter(uncovered))
            block = list(target)
            block_set = set(block)
            
            while len(block) < k:
                best_point = None
                best_gain = -1
                # Sample candidates for speed if too many
                candidates = [x for x in range(v) if x not in block_set]
                rng.shuffle(candidates)
                for x in candidates:
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        triple = frozenset(rest + (x,))
                        if triple in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                    elif gain == best_gain and best_point is not None:
                        # Tie-break randomly
                        if rng.random() < 0.5:
                            best_point = x
                
                if best_point is None or best_gain <= 0:
                    break
                block.append(best_point)
                block_set.add(best_point)
            
            if len(block) < k:
                break
                
            block.sort()
            blocks.append(block)
            
            for triple in itertools.combinations(block, t):
                uncovered.discard(frozenset(triple))
        
        if not uncovered and len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
        
        # If we have time, try local improvement on the best solution
        if best_blocks and time.time() < deadline - 5:
            improved = _local_improve(best_blocks, v, k, t, deadline, rng, max_iterations=100)
            if len(improved) < best_count:
                best_count = len(improved)
                best_blocks = improved
    
    if best_blocks is None:
        # Fallback: simple deterministic greedy
        uncovered = set(all_triples_fs)
        blocks = []
        while uncovered:
            target = min(uncovered)
            block = list(target)
            block_set = set(block)
            while len(block) < k:
                best_point = None
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = sum(1 for rest in itertools.combinations(block, t-1) 
                               if frozenset(rest + (x,)) in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point is None:
                    break
                block.append(best_point)
                block_set.add(best_point)
            if len(block) < k:
                break
            block.sort()
            blocks.append(block)
            for triple in itertools.combinations(block, t):
                uncovered.discard(frozenset(triple))
        best_blocks = blocks
    
    return best_blocks
