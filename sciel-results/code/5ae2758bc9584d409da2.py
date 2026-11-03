"""Covering design C(v,k,t) via fast greedy with precomputed triple maps and random restarts."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict
from math import comb

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using a fast greedy algorithm with
    precomputed block-triple incidence and random restarts.
    
    Returns a list of blocks (each a sorted list of k distinct integers in 0..v-1)
    such that every t-subset is contained in at least one block.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill
    
    # Precompute all t-subsets and map each to an index
    all_triples = list(itertools.combinations(range(v), t))
    n_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # For each point, which triples contain it?
    point_to_triples = defaultdict(list)
    for i, triple in enumerate(all_triples):
        for p in triple:
            point_to_triples[p].append(i)
    
    # Precompute for each pair of points (if t>=2), which triples contain both
    # This helps speed up gain computation
    if t >= 2:
        pair_to_triples = defaultdict(list)
        for i, triple in enumerate(all_triples):
            for a in range(len(triple)):
                for b in range(a+1, len(triple)):
                    pair = (triple[a], triple[b])
                    pair_to_triples[pair].append(i)
    
    def greedy_cover(uncovered_set, rng=None):
        """
        Greedily build a covering starting from the given set of uncovered triple indices.
        Returns the list of blocks added.
        """
        if rng is None:
            rng = random.Random()
        
        blocks = []
        uncovered = set(uncovered_set)
        
        while uncovered:
            # Pick a random uncovered triple to start the block
            seed_triple_idx = rng.choice(list(uncovered))
            seed_triple = all_triples[seed_triple_idx]
            
            # Start the block with one point from the seed triple
            block = [seed_triple[0]]
            
            while len(block) < k:
                # For each candidate point not in block, compute gain
                best_point = -1
                best_gain = -1
                best_tiebreak = -1
                
                for x in range(v):
                    if x in block:
                        continue
                    
                    # Count uncovered triples that would be covered by adding x
                    # A triple is covered if it has >= t-1 points in block and includes x
                    gain = 0
                    
                    if t == 1:
                        # Every triple is just a point; adding x covers triple (x,) if uncovered
                        idx = triple_to_idx.get((x,))
                        if idx is not None and idx in uncovered:
                            gain = 1
                    elif t == 2:
                        # Triples are pairs; adding x covers pairs (x, y) where y in block
                        for y in block:
                            pair = (min(x, y), max(x, y))
                            idx = triple_to_idx.get(pair)
                            if idx is not None and idx in uncovered:
                                gain += 1
                    elif t == 3:
                        # Triples are triples; adding x covers triples (x, y, z) where {y,z} subset of block
                        for y in block:
                            for z in block:
                                if y == z:
                                    continue
                                triple = tuple(sorted((x, y, z)))
                                idx = triple_to_idx.get(triple)
                                if idx is not None and idx in uncovered:
                                    gain += 1
                    else:
                        # General case: check all t-subsets of block ∪ {x} that include x
                        for combo in itertools.combinations(block, t-1):
                            triple = tuple(sorted(combo + (x,)))
                            idx = triple_to_idx.get(triple)
                            if idx is not None and idx in uncovered:
                                gain += 1
                    
                    if gain > best_gain or (gain == best_gain and x < best_tiebreak):
                        best_gain = gain
                        best_point = x
                        best_tiebreak = x
                
                if best_point == -1:
                    candidates = [x for x in range(v) if x not in block]
                    best_point = rng.choice(candidates)
                
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered set: remove all t-subsets of this block
            for combo in itertools.combinations(block, t):
                idx = triple_to_idx.get(combo)
                if idx is not None:
                    uncovered.discard(idx)
        
        return blocks
    
    # Strategy: Try multiple random restarts
    best_blocks = None
    best_count = float('inf')
    
    # Theoretical lower bound
    lb = (comb(v, t) + comb(k, t) - 1) // comb(k, t)
    
    # Number of restarts based on time budget
    # Estimate time per restart
    num_restarts = 10
    base_seed = 42
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:  # Leave 5s margin
            break
        
        rng = random.Random(base_seed + restart * 1000)
        
        blocks = greedy_cover(set(range(n_triples)), rng)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
        
        if best_count <= lb:
            break
    
    if best_blocks is None:
        best_blocks = greedy_cover(set(range(n_triples)))
    
    return best_blocks

# EVOLVE-BLOCK-END
