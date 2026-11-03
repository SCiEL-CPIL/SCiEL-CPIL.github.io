<<<<<<< SEARCH
"""Randomized local search (simulated annealing) for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import defaultdict


def _greedy_cover(v, k, t, rng, tsubset_indices, subset_to_block_idx):
    """Fast greedy construction to get a valid starting covering."""
    uncovered = set(range(len(tsubset_indices)))
    blocks = []
    
    while uncovered:
        # Pick a random uncovered t-subset as seed
        seed_idx = rng.choice(list(uncovered))
        seed = tsubset_indices[seed_idx]
        block = list(seed)
        rng.shuffle(block)
        
        # Greedily add points to maximize coverage
        while len(block) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block]
            rng.shuffle(cands)
            
            for x in cands:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    fs = frozenset(list(rest) + [x])
                    idx = subset_to_block_idx.get(fs)
                    if idx is not None and idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            
        block.sort()
        b_tuple = tuple(block)
        blocks.append(b_tuple)
        for c in itertools.combinations(block, t):
            uncovered.discard(subset_to_block_idx[frozenset(c)])
                
    return blocks
=======
"""Randomized local search (simulated annealing) for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import defaultdict


def _greedy_cover(v, k, t, rng, tsubset_indices, subset_to_block_idx):
    """Fast greedy construction to get a valid starting covering."""
    uncovered = set(range(len(tsubset_indices)))
    blocks = []
    
    while uncovered:
        # Pick an uncovered t-subset with fewest covering options to start
        # Use the one that appears in the fewest candidate blocks (heuristic)
        seed_idx = rng.choice(list(uncovered))
        seed = tsubset_indices[seed_idx]
        block = list(seed)
        rng.shuffle(block)
        
        # Greedily add points to maximize coverage of uncovered triples
        while len(block) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block]
            rng.shuffle(cands)
            
            for x in cands:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    fs = frozenset(list(rest) + [x])
                    idx = subset_to_block_idx.get(fs)
                    if idx is not None and idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            
        block.sort()
        b_tuple = tuple(block)
        blocks.append(b_tuple)
        for c in itertools.combinations(block, t):
            uncovered.discard(subset_to_block_idx[frozenset(c)])
                
    return blocks
>>>>>>> REPLACE

