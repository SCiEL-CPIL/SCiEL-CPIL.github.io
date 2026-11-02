"""Optimized covering design search for C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

# Precompute all 3-subsets of 24 points and map them to indices
_ALL_TRIPLES = list(itertools.combinations(range(24), 3))
_NUM_TRIPLES = len(_ALL_TRIPLES)
_TRIPLE_TO_IDX = {t: i for i, t in enumerate(_ALL_TRIPLES)}

# Precompute triple indices for each possible block
# For each 7-element block, precompute which triples it covers
_BLOCK_TRIPLE_CACHE = {}

def _get_block_triples(block):
    """Return the frozenset of triple-indices covered by a 7-element block."""
    if block in _BLOCK_TRIPLE_CACHE:
        return _BLOCK_TRIPLE_CACHE[block]
    covered = frozenset(_TRIPLE_TO_IDX[combo] for combo in itertools.combinations(block, 3))
    _BLOCK_TRIPLE_CACHE[block] = covered
    return covered

def _get_block_triples_list(block):
    """Return list of triple-indices covered by a 7-element block."""
    return list(_get_block_triples(block))

# Precompute for each pair (a,b) the list of triple indices (a,b,x) for all x
# This helps in fast gain computation
_PAIR_TRIPLE_IDX = {}
for a in range(24):
    for b in range(a+1, 24):
        _PAIR_TRIPLE_IDX[(a, b)] = [_TRIPLE_TO_IDX[tuple(sorted((a, b, x)))] for x in range(24)]

def _greedy_covering(v=24, k=7, t=3, seed=0, time_limit=5, weighted=False):
    """Greedy covering with randomized tie-breaking and optional weighting."""
    rng = random.Random(seed)
    uncovered = np.ones(_NUM_TRIPLES, dtype=np.uint8)
    blocks = []
    start_time = time.time()
    
    while np.any(uncovered):
        if time.time() - start_time > time_limit:
            break
        
        # Find uncovered triples
        uncovered_indices = np.nonzero(uncovered)[0]
        if len(uncovered_indices) == 0:
            break
        
        # Pick a random uncovered triple to start the block
        start_triple_idx = rng.choice(uncovered_indices)
        start_triple = _ALL_TRIPLES[start_triple_idx]
        
        block = list(start_triple)
        block_set = set(block)
        
        # Greedily add points to maximize coverage
        while len(block) < k:
            if time.time() - start_time > time_limit:
                break
            
            # Compute gain for each candidate point
            # For each x not in block, gain = number of uncovered triples (a,b,x) 
            # where a,b are in block
            best_gain = -1
            best_ties = []
            
            # Vectorized: for each candidate x, compute how many uncovered triples it covers
            # A triple (a,b,x) is covered if a,b are both in block and x is not in block
            # We need: for each x, count of pairs (a,b) in block where triple (a,b,x) is uncovered
            
            # Build the set of pairs in block
            block_pairs = list(itertools.combinations(block, 2))
            
            # For each candidate x, compute gain
            gains = np.zeros(v, dtype=np.int32)
            for (a, b) in block_pairs:
                # Get all triple indices for (a,b,x) for x in 0..23
                triple_idxs = _PAIR_TRIPLE_IDX[(a, b)]
                # Check which are uncovered
                gains += uncovered[triple_idxs]
            
            # Zero out gains for points already in block
            for x in block:
                gains[x] = -1
            
            # Find best point(s)
            max_gain = np.max(gains)
            if max_gain <= 0:
                # No gain possible, pick random
                candidates = [x for x in range(v) if x not in block_set]
                if not candidates:
                    break
                best_point = rng.choice(candidates)
            else:
                # Weighted selection: prioritize points that cover more uncovered triples
                # with some randomness for diversity
                best_ties = list(np.nonzero(gains == max_gain)[0])
                if weighted:
                    # Use softmax-like weighting
                    weights = np.exp(gains / max(1, max_gain))
                    weights[block_set] = 0
                    total = np.sum(weights)
                    if total > 0:
                        weights /= total
                        best_point = rng.choices(range(v), weights=weights)[0]
                    else:
                        best_point = rng.choice(best_ties)
                else:
                    best_point = rng.choice(best_ties)
            
            block.append(best_point)
            block_set.add(best_point)
        
        if len(block) < k:
            break
        
        block.sort()
        blocks.append(tuple(block))
        
        # Update uncovered
        for t_idx in _get_block_triples(tuple(block)):
            uncovered[t_idx] = 0
    
    return blocks


def _is_valid_covering(blocks, v=24, k=7, t=3):
    """Check if blocks form a valid covering."""
    if not blocks:
        return False
    covered = np.zeros(_NUM_TRIPLES, dtype=np.uint8)
    for block in blocks:
        if len(block) != k:
            return False
        if len(set(block)) != k:
            return False
        for t_idx in _get_block_triples(tuple(block)):
            covered[t_idx] = 1
    return np.all(covered)


def _remove_redundant(blocks):
    """Remove blocks that don't cover any unique triple."""
    if not blocks:
        return blocks
    
    n = len(blocks)
    triple_count = np.zeros(_NUM_TRIPLES, dtype=np.int32)
    block_triples_list = []
    for block in blocks:
        t_set = _get_block_triples(tuple(block))
        block_triples_list.append(t_set)
        for t_idx in t_set:
            triple_count[t_idx] += 1
    
    # A block is redundant if all its triples are covered by other blocks
    essential = [True] * n
    for i in range(n):
        has_unique = False
        for t_idx in block_triples_list[i]:
            if triple_count[t_idx] == 1:
                has_unique = True
                break
        if not has_unique:
            essential[i] = False
    
    if all(essential):
        return blocks
    
    return [blocks[i] for i in range(n) if essential[i]]


def _local_search(blocks, seed=0, time_limit=30):
    """Local search to remove redundant blocks and try swaps."""
    rng = random.Random(seed)
    blocks = list(blocks)
    start_time = time.time()
    
    # First pass: remove redundant blocks iteratively
    prev_len = len(blocks) + 1
    while len(blocks) < prev_len:
        prev_len = len(blocks)
        blocks = _remove_redundant(blocks)
    
    # Precompute coverage structure
    def compute_coverage(block_list):
        covered = np.zeros(_NUM_TRIPLES, dtype=np.uint8)
        for block in block_list:
            for t_idx in _get_block_triples(tuple(block)):
                covered[t_idx] = 1
        return covered
    
    # Try to remove blocks and repair
    for _ in range(200):
        if time.time() - start_time > time_limit:
            break
        if not blocks:
            break
        
        # Try removing a random block
        idx = rng.randrange(len(blocks))
        candidate = blocks[:idx] + blocks[idx+1:]
        
        if _is_valid_covering(candidate):
            blocks = candidate
        else:
            # Try to repair by adding a block that covers the most uncovered triples
            covered = compute_coverage(candidate)
            uncovered_count = np.sum(1 - covered)
            
            if uncovered_count == 0:
                blocks = candidate
                continue
            
            # Greedily add one block to cover most uncovered triples
            best_block = None
            best_count = 0
            
            # Try some candidate blocks - generate smart candidates
            candidates_to_try = []
            
            # Generate candidates based on uncovered triples
            uncovered_indices = np.nonzero(1 - covered)[0]
            if len(uncovered_indices) > 0:
                # Pick a few random uncovered triples and build blocks around them
                for _ in range(min(30, len(uncovered_indices))):
                    t_idx = rng.choice(uncovered_indices)
                    triple = _ALL_TRIPLES[t_idx]
                    # Build a block containing this triple
                    remaining = [x for x in range(24) if x not in triple]
                    if len(remaining) >= 4:
                        extra = rng.sample(remaining, 4)
                        cand = tuple(sorted(list(triple) + extra))
                        candidates_to_try.append(cand)
            
            # Also try random blocks and existing blocks
            for _ in range(20):
                candidates_to_try.append(tuple(sorted(rng.sample(range(24), 7))))
            candidates_to_try.extend(blocks)
            
            for cand in candidates_to_try:
                count = 0
                for t_idx in _get_block_triples(cand):
                    if not covered[t_idx]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_block = cand
            
            if best_block is not None and best_count > 0:
                new_candidate = candidate + [best_block]
                if _is_valid_covering(new_candidate) and len(new_candidate) < len(blocks):
                    blocks = new_candidate
    
    # Final redundant removal
    blocks = _remove_redundant(blocks)
    
    return blocks


def _simulated_annealing(blocks, seed=0, time_limit=40):
    """Simulated annealing to minimize block count."""
    rng = random.Random(seed)
    blocks = list(blocks)
    start_time = time.time()
    
    # Remove redundant first
    blocks = _remove_redundant(blocks)
    
    best_blocks = list(blocks)
    best_count = len(blocks)
    
    temperature = 3.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        if temperature < 0.1:
            break
        
        # Generate a perturbation
        perturbation_type = rng.random()
        
        if perturbation_type < 0.4 and len(blocks) > 1:
            # Remove a random block
            idx = rng.randrange(len(blocks))
            candidate = blocks[:idx] + blocks[idx+1:]
        elif perturbation_type < 0.8:
            # Swap a block with a random 7-subset
            idx = rng.randrange(len(blocks))
            new_block = tuple(sorted(rng.sample(range(24), 7)))
            candidate = blocks[:idx] + [new_block] + blocks[idx+1:]
        else:
            # Add a random block
            new_block = tuple(sorted(rng.sample(range(24), 7)))
            candidate = blocks + [new_block]
        
        # Evaluate
        if not _is_valid_covering(candidate):
            continue
        
        delta = len(candidate) - len(blocks)
        
        if delta <= 0:
            blocks = candidate
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = list(blocks)
        else:
            if rng.random() < np.exp(-delta / temperature):
                blocks = candidate
        
        temperature *= cooling
    
    best_blocks = _remove_redundant(best_blocks)
    return best_blocks


def construct(v, k, t):
    """Construct a covering design C(v,k,t)."""
    if v != 24 or k != 7 or t != 3:
        return _greedy_covering(v, k, t, seed=0, time_limit=5)
    
    deadline = time.time() + 110
    
    best_blocks = None
    best_count = float('inf')
    
    # Phase 1: Multiple greedy constructions with different seeds
    # Use both standard and weighted greedy
    for seed in range(20):
        if time.time() > deadline - 50:
            break
        weighted = (seed % 2 == 0)
        blocks = _greedy_covering(24, 7, 3, seed=seed, time_limit=2, weighted=weighted)
        if blocks and _is_valid_covering(blocks):
            blocks = _remove_redundant(blocks)
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
    
    if best_blocks is None:
        best_blocks = _greedy_covering(24, 7, 3, seed=0, time_limit=5)
        if not best_blocks or not _is_valid_covering(best_blocks):
            return []
    
    # Phase 2: Local search
    if time.time() < deadline - 20:
        ls_blocks = _local_search(best_blocks, seed=0, time_limit=15)
        if _is_valid_covering(ls_blocks) and len(ls_blocks) < best_count:
            best_blocks = ls_blocks
            best_count = len(ls_blocks)
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 5:
        sa_blocks = _simulated_annealing(best_blocks, seed=0, time_limit=20)
        if _is_valid_covering(sa_blocks) and len(sa_blocks) < best_count:
            best_blocks = sa_blocks
            best_count = len(sa_blocks)
    
    if not _is_valid_covering(best_blocks):
        best_blocks = _greedy_covering(24, 7, 3, seed=0, time_limit=5)
    
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
