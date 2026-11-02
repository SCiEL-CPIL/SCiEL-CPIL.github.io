import itertools
import time
import random
from collections import defaultdict

def _greedy_random(v, k, t, seed):
    rng = random.Random(seed)
    all_points = list(range(v))
    
    # Pre-compute all t-subsets as frozensets for O(1) membership
    uncovered = set(frozenset(combo) for combo in itertools.combinations(range(v), t))
    blocks = []
    
    if not uncovered:
        return []

    while uncovered:
        # Pick a random uncovered triple to start the block
        start_triple = rng.choice(list(uncovered))
        block = list(start_triple)
        block_set = set(block)
        
        while len(block) < k:
            best_gain = -1
            candidates = []
            
            for x in all_points:
                if x in block_set:
                    continue
                
                # Calculate gain: how many uncovered triples are covered by adding x
                gain = 0
                if t == 3:
                    for i in range(len(block)):
                        for j in range(i+1, len(block)):
                            tri = frozenset((block[i], block[j], x))
                            if tri in uncovered:
                                gain += 1
                else:
                    for comb in itertools.combinations(block, t-1):
                        tri = frozenset(list(comb) + [x])
                        if tri in uncovered:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    candidates = [x]
                elif gain == best_gain:
                    candidates.append(x)
            
            # Pick a random candidate from the best ones
            if candidates:
                next_point = rng.choice(candidates)
            else:
                next_point = rng.choice([x for x in all_points if x not in block_set])
            
            block.append(next_point)
            block_set.add(next_point)
        
        # Sort block for consistency
        block.sort()
        blocks.append(block)
        
        # Update uncovered set
        for tri in itertools.combinations(block, t):
            uncovered.discard(frozenset(tri))
            
    return blocks

def _hypergraph_peel(v, k, t, seed, deadline):
    """
    Hypergraph peeling algorithm:
    1. Track degree of each point in residual uncovered hypergraph
    2. Iteratively pick the point with minimum degree (most "isolated" in uncovered)
    3. Build a block around that point that maximally covers triples incident to it
    4. This exploits structural sparsity to escape uniform redundancy traps
    """
    rng = random.Random(seed)
    all_points = list(range(v))
    
    # Initialize uncovered set
    uncovered = set(frozenset(combo) for combo in itertools.combinations(range(v), t))
    if not uncovered:
        return []
    
    # Track degree: for each point, count how many uncovered triples contain it
    point_degree = defaultdict(int)
    for tri in uncovered:
        for p in tri:
            point_degree[p] += 1
    
    # Precompute: for each pair of points, which triples contain both
    # This helps efficiently find triples incident to a point
    point_triples = defaultdict(set)  # point -> set of uncovered triples containing it
    for tri in uncovered:
        for p in tri:
            point_triples[p].add(tri)
    
    blocks = []
    
    while uncovered and time.time() < deadline:
        # Find point with minimum degree among points with degree > 0
        active_points = [p for p in all_points if point_degree[p] > 0]
        if not active_points:
            break
        
        # Pick minimum degree point with some randomness for tie-breaking
        min_deg = min(point_degree[p] for p in active_points)
        min_candidates = [p for p in active_points if point_degree[p] == min_deg]
        anchor = rng.choice(min_candidates)
        
        # Get all uncovered triples containing the anchor
        anchor_triples = list(point_triples[anchor])
        if not anchor_triples:
            continue
        
        # Build block starting with anchor
        block = [anchor]
        block_set = {anchor}
        
        # Greedily add points to maximize coverage of anchor_triples
        # We want to cover as many of the anchor's uncovered triples as possible
        remaining_triples = set(anchor_triples)
        
        while len(block) < k and remaining_triples:
            best_gain = -1
            candidates = []
            
            for x in all_points:
                if x in block_set:
                    continue
                
                # Calculate how many remaining triples would be covered by adding x
                gain = 0
                if t == 3:
                    for i in range(len(block)):
                        for j in range(i+1, len(block)):
                            tri = frozenset((block[i], block[j], x))
                            if tri in remaining_triples:
                                gain += 1
                else:
                    for comb in itertools.combinations(block, t-1):
                        tri = frozenset(list(comb) + [x])
                        if tri in remaining_triples:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    candidates = [x]
                elif gain == best_gain:
                    candidates.append(x)
            
            if candidates and best_gain > 0:
                next_point = rng.choice(candidates)
            else:
                # No positive gain, pick randomly from remaining points
                available = [x for x in all_points if x not in block_set]
                if not available:
                    break
                next_point = rng.choice(available)
            
            block.append(next_point)
            block_set.add(next_point)
            
            # Update remaining_triples
            new_remaining = set()
            for tri in remaining_triples:
                if not block_set.issuperset(tri):
                    new_remaining.add(tri)
            remaining_triples = new_remaining
        
        # Sort block for consistency
        block.sort()
        blocks.append(block)
        
        # Update uncovered set and degrees
        for tri in itertools.combinations(block, t):
            tri_fs = frozenset(tri)
            if tri_fs in uncovered:
                uncovered.discard(tri_fs)
                for p in tri_fs:
                    point_degree[p] -= 1
                    point_triples[p].discard(tri_fs)
    
    return blocks

def _local_search(v, k, t, blocks, rng, deadline):
    """Try to remove blocks if the covering remains valid."""
    blocks = [list(b) for b in blocks]
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        # Shuffle block order for randomness
        rng.shuffle(blocks)
        
        for i in range(len(blocks)):
            if time.time() >= deadline:
                break
            
            # Try removing block i
            candidate_blocks = blocks[:i] + blocks[i+1:]
            
            # Check if covering is still valid
            # Build set of all t-subsets covered by candidate_blocks
            covered = set()
            for b in candidate_blocks:
                for tri in itertools.combinations(b, t):
                    covered.add(frozenset(tri))
            
            # Check if all t-subsets are covered
            all_subs = set(frozenset(combo) for combo in itertools.combinations(range(v), t))
            if all_subs.issubset(covered):
                # Removal successful
                blocks = candidate_blocks
                improved = True
                break
    
    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    # Run multiple restarts
    while time.time() < deadline:
        # Check if we have enough time for a full greedy + local search
        if time.time() + 10.0 > deadline:
            break
        
        # Alternate between greedy and hypergraph peeling strategies
        if seed % 2 == 0:
            blocks = _greedy_random(v, k, t, seed)
        else:
            blocks = _hypergraph_peel(v, k, t, seed, deadline)
        
        # Apply local search if we have time
        if time.time() + 5.0 < deadline:
            rng = random.Random(seed + 1000000)
            blocks = _local_search(v, k, t, blocks, rng, deadline)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
        # Stop early if we hit theoretical lower bound
        if best_len == 69:
            break
            
        seed += 1
    
    return best_blocks
