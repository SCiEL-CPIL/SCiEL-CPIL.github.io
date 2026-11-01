"""Covering design C(v,k,t) via randomized greedy with multi-start and local refinement."""
import itertools
import random
import time

# EVOLVE-BLOCK-START

def _count_uncovered_in_block(block, uncovered, t):
    """Count how many uncovered t-subsets are contained in block."""
    cnt = 0
    for combo in itertools.combinations(block, t):
        if combo in uncovered:
            cnt += 1
    return cnt


def _greedy_construct(v, k, t, rng, time_limit):
    """
    Greedy covering with randomized tie-breaking.
    Returns (blocks, uncovered_set).
    """
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    deadline = time.time() + time_limit

    while uncovered and time.time() < deadline:
        # Pick the uncovered t-subset to cover: choose one at random among the first few
        # to introduce diversity, but bias towards "hard" ones is hard to define simply.
        # Just pick a random uncovered subset.
        target = rng.choice(list(uncovered))
        
        # Start block with the target subset
        block = list(target)
        
        # Add points one by one to maximize coverage of remaining uncovered subsets
        while len(block) < k:
            best_gain = -1
            best_points = []
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                # For each (t-1)-subset of current block, check if adding x covers an uncovered t-set
                for rest in itertools.combinations(block, t - 1):
                    candidate = tuple(sorted(rest + (x,)))
                    if candidate in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_points = [x]
                elif gain == best_gain and gain > 0:
                    best_points.append(x)
            
            if not best_points:
                # No point adds coverage; pick a random unused point
                unused = [x for x in range(v) if x not in block]
                if not unused:
                    break
                block.append(rng.choice(unused))
            else:
                block.append(rng.choice(best_points))
        
        block.sort()
        blocks.append(block)
        
        # Remove covered t-subsets
        for combo in itertools.combinations(block, t):
            uncovered.discard(combo)
    
    return blocks, uncovered


def _local_refine(blocks, v, k, t, rng, time_limit):
    """
    Try to remove redundant blocks or swap points to improve coverage.
    A block is redundant if all its t-subsets are covered by other blocks.
    """
    deadline = time.time() + time_limit
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Build coverage count for each t-subset
        cover_count = {}
        for combo in itertools.combinations(range(v), t):
            cover_count[combo] = 0
        
        for block in blocks:
            for combo in itertools.combinations(block, t):
                cover_count[combo] += 1
        
        # Try to remove redundant blocks
        new_blocks = []
        for i, block in enumerate(blocks):
            if time.time() > deadline:
                break
            # Check if all t-subsets of this block are covered by other blocks
            redundant = True
            for combo in itertools.combinations(block, t):
                if cover_count[combo] <= 1:
                    redundant = False
                    break
            if not redundant:
                new_blocks.append(block)
                # Update cover counts for subsequent checks
                for combo in itertools.combinations(block, t):
                    cover_count[combo] -= 1
                # Wait, we need to be careful. Let's just rebuild.
            else:
                improved = True
                for combo in itertools.combinations(block, t):
                    cover_count[combo] -= 1
        
        if improved:
            blocks = new_blocks
    
    return blocks


def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using multi-start greedy with local refinement.
    """
    start_time = time.time()
    total_deadline = start_time + 115  # Safety margin for 120s limit
    
    rng = random.Random(42)
    
    best_blocks = None
    best_count = float('inf')
    
    # Multi-start greedy
    # Estimate how many starts we can afford
    # Each greedy run takes some time. Let's try a few.
    num_starts = 5
    per_start_time = (total_deadline - start_time) / num_starts
    
    for i in range(num_starts):
        if time.time() > total_deadline - 30:  # Leave time for refinement
            break
        
        remaining_time = min(per_start_time, total_deadline - time.time() - 30)
        if remaining_time < 1:
            break
            
        blocks, uncovered = _greedy_construct(v, k, t, rng, remaining_time)
        
        if not uncovered:
            # Valid covering
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks[:]
        
        # Local refinement on the best so far
        if best_blocks is not None and time.time() < total_deadline - 10:
            refined = _local_refine(best_blocks[:], v, k, t, rng, total_deadline - time.time() - 5)
            if len(refined) < best_count:
                # Verify refinement is still valid
                all_covered = set(itertools.combinations(range(v), t))
                covered = set()
                for block in refined:
                    for combo in itertools.combinations(block, t):
                        covered.add(combo)
                if all_covered.issubset(covered):
                    best_count = len(refined)
                    best_blocks = refined
    
    # If we didn't find a valid covering, fall back to a simple greedy
    if best_blocks is None:
        # Run one more greedy with full remaining time
        remaining = total_deadline - time.time() - 2
        if remaining > 0:
            blocks, uncovered = _greedy_construct(v, k, t, rng, remaining)
            best_blocks = blocks
    
    return best_blocks if best_blocks is not None else []

# EVOLVE-BLOCK-END
