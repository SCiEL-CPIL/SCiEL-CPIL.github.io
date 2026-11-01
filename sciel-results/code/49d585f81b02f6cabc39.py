import itertools
import time
import random
from collections import defaultdict

def _greedy_build(v, k, t, seed, time_budget=None):
    """
    Greedy construction: repeatedly pick the block that covers the most uncovered t-subsets.
    Returns list of blocks.
    """
    rng = random.Random(seed)
    all_points = list(range(v))
    
    # Pre-compute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    uncovered = set(all_t_subs)
    blocks = []
    
    if not uncovered:
        return []
    
    # For speed, we can precompute for each pair (or (t-1)-subset) which points complete it
    # But for simplicity and generality, we'll compute gains on the fly.
    
    # Optimization: For t=3, v=24, we can use a smarter approach.
    # Let's stick to the general greedy but optimize the inner loop.
    
    # We maintain a count of how many times each t-subset is covered? No, we just need to know if it's uncovered.
    # The bottleneck is finding the best block.
    
    # Strategy:
    # 1. Pick a random uncovered t-subset to seed the block? Or pick the "best" block globally?
    # Global best block search is expensive: C(24,7) ~ 346k blocks.
    # For each block, count how many uncovered t-subsets it contains.
    # This is 346k * 35 checks = ~12M operations per step.
    # With ~70 steps, that's 840M operations. Too slow in Python.
    
    # Better strategy: Greedy with local search or randomized greedy.
    # The current seed program uses randomized greedy:
    # 1. Pick random uncovered triple.
    # 2. Build a block around it by greedily adding points that cover the most uncovered triples.
    
    # Let's improve the randomized greedy:
    # - Use a better heuristic for adding points.
    # - Use multiple restarts.
    # - Add a local search phase to remove redundant blocks or swap blocks to reduce count.
    
    while uncovered:
        # Pick a random uncovered t-subset
        start_sub = rng.choice(list(uncovered))
        block = list(start_sub)
        block_set = set(block)
        
        while len(block) < k:
            best_gain = -1
            candidates = []
            
            for x in all_points:
                if x in block_set:
                    continue
                
                # Calculate gain: how many uncovered t-subsets are covered by adding x
                gain = 0
                if t == 3:
                    # Pairs in block
                    # We can precompute pairs of block?
                    # len(block) is small (up to 6), so C(6,2)=15 pairs.
                    for i in range(len(block)):
                        for j in range(i+1, len(block)):
                            tri = (block[i], block[j], x)
                            # Need to sort for set lookup
                            tri_sorted = tuple(sorted(tri))
                            if tri_sorted in uncovered:
                                gain += 1
                else:
                    for comb in itertools.combinations(block, t-1):
                        tri = tuple(sorted(list(comb) + [x]))
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
                # Fallback: pick any point not in block
                next_point = rng.choice([x for x in all_points if x not in block_set])
            
            block.append(next_point)
            block_set.add(next_point)
        
        # Sort block for consistency
        block.sort()
        blocks.append(block)
        
        # Update uncovered set
        for tri in itertools.combinations(block, t):
            uncovered.discard(tuple(tri))
            
    return blocks

def _local_search(blocks, v, k, t, time_budget=None):
    """
    Local search: try to remove blocks that are redundant, or replace blocks with better ones.
    Returns a new list of blocks.
    """
    if not blocks:
        return blocks
    
    # First, remove redundant blocks
    # A block is redundant if all its t-subsets are covered by other blocks.
    # This is O(N^2 * C(k,t)) which might be slow, but N is ~70.
    # 70^2 * 35 = 171,500 checks. Fast enough.
    
    n = len(blocks)
    # Precompute t-subsets for each block
    block_t_subs = [list(itertools.combinations(b, t)) for b in blocks]
    
    # Coverage count: how many blocks cover each t-subset?
    # We only care if count >= 1.
    # Let's maintain a count of how many times each t-subset is covered.
    # Actually, we just need to know if a block is essential.
    
    # Iteratively remove redundant blocks
    changed = True
    while changed:
        changed = False
        n = len(blocks)
        # Build coverage counts
        cover_count = defaultdict(int)
        for b in blocks:
            for tri in itertools.combinations(b, t):
                cover_count[tri] += 1
        
        # Check each block
        to_remove = []
        for i, b in enumerate(blocks):
            is_redundant = True
            for tri in itertools.combinations(b, t):
                if cover_count[tri] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                to_remove.append(i)
        
        if to_remove:
            # Remove the first one found (or all? careful, removing one might make others essential)
            # Remove one at a time to be safe, or all if they are mutually redundant?
            # Removing all redundant blocks at once is safe if we recompute counts.
            # But if block A and B are both redundant, removing A might make B essential?
            # No, if A is redundant, its t-subsets are covered by others.
            # If we remove A, the coverage of those t-subsets decreases by 1.
            # If any of them was covered only by A and B, and B is also being removed, we lose coverage.
            # So we should remove one at a time, or check carefully.
            # Let's remove one at a time for simplicity and safety.
            idx = to_remove[0]
            blocks.pop(idx)
            changed = True
            
    return blocks

def _local_search_swap(blocks, v, k, t, time_budget=None):
    """
    Try to swap blocks to reduce the total number of blocks.
    This is more complex. Maybe just stick to removal for now.
    """
    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    # Run multiple restarts
    while time.time() < deadline:
        # Generate a candidate solution
        blocks = _greedy_build(v, k, t, seed)
        
        # Local search to remove redundant blocks
        blocks = _local_search(blocks, v, k, t)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        # If we found a very good solution, we might want to spend time on local search?
        # For now, just keep restarting.
        # We can stop early if we hit a theoretical lower bound?
        # Lower bound for C(24,7,3) is 69.
        if best_len == 69:
            break
            
        seed += 1
        
    return best_blocks
