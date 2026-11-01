"""
Covering design C(19,8,5): greedy construction with local search refinement.
"""
import itertools
import time
import random
import sys


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit

    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    n_t = len(all_t_subsets)
    
    # Precompute for each point, which t-subsets it participates in (as a tuple of sorted indices)
    # Actually, we need: for a block (set of k points), which t-subsets are covered
    # Precompute: for each point x, list of t-subsets containing x
    point_to_subsets = [set() for _ in range(v)]
    for idx, sub in enumerate(all_t_subsets):
        for p in sub:
            point_to_subsets[p].add(idx)
    
    def gain_for_point(x, current_block_set, uncovered_set):
        """How many uncovered t-subsets would be covered by adding x to current_block_set."""
        if len(current_block_set) < t - 1:
            return 0
        count = 0
        # t-subsets covered by adding x to current_block_set:
        # pick (t-1) from current_block_set, plus x
        for combo in itertools.combinations(sorted(current_block_set), t - 1):
            tsub = tuple(sorted(combo + (x,)))
            if tsub in uncovered_set:
                count += 1
        return count

    def greedy_cover(uncovered, blocks, deadline):
        """Add blocks greedily until all t-subsets are covered or deadline hit."""
        while uncovered and time.time() < deadline:
            # Pick the least-covered uncovered t-subset (or just first)
            first = min(uncovered)
            block_set = set(first)
            
            while len(block_set) < k:
                # Find best point to add
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    g = gain_for_point(x, block_set, uncovered)
                    if g > best_gain:
                        best_gain = g
                        best_x = x
                block_set.add(best_x)
            
            block = sorted(block_set)
            blocks.append(block)
            
            # Remove covered t-subsets
            for tsub in itertools.combinations(block, t):
                uncovered.discard(tsub)
        
        return blocks

    def is_valid(blocks, all_t_subsets):
        """Check if every t-subset is covered."""
        covered = set()
        for block in blocks:
            for tsub in itertools.combinations(block, t):
                covered.add(tsub)
        return len(covered) == len(all_t_subsets)

    def local_search(blocks, deadline):
        """Try to remove redundant blocks and re-cover."""
        if time.time() > deadline:
            return blocks
        
        # Try removing each block and see if we can still cover everything
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline:
                    break
                # Try removing block i
                candidate = blocks[:i] + blocks[i+1:]
                # Check if all t-subsets still covered
                uncovered = set()
                covered = set()
                for block in candidate:
                    for tsub in itertools.combinations(block, t):
                        covered.add(tsub)
                all_set = set(all_t_subsets)
                uncovered = all_set - covered
                if not uncovered:
                    blocks = candidate
                    improved = True
                    break
        
        return blocks

    def randomized_greedy(uncovered, blocks, deadline, seed):
        """Greedy with random tie-breaking."""
        rng = random.Random(seed)
        while uncovered and time.time() < deadline:
            first = min(uncovered)
            block_set = set(first)
            
            while len(block_set) < k:
                candidates = []
                for x in range(v):
                    if x in block_set:
                        continue
                    g = gain_for_point(x, block_set, uncovered)
                    candidates.append((g, x))
                
                if candidates:
                    max_g = max(g for g, _ in candidates)
                    # Pick randomly among those with max gain, or with high gain
                    top = [x for g, x in candidates if g >= max_g]
                    if not top:
                        top = [x for g, x in candidates]
                    block_set.add(rng.choice(top))
            
            block = sorted(block_set)
            blocks.append(block)
            
            for tsub in itertools.combinations(block, t):
                uncovered.discard(tsub)
        
        return blocks

    # Initial greedy cover
    uncovered_initial = set(all_t_subsets)
    blocks = []
    greedy_cover(uncovered_initial, blocks, deadline)
    
    # Local search to remove redundant blocks
    blocks = local_search(blocks, deadline)
    
    # Try multiple randomized restarts if time permits
    best_blocks = blocks
    best_count = len(blocks)
    
    n_restarts = 0
    while time.time() < deadline - 5 and n_restarts < 20:
        n_restarts += 1
        seed = 42 + n_restarts
        uncovered_r = set(all_t_subsets)
        blocks_r = []
        randomized_greedy(uncovered_r, blocks_r, deadline, seed)
        blocks_r = local_search(blocks_r, deadline)
        
        if len(blocks_r) < best_count:
            best_blocks = blocks_r
            best_count = len(blocks_r)
    
    # Verify validity
    if not is_valid(best_blocks, all_t_subsets):
        # Fallback to initial greedy
        uncovered_f = set(all_t_subsets)
        best_blocks = []
        greedy_cover(uncovered_f, best_blocks, deadline)
    
    return best_blocks
