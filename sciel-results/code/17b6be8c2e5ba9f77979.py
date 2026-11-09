"""Covering design C(19,7,4) - Randomized local search with greedy initialization."""
import itertools
import random
import time
from collections import defaultdict


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # For each point, precompute which t-subsets contain it
    point_to_tsubsets = defaultdict(list)
    for i, s in enumerate(all_t_subsets):
        for x in s:
            point_to_tsubsets[x].append(i)
    
    def greedy_cover():
        """Generate initial valid covering using greedy algorithm."""
        uncovered = [True] * n_t_subsets
        uncovered_count = n_t_subsets
        blocks = []
        
        while uncovered_count > 0:
            # Find first uncovered t-subset
            first_uncovered = -1
            for i in range(n_t_subsets):
                if uncovered[i]:
                    first_uncovered = i
                    break
            
            if first_uncovered == -1:
                break
            
            # Start block with points from this t-subset
            block = list(all_t_subsets[first_uncovered])
            
            # Greedily add points to maximize coverage
            while len(block) < k:
                best_point = -1
                best_gain = -1
                block_set = set(block)
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    # Count how many uncovered t-subsets this point would cover
                    gain = 0
                    for tsub_idx in point_to_tsubsets[x]:
                        if uncovered[tsub_idx]:
                            # Check if this t-subset would be covered by adding x
                            tsub = all_t_subsets[tsub_idx]
                            if all(p in block_set or p == x for p in tsub):
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point == -1 or best_gain <= 0:
                    break
                block.append(best_point)
            
            block.sort()
            blocks.append(tuple(block))
            
            # Update uncovered status
            for tsub in itertools.combinations(block, t):
                idx = t_subset_to_idx[tsub]
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
        
        return blocks, uncovered, uncovered_count
    
    # Generate initial covering
    blocks, uncovered, uncovered_count = greedy_cover()
    
    if uncovered_count == 0:
        best_blocks = list(blocks)
        best_count = len(blocks)
    else:
        # If greedy didn't complete, force completion
        while uncovered_count > 0:
            first_uncovered = -1
            for i in range(n_t_subsets):
                if uncovered[i]:
                    first_uncovered = i
                    break
            if first_uncovered == -1:
                break
            block = list(all_t_subsets[first_uncovered])
            while len(block) < k:
                best_point = -1
                best_gain = -1
                block_set = set(block)
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for tsub_idx in point_to_tsubsets[x]:
                        if uncovered[tsub_idx]:
                            tsub = all_t_subsets[tsub_idx]
                            if all(p in block_set or p == x for p in tsub):
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point == -1:
                    # Just pick any unused point
                    for x in range(v):
                        if x not in block_set:
                            best_point = x
                            break
                block.append(best_point)
            block.sort()
            blocks.append(tuple(block))
            for tsub in itertools.combinations(block, t):
                idx = t_subset_to_idx[tsub]
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
        best_blocks = list(blocks)
        best_count = len(blocks)
    
    # Local search
    random.seed(42)
    
    # Build coverage map: for each t-subset, which blocks cover it
    # coverage[tsub_idx] = set of block indices
    coverage = defaultdict(set)
    for bi, block in enumerate(blocks):
        for tsub in itertools.combinations(block, t):
            idx = t_subset_to_idx[tsub]
            coverage[idx].add(bi)
    
    # Count how many times each block is the only cover for some t-subset
    # critical_count[bi] = number of t-subsets where block bi is the only cover
    critical_count = [0] * len(blocks)
    for tsub_idx in range(n_t_subsets):
        if len(coverage[tsub_idx]) == 1:
            bi = next(iter(coverage[tsub_idx]))
            critical_count[bi] += 1
    
    # Objective: minimize len(blocks) + penalty * uncovered_count
    # We want valid coverings, so prioritize reducing block count when valid
    
    def is_valid(uncovered):
        return not any(uncovered)
    
    # Local search moves:
    # 1. Try removing a block (if it's not critical)
    # 2. Try swapping a block for a better one
    # 3. Try adding a block that covers many uncovered t-subsets (if invalid)
    
    current_blocks = list(blocks)
    current_uncovered = list(uncovered)
    current_count = len(current_blocks)
    
    # Rebuild coverage for current state
    coverage = defaultdict(set)
    for bi, block in enumerate(current_blocks):
        for tsub in itertools.combinations(block, t):
            idx = t_subset_to_idx[tsub]
            coverage[idx].add(bi)
    
    uncovered_count = sum(1 for u in current_uncovered if u)
    
    # Precompute for each block, the t-subsets it covers
    block_tsubsets = []
    for block in current_blocks:
        tsubs = set()
        for tsub in itertools.combinations(block, t):
            tsubs.add(t_subset_to_idx[tsub])
        block_tsubsets.append(tsubs)
    
    # Local search
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 5.0:  # Keep 5s margin
        if no_improve_count > max_no_improve:
            break
        
        # Check if valid
        if uncovered_count == 0:
            # Try to remove a block
            removable = [bi for bi in range(len(current_blocks)) if critical_count[bi] == 0]
            if not removable:
                # Try swaps
                # For each block, try replacing with a random 7-subset
                improved = False
                for _ in range(20):  # Limited attempts per iteration
                    bi = random.randrange(len(current_blocks))
                    # Generate random replacement block
                    new_block = tuple(sorted(random.sample(range(v), k)))
                    
                    # Calculate change in uncovered
                    old_tsubs = block_tsubsets[bi]
                    new_tsubs = set(t_subset_to_idx[tsub] for tsub in itertools.combinations(new_block, t))
                    
                    # t-subsets that would become uncovered
                    would_uncover = 0
                    for tsub_idx in old_tsubs:
                        if tsub_idx not in new_tsubs:
                            # Check if covered by other blocks
                            other_covers = coverage[tsub_idx] - {bi}
                            if len(other_covers) == 0:
                                would_uncover += 1
                    
                    if would_uncover == 0:
                        # Accept the swap
                        current_blocks[bi] = new_block
                        block_tsubsets[bi] = new_tsubs
                        # Update coverage
                        for tsub_idx in old_tsubs - new_tsubs:
                            coverage[tsub_idx].discard(bi)
                        for tsub_idx in new_tsubs - old_tsubs:
                            coverage[tsub_idx].add(bi)
                        # Update critical counts
                        for tsub_idx in old_tsubs - new_tsubs:
                            if len(coverage[tsub_idx]) == 0:
                                pass  # Shouldn't happen since would_uncover==0
                        improved = True
                        no_improve_count = 0
                        break
                    
                if not improved:
                    no_improve_count += 1
            else:
                # Remove a random removable block
                bi = random.choice(removable)
                removed_tsubs = block_tsubsets[bi]
                
                # Verify it's still safe
                safe = True
                for tsub_idx in removed_tsubs:
                    if len(coverage[tsub_idx]) == 1:
                        safe = False
                        break
                
                if safe:
                    # Remove the block
                    del current_blocks[bi]
                    del block_tsubsets[bi]
                    
                    # Update coverage
                    for tsub_idx in removed_tsubs:
                        coverage[tsub_idx].discard(bi)
                    
                    # Update critical counts for remaining blocks
                    for tsub_idx in range(n_t_subsets):
                        if len(coverage[tsub_idx]) == 1:
                            bi2 = next(iter(coverage[tsub_idx]))
                            critical_count[bi2] += 1
                    # Recalculate critical counts properly
                    critical_count = [0] * len(current_blocks)
                    for tsub_idx in range(n_t_subsets):
                        if len(coverage[tsub_idx]) == 1:
                            bi2 = next(iter(coverage[tsub_idx]))
                            if bi2 < len(critical_count):
                                critical_count[bi2] += 1
                    
                    current_count -= 1
                    no_improve_count = 0
                    
                    if current_count < best_count:
                        best_count = current_count
                        best_blocks = list(current_blocks)
                else:
                    no_improve_count += 1
        else:
            # Invalid: try to add a block to cover uncovered t-subsets
            # Find block that covers most uncovered t-subsets
            best_new_block = None
            best_gain = 0
            for _ in range(50):
                new_block = tuple(sorted(random.sample(range(v), k)))
                new_tsubs = set(t_subset_to_idx[tsub] for tsub in itertools.combinations(new_block, t))
                gain = sum(1 for tsub_idx in new_tsubs if current_uncovered[tsub_idx])
                if gain > best_gain:
                    best_gain = gain
                    best_new_block = new_block
            
            if best_new_block is not None and best_gain > 0:
                # Add the block
                new_tsubs = set(t_subset_to_idx[tsub] for tsub in itertools.combinations(best_new_block, t))
                current_blocks.append(best_new_block)
                block_tsubsets.append(new_tsubs)
                
                # Update coverage and uncovered
                for tsub_idx in new_tsubs:
                    coverage[tsub_idx].add(len(current_blocks) - 1)
                    if current_uncovered[tsub_idx]:
                        current_uncovered[tsub_idx] = False
                        uncovered_count -= 1
                
                current_count = len(current_blocks)
                no_improve_count = 0
                
                if uncovered_count == 0 and current_count < best_count:
                    best_count = current_count
                    best_blocks = list(current_blocks)
            else:
                no_improve_count += 1
    
    # Return the best valid covering found
    if best_blocks is not None:
        return [list(b) for b in best_blocks]
    else:
        return [list(b) for b in current_blocks]
