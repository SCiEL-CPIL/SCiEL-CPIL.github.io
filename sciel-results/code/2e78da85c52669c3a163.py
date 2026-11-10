"""Covering design C(19,7,4) with time-bounded local search."""
import itertools
import time
import random
import sys
from collections import defaultdict

# Deadline guard
DEADLINE = 115.0  # seconds, leaving 5s safety margin before 120s kill
START_TIME = None

def _now():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    return time.time()

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def construct(v, k, t):
    global START_TIME
    START_TIME = time.time()
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    total_t = len(all_t_subsets)
    
    # Map each t-subset to an index
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Coverage count: how many blocks cover each t-subset
    # We'll use a list of length total_t, initially 0
    coverage = [0] * total_t
    uncovered = [1] * total_t  # 1 if uncovered, 0 if covered
    
    blocks = []
    
    # Helper: get the list of t-subset indices covered by a given block
    def block_t_indices(block):
        return [t_subset_to_idx[s] for s in itertools.combinations(block, t)]
    
    # Helper: add a block, update coverage
    def add_block(block):
        blocks.append(block)
        for idx in block_t_indices(block):
            coverage[idx] += 1
            if coverage[idx] == 1:
                uncovered[idx] = 0
    
    # Helper: remove a block, update coverage
    def remove_block(block):
        blocks.remove(block)
        for idx in block_t_indices(block):
            coverage[idx] -= 1
            if coverage[idx] == 0:
                uncovered[idx] = 1
    
    # Helper: count uncovered
    def count_uncovered():
        return sum(uncovered)
    
    # Greedy initialization
    # Start with all t-subsets uncovered
    uncovered_set = set(range(total_t))
    
    while uncovered_set and _time_left() > 10:
        # Pick a random uncovered t-subset to start a block
        target = random.choice(list(uncovered_set))
        target_subset = all_t_subsets[target]
        
        # Build block greedily
        block = list(target_subset)
        random.shuffle(block)
        
        while len(block) < k:
            # For each candidate point, compute how many uncovered t-subsets it would cover
            best_gain = -1
            best_point = None
            # Precompute (t-1)-subsets of current block
            block_set = set(block)
            t_minus_1_subsets = list(itertools.combinations(block, t - 1))
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in t_minus_1_subsets:
                    # Check if rest + (x,) is an uncovered t-subset
                    new_subset = tuple(sorted(rest + (x,)))
                    if new_subset in t_subset_to_idx:
                        idx = t_subset_to_idx[new_subset]
                        if uncovered[idx]:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point is None:
                break
            block.append(best_point)
        
        block.sort()
        add_block(block)
        
        # Update uncovered_set
        for idx in block_t_indices(block):
            uncovered_set.discard(idx)
    
    # Now we have a valid covering. Try to improve via local search.
    # Local search: try removing redundant blocks, then try swapping.
    
    def is_valid_covering():
        return all(c >= 1 for c in coverage)
    
    # Phase 1: Remove redundant blocks (blocks that don't cover any unique t-subset)
    def remove_redundant_blocks():
        changed = True
        while changed and _time_left() > 5:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                block = blocks[i]
                # Check if this block is the only cover for any t-subset
                redundant = True
                for idx in block_t_indices(block):
                    if coverage[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    remove_block(block)
                    changed = True
                    break
    
    remove_redundant_blocks()
    
    # Phase 2: Local search with simulated annealing
    # Moves:
    # 1. Remove a block (if it doesn't break coverage)
    # 2. Add a random block
    # 3. Swap: remove one block, add another
    
    best_blocks = list(blocks)
    best_count = len(blocks)
    
    temperature = 10.0
    cooling = 0.999
    
    while _time_left() > 3:
        current_count = len(blocks)
        
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.3 and len(blocks) > 0:
            # Try to remove a block
            i = random.randint(0, len(blocks) - 1)
            block = blocks[i]
            # Check if removal would break coverage
            can_remove = True
            for idx in block_t_indices(block):
                if coverage[idx] == 1:
                    can_remove = False
                    break
            if can_remove:
                remove_block(block)
                new_count = len(blocks)
                if new_count < best_count:
                    best_blocks = list(blocks)
                    best_count = new_count
                # Always accept if it reduces count
            else:
                # Try adding a random block to compensate, then remove
                pass
        elif move_type < 0.6 and len(blocks) > 0:
            # Swap: remove a random block, add a new random block
            i = random.randint(0, len(blocks) - 1)
            old_block = blocks[i]
            
            # Generate a new random block
            new_block = sorted(random.sample(range(v), k))
            
            # Check if this swap is valid or improves
            # First, check what happens if we remove old_block
            # We need to be careful: only valid if coverage remains >= 1 everywhere
            can_swap = True
            for idx in block_t_indices(old_block):
                if coverage[idx] == 1:
                    can_swap = False
                    break
            
            if can_swap:
                remove_block(old_block)
                add_block(new_block)
                new_count = len(blocks)
                if new_count < best_count:
                    best_blocks = list(blocks)
                    best_count = new_count
            else:
                # Try adding new block first, then removing old
                add_block(new_block)
                can_remove_now = True
                for idx in block_t_indices(old_block):
                    if coverage[idx] == 1:
                        can_remove_now = False
                        break
                if can_remove_now:
                    remove_block(old_block)
                    new_count = len(blocks)
                    if new_count < best_count:
                        best_blocks = list(blocks)
                        best_count = new_count
                else:
                    # Revert: remove the new block
                    remove_block(new_block)
        else:
            # Add a random block (may help enable future removals)
            new_block = sorted(random.sample(range(v), k))
            add_block(new_block)
            new_count = len(blocks)
            # Accept with probability based on temperature
            delta = new_count - current_count
            if delta > 0:
                import math
                if temperature > 0:
                    prob = math.exp(-delta / temperature)
                    if random.random() > prob:
                        remove_block(new_block)
            
            temperature *= cooling
    
    # Restore best solution
    blocks = list(best_blocks)
    
    # Final: remove redundant blocks again
    remove_redundant_blocks()
    
    return blocks

# Ensure the function is defined at module level
if __name__ == "__main__":
    # Quick test
    import sys
    v, k, t = 19, 7, 4
    result = construct(v, k, t)
    print(f"Blocks: {len(result)}")
