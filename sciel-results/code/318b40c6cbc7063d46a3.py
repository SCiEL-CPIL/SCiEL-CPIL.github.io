import itertools
import time
import random
import sys

def construct(v, k, t):
    # Deadline guard: 110 seconds to be safe
    deadline = time.time() + 110.0
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map t-subset to index for coverage counting
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute for each point x, the list of t-subsets containing x
    # This helps in the greedy step to quickly find candidates
    subsets_with_x = [[] for _ in range(v)]
    for S in all_t_subsets:
        for x in S:
            subsets_with_x[x].append(S)
    
    best_blocks = None
    best_count = float('inf')
    
    def greedy_construct(seed_offset=0):
        random.seed(seed_offset)
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset to start the block
            start_sub = random.choice(list(uncovered))
            block = list(start_sub)
            block_set = set(block)
            
            while len(block) < k:
                best_gain = -1
                best_x = -1
                
                candidates = [x for x in range(v) if x not in block_set]
                
                for x in candidates:
                    gain = 0
                    # Iterate through t-subsets containing x
                    for S in subsets_with_x[x]:
                        if S in uncovered:
                            # Check if S \ {x} is subset of block
                            # Since x is in S, S \ {x} has t-1 elements.
                            # We can check if all elements in S except x are in block_set.
                            # Optimization: Since S is a tuple of sorted ints,
                            # we can just check the other t-1 elements.
                            # A fast way: if S is subset of block_set U {x}
                            # which is equivalent to (S without x) being subset of block_set.
                            # Since x is in S, S without x is S if we ignore x? No.
                            # Let's just do: all(y in block_set for y in S if y != x)
                            # This is O(t) per check. t=5.
                            if all(y in block_set for y in S if y != x):
                                gain += 1
                                
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        
                if best_x == -1:
                    # Fallback: pick random if no gain (should not happen if uncovered is not empty)
                    best_x = random.choice(candidates)
                
                block.append(best_x)
                block_set.add(best_x)
            
            block.sort()
            blocks.append(block)
            
            # Remove covered t-subsets
            # Only check t-subsets of the new block
            for covered_S in itertools.combinations(block, t):
                uncovered.discard(covered_S)
        
        return blocks
    
    def local_search(blocks):
        # Count coverage for each t-subset
        # coverage[i] = number of blocks covering all_t_subsets[i]
        coverage = [0] * num_t_subsets
        
        # First pass: count coverage
        for B in blocks:
            for S in itertools.combinations(B, t):
                idx = subset_to_idx[S]
                coverage[idx] += 1
        
        current_blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            # Try to remove each block
            # We need to be careful with indices if we remove.
            # Let's iterate backwards or mark for removal.
            # Simpler: iterate, check redundancy, if redundant, remove and restart.
            # To make it faster, we can check if any t-subset in B has coverage == 1.
            # If yes, B is NOT redundant.
            # If no, B is redundant.
            
            for i in range(len(current_blocks)):
                B = current_blocks[i]
                is_redundant = True
                for S in itertools.combinations(B, t):
                    idx = subset_to_idx[S]
                    if coverage[idx] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove B
                    # Update coverage
                    for S in itertools.combinations(B, t):
                        idx = subset_to_idx[S]
                        coverage[idx] -= 1
                    current_blocks.pop(i)
                    changed = True
                    break # Restart loop
        
        return current_blocks

    # Main Loop
    start_time = time.time()
    attempts = 0
    
    while time.time() < deadline:
        attempts += 1
        seed = attempts
        blocks = greedy_construct(seed)
        
        # Apply local search to remove redundant blocks
        blocks = local_search(blocks)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
        # Early termination if we reach a very low count?
        if best_count <= 265:
            break
            
    return best_blocks

if __name__ == "__main__":
    pass
