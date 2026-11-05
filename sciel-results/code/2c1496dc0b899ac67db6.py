import itertools
import time
import random
import sys

def construct(v, k, t):
    # Deadline guard: 110 seconds to be safe
    deadline = time.time() + 110.0
    
    # Precompute all t-subsets for fast lookup
    # For v=19, t=5, C(19,5) = 11628. This is small enough to store in a set or list.
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for faster bit manipulation if needed, 
    # but for v=19, simple sets of tuples are likely fast enough in Python if we are careful.
    # Let's use a set of tuples for uncovered subsets.
    
    best_blocks = None
    best_count = float('inf')
    
    # Helper: Greedy construction with randomization
    def greedy_construct(seed_offset=0):
        random.seed(seed_offset)
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset to start the block
            # This breaks the determinism of picking min(uncovered)
            start_sub = random.choice(list(uncovered))
            block = list(start_sub)
            
            while len(block) < k:
                # Find the point x that covers the most uncovered t-subsets
                # A t-subset is covered if it is a subset of the block.
                # When adding x to block, we cover t-subsets that contain x and are subsets of block U {x}.
                # Specifically, a t-subset S is newly covered if S is a subset of block U {x} 
                # and S is NOT a subset of block (which means x in S).
                # So we look for S in uncovered such that x in S and S \ {x} is a subset of block.
                
                best_gain = -1
                best_x = -1
                
                # Optimize: Precompute which t-subsets in uncovered are "close" to the current block?
                # For v=19, k=8, t=5.
                # Current block size b. We need to add x.
                # Gain(x) = number of S in uncovered such that x in S and S \ {x} subset of block.
                
                candidates = [x for x in range(v) if x not in block]
                
                # For speed, we can iterate through uncovered and count, but that's O(|uncovered| * k).
                # |uncovered| can be large.
                # Alternative: Iterate through candidates x, and check how many S in uncovered contain x and are "compatible".
                
                # Let's try a slightly smarter heuristic:
                # For each candidate x, count how many S in uncovered satisfy:
                # 1. x in S
                # 2. S \ {x} is a subset of block
                
                # We can precompute for the current block state? No, block changes.
                # Let's just iterate. With v=19, candidates ~11. Uncovered ~10k. 11*10k = 110k ops per step.
                # Steps: (k-t) per block. Blocks ~300. 300 * 3 * 110k = 100M ops. Might be too slow in Python.
                
                # Optimization:
                # Instead of iterating all uncovered, we can filter uncovered to those that intersect with potential extensions?
                # Or, we can maintain a list of "active" uncovered subsets that are subsets of (block U candidates)?
                
                # Let's stick to a simpler, faster approximation if needed, but try the direct method first with optimization.
                # We can break early if gain is 0? No, we need max.
                
                # Let's try to speed up the gain calculation.
                # Precompute for each x, the list of t-subsets containing x?
                # subsets_with_x[x] = [S for S in all_t_subsets if x in S]
                # Then gain(x) = sum(1 for S in subsets_with_x[x] if S in uncovered and S \ {x} subset of block)
                
                # This is better.
                
                if not hasattr(greedy_construct, 'subsets_with_x'):
                    greedy_construct.subsets_with_x = [[] for _ in range(v)]
                    for S in all_t_subsets:
                        for x in S:
                            greedy_construct.subsets_with_x[x].append(S)
                
                for x in candidates:
                    gain = 0
                    # Check subsets containing x
                    for S in greedy_construct.subsets_with_x[x]:
                        if S in uncovered:
                            # Check if S \ {x} is subset of block
                            # S is tuple. Remove x.
                            # Since S is a tuple of sorted ints, and x is in S.
                            # We can check if all elements of S except x are in block.
                            # To make this fast, convert block to a set? 
                            # Block is small (size < 8).
                            block_set = set(block)
                            # S \ {x}
                            # We can check: all(y in block_set for y in S if y != x)
                            if all(y in block_set for y in S if y != x):
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        if gain == 0:
                            # If no gain, we are stuck? Should not happen if uncovered is not empty.
                            # But if all gains are 0, it means no single point addition covers any new subset?
                            # That implies the uncovered subsets are not "reachable" by adding 1 point to current block?
                            # This can happen if the uncovered subset is disjoint from block?
                            # Wait, we started with start_sub in block. So start_sub is covered.
                            # Any uncovered S must be different.
                            # If S is not a superset of any block extension, we might have an issue.
                            # But we can always add points.
                            # If gain is 0 for all, we pick a random one to break tie?
                            # Actually, if gain is 0, it means no S in uncovered is a subset of block U {x}.
                            # This implies S has elements outside block U {x}.
                            # We should still add x to grow the block.
                            pass
                
                if best_x == -1:
                    # Fallback: pick random
                    best_x = random.choice(candidates)
                
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            
            # Remove covered t-subsets
            # For each S in all_t_subsets, if S is subset of block, remove from uncovered.
            # Optimization: Only check S that were not removed before?
            # We can iterate through all_t_subsets and check.
            # Or, iterate through combinations of block of size t.
            # Combinations of block of size t are the ones covered.
            # So:
            for covered_S in itertools.combinations(block, t):
                uncovered.discard(covered_S)
        
        return blocks
    
    # Helper: Local Search (Remove redundant blocks)
    def local_search(blocks):
        # Check if any block is redundant.
        # A block B is redundant if for every t-subset S of B, there exists another block B' != B such that S is subset of B'.
        
        # Precompute for each t-subset S, which blocks cover it.
        # This is expensive if done from scratch every time.
        # But v=19, k=8, t=5.
        # Let's try a simpler approach:
        # For each block B, check if it can be removed.
        
        current_blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(current_blocks)):
                B = current_blocks[i]
                # Check if B is redundant
                is_redundant = True
                for S in itertools.combinations(B, t):
                    # Check if S is covered by any other block
                    covered_by_other = False
                    for j in range(len(current_blocks)):
                        if i == j:
                            continue
                        Bj = current_blocks[j]
                        # Check if S is subset of Bj
                        # S is tuple, Bj is list.
                        if all(x in Bj for x in S):
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        is_redundant = False
                        break
                
                if is_redundant:
                    current_blocks.pop(i)
                    changed = True
                    break # Restart loop as indices changed
        
        return current_blocks

    # Main Loop
    start_time = time.time()
    attempts = 0
    
    while time.time() < deadline:
        attempts += 1
        # Randomize the greedy construction
        seed = attempts
        blocks = greedy_construct(seed)
        
        # Apply local search to remove redundant blocks
        blocks = local_search(blocks)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
        # Early termination if we reach a very low count?
        # Theoretical lower bound is 264. If we get close, we can stop.
        # But let's keep searching for a bit longer to be sure.
        if best_count <= 265:
            break
            
    return best_blocks

# To ensure the function is available immediately
if __name__ == "__main__":
    # Small test
    # blocks = construct(19, 8, 5)
    # print(len(blocks))
    pass
