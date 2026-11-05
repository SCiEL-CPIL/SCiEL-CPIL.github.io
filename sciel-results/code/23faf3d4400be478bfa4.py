"""Covering design C(24,7,3) using greedy + local search refinement."""
import itertools
import random
import time
import sys

# Precompute all 3-subsets for v=24
# But we need to be general for the interface, though the task is specific.
# Let's write a general solution that works well for v=24, k=7, t=3.

def construct(v, k, t):
    start_time = time.time()
    timeout = 110.0  # Leave 10s buffer
    
    # If parameters are small, we can be more exhaustive.
    # For v=24, k=7, t=3, the number of t-subsets is C(24,3) = 2024.
    # The number of k-subsets is C(24,7) = 346104.
    
    # Strategy:
    # 1. Greedy construction to get an initial set of blocks.
    # 2. Local search: try to remove redundant blocks or swap blocks to cover more.
    # 3. Since we want to minimize the number of blocks, we can try to find a smaller set.
    
    # Step 1: Greedy Construction
    # We represent uncovered t-subsets as a set of frozensets or tuples.
    # For speed, we can use a bitset or just a set of tuples.
    # C(24,3) = 2024, so a set of 2024 tuples is manageable.
    
    all_t_subsets = set(itertools.combinations(range(v), t))
    uncovered = set(all_t_subsets)
    blocks = []
    
    # Precompute for each potential block (k-subset) the set of t-subsets it covers?
    # That's too expensive to precompute all C(24,7).
    # Instead, during greedy, we evaluate gain on the fly.
    
    # To speed up greedy, we can maintain a count of how many times each t-subset is covered?
    # No, we just want to cover the uncovered ones.
    
    # Optimization: 
    # For a given partial block B, adding point x covers new t-subsets of the form {x} U C(B, t-1).
    # We need to check if these are in 'uncovered'.
    
    while uncovered:
        if time.time() - start_time > timeout:
            break
        
        # Pick the first uncovered t-subset to start the block
        # To be more effective, we could pick the uncovered t-subset with the highest "frequency" 
        # of points that appear in other uncovered t-subsets? 
        # Standard greedy: just pick any, usually the first.
        
        # Let's pick the uncovered t-subset that has the most "popular" points?
        # Actually, just picking min(uncovered) is fast.
        
        block = list(min(uncovered))
        
        # Extend block to size k
        while len(block) < k:
            if time.time() - start_time > timeout:
                break
            
            best_x = -1
            best_gain = -1
            
            # Candidate points: those not in block
            # We can restrict candidates to those that appear in many uncovered t-subsets involving the current block?
            # Simple approach: iterate all x not in block.
            
            block_set = set(block)
            # Precompute combinations of block of size t-1
            # t=3, so t-1=2. C(6,2)=15 pairs max.
            pairs = list(itertools.combinations(block, t - 1))
            
            for x in range(v):
                if x in block_set:
                    continue
                
                gain = 0
                for p in pairs:
                    # Check if (x, p[0], p[1]) is in uncovered
                    # We need to form a sorted tuple to match the set representation
                    tri = (x, p[0], p[1])
                    tri_sorted = tuple(sorted(tri))
                    if tri_sorted in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x == -1:
                # Should not happen unless uncovered is empty or no points left
                break
                
            block.append(best_x)
        
        if len(block) < k:
            break
            
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for comb in itertools.combinations(block, t):
            uncovered.discard(tuple(sorted(comb)))
            
        # Check timeout
        if time.time() - start_time > timeout:
            break
    
    if not uncovered:
        # We have a valid covering. Now try to minimize the number of blocks.
        # Local Search: Try to remove blocks if they are redundant.
        # Or try to swap blocks.
        
        # Simple reduction:
        # Iterate through blocks, check if removing one still leaves all t-subsets covered.
        # This is O(N * C(24,3)) per removal attempt, which is fine.
        
        # To check if a block is redundant, we need to know if all its t-subsets are covered by other blocks.
        # We can maintain a count of how many blocks cover each t-subset.
        
        # Let's build a counter for the current set of blocks.
        # This might be expensive if we do it from scratch, but N is small (~76).
        
        def is_valid_covering(b_list):
            # Check if b_list covers all t-subsets
            covered = set()
            for b in b_list:
                for c in itertools.combinations(b, t):
                    covered.add(tuple(sorted(c)))
            return covered == all_t_subsets
        
        # Greedy removal
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() - start_time > timeout:
                    return blocks
                
                # Try removing blocks[i]
                new_blocks = blocks[:i] + blocks[i+1:]
                if is_valid_covering(new_blocks):
                    blocks = new_blocks
                    changed = True
                    break # Restart loop as indices changed
        
        # Further optimization: try to replace a block with a different one that covers the same or more?
        # This is harder. 
        # Another strategy: start with a larger set and remove? 
        # Or use a different greedy order?
        
        # Let's try a randomized restart approach if time permits.
        # But for now, greedy + removal is a solid baseline.
        
        return blocks
    else:
        # Timeout during construction. 
        # We might not have a valid covering. 
        # The task says "Valid when every t-subset... lies in at least one block".
        # If we timeout, we should probably return what we have, but it will score 0 if invalid.
        # We should try to finish the last block or just return invalid?
        # Better to try to complete the current block if we are close.
        
        # Actually, if we timeout in the while loop, 'uncovered' is not empty.
        # The returned blocks will be invalid.
        # We can try a faster heuristic if we detect we are running out of time?
        # For now, just return what we have.
        
        return blocks

# Test locally
if __name__ == "__main__":
    # Quick test
    blocks = construct(24, 7, 3)
    print(f"Number of blocks: {len(blocks)}")
    
    # Verify
    covered = set()
    for b in blocks:
        for c in itertools.combinations(b, 3):
            covered.add(tuple(sorted(c)))
    
    all_subsets = set(itertools.combinations(range(24), 3))
    if covered == all_subsets:
        print("Valid covering")
    else:
        print("Invalid covering")
        print(f"Missing: {len(all_subsets - covered)}")
