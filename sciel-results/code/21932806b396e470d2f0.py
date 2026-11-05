import time
import sys
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Step 1: Find the S(4,5,11) design
    # Points 0..10
    # Blocks are 5-subsets
    # Constraint: Each 4-subset is in exactly one block
    
    points = list(range(11))
    all_4_subsets = list(combinations(points, 4))
    all_5_subsets = list(combinations(points, 5))
    
    # Map 4-subsets to indices for bitset representation
    # There are 330 4-subsets
    # We can use an integer bitmask of 330 bits (fits in Python int)
    # Or a set of indices.
    
    # Precompute which 4-subsets are covered by each 5-subset
    # 5-subset b covers 5 4-subsets (b with one element removed)
    
    # To speed up search, we can fix the first block to be (0,1,2,3,4)
    # due to symmetry (automorphism group is transitive on blocks? 
    # Actually, the automorphism group of S(4,5,11) is M11, which is 5-transitive.
    # So we can assume the first block contains 0,1,2,3,4.
    
    # Let's use a simple backtracking with sets.
    
    covered_4s = set()
    blocks_found = []
    
    # Candidate blocks: all 5-subsets
    # We will pick one, add to covered, recurse.
    
    # Optimization: 
    # 1. Fix first block to (0,1,2,3,4)
    # 2. Use a list of available blocks, filter as we go.
    
    def solve_s4511():
        # State: covered_4s (set of tuples), blocks_found (list of tuples)
        # We need to find 66 blocks.
        
        # Initial block
        first_block = (0, 1, 2, 3, 4)
        covered = set()
        for i in range(5):
            # Remove i-th element from first_block
            b = list(first_block)
            b.pop(i)
            covered.add(tuple(b))
        
        blocks = [first_block]
        
        # Remaining 4-subsets to cover
        # We can just check against 'covered'
        
        # List of all potential blocks
        # We can filter them dynamically
        # To be efficient, we can precompute for each 4-subset which blocks contain it
        # But with only 462 blocks, simple filtering is fast enough.
        
        # Let's store all 5-subsets
        all_blocks = list(combinations(points, 5))
        
        # Recursive function
        # To avoid global state issues, we pass state or use nonlocal
        # Python recursion limit might be an issue if depth is 66, but it's fine.
        
        def backtrack(covered, blocks):
            if time.time() > deadline:
                return False
            
            if len(blocks) == 66:
                return True
            
            # Find the first uncovered 4-subset
            # To be deterministic, we can just iterate through all_4_subsets
            # But it's faster to find an uncovered one.
            # Since we are adding blocks, we can just pick an uncovered 4-subset.
            
            # Heuristic: Pick the 4-subset with the fewest available blocks
            # This is the "MRV" (Minimum Remaining Values) heuristic from CSP.
            
            # However, calculating this for all 330 subsets every step might be slow in Python.
            # Given the small size, let's just pick the first uncovered one.
            
            target_4 = None
            for s in all_4_subsets:
                if s not in covered:
                    target_4 = s
                    break
            
            if target_4 is None:
                return len(blocks) == 66 # Should be true if we didn't fail
            
            # Find all blocks that contain target_4
            # A block b (5-subset) contains target_4 (4-subset) if target_4 is a subset of b.
            candidates = []
            for b in all_blocks:
                if target_4[0] in b and target_4[1] in b and target_4[2] in b and target_4[3] in b:
                    # Check if adding b conflicts with existing covered 4s
                    # b covers 5 4-subsets.
                    # If any of them are already covered (and not by b, but b is new), conflict.
                    # Actually, if a 4-subset is in 'covered', it was covered by a previous block.
                    # If b also covers it, that's a conflict (unless it's the same block, but blocks are distinct).
                    valid = True
                    new_4s = []
                    for i in range(5):
                        bb = list(b)
                        bb.pop(i)
                        t = tuple(bb)
                        if t in covered:
                            valid = False
                            break
                        new_4s.append(t)
                    
                    if valid:
                        candidates.append((b, new_4s))
                        
            # If no candidates, fail
            if not candidates:
                return False
                
            # Try each candidate
            # Sort candidates? Maybe by some heuristic.
            # For now, just iterate.
            for b, new_4s in candidates:
                # Add b
                new_covered = covered.union(new_4s)
                new_blocks = blocks + [b]
                
                if backtrack(new_covered, new_blocks):
                    blocks_found.extend(new_blocks)
                    return True
                    
                # Undo is handled by passing new lists (immutable-ish approach for sets/lists in recursion)
                # But creating new sets is O(330). 66 levels * 330 ops is ~20k ops. Very fast.
                
            return False

        if backtrack(covered, blocks):
            return blocks_found
        else:
            return None

    s4511_blocks = solve_s4511()
    
    if s4511_blocks is None:
        # Fallback: return empty or partial? Task requires maximizing.
        # If solver fails, we return what we have? 
        # But the solver should succeed.
        return []

    # Step 2: Construct S(5,6,12) from S(4,5,11)
    # Points 0..11
    # Type 1: b U {11} for b in s4511_blocks
    # Type 2: {0..10} \ b for b in s4511_blocks
    
    s5612_blocks = []
    
    for b in s4511_blocks:
        # b is a tuple of 5 ints from 0..10
        # Type 1
        b1 = list(b)
        b1.append(11)
        s5612_blocks.append(b1)
        
        # Type 2
        # Complement of b in {0..10}
        # b is a set of 5, so complement is 6 elements
        b2 = [x for x in range(11) if x not in b]
        s5612_blocks.append(b2)
        
    # Verify distinctness and count
    # There should be 132 blocks
    # We can return them as lists of ints
    
    return s5612_blocks
# EVOLVE-BLOCK-END
