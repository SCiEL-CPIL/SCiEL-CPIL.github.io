import itertools
import time

def construct():
    """
    Constructs the Steiner System S(5,6,12) (Witt Design) on 12 points.
    Returns a list of 132 distinct 6-subsets of {0, ..., 11}.
    """
    
    # Step 1: Construct S(4,5,11)
    # We need a collection of 5-subsets of {0..10} such that every 4-subset is in exactly one block.
    # Total 4-subsets: C(11, 4) = 330.
    # Each 5-block covers 5 4-subsets.
    # Number of blocks: 330 / 5 = 66.
    
    def find_s4511():
        points = list(range(11))
        all_4sets = list(itertools.combinations(points, 4))
        
        # Map 4-set to index for fast lookup
        # We will maintain a boolean array 'covered' of size 330
        # And a list 'block_4sets' where block_4sets[i] is the list of indices of 4-sets in block i
        
        # To make the backtracking efficient, we fix the first block.
        # By symmetry, we can assume the first block is (0, 1, 2, 3, 4).
        first_block = (0, 1, 2, 3, 4)
        
        # Pre-compute the 4-subsets of any 5-set
        # We'll use a dictionary to map 5-set to its 4-subset indices
        # Actually, we can just compute on the fly, but pre-computing all 5-sets' 4-subsets is better.
        
        # Let's build a map: 4-set -> list of 5-sets containing it.
        # This allows us to quickly find candidates for a specific uncovered 4-set.
        
        # 1. Generate all 5-sets
        all_5sets = list(itertools.combinations(points, 5))
        
        # 2. Map each 4-set to a list of 5-sets that contain it
        # 4-set -> list of 5-set tuples
        four_to_fives = {s: [] for s in all_4sets}
        
        # Index mapping for 4-sets
        four_set_index = {s: i for i, s in enumerate(all_4sets)}
        
        for b5 in all_5sets:
            # Get all 4-subsets of b5
            subs = list(itertools.combinations(b5, 4))
            for s4 in subs:
                four_to_fives[s4].append(b5)
        
        # Backtracking state
        covered_4sets = [False] * 330
        blocks = []
        
        # Recursive helper
        def backtrack():
            # Find the first uncovered 4-set (lexicographically smallest index)
            target_idx = -1
            for i in range(330):
                if not covered_4sets[i]:
                    target_idx = i
                    break
            
            if target_idx == -1:
                # All 4-sets covered
                return True
            
            target_4set = all_4sets[target_idx]
            
            # Try all 5-sets that contain this 4-set
            candidates = four_to_fives[target_4set]
            
            for b5 in candidates:
                # Check if this 5-set is compatible with existing blocks
                # i.e., none of its other 4-subsets are already covered
                
                # Get indices of all 4-subsets in b5
                subs = list(itertools.combinations(b5, 4))
                sub_indices = [four_set_index[s] for s in subs]
                
                valid = True
                for idx in sub_indices:
                    if covered_4sets[idx]:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Place the block
                for idx in sub_indices:
                    covered_4sets[idx] = True
                blocks.append(b5)
                
                if backtrack():
                    return True
                
                # Undo
                blocks.pop()
                for idx in sub_indices:
                    covered_4sets[idx] = False
            
            return False
        
        # Initialize with the first block fixed
        # first_block = (0, 1, 2, 3, 4)
        first_subs = list(itertools.combinations(first_block, 4))
        for s in first_subs:
            covered_4sets[four_set_index[s]] = True
        blocks.append(first_block)
        
        if backtrack():
            return blocks
        else:
            return None

    s4511_blocks = find_s4511()
    
    if s4511_blocks is None:
        # Fallback if backtracking fails (should not happen for S(4,5,11))
        # Return empty or a small valid packing
        return []

    # Step 2: Construct S(5,6,12) from S(4,5,11)
    # Points 0..10 in S(4,5,11), plus point 11.
    # For each block B in S(4,5,11):
    # 1. B + {11}
    # 2. (0..10 \ B)
    
    w12_blocks = []
    all_points_11 = set(range(11))
    
    for b in s4511_blocks:
        # b is a tuple of 5 elements from 0..10
        # Block 1: b plus 11
        w12_blocks.append(tuple(sorted(list(b) + [11])))
        
        # Block 2: complement of b in 0..10
        comp = tuple(sorted(all_points_11 - set(b)))
        w12_blocks.append(comp)
    
    # The result should be 132 blocks
    # Convert to list of lists as per interface
    return [list(b) for b in w12_blocks]

if __name__ == "__main__":
    # Quick self-test
    blocks = construct()
    print(f"Constructed {len(blocks)} blocks")
    # Verify validity
    if len(blocks) == 132:
        seen_5sets = set()
        valid = True
        for b in blocks:
            if len(set(b)) != 6:
                valid = False
                break
            for s in itertools.combinations(b, 5):
                if s in seen_5sets:
                    valid = False
                    break
                seen_5sets.add(s)
            if not valid:
                break
        print(f"Valid: {valid}, Unique 5-sets: {len(seen_5sets)}")
    else:
        print("Invalid count")
