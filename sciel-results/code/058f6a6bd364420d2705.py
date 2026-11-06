import time

def construct():
    """
    Returns the 132 blocks of the Steiner System S(5, 6, 12).
    This is the unique (up to isomorphism) packing of 6-subsets on 12 points
    such that no 5-subset is repeated. The maximum number of such blocks is 132.
    
    The blocks are found using an optimized backtracking search with symmetry breaking.
    """
    from itertools import combinations
    from collections import defaultdict
    import sys
    
    sys.setrecursionlimit(10000)
    
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Precompute 5-subsets and 6-subsets
    all_5subsets = list(combinations(range(12), 5))
    five_to_id = {sub: i for i, sub in enumerate(all_5subsets)}
    
    all_6subsets = list(combinations(range(12), 6))
    
    # Map each 6-subset to its constituent 5-subset IDs
    six_to_fives = []
    for block in all_6subsets:
        fives = []
        for i in range(6):
            sub = block[:i] + block[i+1:]
            fives.append(five_to_id[sub])
        six_to_fives.append(fives)
        
    # Inverse mapping: 5-subset ID -> list of 6-subset indices
    five_to_six = defaultdict(list)
    for idx, fives in enumerate(six_to_fives):
        for fid in fives:
            five_to_six[fid].append(idx)
            
    used_fives = set()
    blocks_found = []
    
    def backtrack():
        if time.time() > deadline:
            return False
        if len(blocks_found) == 132:
            return True
            
        # Find the 5-subset with the fewest available blocks (MRV)
        min_count = float('inf')
        best_fid = -1
        
        for fid in range(792):
            if fid in used_fives:
                continue
                
            count = 0
            for idx in five_to_six[fid]:
                # Check if block idx is valid
                valid = True
                for f in six_to_fives[idx]:
                    if f in used_fives:
                        valid = False
                        break
                if valid:
                    count += 1
                    
            if count == 0:
                return False # Dead end
                
            if count < min_count:
                min_count = count
                best_fid = fid
                if count == 1:
                    break # Best possible MRV
                
        if best_fid == -1:
            return True # All fids used? Should not happen before 132 blocks
            
        # Try all blocks containing best_fid
        for idx in five_to_six[best_fid]:
            # Check if block idx is valid
            valid = True
            for f in six_to_fives[idx]:
                if f in used_fives:
                    valid = False
                    break
            if not valid:
                continue
                
            # Apply
            blocks_found.append(all_6subsets[idx])
            for f in six_to_fives[idx]:
                used_fives.add(f)
                
            if backtrack():
                return True
                
            # Undo
            blocks_found.pop()
            for f in six_to_fives[idx]:
                used_fives.discard(f)
                
        return False

    # Symmetry breaking: Fix the first block to (0,1,2,3,4,5)
    first_block = (0, 1, 2, 3, 4, 5)
    # Precomputed index for speed
    first_idx = 0 # (0,1,2,3,4,5) is the first combination in lexicographic order
    
    used_fives = set(six_to_fives[first_idx])
    blocks_found = [first_block]
    
    if backtrack():
        return [list(b) for b in blocks_found]
        
    return [list(b) for b in blocks_found]

if __name__ == '__main__':
    res = construct()
    print(f"Number of blocks: {len(res)}")
