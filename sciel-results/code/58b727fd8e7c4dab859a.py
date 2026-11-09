import itertools

def construct():
    points = list(range(12))
    all_6_subsets = list(itertools.combinations(points, 6))
    
    # Map 5-subsets to indices
    all_5_subsets = list(itertools.combinations(points, 5))
    subset_to_idx = {s: i for i, s in enumerate(all_5_subsets)}
    
    # Precompute 5-subset indices for each 6-subset
    block_5_indices = {}
    for b in all_6_subsets:
        block_5_indices[b] = [subset_to_idx[s] for s in itertools.combinations(b, 5)]
        
    # Complement map
    complement_map = {}
    for b in all_6_subsets:
        b_set = set(b)
        complement_map[b] = tuple(p for p in points if p not in b_set)

    covered = [False] * 792
    blocks = []
    blocks_set = set()
    
    def add_block(b):
        blocks.append(b)
        blocks_set.add(b)
        for idx in block_5_indices[b]:
            covered[idx] = True

    def remove_block(b):
        blocks.pop() # Assumes LIFO order for removal in backtracking
        blocks_set.remove(b)
        for idx in block_5_indices[b]:
            covered[idx] = False

    # Fix first block
    first_block = (0, 1, 2, 3, 4, 5)
    add_block(first_block)
    add_block(complement_map[first_block])
    
    def is_valid_pair(b1, b2):
        for idx in block_5_indices[b1]:
            if covered[idx]:
                return False
        for idx in block_5_indices[b2]:
            if covered[idx]:
                return False
        return True

    def backtrack(start_idx):
        if len(blocks) == 132:
            return True
            
        for i in range(start_idx, len(all_6_subsets)):
            b = all_6_subsets[i]
            bc = complement_map[b]
            
            if b in blocks_set or bc in blocks_set:
                continue
                
            if not is_valid_pair(b, bc):
                continue
                
            add_block(b)
            add_block(bc)
            
            if backtrack(i + 1):
                return True
                
            remove_block(bc)
            remove_block(b)
                
        return False

    if backtrack(0):
        return [list(b) for b in blocks]
    else:
        return [list(b) for b in blocks]
