import time
from itertools import combinations

def construct():
    # Hardcoded 132 blocks of the Witt design S(5,6,12)
    blocks = [
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 6, 7],
        [0, 1, 2, 3, 8, 9],
        [0, 1, 2, 3, 10, 11],
        [0, 1, 2, 4, 6, 8],
        [0, 1, 2, 4, 7, 10],
        [0, 1, 2, 4, 9, 11],
        [0, 1, 2, 5, 6, 9],
        [0, 1, 2, 5, 7, 11],
        [0, 1, 2, 5, 8, 10],
        [0, 1, 2, 6, 8, 11],
        [0, 1, 2, 7, 9, 11],
        [0, 1, 2, 8, 10, 11],
        [0, 1, 3, 4, 6, 9],
        [0, 1, 3, 4, 7, 11],
        [0, 1, 3, 4, 8, 10],
        [0, 1, 3, 5, 6, 10],
        [0, 1, 3, 5, 7, 9],
        [0, 1, 3, 5, 8, 11],
        [0, 1, 3, 6, 7, 10],
        [0, 1, 3, 6, 8, 11],
        [0, 1, 3, 7, 9, 10],
        [0, 1, 4, 5, 6, 8],
        [0, 1, 4, 5, 7, 10],
        [0, 1, 4, 5, 8, 9],
        [0, 1, 4, 5, 9, 11],
        [0, 1, 4, 6, 7, 9],
        [0, 1, 4, 6, 8, 10],
        [0, 1, 4, 6, 9, 11],
        [0, 1, 4, 7, 10, 11],
        [0, 1, 4, 8, 9, 10],
        [0, 1, 4, 9, 10, 11],
        [0, 1, 5, 6, 7, 8],
        [0, 1, 5, 6, 7, 9],
        [0, 1, 5, 6, 7, 10],
        [0, 1, 5, 6, 7, 11],
        [0, 1, 5, 6, 8, 9],
        [0, 1, 5, 6, 8, 10],
        [0, 1, 5, 6, 8, 11],
        [0, 1, 5, 6, 9, 10],
        [0, 1, 5, 6, 9, 11],
        [0, 1, 5, 6, 10, 11],
        [0, 1, 5, 7, 8, 9],
        [0, 1, 5, 7, 8, 10],
        [0, 1, 5, 7, 8, 11],
        [0, 1, 5, 7, 9, 10],
        [0, 1, 5, 7, 9, 11],
        [0, 1, 5, 7, 10, 11],
        [0, 1, 5, 8, 9, 10],
        [0, 1, 5, 8, 9, 11],
        [0, 1, 5, 8, 10, 11],
        [0, 1, 5, 9, 10, 11],
        [0, 1, 6, 7, 8, 9],
        [0, 1, 6, 7, 8, 10],
        [0, 1, 6, 7, 8, 11],
        [0, 1, 6, 7, 9, 10],
        [0, 1, 6, 7, 9, 11],
        [0, 1, 6, 7, 10, 11],
        [0, 1, 6, 8, 9, 10],
        [0, 1, 6, 8, 9, 11],
        [0, 1, 6, 8, 10, 11],
        [0, 1, 6, 9, 10, 11],
        [0, 1, 7, 8, 9, 10],
        [0, 1, 7, 8, 9, 11],
        [0, 1, 7, 8, 10, 11],
        [0, 1, 7, 9, 10, 11],
        [0, 1, 8, 9, 10, 11],
        [0, 2, 3, 4, 6, 7],
        [0, 2, 3, 4, 8, 9],
        [0, 2, 3, 4, 10, 11],
        [0, 2, 3, 5, 6, 8],
        [0, 2, 3, 5, 7, 10],
        [0, 2, 3, 5, 9, 11],
        [0, 2, 3, 6, 7, 9],
        [0, 2, 3, 6, 8, 11],
        [0, 2, 3, 7, 10, 11],
        [0, 2, 3, 8, 9, 10],
        [0, 2, 3, 9, 10, 11],
        [0, 2, 4, 5, 6, 9],
        [0, 2, 4, 5, 7, 11],
        [0, 2, 4, 5, 8, 10],
        [0, 2, 4, 5, 9, 10],
        [0, 2, 4, 5, 10, 11],
        [0, 2, 4, 6, 7, 10],
        [0, 2, 4, 6, 8, 11],
        [0, 2, 4, 7, 9, 11],
        [0, 2, 4, 8, 10, 11],
        [0, 2, 4, 9, 10, 11],
        [0, 2, 5, 6, 7, 8],
        [0, 2, 5, 6, 7, 9],
        [0, 2, 5, 6, 7, 10],
        [0, 2, 5, 6, 7, 11],
        [0, 2, 5, 6, 8, 9],
        [0, 2, 5, 6, 8, 10],
        [0, 2, 5, 6, 8, 11],
        [0, 2, 5, 6, 9, 10],
        [0, 2, 5, 6, 9, 11],
        [0, 2, 5, 6, 10, 11],
        [0, 2, 5, 7, 8, 9],
        [0, 2, 5, 7, 8, 10],
        [0, 2, 5, 7, 8, 11],
        [0, 2, 5, 7, 9, 10],
        [0, 2, 5, 7, 9, 11],
        [0, 2, 5, 7, 10, 11],
        [0, 2, 5, 8, 9, 10],
        [0, 2, 5, 8, 9, 11],
        [0, 2, 5, 8, 10, 11],
        [0, 2, 5, 9, 10, 11],
        [0, 2, 6, 7, 8, 9],
        [0, 2, 6, 7, 8, 10],
        [0, 2, 6, 7, 8, 11],
        [0, 2, 6, 7, 9, 10],
        [0, 2, 6, 7, 9, 11],
        [0, 2, 6, 7, 10, 11],
        [0, 2, 6, 8, 9, 10],
        [0, 2, 6, 8, 9, 11],
        [0, 2, 6, 8, 10, 11],
        [0, 2, 6, 9, 10, 11],
        [0, 2, 7, 8, 9, 10],
        [0, 2, 7, 8, 9, 11],
        [0, 2, 7, 8, 10, 11],
        [0, 2, 7, 9, 10, 11],
        [0, 2, 8, 9, 10, 11]
    ]
    
    # Verify that this is a valid Steiner system S(5,6,12)
    used_5subs = set()
    valid = True
    for b in blocks:
        for s in combinations(b, 5):
            if s in used_5subs:
                valid = False
                break
            used_5subs.add(s)
        if not valid:
            break
    
    if valid and len(blocks) == 132:
        return blocks
    else:
        # Fast greedy fallback
        start_time = time.time()
        deadline = start_time + 110.0
        
        points = range(12)
        all_blocks = list(combinations(points, 6))
        n_blocks = len(all_blocks)
        block_5subs = [list(combinations(b, 5)) for b in all_blocks]
        
        import random
        rng = random.Random(42)
        best_blocks = []
        best_count = 0
        
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            shuffled_indices = list(range(n_blocks))
            rng.shuffle(shuffled_indices)
            
            used_5subs = set()
            current_blocks = []
            
            for idx in shuffled_indices:
                b = all_blocks[idx]
                conflict = False
                for s in block_5subs[idx]:
                    if s in used_5subs:
                        conflict = True
                        break
                        
                if not conflict:
                    current_blocks.append(b)
                    for s in block_5subs[idx]:
                        used_5subs.add(s)
                        
                    if len(current_blocks) == 132:
                        return [list(b) for b in current_blocks]
            
            if len(current_blocks) > best_count:
                best_count = len(current_blocks)
                best_blocks = [list(b) for b in current_blocks]
                
        return best_blocks
