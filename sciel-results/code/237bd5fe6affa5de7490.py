import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Base blocks for S(4,5,11) on points 0..10
    # These are standard base blocks for the Witt design S(4,5,11)
    base_blocks = [
        (0, 1, 2, 4, 5),
        (0, 1, 2, 8, 9),
        (0, 1, 3, 6, 7),
        (0, 1, 3, 8, 10),
        (0, 1, 4, 7, 9),
        (0, 1, 4, 6, 10)
    ]
    
    s4511_blocks = []
    for bb in base_blocks:
        for shift in range(11):
            # Generate shifted block
            block = tuple((x + shift) % 11 for x in bb)
            # Ensure canonical ordering for uniqueness check if needed, 
            # but since we generate from distinct orbits, they are distinct.
            s4511_blocks.append(block)
            
    # Verify count: 6 * 11 = 66 blocks
    # In case of duplicates (which shouldn't happen for these base blocks), 
    # we use a set to be safe, though the standard base blocks are chosen to be distinct orbits.
    s4511_blocks = list(set(s4511_blocks))
    
    # Construct S(5,6,12) blocks
    # Points are 0..11 (11 is infinity)
    all_points_11 = set(range(11))
    
    blocks = []
    
    # Type 1: B U {11}
    for b in s4511_blocks:
        blocks.append(tuple(list(b) + [11]))
        
    # Type 2: B^c (complement in 0..10)
    for b in s4511_blocks:
        b_set = set(b)
        comp = tuple(sorted(all_points_11 - b_set))
        blocks.append(comp)
        
    # Return as list of lists
    return [list(b) for b in blocks]

