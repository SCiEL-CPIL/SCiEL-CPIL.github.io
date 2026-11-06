# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 59.0
    
    # Pre-compute roots for all quadratic residues
    # Store in a dictionary for O(1) lookup of roots
    # Also store in a set/list for sorted sliding window
    roots = {}
    residues = []
    
    for x in range(1, 1009):
        r = (x * x) % 1009
        # r is never 0 for 1 <= x < 1009
        if r not in roots:
            roots[r] = x
            residues.append(r)
            
    # Sort residues to enable sliding window check for consecutive integers
    residues.sort()
    
    # Sliding window of size 5
    # We look for 5 consecutive integers in the sorted list
    # If residues[i], ..., residues[i+4] are consecutive, 
    # then residues[i+4] - residues[i] == 4
    # Since the list is sorted and unique, this implies they are i, i+1, i+2, i+3, i+4
    
    for i in range(len(residues) - 4):
        if time.time() > deadline:
            # Return best found so far? 
            # Since we iterate in ascending order of residue value (sorted list),
            # the first valid window is the minimum s.
            # If we break, we might not have found one yet.
            # But finding s=1 is very fast.
            break
            
        if residues[i + 4] - residues[i] == 4:
            s = residues[i]
            # Verify s is within bounds (1 <= s <= 1004)
            # s is a residue, so 1 <= s <= 1008.
            # s+4 must be <= 1008? 
            # The task says 1 <= s <= 1004 and 1 <= ri <= 1008.
            # If s=1005, s+4=1009=0. 0 is excluded.
            # So s must be such that s+4 is a non-zero residue.
            # Our list contains only non-zero residues.
            # So if s+4 is in the list, it is non-zero.
            # Max residue is 1008.
            # If s=1005, s+4=1009 -> 0. Not in list.
            # So the condition residues[i+4] - residues[i] == 4 naturally handles bounds.
            
            return {'start': s, 'roots': [roots[s + j] for j in range(5)]}
            
    raise RuntimeError('No valid run found within deadline')
# EVOLVE-BLOCK-END
