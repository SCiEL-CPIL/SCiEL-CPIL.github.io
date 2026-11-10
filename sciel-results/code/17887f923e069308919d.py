"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    # Greedy algorithm with time guard
    start_time = time.time()
    deadline = start_time + 85  # 85 seconds out of 90 second limit
    
    selected = set()
    for x in range(1, 513):
        if time.time() > deadline:
            break
        # Check if adding x creates a 4-AP
        can_add = True
        for d in range(1, (x-1)//3 + 1):
            # Check if x, x-d, x-2d, x-3d are all in selected (or being added)
            if x - d in selected and x - 2*d in selected and x - 3*d in selected:
                can_add = False
                break
        if can_add:
            selected.add(x)
    
    return sorted(selected)
# EVOLVE-BLOCK-END
