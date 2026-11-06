import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Deadline guard: ensure we return well before the 120s limit.
    # The construction is deterministic and fast, so this is mostly a safety net.
    deadline = start_time + 110.0
    
    # Design: 8 disjoint copies of K4.
    # n = 8 * 4 = 32.
    # Properties:
    # - No C5: Components are K4 (max cycle length 4).
    # - No independent set of 9: alpha = 8 * alpha(K4) = 8 * 1 = 8.
    # This satisfies n <= 35, no C5, and alpha < 9.
    # This matches the "best known" lower bound mentioned in the prompt.
    
    blocks = 8
    size_per_block = 4
    n = blocks * size_per_block
    
    edges = []
    for b in range(blocks):
        base = b * size_per_block
        for i in range(size_per_block):
            for j in range(i + 1, size_per_block):
                edges.append([base + i, base + j])
                
    # Check if time is running out (unlikely for this static construction)
    if time.time() > deadline:
         # Fallback to a smaller valid graph if something went wrong (shouldn't happen)
         # But since it's static, we just return.
         pass

    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
