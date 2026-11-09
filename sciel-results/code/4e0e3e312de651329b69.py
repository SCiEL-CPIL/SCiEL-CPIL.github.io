# EVOLVE-BLOCK-START
import time

def construct():
    """Construct densest graph on 16 vertices with diameter 2 where every edge is critical.
    
    Uses K_{8,8} bipartite graph: by Mantel's theorem it's the max-edge triangle-free
    graph, and no two vertices in the same part share a common neighbor, so removing
    any edge breaks all length-2 paths between its endpoints, raising diameter to 3.
    """
    deadline = time.monotonic() + 85.0  # safety margin before 90s wall-clock kill
    
    # Partition vertices into two sets of 8
    left_part = tuple(i for i in range(8))
    right_part = tuple(8 + i for i in range(8))
    
    # Build all 64 edges between the two partitions
    result = [[u, v] for u in left_part for v in right_part]
    
    # Verify types are exactly Python int (not numpy, not bool)
    for e in result:
        if not (isinstance(e[0], int) and not isinstance(e[0], bool) and
                isinstance(e[1], int) and not isinstance(e[1], bool)):
            # Fallback: force plain int conversion
            result = [[int(u), int(v)] for u in left_part for v in right_part]
            break
    
    # Final safety check: return within deadline
    if time.monotonic() > deadline:
        return result
    
    return result
# EVOLVE-BLOCK-START
