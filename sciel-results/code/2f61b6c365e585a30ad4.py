import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s buffer for safety

    # We'll work with n=9 (K_9^5 has 126 edges, which is non-2-colorable by pigeonhole)
    n = 9
    all_edges = [list(e) for e in combinations(range(n), 5)]
    num_edges = len(all_edges)  # 126

    # Precompute edge masks (bitmask of vertices in each edge)
    edge_masks = []
    for e in all_edges:
        mask_e = 0
        for v in e:
            mask_e |= (1 << v)
        edge_masks.append(mask_e)

    # Precompute bad_masks: for each coloring c (0..2^9-1), bad_masks[c] is a bitmask
    # of edges that are monochromatic under coloring c
    bad_masks = [0] * (1 << n)
    for c in range(1 << n):
        bm = 0
        for i in range(num_edges):
            mask_e = edge_masks[i]
            inter = c & mask_e
            if inter == 0 or inter == mask_e:
                bm |= (1 << i)
        bad_masks[c] = bm

    # Quick check: if any coloring has no monochromatic edges, the full graph is colorable
    if any(bm == 0 for bm in bad_masks):
        # Should not happen for n=9, but just in case
        # Fallback: return full K_9^5
        return {'n': n, 'edges': all_edges}

    def is_non_colorable(active_mask):
        """Check if the subgraph defined by active_mask is non-2-colorable."""
        for bm in bad_masks:
            if (bm & active_mask) == 0:
                return False
        return True

    # Start with all edges
    current_mask = (1 << num_edges) - 1

    # Verify start is non-colorable
    if not is_non_colorable(current_mask):
        return {'n': n, 'edges': all_edges}

    best_local_mask = current_mask
    best_local_count = num_edges

    # Greedy reduction with multiple restarts
    # Use a set of edge indices for fast membership
    def count_bits(m):
        return bin(m).count('1')

    # Try multiple random reduction passes
    restarts = 0
    while time.time() < deadline - 2:
        restarts += 1
        curr = current_mask
        changed = True
        while changed:
            changed = False
            # Shuffle edge indices
            indices = list(range(num_edges))
            random.shuffle(indices)
            for i in indices:
                # Try removing edge i
                if curr & (1 << i):
                    new_mask = curr & ~(1 << i)
                    if is_non_colorable(new_mask):
                        curr = new_mask
                        changed = True
                        # Continue trying to remove more

        # Update best
        count = count_bits(curr)
        if count < best_local_count:
            best_local_count = count
            best_local_mask = curr

        # If we reach 51 or fewer edges, we can stop early
        if best_local_count <= 51:
            break

    # Extract edges
    best_edges = [all_edges[i] for i in range(num_edges) if best_local_mask & (1 << i)]

    return {'n': n, 'edges': best_edges}
