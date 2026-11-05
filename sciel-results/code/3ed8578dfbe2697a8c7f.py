"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(unv[0])
    
    # Deadline guard: fall back to nearest-neighbor if time is running out
    if time.time() > _DEADLINE:
        return int(unv[np.argmin(distance_matrix[current_node, unv])])
    
    dm = distance_matrix
    best_score = np.inf
    best_node = int(unv[0])
    
    for c in unv:
        d1 = dm[current_node, c]
        # Pruning: if immediate cost is already worse than best total, skip
        if d1 >= best_score:
            continue
            
        if n > 2:
            # Find next best node x from c among remaining
            remaining = unv[unv != c]
            dists_c = dm[c, remaining]
            idx_x = np.argmin(dists_c)
            x = remaining[idx_x]
            d2 = dists_c[idx_x]
            
            score = d1 + 0.5 * d2
            
            if n > 3:
                # Find next best node y from x among remaining (excluding c and x)
                remaining2 = remaining[remaining != x]
                dists_x = dm[x, remaining2]
                idx_y = np.argmin(dists_x)
                d3 = dists_x[idx_y]
                score += 0.25 * d3
            
            if score < best_score:
                best_score = score
                best_node = int(c)
        else:
            if d1 < best_score:
                best_score = d1
                best_node = int(c)
                
    return best_node
# EVOLVE-BLOCK-END
