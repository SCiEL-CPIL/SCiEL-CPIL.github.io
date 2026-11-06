import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 110.0

    def elapsed():
        return time.time() - start_time

    def build_paley(p):
        squares = set()
        for x in range(1, p):
            squares.add((x*x) % p)
        
        orientation = []
        for i in range(p):
            for j in range(i+1, p):
                diff = (j - i) % p
                if diff in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return p, orientation

    def get_in_nb(n, orientation, active):
        # active is a list of original indices
        # We need in_nb for the active vertices, indexed 0..len(active)-1
        # in_nb[i] is a bitmask of active vertices that point to active[i]
        m = len(active)
        # Map original index to new index
        orig_to_new = {orig: new for new, orig in enumerate(active)}
        
        in_nb = [0] * m
        
        # We need to check all pairs in active
        # It's faster to iterate over the original orientation and filter?
        # Or just iterate i, j in active.
        # m is ~50-60. m^2 is 3000. Very fast.
        
        # Build adj matrix for active
        # adj[i][j] = 1 if active[i] -> active[j]
        adj = [[0]*m for _ in range(m)]
        
        # Reconstruct adj from orientation
        # Orientation is for original indices 0..n-1
        # We need a fast lookup.
        # Since n is small (67), we can build a full adj matrix once for the original graph.
        # Let's do that outside the loop.
        
        pass # Placeholder

    # Build original adj matrix
    p_init, orientation = build_paley(67)
    n_orig = p_init
    adj_orig = np.zeros((n_orig, n_orig), dtype=np.uint8)
    idx = 0
    for i in range(n_orig):
        for j in range(i+1, n_orig):
            if orientation[idx] == 0:
                adj_orig[i, j] = 1
            else:
                adj_orig[j, i] = 1
            idx += 1

    active = list(range(n_orig))
    
    while elapsed() < DEADLINE:
        m = len(active)
        if m < 5: break
        
        # Map active indices to 0..m-1
        orig_to_new = {orig: new for new, orig in enumerate(active)}
        
        # Build in_nb for active subgraph
        in_nb = [0] * m
        for i_new, i_orig in enumerate(active):
            mask = 0
            for j_new, j_orig in enumerate(active):
                if i_new == j_new: continue
                # If j -> i, then j is in in_nb of i
                if adj_orig[j_orig, i_orig] == 1:
                    mask |= (1 << j_new)
            in_nb[i_new] = mask
            
        # Find critical vertices
        critical = [False] * m
        
        # Iterate over all 4-sets in active
        # Using indices 0..m-1
        for a in range(m - 3):
            if elapsed() > DEADLINE: break
            for b in range(a + 1, m - 2):
                for c in range(b + 1, m - 1):
                    abc = in_nb[a] & in_nb[b] & in_nb[c]
                    if abc == 0: continue
                    for d in range(c + 1, m):
                        D = abc & in_nb[d]
                        if D == 0:
                            # Invalid tournament (should not happen if we started valid and only removed safe vertices)
                            # But if it happens, we stop
                            critical = [True] * m
                            break
                        if D & (D - 1) == 0:
                            w = D.bit_length() - 1
                            critical[w] = True
                    if D == 0: break
                if D == 0: break
            if D == 0: break
            
        if all(critical):
            break
            
        # Find a non-critical vertex
        removable_idx = -1
        for i in range(m):
            if not critical[i]:
                removable_idx = i
                break
                
        if removable_idx == -1:
            break
            
        # Remove active[removable_idx]
        active.pop(removable_idx)

    # Construct final orientation
    n_final = len(active)
    # Map active to 0..n_final-1 is implicit in order
    final_orientation = []
    for i in range(n_final):
        for j in range(i+1, n_final):
            orig_i = active[i]
            orig_j = active[j]
            if adj_orig[orig_i, orig_j] == 1:
                final_orientation.append(0)
            else:
                final_orientation.append(1)
                
    return {'n': n_final, 'orientation': final_orientation}
