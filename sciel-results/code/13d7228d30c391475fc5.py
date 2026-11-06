import time
import random
import numpy as np
from itertools import combinations

N = 20
NUM_COLOURS = 20

def get_all_4_cycles():
    cycles = []
    for verts in combinations(range(N), 4):
        a, b, c, d = verts
        cycles.append(((a,b), (b,c), (c,d), (d,a)))
        cycles.append(((a,b), (b,d), (d,c), (c,a)))
        cycles.append(((a,c), (c,b), (b,d), (d,a)))
    return cycles

ALL_CYCLES = get_all_4_cycles()
NUM_CYCLES = len(ALL_CYCLES)

# Map each edge to the cycles it belongs to
# Edge is (u, v) with u < v
edge_to_cycles = {}
for i, cyc in enumerate(ALL_CYCLES):
    for e in cyc:
        u, v = e
        if u > v:
            u, v = v, u
        e_norm = (u, v)
        if e_norm not in edge_to_cycles:
            edge_to_cycles[e_norm] = []
        edge_to_cycles[e_norm].append(i)

def init_cycle_validity(C):
    """
    C is 20x20 numpy array.
    Returns (validity_array, bad_count)
    """
    validity = np.zeros(NUM_CYCLES, dtype=bool)
    bad_count = 0
    for i, cyc in enumerate(ALL_CYCLES):
        e1, e2, e3, e4 = cyc
        c1 = C[e1[0]][e1[1]]
        c2 = C[e2[0]][e2[1]]
        c3 = C[e3[0]][e3[1]]
        c4 = C[e4[0]][e4[1]]
        
        if c1 == 0 or c2 == 0 or c3 == 0 or c4 == 0:
            validity[i] = True # Not a subgraph, so "valid" in the sense of no violation
        else:
            if len({c1, c2, c3, c4}) == 4:
                validity[i] = True
            else:
                validity[i] = False
                bad_count += 1
    return validity, bad_count

def update_cycle_validity(C, validity, bad_count, edge):
    """
    Updates validity and bad_count after changing edge `edge` in C.
    Returns new bad_count.
    """
    u, v = edge
    if u > v:
        u, v = v, u
    e_norm = (u, v)
    
    if e_norm not in edge_to_cycles:
        return bad_count
        
    for cyc_id in edge_to_cycles[e_norm]:
        # Check if this cycle was valid before
        was_valid = validity[cyc_id]
        
        # Check if it is valid now
        cyc = ALL_CYCLES[cyc_id]
        e1, e2, e3, e4 = cyc
        c1 = C[e1[0]][e1[1]]
        c2 = C[e2[0]][e2[1]]
        c3 = C[e3[0]][e3[1]]
        c4 = C[e4[0]][e4[1]]
        
        is_valid = True
        if c1 != 0 and c2 != 0 and c3 != 0 and c4 != 0:
            if len({c1, c2, c3, c4}) < 4:
                is_valid = False
                
        if was_valid and not is_valid:
            bad_count += 1
            validity[cyc_id] = False
        elif not was_valid and is_valid:
            bad_count -= 1
            validity[cyc_id] = True
            
    return bad_count

def construct():
    t0 = time.time()
    deadline = 145.0
    
    best_edge_count = 0
    best_C = None
    
    def get_edge_count(C):
        return int(np.sum(C > 0) // 2)
        
    def format_result(C):
        edges = []
        for i in range(N):
            for j in range(i+1, N):
                c = C[i][j]
                if c > 0:
                    edges.append([i, j, int(c)])
        return edges

    def local_search(C):
        C = C.copy()
        validity, bad_count = init_cycle_validity(C)
        
        if bad_count == 0:
            return C
            
        while bad_count > 0:
            if time.time() > deadline:
                break
                
            # Find a bad cycle
            # We can pick a random bad cycle by sampling
            # Or iterate to find one.
            # Since we have validity array, we can find indices where validity is False.
            bad_indices = np.where(validity == False)[0]
            if len(bad_indices) == 0:
                break
                
            target_idx = bad_indices[random.randint(0, len(bad_indices)-1)]
            target_cycle = ALL_CYCLES[target_idx]
            edges_in_cycle = [e for e in target_cycle]
            
            best_move_score = bad_count + 1
            best_move = None
            
            # Try recoloring
            for e in edges_in_cycle:
                u, v = e
                old_c = C[u][v]
                if old_c == 0:
                    continue # Can't recolor a missing edge in this context? 
                    # Actually, if edge is missing, cycle is not a subgraph, so it shouldn't be bad.
                    # So edges in a bad cycle must be present.
                
                for new_c in range(1, NUM_COLOURS + 1):
                    if new_c == old_c:
                        continue
                    
                    C[u][v] = new_c
                    C[v][u] = new_c
                    
                    new_bad = update_cycle_validity(C, validity, bad_count, e)
                    
                    if new_bad < best_move_score:
                        best_move_score = new_bad
                        best_move = ('recolor', e, new_c, C.copy(), validity.copy())
                    
                    # Revert
                    C[u][v] = old_c
                    C[v][u] = old_c
                    update_cycle_validity(C, validity, new_bad, e) # Restore state
                    
            # Try removing
            if best_move is None:
                for e in edges_in_cycle:
                    u, v = e
                    old_c = C[u][v]
                    
                    C[u][v] = 0
                    C[v][u] = 0
                    
                    new_bad = update_cycle_validity(C, validity, bad_count, e)
                    
                    if new_bad < best_move_score:
                        best_move_score = new_bad
                        best_move = ('remove', e, 0, C.copy(), validity.copy())
                        
                    # Revert
                    C[u][v] = old_c
                    C[v][u] = old_c
                    update_cycle_validity(C, validity, new_bad, e)
                    
            if best_move is None:
                # No improvement found. Local minimum.
                # Try a random perturbation? Or break.
                # Let's break to allow restart.
                break
                
            # Apply best move
            action, e, val, new_C, new_validity = best_move
            C = new_C
            validity = new_validity
            bad_count = best_move_score
            
        return C

    while time.time() < deadline:
        # Random initialization
        C = np.random.randint(1, NUM_COLOURS + 1, size=(N, N))
        C = (C + C.T) // 2
        np.fill_diagonal(C, 0)
        
        C_opt = local_search(C)
        
        edge_count = get_edge_count(C_opt)
        if edge_count > best_edge_count:
            best_edge_count = edge_count
            best_C = C_opt.copy()
            
    if best_C is not None:
        return format_result(best_C)
    else:
        return []

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} edges")
