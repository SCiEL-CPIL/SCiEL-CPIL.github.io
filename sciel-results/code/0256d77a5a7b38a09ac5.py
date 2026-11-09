"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _edge_key(x, y):
    return (min(x, y), max(x, y))

def _face_conflicts(x, y, chosen):
    """Return set of edges in chosen that would form a C4 if (x,y) were added."""
    d = 7
    conflicts = set()
    for j in range(d):
        if (1 << j) & (x ^ y):
            continue  # j is the direction of edge (x,y)
        c = x ^ (1 << j)
        e = y ^ (1 << j)
        ec1 = _edge_key(x, c)
        ec2 = _edge_key(y, e)
        ec3 = _edge_key(c, e)
        if ec1 in chosen and ec2 in chosen and ec3 in chosen:
            conflicts.update([ec1, ec2, ec3])
    return conflicts

def _greedy_init():
    d = 7
    chosen = set()
    out = []
    for x in range(2 ** d):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                ok = True
                for j in range(d):
                    if j != i:
                        c, e = x ^ (1 << j), y ^ (1 << j)
                        if all(p in chosen for p in ((_edge_key(x, c)), (_edge_key(y, e)), (_edge_key(c, e)))):
                            ok = False
                            break
                if ok:
                    key = _edge_key(x, y)
                    chosen.add(key)
                    out.append([x, y])
    return chosen, out

def _local_search(chosen, deadline):
    """Local search: try to add missing edges, removing conflicting ones if beneficial."""
    d = 7
    all_edges = []
    for x in range(2 ** d):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append(_edge_key(x, y))
    
    best_count = len(chosen)
    best_chosen = set(chosen)
    
    while time.time() < deadline - 5:
        # Find missing edges
        missing = [e for e in all_edges if e not in chosen]
        if not missing:
            break
        
        random.shuffle(missing)
        improved = False
        
        for edge in missing:
            if time.time() > deadline - 5:
                break
            conflicts = _face_conflicts(edge[0], edge[1], chosen)
            if not conflicts:
                # Can add directly
                chosen.add(edge)
                improved = True
            elif len(conflicts) == 1:
                # Removing one edge to add one is neutral; skip unless we want diversity
                pass
            elif len(conflicts) == 2:
                # Remove 2, add 1: net -1, skip
                pass
            # len(conflicts) >= 3: net loss, skip
        
        if improved:
            best_count = len(chosen)
            best_chosen = set(chosen)
        else:
            # No improvement possible via single additions; try 2-opt style
            # Try removing a pair of edges that are in the same face, then see if we can add more
            # This is complex; for now, just restart with different random order
            if best_count > len(chosen):
                chosen = set(best_chosen)
            break
    
    return best_chosen, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin: evaluator kills at 150s
    
    # Phase 1: Greedy initialization
    chosen, out = _greedy_init()
    
    # Phase 2: Local search to improve
    if time.time() < deadline - 10:
        chosen, count = _local_search(chosen, deadline)
    
    # Build output
    result = [[u, v] for (u, v) in chosen]
    return result
# EVOLVE-BLOCK-END
