"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def _edge_index(u, v):
    """Return (axis, u) for edge between u and v (differ in one bit)."""
    diff = u ^ v
    axis = diff.bit_length() - 1
    return axis, u

def _matrices_to_edges(mats):
    """Convert list of 7 binary matrices to list of [u, v] edges."""
    edges = []
    for i in range(7):
        M = mats[i]
        for u in range(128):
            if M[u]:
                v = u ^ (1 << i)
                if u < v:
                    edges.append([u, v])
    return edges

def _count_edges(mats):
    return sum(int(M.sum()) for M in mats)

def _check_c4(mats, axis, u):
    """Check if adding edge (u, u^(1<<axis)) would create a C4.
    For each other axis j, check if there exists a neighbor direction such that
    the 4-cycle is completed.
    """
    v = u ^ (1 << axis)
    M_axis = mats[axis]
    for j in range(7):
        if j == axis:
            continue
        M_j = mats[j]
        # The four vertices of the face: u, v, u^bit_j, v^bit_j
        c = u ^ (1 << j)
        e = v ^ (1 << j)
        # Check if all three other edges exist
        # Edge u-c: axis j, at position min(u,c)
        # Edge v-e: axis j, at position min(v,e)
        # Edge c-e: axis axis, at position min(c,e)
        if M_j[min(u, c)] and M_j[min(v, e)] and M_axis[min(c, e)]:
            return False
    return True

def _remove_edge(mats, axis, u):
    """Remove edge from u in direction axis."""
    mats[axis][u] = 0

def _add_edge(mats, axis, u):
    """Add edge from u in direction axis (must be valid)."""
    mats[axis][u] = 1

def _greedy_random(mats, rng, order):
    """Greedy: process edges in given order, add if no C4 created."""
    for (axis, u) in order:
        if mats[axis][u]:
            continue
        if _check_c4(mats, axis, u):
            mats[axis][u] = 1

def _local_search_1flip(mats, rng, max_iters=500):
    """Try to add random missing edges or swap to improve."""
    improved = True
    while improved and max_iters > 0:
        improved = False
        max_iters -= 1
        # Try adding a random missing edge
        for _ in range(50):
            axis = rng.randint(0, 6)
            u = rng.randint(0, 127)
            if mats[axis][u]:
                continue
            if _check_c4(mats, axis, u):
                mats[axis][u] = 1
                improved = True
                break
    return mats

def _local_search_2swap(mats, rng, max_iters=200):
    """Try 2-swap: remove one edge, add another."""
    improved = True
    while improved and max_iters > 0:
        improved = False
        max_iters -= 1
        for _ in range(30):
            # Pick a random existing edge to potentially remove
            axis_rm = rng.randint(0, 6)
            u_rm = rng.randint(0, 127)
            if not mats[axis_rm][u_rm]:
                continue
            # Pick a random missing edge to add
            axis_add = rng.randint(0, 6)
            u_add = rng.randint(0, 127)
            if mats[axis_add][u_add]:
                continue
            # Try the swap
            mats[axis_rm][u_rm] = 0
            if _check_c4(mats, axis_add, u_add):
                mats[axis_add][u_add] = 1
                improved = True
            else:
                mats[axis_rm][u_rm] = 1
            if improved:
                break
    return mats

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    rng = random.Random(42)
    
    best_mats = None
    best_count = 0
    
    # Generate all edges
    all_edges = []
    for axis in range(7):
        for u in range(128):
            all_edges.append((axis, u))
    
    # Multiple restarts with different random orderings
    restart = 0
    while time.time() < deadline:
        restart += 1
        
        # Randomize edge order
        order = all_edges[:]
        rng.shuffle(order)
        
        # Initialize empty matrices
        mats = [np.zeros(128, dtype=np.uint8) for _ in range(7)]
        
        # Greedy construction
        _greedy_random(mats, rng, order)
        
        count = _count_edges(mats)
        
        # Local search: 1-flip (add edges)
        mats = _local_search_1flip(mats, rng, max_iters=300)
        count = _count_edges(mats)
        
        # Local search: 2-swap
        if time.time() < deadline:
            mats = _local_search_2swap(mats, rng, max_iters=100)
            count = _count_edges(mats)
        
        if count > best_count:
            best_count = count
            best_mats = [M.copy() for M in mats]
        
        # If we've hit a high count, do more local search
        if best_count >= 300 and time.time() < deadline - 5:
            mats = [M.copy() for M in best_mats]
            mats = _local_search_1flip(mats, rng, max_iters=500)
            count = _count_edges(mats)
            if count > best_count:
                best_count = count
                best_mats = [M.copy() for M in mats]
            
            if time.time() < deadline - 5:
                mats = _local_search_2swap(mats, rng, max_iters=200)
                count = _count_edges(mats)
                if count > best_count:
                    best_count = count
                    best_mats = [M.copy() for M in mats]
    
    # Convert best matrices to edge list
    if best_mats is None:
        # Fallback: return empty or minimal valid set
        return []
    
    return _matrices_to_edges(best_mats)

# EVOLVE-BLOCK-END
