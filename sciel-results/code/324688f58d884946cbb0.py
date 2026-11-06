"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations
import random

# EVOLVE-BLOCK-START
def _build_bitmask(n, edges):
    """Build adjacency as list of bitmasks for fast triangle detection."""
    mask = [0] * n
    for u, v in edges:
        mask[u] |= (1 << v)
        mask[v] |= (1 << u)
    return mask

def _compute_score_fast(n, mask, deg):
    """Compute max triangle degree sum using bitmask operations."""
    max_score = 0
    for i in range(n):
        mi = mask[i]
        # For each j > i adjacent to i
        m = mi & ~((1 << (i + 1)) - 1)  # only j > i
        while m:
            j_bit = m & (-m)
            j = j_bit.bit_length() - 1
            m ^= j_bit
            # Common neighbors of i and j with index > j
            common = mask[i] & mask[j] & ~((1 << (j + 1)) - 1)
            while common:
                k_bit = common & (-common)
                k = k_bit.bit_length() - 1
                common ^= k_bit
                s = deg[i] + deg[j] + deg[k]
                if s > max_score:
                    max_score = s
    return max_score

def _get_non_edges(n, edge_set):
    return [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]

def _local_search(n, edges, max_time=100, seed=42):
    start_time = time.time()
    deadline = start_time + max_time
    
    edge_set = set(tuple(sorted(e)) for e in edges)
    mask = _build_bitmask(n, edges)
    deg = np.zeros(n, dtype=np.int64)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    
    current_score = _compute_score_fast(n, mask, deg)
    best_score = current_score
    best_edge_set = set(edge_set)
    
    rng = random.Random(seed)
    
    no_improve_count = 0
    max_no_improve = 20 # After 20 full passes without improvement, perturb
    
    while time.time() < deadline:
        non_edges = _get_non_edges(n, edge_set)
        if not non_edges:
            break
        
        improved = False
        shuffled_remove = list(edge_set)
        rng.shuffle(shuffled_remove)
        
        for e_remove in shuffled_remove:
            if time.time() >= deadline:
                break
            u1, v1 = e_remove
            
            shuffled_add = list(non_edges)
            rng.shuffle(shuffled_add)
            
            for e_add in shuffled_add:
                u2, v2 = e_add
                if (u2, v2) in edge_set:
                    continue
                if u2 == u1 and v2 == v1:
                    continue
                
                # Tentative swap
                mask[u1] &= ~(1 << v1)
                mask[v1] &= ~(1 << u1)
                mask[u2] |= (1 << v2)
                mask[v2] |= (1 << u2)
                deg[u1] -= 1
                deg[v1] -= 1
                deg[u2] += 1
                deg[v2] += 1
                
                new_score = _compute_score_fast(n, mask, deg)
                
                if new_score < best_score:
                    best_score = new_score
                    best_edge_set = set(edge_set)
                    best_edge_set.discard(e_remove)
                    best_edge_set.add(tuple(sorted(e_add)))
                    current_score = new_score
                    improved = True
                    edge_set.discard(e_remove)
                    edge_set.add(tuple(sorted(e_add)))
                else:
                    # Revert
                    mask[u1] |= (1 << v1)
                    mask[v1] |= (1 << u1)
                    mask[u2] &= ~(1 << v2)
                    mask[v2] &= ~(1 << u2)
                    deg[u1] += 1
                    deg[v1] += 1
                    deg[u2] -= 1
                    deg[v2] -= 1
                
                if improved:
                    break
        
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturb: swap 2-3 random edges
                # This might increase score temporarily, but we only keep if it leads to better best_score later?
                # No, local search usually keeps current state.
                # If we perturb, we change current state.
                # We should only perturb if we are stuck.
                # Let's do a random walk of 5 steps.
                for _ in range(5):
                    if time.time() >= deadline: break
                    e_remove = rng.choice(list(edge_set))
                    non_edges = _get_non_edges(n, edge_set)
                    if not non_edges: break
                    e_add = rng.choice(non_edges)
                    if e_add in edge_set: continue
                    if e_add == e_remove: continue
                    
                    u1, v1 = e_remove
                    u2, v2 = e_add
                    
                    mask[u1] &= ~(1 << v1)
                    mask[v1] &= ~(1 << u1)
                    mask[u2] |= (1 << v2)
                    mask[v2] |= (1 << u2)
                    deg[u1] -= 1
                    deg[v1] -= 1
                    deg[u2] += 1
                    deg[v2] += 1
                    
                    edge_set.discard(e_remove)
                    edge_set.add(tuple(sorted(e_add)))
                    
                no_improve_count = 0
                
    best_edges = [list(e) for e in best_edge_set]
    return best_edges, best_score

def construct():
    n = 16
    target_edges = 65
    start_time = time.time()
    
    # Start from K_{6,10} + 5 internal edges in the 10-vertex part
    initial_edges = []
    for u in range(6):
        for v in range(6, 16):
            initial_edges.append([u, v])
    
    internal_edges = [(6,7), (8,9), (10,11), (12,13), (14,15)]
    for u, v in internal_edges:
        initial_edges.append([u, v])
    
    assert len(initial_edges) == 65
    
    # Run local search
    # Use 140s budget
    best_edges, best_score = _local_search(n, initial_edges, max_time=140, seed=42)
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
