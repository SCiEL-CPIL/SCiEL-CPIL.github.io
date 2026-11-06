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
    """Compute max triangle degree sum using bitmask operations with pruning."""
    max_score = 0
    for i in range(n):
        mi = mask[i]
        # For each j > i adjacent to i
        m = mi & ~((1 << (i + 1)) - 1)  # only j > i
        while m:
            j_bit = m & (-m)
            j = j_bit.bit_length() - 1
            m ^= j_bit
            
            # Pruning: if deg[i] + deg[j] + max_possible_deg <= max_score, skip
            # max_possible_deg is max(deg[k]) for k > j, but we don't have that precomputed easily.
            # Instead, just compute the sum and check.
            
            # Common neighbors of i and j with index > j
            common = mask[i] & mask[j] & ~((1 << (j + 1)) - 1)
            while common:
                k_bit = common & (-common)
                k = k_bit.bit_length() - 1
                common ^= k_bit
                s = deg[i] + deg[j] + deg[k]
                if s > max_score:
                    max_score = s
                    # Early exit if we hit a theoretical lower bound? 
                    # Theoretical min for n=16 is ~23.4. If we hit 23, we can stop?
                    # But we want to minimize, so 23 is better than 24.
                    # If we find 23, we can break early if we assume it's optimal? 
                    # Let's not break, but if max_score <= 23, it's very good.
    return max_score

def _local_search(n, edges, max_time=100):
    start_time = time.time()
    deadline = start_time + max_time
    
    edge_set = set(tuple(sorted(e)) for e in edges)
    mask = _build_bitmask(n, edges)
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    
    current_score = _compute_score_fast(n, mask, deg)
    best_score = current_score
    best_edge_set = set(edge_set)
    
    all_pairs = set(tuple(sorted(p)) for p in combinations(range(n), 2))
    rng = random.Random(42)
    
    while time.time() < deadline:
        non_edges = list(all_pairs - edge_set)
        if not non_edges:
            break
        
        improved = False
        edges_list = list(edge_set)
        rng.shuffle(edges_list)
        rng.shuffle(non_edges)
        
        for e_remove in edges_list:
            if time.time() >= deadline:
                break
            u1, v1 = e_remove
            
            for e_add in non_edges:
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
                    # Update non_edges list? It's static for now, which is fine for 1-swap
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
        
        if not improved:
            break
    
    best_edges = [list(e) for e in best_edge_set]
    return best_edges, best_score

def _polish(n, edges, max_time=30):
    """Aggressive local search with 2-swaps and simulated annealing."""
    start_time = time.time()
    deadline = start_time + max_time
    
    edge_set = set(tuple(sorted(e)) for e in edges)
    mask = _build_bitmask(n, edges)
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    
    current_score = _compute_score_fast(n, mask, deg)
    best_score = current_score
    best_edge_set = set(edge_set)
    
    all_pairs = set(tuple(sorted(p)) for p in combinations(range(n), 2))
    rng = random.Random(42)
    
    # Simulated Annealing parameters
    T = 1.0
    T_min = 0.01
    cooling = 0.99
    
    while time.time() < deadline:
        non_edges = list(all_pairs - edge_set)
        if not non_edges:
            break
        
        improved = False
        edges_list = list(edge_set)
        rng.shuffle(edges_list)
        
        for e_remove in edges_list:
            if time.time() >= deadline:
                break
            u1, v1 = e_remove
            
            # 1-Swap
            for e_add in non_edges:
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
                
                if new_score < current_score:
                    # Accept improvement
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_edge_set = set(edge_set)
                        best_edge_set.discard(e_remove)
                        best_edge_set.add(tuple(sorted(e_add)))
                    edge_set.discard(e_remove)
                    edge_set.add(tuple(sorted(e_add)))
                    improved = True
                else:
                    # Simulated Annealing: accept worse with probability exp(-delta/T)
                    delta = new_score - current_score
                    if delta > 0 and rng.random() < np.exp(-delta / T):
                        current_score = new_score
                        edge_set.discard(e_remove)
                        edge_set.add(tuple(sorted(e_add)))
                        improved = True
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
                T *= cooling
                continue
                
            # 2-Swap: Remove e_remove1, e_remove2, Add e_add1, e_add2
            # This is expensive, so we only try it occasionally or if stuck
            if rng.random() < 0.1:
                # Pick a second edge to remove
                other_edges = [e for e in edge_set if e != e_remove]
                if other_edges:
                    e_remove2 = rng.choice(other_edges)
                    u3, v3 = e_remove2
                    
                    # Pick two edges to add
                    if len(non_edges) >= 2:
                        e_add1, e_add2 = rng.sample(non_edges, 2)
                        u4, v4 = e_add1
                        u5, v5 = e_add2
                        
                        # Check validity: no duplicate edges in the new set
                        # We remove (u1,v1) and (u3,v3). We add (u4,v4) and (u5,v5).
                        # Ensure added edges are not in the current set (they are from non_edges)
                        # Ensure added edges are distinct from removed edges (they are)
                        # Ensure added edges are distinct from each other
                        if (u4, v4) != (u5, v5) and (u4, v4) != (v5, u4) and (u5, v5) != (u4, v4):
                             # Also ensure we don't create multi-edges (simple graph constraint)
                             # Since we start with a simple graph and only swap, 
                             # we just need to ensure (u4,v4) is not (u3,v3) or (u1,v1)
                             # which is true since they are in non_edges.
                             
                            # Apply 2-swap
                            mask[u1] &= ~(1 << v1)
                            mask[v1] &= ~(1 << u1)
                            mask[u3] &= ~(1 << v3)
                            mask[v3] &= ~(1 << u3)
                            mask[u4] |= (1 << v4)
                            mask[v4] |= (1 << u4)
                            mask[u5] |= (1 << v5)
                            mask[v5] |= (1 << u5)
                            deg[u1] -= 1
                            deg[v1] -= 1
                            deg[u3] -= 1
                            deg[v3] -= 1
                            deg[u4] += 1
                            deg[v4] += 1
                            deg[u5] += 1
                            deg[v5] += 1
                            
                            new_score = _compute_score_fast(n, mask, deg)
                            
                            if new_score < current_score:
                                current_score = new_score
                                if new_score < best_score:
                                    best_score = new_score
                                    best_edge_set = set(edge_set)
                                    best_edge_set.discard(e_remove)
                                    best_edge_set.discard(e_remove2)
                                    best_edge_set.add(tuple(sorted(e_add1)))
                                    best_edge_set.add(tuple(sorted(e_add2)))
                                edge_set.discard(e_remove)
                                edge_set.discard(e_remove2)
                                edge_set.add(tuple(sorted(e_add1)))
                                edge_set.add(tuple(sorted(e_add2)))
                                improved = True
                            else:
                                # Revert
                                mask[u1] |= (1 << v1)
                                mask[v1] |= (1 << u1)
                                mask[u3] |= (1 << v3)
                                mask[v3] |= (1 << u3)
                                mask[u4] &= ~(1 << v4)
                                mask[v4] &= ~(1 << u4)
                                mask[u5] &= ~(1 << v5)
                                mask[v5] &= ~(1 << u5)
                                deg[u1] += 1
                                deg[v1] += 1
                                deg[u3] += 1
                                deg[v3] += 1
                                deg[u4] -= 1
                                deg[v4] -= 1
                                deg[u5] -= 1
                                deg[v5] -= 1
        
        if not improved:
            T *= cooling
            if T < T_min:
                break
    
    best_edges = [list(e) for e in best_edge_set]
    return best_edges, best_score

def construct():
    n = 16
    target_edges = 65
    start_time = time.time()
    total_deadline = 145  # safety margin before 150s limit
    
    # Start from K_{6,10} + 5 internal edges in the 10-vertex part
    initial_edges = []
    for u in range(6):
        for v in range(6, 16):
            initial_edges.append([u, v])
    
    internal_edges = [(6,7), (8,9), (10,11), (12,13), (14,15)]
    for u, v in internal_edges:
        initial_edges.append([u, v])
    
    assert len(initial_edges) == 65
    
    elapsed = time.time() - start_time
    search_time = min(100, total_deadline - 20)
    best_edges, best_score = _local_search(n, initial_edges, max_time=search_time)
    
    elapsed = time.time() - start_time
    polish_time = min(40, max(10, total_deadline - 10 - elapsed))
    best_edges, best_score = _polish(n, best_edges, max_time=polish_time)
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
