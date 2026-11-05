"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import random
import itertools
import numpy as np


def _dist_sq(p, q):
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def _precompute_conflicts():
    """
    Precompute for each point q, the set of pairs (a, b) such that
    triangle (q, a, b) is isosceles.
    Returns a list of sets, indexed by point index.
    Point index = x * N + y.
    Pair represented as (min(idx_a, idx_b), max(idx_a, idx_b)).
    """
    points = list(itertools.product(range(N), repeat=2))
    num_points = len(points)
    
    # Map coordinate to index
    coord_to_idx = {p: i for i, p in enumerate(points)}
    
    # Precompute squared distances between all pairs
    # dist[i][j]
    dist = [[0] * num_points for _ in range(num_points)]
    for i in range(num_points):
        for j in range(i + 1, num_points):
            d = _dist_sq(points[i], points[j])
            dist[i][j] = d
            dist[j][i] = d

    conflicts = [set() for _ in range(num_points)]
    
    for q_idx in range(num_points):
        q = points[q_idx]
        # Check all pairs (a, b) with a < b, a != q, b != q
        # Actually, we can just iterate all pairs a, b
        # Optimization: Iterate a, b such that a < b
        for a_idx in range(num_points):
            if a_idx == q_idx:
                continue
            for b_idx in range(a_idx + 1, num_points):
                if b_idx == q_idx:
                    continue
                # Check if triangle (q, a, b) is isosceles
                d_qa = dist[q_idx][a_idx]
                d_qb = dist[q_idx][b_idx]
                d_ab = dist[a_idx][b_idx]
                
                if d_qa == d_qb or d_qa == d_ab or d_qb == d_ab:
                    # Add pair (a, b) to conflicts of q
                    # Ensure a_idx < b_idx for consistent representation
                    pair = (a_idx, b_idx)
                    conflicts[q_idx].add(pair)
                    
    return points, conflicts


def _is_maximal(S_indices, points, conflicts, all_indices):
    """
    Check if set S_indices is maximal.
    S_indices: set of point indices in S.
    Returns True if maximal, False otherwise.
    """
    # Generate all pairs in S
    S_list = list(S_indices)
    s_pairs = set()
    for i in range(len(S_list)):
        for j in range(i + 1, len(S_list)):
            a, b = S_list[i], S_list[j]
            if a > b:
                a, b = b, a
            s_pairs.add((a, b))
            
    # Check if any point q NOT in S can be added
    # q can be added if conflicts[q] is disjoint from s_pairs
    for q_idx in all_indices:
        if q_idx in S_indices:
            continue
        # Check intersection
        # If conflicts[q_idx] has no common pair with s_pairs, then q can be added
        # If any q can be added, S is not maximal
        # Optimization: If conflicts[q_idx] is empty, it's definitely addable
        if not conflicts[q_idx]:
            return False
        # Check intersection
        # Since s_pairs might be large, and conflicts[q_idx] might be large
        # We can iterate over the smaller set
        if len(conflicts[q_idx]) < len(s_pairs):
            for pair in conflicts[q_idx]:
                if pair in s_pairs:
                    break
            else:
                return False # No intersection, q can be added
        else:
            for pair in s_pairs:
                if pair in conflicts[q_idx]:
                    break
            else:
                return False # No intersection, q can be added
                
    return True


def _greedy_add_initial(points, conflicts, rng):
    """
    Greedily add points to form a valid, maximal set.
    """
    S = set()
    remaining = list(range(len(points)))
    rng.shuffle(remaining)
    
    s_pairs = set()
    
    while remaining:
        # Find a point that can be added
        # A point p can be added if conflicts[p] is disjoint from s_pairs
        added = False
        for p_idx in remaining:
            # Check if p can be added
            # If conflicts[p_idx] intersects s_pairs, it's invalid
            valid = True
            if len(conflicts[p_idx]) < len(s_pairs):
                for pair in conflicts[p_idx]:
                    if pair in s_pairs:
                        valid = False
                        break
            else:
                for pair in s_pairs:
                    if pair in conflicts[p_idx]:
                        valid = False
                        break
                        
            if valid:
                # Add p
                S.add(p_idx)
                # Update s_pairs
                for q_idx in S:
                    if q_idx == p_idx:
                        continue
                    a, b = p_idx, q_idx
                    if a > b:
                        a, b = b, a
                    s_pairs.add((a, b))
                remaining.remove(p_idx)
                added = True
                break
        
        if not added:
            break
            
    return S, s_pairs


def construct():
    points, conflicts = _precompute_conflicts()
    all_indices = set(range(len(points)))
    
    best_set = None
    best_size = float('inf')
    
    # Multiple restarts
    for seed in range(20):
        rng = random.Random(seed)
        
        # 1. Greedy Add to get initial maximal valid set
        S, s_pairs = _greedy_add_initial(points, conflicts, rng)
        
        # 2. Greedy Removal to minimize size while maintaining maximality
        improved = True
        while improved:
            improved = False
            S_list = list(S)
            rng.shuffle(S_list)
            
            for p_idx in S_list:
                if p_idx not in S:
                    continue
                
                # Tentatively remove p_idx
                S_new = S - {p_idx}
                
                # Check if S_new is maximal
                # We need s_pairs for S_new
                # Instead of rebuilding s_pairs from scratch, we can remove pairs involving p_idx
                # But _is_maximal takes S_indices and rebuilds s_pairs internally currently.
                # Let's optimize _is_maximal to accept s_pairs?
                # Or just call it. For N=16, |S| is small.
                
                if _is_maximal(S_new, points, conflicts, all_indices):
                    S = S_new
                    improved = True
                    break # Restart the loop with new S
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S
            
    # Convert indices back to coordinates
    result = [points[i] for i in best_set]
    return result

# EVOLVE-BLOCK-END
