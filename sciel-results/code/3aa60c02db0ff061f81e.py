# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations

def kissing_number11() -> np.ndarray:
    """
    Find a large kissing configuration in dimension 11 using:
    - Weight-2 integer vectors (norm^2 = 2)
    - Precomputed conflict graph where two vectors conflict if their squared distance < max_norm_sq
    - Multiple random restarts with greedy + local search to find large independent sets
    """
    d = 11
    deadline = time.time() + 590  # Safety margin before 600s limit
    
    # Generate all weight-2 vectors: (i, j) with signs (si, sj), i < j
    all_vectors = []
    all_indices = []  # store (i, j) for each vector
    all_signs = []    # store (si, sj)
    
    for i in range(d):
        for j in range(i + 1, d):
            for si in (-1, 1):
                for sj in (-1, 1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    all_vectors.append(p)
                    all_indices.append((i, j))
                    all_signs.append((si, sj))
    
    n = len(all_vectors)  # 220 vectors
    norm_sq = 2  # All have norm^2 = 2
    
    # Precompute conflict graph: two vectors conflict if dist^2 < norm_sq (=2)
    # dist^2 = |p - q|^2. For weight-2 vectors:
    # - If they share index i with same sign: diff has 0 at i, and other entries
    # - Let's just compute directly
    # dist^2 < 2 means dist^2 <= 1 (integer), so dist^2 = 0 or 1
    # dist^2 = 0 means same vector (won't happen since distinct)
    # dist^2 = 1 means they differ in exactly one coordinate by 1
    
    # Build adjacency list: conflicts[a] = set of b where dist(a,b)^2 < 2
    conflicts = [set() for _ in range(n)]
    
    for a in range(n):
        for b in range(a + 1, n):
            diff = all_vectors[a] - all_vectors[b]
            ds = int(np.dot(diff, diff))
            if ds < norm_sq:  # ds < 2, so ds <= 1
                conflicts[a].add(b)
                conflicts[b].add(a)
    
    # Now find a large independent set in this conflict graph
    # Using greedy with random ordering + local search
    
    best_set = []
    best_size = 0
    
    np.random.seed(42)
    random.seed(42)
    
    # Try multiple random restarts with greedy + local improvement
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() > deadline - 30:
            break
        
        # Random ordering
        order = list(range(n))
        random.shuffle(order)
        
        # Greedy: add vertex if not conflicting with already selected
        selected = set()
        for v in order:
            # Check if v conflicts with any selected
            if not conflicts[v].intersection(selected):
                selected.add(v)
        
        # Local search: try to swap or add to improve
        improved = True
        while improved:
            improved = False
            
            # Try adding a non-selected vertex
            for v in order:
                if v not in selected and not conflicts[v].intersection(selected):
                    selected.add(v)
                    improved = True
                    break
            
            # Try removing a selected vertex that allows adding multiple others
            # Simple version: try swapping one selected with one non-selected
            for v_sel in list(selected):
                for v_add in order:
                    if v_add in selected:
                        continue
                    # If we remove v_sel, can we add v_add?
                    new_selected = selected - {v_sel}
                    if not conflicts[v_add].intersection(new_selected):
                        # Check if this improves size
                        # We're replacing 1 with 1, so no net gain unless we can chain
                        # Skip simple swaps for now
                        pass
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = list(selected)
        
        # Early stop if we found a large set
        if best_size > 150:
            break
    
    # Additional local search on best set: try to add more vertices
    if time.time() < deadline - 20:
        selected = set(best_set)
        # Try each non-selected vertex
        for v in range(n):
            if v not in selected and not conflicts[v].intersection(selected):
                selected.add(v)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = list(selected)
    
    # Convert to array
    if not best_set:
        # Fallback: just return all 220 (they might not all be valid together)
        # Actually, let's just return the weight-2 vectors as before
        return np.array(all_vectors[:2], dtype=np.int64)
    
    result = np.array([all_vectors[i] for i in best_set], dtype=np.int64)
    
    # Final verification
    k = len(result)
    if k < 2:
        return np.array(all_vectors[:2], dtype=np.int64)
    
    max_norm_sq = 0
    for p in result:
        ns = int(np.dot(p, p))
        if ns > max_norm_sq:
            max_norm_sq = ns
    
    min_dist_sq = float('inf')
    for i in range(k):
        for j in range(i + 1, k):
            diff = result[i] - result[j]
            ds = int(np.dot(diff, diff))
            if ds < min_dist_sq:
                min_dist_sq = ds
    
    if min_dist_sq < max_norm_sq:
        # Shouldn't happen, but fallback
        # Return just the first two non-conflicting
        for i in range(k):
            for j in range(i+1, k):
                diff = result[i] - result[j]
                if int(np.dot(diff, diff)) >= max_norm_sq:
                    return result[[i, j]]
        return np.array(all_vectors[:2], dtype=np.int64)
    
    return result

# EVOLVE-BLOCK-END
