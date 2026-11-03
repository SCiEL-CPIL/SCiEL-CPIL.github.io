import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    start_time = time.time()
    deadline = 130.0  # 20s margin before 150s kill
    
    # Generate all weight-3 vectors in R^11
    # C(11,3) * 2^3 = 165 * 8 = 1320 vectors
    vectors_list = []
    for support in itertools.combinations(range(D), 3):
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[support[0]] = s1
                    v[support[1]] = s2
                    v[support[2]] = s3
                    vectors_list.append(v)
    
    vectors = np.array(vectors_list)
    N = len(vectors)
    
    # Condition: for all distinct u, v in set, u·v <= 1
    # Precompute dot product matrix
    M = vectors @ vectors.T
    
    # Conflict: M[i,j] > 1 (and i != j)
    # Build conflict as boolean matrix
    conflict = (M > 1).astype(np.int32)
    np.fill_diagonal(conflict, 0)
    
    # Degree of each node
    degrees = conflict.sum(axis=1)
    
    best_count = 0
    best_selected = None
    
    # Strategy 1: Multiple greedy approaches with different orderings
    rng = np.random.default_rng(12345)
    
    num_greedy_runs = 500
    for run in range(num_greedy_runs):
        if time.time() > deadline:
            break
        
        if run == 0:
            # First run: greedy by ascending degree (fewest conflicts first)
            order = np.argsort(degrees)
        elif run == 1:
            # Second run: greedy by descending degree
            order = np.argsort(degrees)[::-1]
        else:
            # Random restarts
            order = rng.permutation(N)
        
        # Greedy: process in order, add if no conflict with already selected
        selected = []
        selected_set = set()
        # For speed, maintain a bitset of conflicts with selected
        # Since N=1320, use a boolean array
        forbidden = np.zeros(N, dtype=bool)
        
        for idx in order:
            if forbidden[idx]:
                continue
            selected.append(idx)
            selected_set.add(idx)
            # Mark all conflicts of idx as forbidden
            forbidden |= (conflict[idx] == 1)
        
        count = len(selected)
        if count > best_count:
            best_count = count
            best_selected = selected[:]
        
        # Local search: try to improve
        if time.time() > deadline - 10:
            break
        
        # Local search phase
        for ls_iter in range(200):
            if time.time() > deadline - 5:
                break
            
            improved = False
            selected_set = set(best_selected)
            # Build forbidden from selected
            forbidden = np.zeros(N, dtype=bool)
            for idx in best_selected:
                forbidden |= (conflict[idx] == 1)
            # Remove self from forbidden for selected
            for idx in best_selected:
                forbidden[idx] = False
            
            # Try to add each non-selected node
            added_any = False
            new_adds = []
            for idx in range(N):
                if idx in selected_set:
                    continue
                if forbidden[idx]:
                    continue
                # Can add idx directly
                new_adds.append(idx)
            
            if len(new_adds) > 0:
                # Add the one with fewest conflicts with remaining non-selected
                # Just add all that can be added greedily
                for idx in new_adds:
                    if idx not in selected_set and not forbidden[idx]:
                        selected_set.add(idx)
                        best_selected.append(idx)
                        forbidden |= (conflict[idx] == 1)
                        forbidden[idx] = False
                        improved = True
                
                if improved:
                    best_count = len(best_selected)
            
            # Try swaps: remove one selected, add one or more non-selected
            if not improved and ls_iter < 100:
                # Try to find a swap that improves
                # For each selected node, check if removing it allows adding 2+ new nodes
                # This is expensive, so do a limited search
                if time.time() > deadline - 3:
                    break
                
                # Pick a random selected node to try removing
                if len(best_selected) > 0:
                    try_idx = best_selected[rng.integers(len(best_selected))]
                    # Compute what we'd gain
                    # Remove try_idx: un-forbid its conflicts
                    # But we need to track carefully
                    # Simpler: just try pairwise swaps
                    pass
        
        # Periodically try a different strategy: greedy with random tie-breaking
        if run % 50 == 0 and run > 0:
            # Weighted random: prefer nodes with fewer conflicts among unselected
            order = np.arange(N)
            for i in range(N):
                # With probability proportional to (1/degree), swap
                if i > 0 and rng.random() < 0.3:
                    j = rng.integers(0, i + 1)
                    order[i], order[j] = order[j], order[i]
            
            selected = []
            forbidden = np.zeros(N, dtype=bool)
            for idx in order:
                if forbidden[idx]:
                    continue
                selected.append(idx)
                forbidden |= (conflict[idx] == 1)
            
            if len(selected) > best_count:
                best_count = len(selected)
                best_selected = selected[:]
    
    # Final local search polish
    if best_selected is not None and time.time() < deadline - 2:
        # Try to add any remaining compatible vectors
        selected_set = set(best_selected)
        forbidden = np.zeros(N, dtype=bool)
        for idx in best_selected:
            forbidden |= (conflict[idx] == 1)
        for idx in best_selected:
            forbidden[idx] = False
        
        for idx in range(N):
            if idx in selected_set:
                continue
            if not forbidden[idx]:
                best_selected.append(idx)
                selected_set.add(idx)
                forbidden |= (conflict[idx] == 1)
                forbidden[idx] = False
                best_count = len(best_selected)
    
    if best_count > 0:
        result = vectors[np.array(best_selected)]
    else:
        # Fallback: just return all vectors (invalid but shouldn't happen)
        result = vectors[:10]
    
    return result
