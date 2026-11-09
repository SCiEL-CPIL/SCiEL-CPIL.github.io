"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from itertools import combinations

def _score_from_sizes(n_diff, n_sum, max_u):
    """Compute score from set sizes and max element."""
    if max_u <= 0 or n_sum <= 0 or n_diff <= 0:
        return 0.0
    ratio = n_diff / n_sum
    if ratio <= 1.0:
        return 1.0
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom

def _compute_score(U):
    """Compute the score for a given set U."""
    U = sorted(set(U))
    if 0 not in U or len(U) == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    n = len(U)
    # Use numpy for efficient set operations
    arr = np.array(U, dtype=np.int64)
    # Sumset: all pairwise sums
    sums = arr[:, None] + arr[None, :]
    n_sum = len(np.unique(sums))
    # Difference set: all pairwise differences
    diffs = arr[:, None] - arr[None, :]
    n_diff = len(np.unique(diffs))
    return _score_from_sizes(n_diff, n_sum, max_u)

def compute_score(U):
    """Compute the score for a given set U (public interface)."""
    return _compute_score(U)

def _greedy_construct(start_time, deadline, max_elements=50, max_value=200):
    """Greedy constructive algorithm: iteratively select next element to maximize
    the immediate gain in |U-U|/|U+U| while penalizing log increase in max(U)."""
    U = [0, 1]
    arr = np.array(U, dtype=np.int64)
    
    current_score = _compute_score(U)
    
    # Pre-compute current sumset and diffset sizes
    sums = arr[:, None] + arr[None, :]
    diffs = arr[:, None] - arr[None, :]
    sumset = set(np.unique(sums).tolist())
    diffset = set(np.unique(diffs).tolist())
    n_sum = len(sumset)
    n_diff = len(diffset)
    max_u = 1
    
    for step in range(max_elements - 2):
        if time.time() > deadline:
            break
        
        # Try candidate values
        # Focus on values that might expand the difference set more than the sumset
        # Candidates: small values near existing elements, and some larger gaps
        candidates = set()
        # Small values
        for v in range(1, min(max_u + 10, max_value + 1)):
            if v not in U:
                candidates.add(v)
        # Larger gaps - geometric-like extensions
        for v in range(max_u + 1, min(max_u + 50, max_value + 1)):
            candidates.add(v)
        # Some strategic larger values
        for v in [max_u * 2, max_u * 3, max_u + max_u // 2]:
            if v < max_value and v not in U:
                candidates.add(v)
        
        if not candidates:
            break
        
        best_candidate = None
        best_gain = -1e18
        best_new_score = current_score
        
        for cand in sorted(candidates):
            if time.time() > deadline:
                break
            
            # Compute new sumset and diffset sizes
            # New sums: existing sums + (cand + u for u in U)
            new_sums = set(sumset)
            for u in U:
                new_sums.add(cand + u)
            n_new_sum = len(new_sums)
            
            # New diffs: existing diffs + (cand - u for u in U) + (u - cand for u in U)
            new_diffs = set(diffset)
            for u in U:
                new_diffs.add(cand - u)
                new_diffs.add(u - cand)
            n_new_diff = len(new_diffs)
            
            new_max_u = max(max_u, cand)
            
            # Compute the score with the new element
            if n_new_sum > 0 and n_new_diff > 0 and new_max_u > 0:
                ratio = n_new_diff / n_new_sum
                if ratio > 1.0:
                    denom = math.log(2 * new_max_u + 1)
                    if denom > 0:
                        new_score = 1.0 + math.log(ratio) / denom
                    else:
                        new_score = 0.0
                else:
                    new_score = 1.0
            else:
                new_score = 0.0
            
            # We want to maximize new_score, but also consider the gain
            if new_score > best_new_score + 1e-12:
                best_new_score = new_score
                best_candidate = cand
                best_gain = new_score - current_score
        
        if best_candidate is None:
            break
        
        # Add the best candidate
        U.append(best_candidate)
        U.sort()
        arr = np.array(U, dtype=np.int64)
        
        # Update sumset and diffset
        sums = arr[:, None] + arr[None, :]
        diffs = arr[:, None] - arr[None, :]
        sumset = set(np.unique(sums).tolist())
        diffset = set(np.unique(diffs).tolist())
        n_sum = len(sumset)
        n_diff = len(diffset)
        max_u = max(U)
        current_score = _compute_score(U)
    
    return U, current_score

def _local_search_numpy(U, deadline, max_steps=200):
    """Local search using numpy for efficient evaluation."""
    U = sorted(set(U))
    best_score = _compute_score(U)
    best_set = U[:]
    max_u = max(U)
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        improved = False
        n = len(U)
        
        # Try adding a new element
        for new_val in range(1, max_u + 30):
            if new_val not in U:
                U_new = sorted(U + [new_val])
                score = _compute_score(U_new)
                if score > best_score + 1e-12:
                    best_score = score
                    best_set = U_new
                    U = U_new
                    max_u = max(U)
                    improved = True
                    break
        
        # Try removing an element (if size > 2)
        if len(U) > 2:
            for i in range(1, len(U)):  # Don't remove 0
                U_new = U[:i] + U[i+1:]
                score = _compute_score(U_new)
                if score > best_score + 1e-12:
                    best_score = score
                    best_set = U_new
                    U = U_new
                    max_u = max(U)
                    improved = True
                    break
        
        # Try replacing an element
        for i in range(1, len(U)):
            old_val = U[i]
            for new_val in range(1, max_u + 40):
                if new_val == old_val or new_val in U:
                    continue
                U_new = sorted(U[:i] + [new_val] + U[i+1:])
                score = _compute_score(U_new)
                if score > best_score + 1e-12:
                    best_score = score
                    best_set = U_new
                    U = U_new
                    max_u = max(U)
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_set, best_score

def construct():
    """Main construct function."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with known good small sets
    candidates = [
        [0, 1, 3],
        [0, 1, 4, 9],
        [0, 1, 3, 7],
        [0, 1, 4, 13],
    ]
    
    best_set = None
    best_score = 0.0
    
    for U in candidates:
        score = _compute_score(U)
        if score > best_score:
            best_score = score
            best_set = U[:]
    
    # Greedy constructive algorithm
    if time.time() < deadline:
        greedy_set, greedy_score = _greedy_construct(start_time, deadline, max_elements=30, max_value=300)
        if greedy_score > best_score:
            best_score = greedy_score
            best_set = greedy_set[:]
    
    # Local search from the best set found so far
    if best_set and time.time() < deadline:
        ls_set, ls_score = _local_search_numpy(best_set, deadline, max_steps=300)
        if ls_score > best_score:
            best_score = ls_score
            best_set = ls_set
    
    # Try geometric constructions and local search
    for k in range(3, 12):
        if time.time() > deadline:
            break
        for r in [2, 3, 4, 5]:
            U = [0, 1]
            for i in range(2, k):
                U.append(U[-1] * r)
            score = _compute_score(U)
            if score > best_score:
                best_score = score
                best_set = U[:]
            
            # Local search on geometric
            if time.time() < deadline:
                ls_set, ls_score = _local_search_numpy(best_set, deadline, max_steps=150)
                if ls_score > best_score:
                    best_score = ls_score
                    best_set = ls_set
    
    # Final local search polish
    if best_set and time.time() < deadline:
        ls_set, ls_score = _local_search_numpy(best_set, deadline, max_steps=400)
        if ls_score > best_score:
            best_score = ls_score
            best_set = ls_set
    
    return best_set if best_set else [0, 1, 3]
# EVOLVE-BLOCK-END
