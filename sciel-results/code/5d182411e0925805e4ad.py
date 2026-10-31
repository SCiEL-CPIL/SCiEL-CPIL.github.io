"""Dual-view construction: define U via Sidon-like difference structure, minimize additive energy."""
import numpy as np
import math
import random
import time

def construct():
    """
    Construct U using dual-view approach:
    - Prioritize compact sets (low max U) to minimize denominator.
    - Use Sidon-like constraints to maximize |U-U|.
    - Use local search to minimize |U+U| (additive energy).
    - Score = 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)
    """
    
    def compute_score(U):
        if not U or 0 not in U:
            return 0.0
        U_set = set(U)
        max_u = max(U_set)
        if max_u == 0:
            return 0.0
        sum_set = set()
        diff_set = set()
        # Optimize loops
        for a in U_set:
            for b in U_set:
                sum_set.add(a + b)
                diff_set.add(a - b)
        n_diff = len(diff_set)
        n_sum = len(sum_set)
        if n_sum == 0:
            return 0.0
        log_denom = math.log(2 * max_u + 1)
        if log_denom <= 0:
            return 0.0
        return 1.0 + math.log(n_diff / n_sum) / log_denom

    def local_search(U, max_iters=1000, max_val=1000):
        """
        Local search with focus on compactness and sum minimization.
        """
        U = sorted(set(U))
        if 0 not in U:
            U = sorted([0] + U)
        
        best_U = list(U)
        best_score = compute_score(U)
        current_score = best_score
        
        for iteration in range(max_iters):
            improved = False
            U_set = set(best_U)
            n = len(best_U)
            
            # 1. Try moving each element (except 0) to a nearby value
            for i, elem in enumerate(best_U):
                if elem == 0:
                    continue
                
                # Try a range of moves
                candidates_to_try = set()
                # Neighbors
                for delta in range(-10, 11):
                    nv = elem + delta
                    if nv >= 0 and nv not in U_set and nv < max_val:
                        candidates_to_try.add(nv)
                
                # Random global candidates (biased towards lower values for compactness)
                for _ in range(20):
                    nv = random.randint(1, max_val)
                    if nv not in U_set:
                        candidates_to_try.add(nv)
                
                # Evaluate candidates
                best_candidate = None
                best_candidate_score = current_score
                
                for nv in candidates_to_try:
                    new_U = list(best_U)
                    new_U[i] = nv
                    new_U = sorted(set(new_U))
                    if 0 not in new_U or len(new_U) < 2:
                        continue
                    s = compute_score(new_U)
                    if s > best_candidate_score:
                        best_candidate_score = s
                        best_candidate = nv
                
                if best_candidate is not None:
                    best_U[i] = best_candidate
                    best_U = sorted(set(best_U))
                    current_score = best_candidate_score
                    if current_score > best_score:
                        best_score = current_score
                    improved = True
                    break # Restart loop with new state
            
            # 2. Try adding a new element
            if not improved:
                # Find the best element to add
                best_add = None
                best_add_score = current_score
                U_set = set(best_U)
                
                # Try small values first (compactness)
                for nv in range(1, min(50, max_val)):
                    if nv in U_set:
                        continue
                    new_U = sorted(best_U + [nv])
                    s = compute_score(new_U)
                    if s > best_add_score:
                        best_add_score = s
                        best_add = nv
                
                # Try some larger random values
                for _ in range(50):
                    nv = random.randint(1, max_val)
                    if nv in U_set:
                        continue
                    new_U = sorted(best_U + [nv])
                    s = compute_score(new_U)
                    if s > best_add_score:
                        best_add_score = s
                        best_add = nv
                        
                if best_add is not None:
                    best_U.append(best_add)
                    best_U = sorted(set(best_U))
                    current_score = best_add_score
                    if current_score > best_score:
                        best_score = current_score
                    improved = True

            # 3. Try removing an element
            if not improved and n > 2:
                best_remove = None
                best_remove_score = current_score
                for i, elem in enumerate(best_U):
                    if elem == 0:
                        continue
                    new_U = [x for j, x in enumerate(best_U) if j != i]
                    if len(new_U) < 2:
                        continue
                    s = compute_score(new_U)
                    if s > best_remove_score:
                        best_remove_score = s
                        best_remove = i
                        
                if best_remove is not None:
                    best_U.pop(best_remove)
                    current_score = best_remove_score
                    if current_score > best_score:
                        best_score = current_score
                    improved = True

            if not improved:
                break
        
        return best_U, best_score

    def generate_mian_chowla(length):
        """Generate Mian-Chowla sequence (greedy Sidon)."""
        seq = [0]
        used_diffs = set()
        for _ in range(length - 1):
            found = False
            candidate = 1
            while not found:
                # Check if candidate forms a Sidon sequence with existing
                valid = True
                for a in seq:
                    d = candidate - a
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    # Update used_diffs
                    for a in seq:
                        used_diffs.add(candidate - a)
                    seq.append(candidate)
                    found = True
                else:
                    candidate += 1
        return seq

    def greedy_compact_sidon(target_len, max_val=100):
        U = [0]
        used_diffs = set()
        for _ in range(target_len - 1):
            best_cand = None
            # Try smallest candidates first to keep max(U) low
            for c in range(1, max_val + 1):
                valid = True
                for a in U:
                    d = c - a
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    best_cand = c
                    break
            if best_cand is None:
                break
            U.append(best_cand)
            for a in U[:-1]: # Update diffs with new element
                used_diffs.add(best_cand - a)
        return U

    # Generate diverse initial candidates
    candidates = []
    
    # 1. Mian-Chowla sequences (strong Sidon property)
    for length in range(5, 15):
        candidates.append(generate_mian_chowla(length))
        
    # 2. Compact Sidon-like: Greedy construction prioritizing small values
    for length in range(5, 15):
        candidates.append(greedy_compact_sidon(length, 200))

    # 3. Randomized Sidon construction
    random.seed(123)
    for trial in range(20):
        target_len = random.randint(8, 15)
        max_val = random.randint(50, 200)
        U = [0]
        used_diffs = set()
        for _ in range(target_len - 1):
            found = False
            # Shuffle candidates to avoid bias
            cands = list(range(1, max_val + 1))
            random.shuffle(cands)
            for c in cands:
                valid = True
                for a in U:
                    d = c - a
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    U.append(c)
                    for a in U[:-1]:
                        used_diffs.add(c - a)
                    found = True
                    break
            if not found:
                break
        if len(U) >= 5:
            candidates.append(U)

    # Run local search on each candidate
    best_U = [0, 1]
    best_score = 0.0
    
    # Limit time
    start_time = time.time()
    time_limit = 100 # seconds
    
    for U in candidates:
        if time.time() - start_time > time_limit:
            break
        U = sorted(set(U))
        if 0 not in U:
            U = [0] + U
        if len(U) < 2:
            continue
        
        # Use a tighter max_val for local search to encourage compactness
        searched_U, searched_score = local_search(U, max_iters=500, max_val=500)
        
        if searched_score > best_score:
            best_score = searched_score
            best_U = searched_U

    return best_U

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
