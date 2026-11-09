"""
Program to construct a 32-element set in 1..512 maximizing the number of distinct
3-term arithmetic progression differences.
"""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _score_set(S):
    """Return the number of distinct d > 0 such that a, a+d, a+2d all in S."""
    S_set = set(S)
    diffs = set()
    sorted_S = sorted(S)
    n = len(sorted_S)
    for i in range(n):
        for j in range(i + 1, n):
            diff = sorted_S[j] - sorted_S[i]
            if diff % 2 == 0:
                d = diff // 2
                mid = sorted_S[i] + d
                if mid in S_set:
                    diffs.add(d)
    return len(diffs)


def _score_set_fast(S):
    """Faster scoring using bitset-like approach with sorted array and binary search."""
    S_set = set(S)
    diffs = set()
    sorted_S = sorted(S)
    n = len(sorted_S)
    for i in range(n):
        for j in range(i + 1, n):
            diff = sorted_S[j] - sorted_S[i]
            if diff % 2 == 0:
                d = diff // 2
                mid = sorted_S[i] + d
                if mid in S_set:
                    diffs.add(d)
    return len(diffs)


def _random_set():
    """Generate a random 32-element set in 1..512."""
    return random.sample(range(1, 513), 32)


def _greedy_construct(time_budget=30.0):
    """
    Greedy construction: start with empty set, add elements one by one
    to maximize the number of new differences covered.
    """
    start_time = time.time()
    S = []
    S_set = set()
    covered_diffs = set()
    
    for _ in range(32):
        if time.time() - start_time > time_budget:
            break
        
        best_elem = None
        best_new_diffs = -1
        best_total = 0
        
        # Try each candidate element
        for elem in range(1, 513):
            if elem in S_set:
                continue
            
            # Compute which new differences this element would cover
            new_diffs = set()
            for existing in S:
                # Check if elem forms a 3-term AP with two existing elements
                # Case 1: elem is the end: a, a+d, elem => d = (elem - a) / 2
                # Case 2: elem is the middle: a, elem, c => d = elem - a = c - elem
                # Case 3: elem is the start: elem, elem+d, c => d = c - elem
                
                # For each pair (x, y) in S, check if adding elem creates new APs
                # Actually, let's just compute the full score incrementally
                pass
            
            # Simpler: just compute the score of S + [elem]
            # But that's O(n^2) per candidate, and 512 candidates = expensive
            # Let's use a smarter approach: for each existing pair, check if elem completes an AP
            
            # For each existing element x in S:
            #   If elem > x: d = (elem - x) / 2 (if integer), check if x + d in S
            #   If elem < x: d = (x - elem) / 2 (if integer), check if x - d in S
            #   Also check if elem is the middle: for each y in S, y != elem:
            #     if elem > y: d = elem - y, check if elem + d in S
            #     if elem < y: d = y - elem, check if elem - d in S
            
            new_d = set()
            for x in S:
                if elem > x:
                    diff = elem - x
                    if diff % 2 == 0:
                        d = diff // 2
                        mid = x + d
                        if mid in S_set and d > 0:
                            new_d.add(d)
                elif elem < x:
                    diff = x - elem
                    if diff % 2 == 0:
                        d = diff // 2
                        mid = elem + d
                        if mid in S_set and d > 0:
                            new_d.add(d)
            
            # Check if elem is the middle
            for y in S:
                if y == elem:
                    continue
                if elem > y:
                    d = elem - y
                    end = elem + d
                    if end in S_set and d > 0:
                        new_d.add(d)
                elif elem < y:
                    d = y - elem
                    start = elem - d
                    if start in S_set and d > 0:
                        new_d.add(d)
            
            # Count how many of these new differences are not already covered
            truly_new = new_d - covered_diffs
            if len(truly_new) > best_new_diffs:
                best_new_diffs = len(truly_new)
                best_elem = elem
                best_total = len(covered_diffs | new_d)
        
        if best_elem is None:
            # No improvement possible, pick random
            remaining = [x for x in range(1, 513) if x not in S_set]
            if remaining:
                best_elem = random.choice(remaining)
            else:
                break
        
        S.append(best_elem)
        S_set.add(best_elem)
        # Update covered_diffs
        for x in S:
            if best_elem > x:
                diff = best_elem - x
                if diff % 2 == 0:
                    d = diff // 2
                    mid = x + d
                    if mid in S_set and d > 0:
                        covered_diffs.add(d)
            elif best_elem < x:
                diff = x - best_elem
                if diff % 2 == 0:
                    d = diff // 2
                    mid = best_elem + d
                    if mid in S_set and d > 0:
                        covered_diffs.add(d)
        
        for y in S:
            if y == best_elem:
                continue
            if best_elem > y:
                d = best_elem - y
                end = best_elem + d
                if end in S_set and d > 0:
                    covered_diffs.add(d)
            elif best_elem < y:
                d = y - best_elem
                start = best_elem - d
                if start in S_set and d > 0:
                    covered_diffs.add(d)
    
    if len(S) < 32:
        # Fill with random elements
        for x in range(1, 513):
            if len(S) >= 32:
                break
            if x not in S_set:
                S.append(x)
                S_set.add(x)
    
    return S, _score_set(S)


def _local_search(S, time_budget=20.0):
    """
    Local search: try single-element swaps to improve the score.
    """
    start_time = time.time()
    S = list(S)
    S_set = set(S)
    current_score = _score_set(S)
    best_S = list(S)
    best_score = current_score
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < time_budget:
        if no_improve_count > max_no_improve:
            break
        
        improved = False
        # Try swapping each element with a random candidate
        for idx in range(32):
            if time.time() - start_time > time_budget:
                break
            
            old_val = S[idx]
            
            # Try a few random candidates
            for _ in range(10):
                new_val = random.randint(1, 512)
                if new_val in S_set or new_val == old_val:
                    continue
                
                # Temporarily swap
                S[idx] = new_val
                S_set.remove(old_val)
                S_set.add(new_val)
                
                new_score = _score_set(S)
                
                if new_score > current_score:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_S = list(S)
                    improved = True
                    break
                else:
                    # Revert
                    S[idx] = old_val
                    S_set.remove(new_val)
                    S_set.add(old_val)
            
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
        else:
            no_improve_count = 0
    
    return best_S, best_score


def _simulated_annealing(S, time_budget=25.0):
    """
    Simulated annealing: accept worse moves with decreasing probability.
    """
    start_time = time.time()
    S = list(S)
    S_set = set(S)
    current_score = _score_set(S)
    best_S = list(S)
    best_score = current_score
    
    T = 10.0  # Initial temperature
    T_min = 0.1
    cooling_rate = 0.999
    
    while time.time() - start_time < time_budget and T > T_min:
        # Pick a random element to change
        idx = random.randint(0, 31)
        old_val = S[idx]
        new_val = random.randint(1, 512)
        
        if new_val in S_set:
            T *= cooling_rate
            continue
        
        # Temporarily swap
        S[idx] = new_val
        S_set.remove(old_val)
        S_set.add(new_val)
        
        new_score = _score_set(S)
        delta = new_score - current_score
        
        if delta >= 0 or random.random() < pow(2.718281828, delta / T):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_S = list(S)
        else:
            # Revert
            S[idx] = old_val
            S_set.remove(new_val)
            S_set.add(old_val)
        
        T *= cooling_rate
    
    return best_S, best_score


def _structured_candidates():
    """
    Generate structured candidate sets: APs, Cantor-like, etc.
    """
    candidates = []
    
    # Single APs
    for step in range(1, 17):
        S = [1 + i * step for i in range(32)]
        if max(S) <= 512:
            candidates.append(S)
    
    # Two APs
    for d1 in range(1, 12):
        for d2 in range(1, 12):
            if d1 == d2:
                continue
            for n1 in range(4, 29):
                n2 = 32 - n1
                a1 = 1
                S1 = [a1 + i * d1 for i in range(n1)]
                if max(S1) > 512:
                    continue
                a2 = max(S1) + 1
                S2 = [a2 + i * d2 for i in range(n2)]
                if max(S2) > 512:
                    continue
                S = S1 + S2
                if len(set(S)) == 32:
                    candidates.append(S)
    
    # Cantor-like: base 4, digits {0, 1}
    S_cantor = []
    for d1 in (0, 1):
        for d2 in (0, 1):
            for d3 in (0, 1):
                for d4 in (0, 1):
                    val = d1 + d2 * 4 + d3 * 16 + d4 * 64
                    if 1 <= val <= 512:
                        S_cantor.append(val)
    if len(S_cantor) >= 32:
        candidates.append(S_cantor[:32])
    
    # Base 8, digits {0, 1}
    S_base8 = []
    for d1 in (0, 1):
        for d2 in (0, 1):
            for d3 in (0, 1):
                for d4 in (0, 1):
                    val = d1 + d2 * 8 + d3 * 64 + d4 * 512
                    if 1 <= val <= 512:
                        S_base8.append(val)
    if len(S_base8) >= 32:
        candidates.append(S_base8[:32])
    
    return candidates


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    best_S = list(range(1, 33))
    best_score = _score_set(best_S)
    
    # Phase 1: Try structured candidates
    if time.time() < deadline - 60:
        for S_cand in _structured_candidates():
            if time.time() < deadline - 60:
                score = _score_set(S_cand)
                if score > best_score:
                    best_score = score
                    best_S = list(S_cand)
    
    # Phase 2: Greedy construction
    if time.time() < deadline - 40:
        S_greedy, score_greedy = _greedy_construct(time_budget=30.0)
        if score_greedy > best_score:
            best_score = score_greedy
            best_S = S_greedy
    
    # Phase 3: Local search
    if time.time() < deadline - 20:
        S_ls, score_ls = _local_search(best_S, time_budget=15.0)
        if score_ls > best_score:
            best_score = score_ls
            best_S = S_ls
    
    # Phase 4: Simulated annealing
    if time.time() < deadline - 10:
        S_sa, score_sa = _simulated_annealing(best_S, time_budget=10.0)
        if score_sa > best_score:
            best_score = score_sa
            best_S = S_sa
    
    # Phase 5: Final local search polish
    remaining = deadline - time.time()
    if remaining > 5:
        S_final, score_final = _local_search(best_S, time_budget=remaining - 3)
        if score_final > best_score:
            best_score = score_final
            best_S = S_final
    
    return list(best_S)

# EVOLVE-BLOCK-END
