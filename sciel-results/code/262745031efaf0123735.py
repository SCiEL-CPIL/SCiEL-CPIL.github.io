import time
import random
import sys

def _score_set(S):
    """
    Calculate the number of distinct d > 0 such that 
    a, a+d, a+2d are in S for some a.
    S is a set of integers.
    """
    diffs = set()
    elements = list(S)
    n = len(elements)
    for i in range(n):
        a = elements[i]
        for j in range(n):
            if i == j:
                continue
            b = elements[j]
            if a > b:
                continue
            c = 2 * b - a
            if c in S:
                d = b - a
                if d > 0:
                    diffs.add(d)
    return len(diffs)

def _fast_score(S):
    """Fast scoring using a boolean array for O(1) membership."""
    in_set = [False] * 513
    for x in S:
        in_set[x] = True
    
    covered = set()
    for a in S:
        for b in S:
            if a >= b:
                continue
            d = b - a
            c = 2 * b - a
            if c <= 512 and in_set[c]:
                covered.add(d)
    return len(covered)

def _greedy_construct():
    """
    Greedy construction: build set element by element,
    choosing at each step the element that maximizes new differences covered.
    """
    current_set = set()
    in_set = [False] * 513
    covered_diffs = set()
    
    all_elements = list(range(1, 513))
    
    for step in range(32):
        best_elem = None
        best_new_count = -1
        
        for elem in all_elements:
            if in_set[elem]:
                continue
            
            new_diffs = set()
            
            # elem is the start: elem, elem+d, elem+2d
            for b in current_set:
                if b > elem:
                    d = b - elem
                    c = elem + 2 * d
                    if c <= 512 and in_set[c]:
                        new_diffs.add(d)
            
            # elem is the middle: a, elem, a+2d where d = elem - a
            for a in current_set:
                if a < elem:
                    d = elem - a
                    c = elem + d
                    if c <= 512 and in_set[c]:
                        new_diffs.add(d)
            
            # elem is the end: a, a+d, elem where d = (elem - a)/2 must be integer
            for a in current_set:
                if a < elem and (elem - a) % 2 == 0:
                    d = (elem - a) // 2
                    b = a + d
                    if b <= 512 and in_set[b]:
                        new_diffs.add(d)
            
            new_diffs -= covered_diffs
            count = len(new_diffs)
            
            if count > best_new_count:
                best_new_count = count
                best_elem = elem
        
        if best_elem is not None:
            current_set.add(best_elem)
            in_set[best_elem] = True
            for a in current_set:
                for b in current_set:
                    if a >= b:
                        continue
                    d = b - a
                    c = 2 * b - a
                    if c <= 512 and in_set[c]:
                        covered_diffs.add(d)
    
    return current_set

def _local_refine(S, time_limit):
    """
    Local refinement using fast scoring.
    Try single-element swaps to improve the score.
    """
    current_set = set(S)
    current_score = _fast_score(current_set)
    best_set = set(current_set)
    best_score = current_score
    
    all_elements = list(range(1, 513))
    
    while time.time() < time_limit:
        improved = False
        
        candidates = list(current_set)
        random.shuffle(candidates)
        
        for x in candidates:
            for _ in range(20):
                y = random.choice(all_elements)
                if y in current_set:
                    continue
                
                new_set = set(current_set)
                new_set.remove(x)
                new_set.add(y)
                
                new_score = _fast_score(new_set)
                
                if new_score > current_score:
                    current_set = new_set
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = set(current_set)
                    improved = True
                    break
        
        if not improved:
            for x in list(current_set):
                for y in random.sample(all_elements, min(50, len(all_elements) - 32)):
                    if y in current_set:
                        continue
                    
                    new_set = set(current_set)
                    new_set.remove(x)
                    new_set.add(y)
                    
                    new_score = _fast_score(new_set)
                    
                    if new_score > current_score:
                        current_set = new_set
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_set = set(current_set)
                        improved = True
                        break
                if improved:
                    break
        
        if time.time() >= time_limit:
            break
    
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_score = -1
    best_set = None
    
    greedy_set = _greedy_construct()
    greedy_score = _fast_score(greedy_set)
    
    if greedy_score > best_score:
        best_score = greedy_score
        best_set = greedy_set
    
    refine_deadline = deadline - 5
    refined_set, refined_score = _local_refine(greedy_set, refine_deadline)
    
    if refined_score > best_score:
        best_score = refined_score
        best_set = refined_set
    
    if time.time() < deadline - 20:
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            rand_set = set(random.sample(range(1, 513), 32))
            
            quick_deadline = min(deadline - 10, time.time() + 15)
            q_set, q_score = _local_refine(rand_set, quick_deadline)
            
            if q_score > best_score:
                best_score = q_score
                best_set = q_set
    
    if best_set is None:
        return list(range(1, 33))
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
