"""Optimized construction for max 3-AP, no 4-AP subset of [1,256]."""
import time
import random
from collections import defaultdict

N = 256

def count_3aps(S):
    """Count distinct 3-term APs (positive difference) in set S."""
    S_set = set(S)
    count = 0
    for i in range(1, N+1):
        if i not in S_set:
            continue
        d = 1
        while i + 2*d <= N:
            if i+d in S_set and i+2*d in S_set:
                count += 1
            d += 1
    return count

def has_4ap(S):
    """Check if S contains a nonconstant 4-term AP."""
    S_set = set(S)
    for i in range(1, N+1):
        if i not in S_set:
            continue
        d = 1
        while i + 3*d <= N:
            if i+d in S_set and i+2*d in S_set and i+3*d in S_set:
                return True
            d += 1
    return False

def greedy_construct():
    """Greedy construction: add elements that maximize 3-AP gain without creating 4-AP."""
    selected = set()
    # Precompute: for each candidate x, what 4-APs would it create?
    # We'll do a greedy: at each step, pick x that maximizes new 3-APs added,
    # subject to not creating any 4-AP.
    
    candidates = list(range(1, N+1))
    random.shuffle(candidates)
    
    for x in candidates:
        # Check if adding x creates a 4-AP
        creates_4ap = False
        # x could be first, second, third, or fourth term of a 4-AP
        # x as first: x, x+d, x+2d, x+3d all in selected after adding x
        #   => x+d, x+2d, x+3d all in selected
        for d in range(1, (N - x) // 3 + 1):
            if (x+d) in selected and (x+2*d) in selected and (x+3*d) in selected:
                creates_4ap = True
                break
        if creates_4ap:
            continue
        # x as second: x-d, x, x+d, x+2d => x-d, x+d, x+2d in selected
        for d in range(1, x // 3 + 1):
            if (x-d) in selected and (x+d) in selected and (x+2*d) in selected:
                creates_4ap = True
                break
        if creates_4ap:
            continue
        # x as third: x-2d, x-d, x, x+d => x-2d, x-d, x+d in selected
        for d in range(1, (x-1) // 2 + 1):
            if (x-2*d) in selected and (x-d) in selected and (x+d) in selected:
                creates_4ap = True
                break
        if creates_4ap:
            continue
        # x as fourth: x-3d, x-2d, x-d, x => x-3d, x-2d, x-d in selected
        for d in range(1, (x-1) // 3 + 1):
            if (x-3*d) in selected and (x-2*d) in selected and (x-d) in selected:
                creates_4ap = True
                break
        if creates_4ap:
            continue
        selected.add(x)
    
    return sorted(selected)

def local_search(S, max_time=60.0):
    """Local search: try adding, removing, and swapping elements to improve 3-AP count."""
    S_set = set(S)
    current_3ap = count_3aps(S)
    start_time = time.time()
    deadline = start_time + max_time
    
    best_S = list(S_set)
    best_3ap = current_3ap
    
    while time.time() < deadline:
        improved = False
        
        # Try adding elements
        for x in range(1, N+1):
            if x in S_set:
                continue
            # Check 4-AP constraint
            creates_4ap = False
            for d in range(1, (N - x) // 3 + 1):
                if (x+d) in S_set and (x+2*d) in S_set and (x+3*d) in S_set:
                    creates_4ap = True
                    break
            if not creates_4ap:
                for d in range(1, x // 3 + 1):
                    if (x-d) in S_set and (x+d) in S_set and (x+2*d) in S_set:
                        creates_4ap = True
                        break
            if not creates_4ap:
                for d in range(1, (x-1) // 2 + 1):
                    if (x-2*d) in S_set and (x-d) in S_set and (x+d) in S_set:
                        creates_4ap = True
                        break
            if not creates_4ap:
                for d in range(1, (x-1) // 3 + 1):
                    if (x-3*d) in S_set and (x-2*d) in S_set and (x-d) in S_set:
                        creates_4ap = True
                        break
            
            if creates_4ap:
                continue
            
            # Count new 3-APs gained by adding x
            new_3aps = 0
            # x as first term: x, x+d, x+2d
            for d in range(1, (N - x) // 2 + 1):
                if (x+d) in S_set and (x+2*d) in S_set:
                    new_3aps += 1
            # x as second term: x-d, x, x+d
            for d in range(1, x // 2 + 1):
                if (x-d) in S_set and (x+d) in S_set:
                    new_3aps += 1
            # x as third term: x-2d, x-d, x
            for d in range(1, (x-1) // 2 + 1):
                if (x-2*d) in S_set and (x-d) in S_set:
                    new_3aps += 1
            
            if new_3aps > 0:
                S_set.add(x)
                current_3ap += new_3aps
                improved = True
                if current_3ap > best_3ap:
                    best_3ap = current_3ap
                    best_S = list(S_set)
                break
        
        # Try removing elements that allow adding more
        if not improved:
            for x in list(S_set):
                S_set.remove(x)
                current_3ap = count_3aps(S_set)
                # Try to add back or add others
                for y in range(1, N+1):
                    if y in S_set:
                        continue
                    creates_4ap = False
                    for d in range(1, (N - y) // 3 + 1):
                        if (y+d) in S_set and (y+2*d) in S_set and (y+3*d) in S_set:
                            creates_4ap = True
                            break
                    if not creates_4ap:
                        for d in range(1, y // 3 + 1):
                            if (y-d) in S_set and (y+d) in S_set and (y+2*d) in S_set:
                                creates_4ap = True
                                break
                    if not creates_4ap:
                        for d in range(1, (y-1) // 2 + 1):
                            if (y-2*d) in S_set and (y-d) in S_set and (y+d) in S_set:
                                creates_4ap = True
                                break
                    if not creates_4ap:
                        for d in range(1, (y-1) // 3 + 1):
                            if (y-3*d) in S_set and (y-2*d) in S_set and (y-d) in S_set:
                                creates_4ap = True
                                break
                    if creates_4ap:
                        continue
                    new_3aps = 0
                    for d in range(1, (N - y) // 2 + 1):
                        if (y+d) in S_set and (y+2*d) in S_set:
                            new_3aps += 1
                    for d in range(1, y // 2 + 1):
                        if (y-d) in S_set and (y+d) in S_set:
                            new_3aps += 1
                    for d in range(1, (y-1) // 2 + 1):
                        if (y-2*d) in S_set and (y-d) in S_set:
                            new_3aps += 1
                    if new_3aps > 0:
                        S_set.add(y)
                        current_3ap += new_3aps
                        if current_3ap > best_3ap:
                            best_3ap = current_3ap
                            best_S = list(S_set)
                        improved = True
                        break
                if not improved:
                    S_set.add(x)
                    current_3ap = count_3aps(S_set)
        
        if not improved and time.time() > start_time + 5:
            break
    
    return best_S

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s
    
    # Multiple random restarts with greedy + local search
    best_result = []
    best_score = 0
    
    # First, get a baseline
    baseline = greedy_construct()
    best_result = baseline
    best_score = count_3aps(baseline)
    
    # Try multiple randomizations
    for restart in range(20):
        if time.time() > deadline:
            break
        
        random.seed(restart * 12345 + 67890)
        S = greedy_construct()
        S = local_search(S, max_time=3.0)
        score = count_3aps(S)
        
        if score > best_score:
            best_score = score
            best_result = S
    
    # Final local search on best
    if time.time() < deadline - 5:
        best_result = local_search(best_result, max_time=deadline - time.time() - 2)
    
    return sorted(set(best_result))
