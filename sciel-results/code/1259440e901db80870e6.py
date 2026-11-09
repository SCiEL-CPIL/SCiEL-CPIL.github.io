import random
import time

N = 256

def is_sidon(S):
    sums = set()
    for i, a in enumerate(S):
        for j in range(i, len(S)):
            b = S[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def is_maximal(S, N):
    S_set = set(S)
    sums = set()
    for i, a in enumerate(S):
        for j in range(i, len(S)):
            sums.add(a + S[j])
    
    blocked = [False] * (N + 1)
    for s in sums:
        for a in S:
            y = s - a
            if 1 <= y <= N:
                blocked[y] = True
        if s % 2 == 0:
            y = s // 2
            if 1 <= y <= N:
                blocked[y] = True
                
    for y in range(1, N + 1):
        if y in S_set:
            continue
        if not blocked[y]:
            return False
    return True

def greedy_sidon(order, N):
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def reduce_maximal(S, N):
    S = list(S)
    improved = True
    while improved:
        improved = False
        S_set = set(S)
        sums = set()
        for i, a in enumerate(S):
            for j in range(i, len(S)):
                sums.add(a + S[j])
        
        # Try removals
        candidates = list(S)
        random.shuffle(candidates)
        for x in candidates:
            candidate_S = [s for s in S if s != x]
            if is_maximal(candidate_S, N):
                S = candidate_S
                improved = True
                break
        if improved:
            continue
            
        # Pre-compute conflicts for all y not in S
        # conflict[y] = set of elements in S that participate in a sum blocking y
        # If removing x makes conflict[y] empty (or only x), then y is a candidate
        # Actually, we need to check if y is blocked by S\{x}.
        # y is blocked by S\{x} if:
        # 1. 2y is a sum of two elements in S\{x}
        # 2. y+a is a sum of two elements in S\{x} for some a in S\{x}
        
        # Let's identify candidates for each x more efficiently.
        # For each x, find y such that all conflicts of y with S involve x.
        
        # conflicts[y] = set of elements z in S such that y is blocked due to z.
        # If x is in conflicts[y] for ALL conflicts, then y might be unblocked.
        
        # To be safe, we will just try random swaps but with a better filter.
        # Or we can try all y for a specific x if the set of candidates is small.
        
        for x in S:
            # Find y candidates
            # A y is a candidate if S\{x} U {y} is Sidon.
            # This is equivalent to:
            # 1. y not in S\{x}
            # 2. 2y not in sums(S\{x})
            # 3. y+a not in sums(S\{x}) for all a in S\{x}
            
            # sums(S\{x}) = sums - {x+a for a in S, a!=x} - {2x}
            # Let's compute sums_no_x
            sums_no_x = set(sums)
            for a in S:
                if a == x: continue
                s = x + a
                sums_no_x.discard(s)
            sums_no_x.discard(2 * x)
            
            # Now check y
            # We only need to check y that were blocked in S.
            # But checking all 256 y is O(256 * |S|) = O(5000) per x.
            # 20 x's * 5000 = 100,000 ops. Very fast.
            
            # We also need to check if the new set is MAXIMAL.
            # This is the expensive part.
            # So we should filter y to only those that are Sidon-compatible with S\{x}.
            
            valid_y = []
            for y in range(1, N + 1):
                if y in S_set:
                    continue
                # Check Sidon condition for S\{x} U {y}
                # 1. 2y in sums_no_x?
                if 2 * y in sums_no_x:
                    continue
                # 2. y+a in sums_no_x for any a in S\{x}?
                ok = True
                for a in S:
                    if a == x: continue
                    if y + a in sums_no_x:
                        ok = False
                        break
                if ok:
                    valid_y.append(y)
            
            # Now try valid_y
            # Shuffle valid_y
            random.shuffle(valid_y)
            for y in valid_y:
                candidate_S = [s for s in S if s != x] + [y]
                # Check Maximal
                if is_maximal(candidate_S, N):
                    S = candidate_S
                    S_set = set(S)
                    improved = True
                    break
            if improved:
                break
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 25  # Safety margin
    
    best = None
    best_size = N + 1
    
    # Baseline: Ascending greedy
    order = list(range(1, N + 1))
    S = greedy_sidon(order, N)
    S = reduce_maximal(S, N)
    if len(S) < best_size:
        best = S
        best_size = len(S)
        
    random.seed(42)
    
    # Try multiple random orderings
    for trial in range(5000):
        if time.time() > deadline:
            break
            
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        S = greedy_sidon(order, N)
        
        # Reduce
        if time.time() < deadline - 2:
            S = reduce_maximal(S, N)
        
        if len(S) < best_size:
            best = S
            best_size = len(S)
            
    if best is None:
        best = greedy_sidon(list(range(1, N + 1)), N)
        
    return best
