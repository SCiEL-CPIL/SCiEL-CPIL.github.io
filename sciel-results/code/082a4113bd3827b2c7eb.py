import time
import random
import bisect

N = 256

def construct():
    start_time = time.time()
    deadline = 85.0

    def can_add(S, a):
        """Check if 'a' can be added to sorted list S.
        For every a'<b<=c in S∪{a}, a' must NOT divide b+c.
        We check three cases: a as smallest, a as middle, a as largest."""
        n = len(S)
        pos = bisect.bisect_left(S, a)
        
        # Case 1: a is the smallest element (a' = a)
        # For all i, j with pos <= i <= j < n: a does not divide S[i] + S[j]
        for i in range(pos, n):
            b = S[i]
            target = (-b) % a
            for j in range(i, n):
                if S[j] % a == target:
                    return False
        
        # Case 2: a is the middle element (b = a)
        # For all i < pos and j >= pos (or j = pos for c = a): a' does not divide a + c
        # c ranges over S[pos:] and a itself
        for i in range(pos):
            a_small = S[i]
            # c = a: check (a + a) % a_small != 0
            if (a + a) % a_small == 0:
                return False
            # c = S[j] for j >= pos
            target = (-a) % a_small
            for j in range(pos, n):
                if S[j] % a_small == target:
                    return False
        
        # Case 3: a is the largest element (c = a)
        # For all i < j < pos: S[i] does not divide S[j] + a
        for i in range(pos):
            a_small = S[i]
            target = (-a) % a_small
            for j in range(i + 1, pos):
                if S[j] % a_small == target:
                    return False
        
        return True

    def greedy_from_upper(min_start):
        """Start with [min_start, N], repair conflicts, then greedily add smaller elements."""
        S = list(range(min_start, N + 1))
        S_set = set(S)
        
        # Repair phase: remove elements that cause conflicts
        changed = True
        while changed:
            changed = False
            for i in range(len(S)):
                a = S[i]
                if a == 0:
                    continue
                # Check if a causes conflicts with elements after it
                conflict_found = False
                for j in range(i + 1, len(S)):
                    b = S[j]
                    target = (-b) % a
                    for k in range(j, len(S)):
                        if S[k] % a == target:
                            # Remove S[k] (the largest in the conflicting triple)
                            S.pop(k)
                            S_set.discard(S[k])
                            changed = True
                            conflict_found = True
                            break
                    if conflict_found:
                        break
                if conflict_found:
                    break
        
        # Greedily add smaller elements from 1 to min_start-1
        for candidate in range(1, min_start):
            if candidate in S_set:
                continue
            if can_add(S, candidate):
                pos = bisect.bisect_left(S, candidate)
                S.insert(pos, candidate)
                S_set.add(candidate)
        
        return S

    def local_search(S, max_iterations=50):
        """Local search: try to add missing elements, swap if needed."""
        S_set = set(S)
        best_len = len(S)
        
        for iteration in range(max_iterations):
            if time.time() - start_time > deadline:
                break
            
            improved = False
            
            # Try to add any missing element directly
            for a in range(1, N + 1):
                if a in S_set:
                    continue
                if can_add(S, a):
                    pos = bisect.bisect_left(S, a)
                    S.insert(pos, a)
                    S_set.add(a)
                    best_len = len(S)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swap: remove one element, add a missing one
            # Find which missing elements are "close" to being addable
            for a in range(1, N + 1):
                if a in S_set:
                    continue
                
                # Find conflicts for a
                n = len(S)
                pos = bisect.bisect_left(S, a)
                conflicts = []
                
                # Conflicts where a is smallest
                for i in range(pos, n):
                    b = S[i]
                    target = (-b) % a
                    for j in range(i, n):
                        if S[j] % a == target:
                            conflicts.append((i, j, 'a_smallest'))
                
                # Conflicts where a is middle
                for i in range(pos):
                    a_small = S[i]
                    if (a + a) % a_small == 0:
                        conflicts.append((i, -1, 'a_middle_self'))
                    target = (-a) % a_small
                    for j in range(pos, n):
                        if S[j] % a_small == target:
                            conflicts.append((i, j, 'a_middle'))
                
                # Conflicts where a is largest
                for i in range(pos):
                    a_small = S[i]
                    target = (-a) % a_small
                    for j in range(i + 1, pos):
                        if S[j] % a_small == target:
                            conflicts.append((i, j, 'a_largest'))
                
                if not conflicts:
                    continue
                
                # Try removing each conflicting element
                freq = {}
                for idx1, idx2, _ in conflicts:
                    if idx1 >= 0:
                        freq[idx1] = freq.get(idx1, 0) + 1
                    if idx2 >= 0:
                        freq[idx2] = freq.get(idx2, 0) + 1
                
                sorted_indices = sorted(freq.keys(), key=lambda x: freq[x], reverse=True)
                
                for rem_idx in sorted_indices[:3]:
                    S_new = S[:]
                    removed = S_new.pop(rem_idx)
                    S_new_set = S_set - {removed}
                    
                    if can_add(S_new, a):
                        pos_new = bisect.bisect_left(S_new, a)
                        S_new.insert(pos_new, a)
                        S = S_new
                        S_set = S_new_set | {a}
                        improved = True
                        break
                
                if improved:
                    break
        
            if not improved:
                # Try random perturbation
                if len(S) > 2:
                    rem_idx = random.randint(0, len(S) - 1)
                    S_new = S[:]
                    removed = S_new.pop(rem_idx)
                    S_new_set = S_set - {removed}
                    
                    # Try to add back removed or another element
                    candidates = [removed] + [x for x in range(1, N + 1) if x not in S_new_set and random.random() < 0.2]
                    random.shuffle(candidates)
                    
                    for candidate in candidates:
                        if candidate in S_new_set:
                            continue
                        if can_add(S_new, candidate):
                            pos_new = bisect.bisect_left(S_new, candidate)
                            S_new.insert(pos_new, candidate)
                            S = S_new
                            S_set = S_new_set | {candidate}
                            improved = True
                            break
                    if not improved:
                        break
        
        return S

    # Multi-start: try different upper-tail starting points
    best_set = []
    best_len = 0
    
    start_points = [128, 144, 160, 171, 180, 192, 200, 210, 220, 230, 240]
    
    for min_start in start_points:
        if time.time() - start_time > deadline:
            break
        
        S = greedy_from_upper(min_start)
        
        # Local search on this solution
        S = local_search(S)
        
        if len(S) > best_len:
            best_set = S[:]
            best_len = len(S)
    
    # Additional local search on the best solution
    if time.time() - start_time < deadline:
        best_set = local_search(best_set, max_iterations=100)
    
    # Final greedy pass
    if time.time() - start_time < deadline:
        S_set = set(best_set)
        for a in range(1, N + 1):
            if a in S_set:
                continue
            if can_add(best_set, a):
                pos = bisect.bisect_left(best_set, a)
                best_set.insert(pos, a)
                S_set.add(a)
    
    return best_set
