"""Direct construction of two Sidon sets with disjoint difference sets."""
import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s limit
    n = 256
    
    def elapsed():
        return time.time() - start_time
    
    def is_valid_solution(A, B):
        if not A or not B:
            return False
        if len(A) > 64 or len(B) > 64:
            return False
        # Check Sidon for A
        sums_A = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sums_A:
                    return False
                sums_A.add(s)
        # Check Sidon for B
        sums_B = set()
        for i in range(len(B)):
            for j in range(i, len(B)):
                s = B[i] + B[j]
                if s in sums_B:
                    return False
                sums_B.add(s)
        
        # Check disjoint differences
        diffs_A = set()
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                d = A[j] - A[i]
                diffs_A.add(d)
        diffs_B = set()
        for i in range(len(B)):
            for j in range(i+1, len(B)):
                d = B[j] - B[i]
                diffs_B.add(d)
                
        if diffs_A & diffs_B:
            return False
            
        return True

    def can_add_to_set(x, S, sums_S, diffs_S, diffs_other):
        """Check if x can be added to set S."""
        for s in S:
            if (x + s) in sums_S:
                return False
        if (2 * x) in sums_S:
            return False
        for s in S:
            d = abs(x - s)
            if d != 0 and d in diffs_other:
                return False
        return True

    def add_to_set(x, S, sums_S, diffs_S):
        """Add x to set S and update auxiliary structures."""
        S.append(x)
        for s in S[:-1]:
            sums_S.add(x + s)
            d = abs(x - s)
            if d != 0:
                diffs_S.add(d)
        sums_S.add(2 * x)

    def remove_from_set(x, S, sums_S, diffs_S):
        """Remove x from set S and update auxiliary structures."""
        if x not in S:
            return False
        idx = S.index(x)
        S.pop(idx)
        # Rebuild sums and diffs for remaining elements
        new_sums = set()
        new_diffs = set()
        for i in range(len(S)):
            new_sums.add(2 * S[i])
            for j in range(i+1, len(S)):
                new_sums.add(S[i] + S[j])
                new_diffs.add(S[j] - S[i])
        sums_S.clear()
        sums_S.update(new_sums)
        diffs_S.clear()
        diffs_S.update(new_diffs)
        return True

    def rebuild(A, B):
        """Rebuild all auxiliary structures from scratch."""
        sums_A = set()
        diffs_A = set()
        for i in range(len(A)):
            sums_A.add(2 * A[i])
            for j in range(i+1, len(A)):
                sums_A.add(A[i] + A[j])
                diffs_A.add(A[j] - A[i])
        sums_B = set()
        diffs_B = set()
        for i in range(len(B)):
            sums_B.add(2 * B[i])
            for j in range(i+1, len(B)):
                sums_B.add(B[i] + B[j])
                diffs_B.add(B[j] - B[i])
        return sums_A, sums_B, diffs_A, diffs_B

    def local_search_polish(A, B, sums_A, sums_B, diffs_A, diffs_B):
        """Aggressive local search: insert, swap, kick strategies."""
        all_elements = set(range(1, n + 1))
        
        # Phase 1: Try to insert skipped elements
        improved = True
        while improved and elapsed() < deadline - 3.0:
            improved = False
            used = set(A) | set(B)
            skipped = sorted(all_elements - used)
            for x in skipped:
                if elapsed() > deadline - 3.0:
                    break
                # Try adding to A
                if len(A) < 64 and can_add_to_set(x, A, sums_A, diffs_A, diffs_B):
                    add_to_set(x, A, sums_A, diffs_A)
                    improved = True
                    break
                # Try adding to B
                if len(B) < 64 and can_add_to_set(x, B, sums_B, diffs_B, diffs_A):
                    add_to_set(x, B, sums_B, diffs_B)
                    improved = True
                    break
        
        # Phase 2: Cross-set moves (move element from one set to the other)
        for _ in range(5):
            if elapsed() > deadline - 3.0:
                break
            moved = False
            # Try moving from A to B
            for x in list(A):
                if elapsed() > deadline - 3.0:
                    break
                if len(B) >= 64:
                    continue
                # Save state
                old_A = A[:]
                old_B = B[:]
                remove_from_set(x, A, sums_A, diffs_A)
                if can_add_to_set(x, B, sums_B, diffs_B, diffs_A):
                    add_to_set(x, B, sums_B, diffs_B)
                    moved = True
                    break
                else:
                    # Revert
                    A.clear()
                    A.extend(old_A)
                    sums_A.clear()
                    diffs_A.clear()
                    sums_A, sums_B, diffs_A, diffs_B = rebuild(A, B)
            # Try moving from B to A
            if not moved:
                for x in list(B):
                    if elapsed() > deadline - 3.0:
                        break
                    if len(A) >= 64:
                        continue
                    old_A = A[:]
                    old_B = B[:]
                    remove_from_set(x, B, sums_B, diffs_B)
                    if can_add_to_set(x, A, sums_A, diffs_A, diffs_B):
                        add_to_set(x, A, sums_A, diffs_A)
                        moved = True
                        break
                    else:
                        B.clear()
                        B.extend(old_B)
                        sums_A, sums_B, diffs_A, diffs_B = rebuild(A, B)
            if not moved:
                break
        
        # Phase 3: Kick strategy - remove one element, try to add two
        for _ in range(3):
            if elapsed() > deadline - 3.0:
                break
                kicked = False
                # Try kicking from A
                for x in list(A):
                    if elapsed() > deadline - 3.0:
                        break
                    old_A = A[:]
                    old_B = B[:]
                    remove_from_set(x, A, sums_A, diffs_A)
                    # Try to add two elements to A or B
                    added_count = 0
                    used = set(A) | set(B)
                    candidates = sorted(set(range(1, n+1)) - used)
                    for y in candidates:
                        if elapsed() > deadline - 3.0:
                            break
                        if added_count >= 2:
                            break
                        if len(A) < 64 and can_add_to_set(y, A, sums_A, diffs_A, diffs_B):
                            add_to_set(y, A, sums_A, diffs_A)
                            added_count += 1
                        elif len(B) < 64 and can_add_to_set(y, B, sums_B, diffs_B, diffs_A):
                            add_to_set(y, B, sums_B, diffs_B)
                            added_count += 1
                    if added_count >= 2:
                        kicked = True
                        break
                    else:
                        # Revert
                        A.clear()
                        A.extend(old_A)
                        B.clear()
                        B.extend(old_B)
                        sums_A, sums_B, diffs_A, diffs_B = rebuild(A, B)
                
                # Try kicking from B
                if not kicked:
                    for x in list(B):
                        if elapsed() > deadline - 3.0:
                            break
                        old_A = A[:]
                        old_B = B[:]
                        remove_from_set(x, B, sums_B, diffs_B)
                        added_count = 0
                        used = set(A) | set(B)
                        candidates = sorted(set(range(1, n+1)) - used)
                        for y in candidates:
                            if elapsed() > deadline - 3.0:
                                break
                            if added_count >= 2:
                                break
                            if len(A) < 64 and can_add_to_set(y, A, sums_A, diffs_A, diffs_B):
                                add_to_set(y, A, sums_A, diffs_A)
                                added_count += 1
                            elif len(B) < 64 and can_add_to_set(y, B, sums_B, diffs_B, diffs_A):
                                add_to_set(y, B, sums_B, diffs_B)
                                added_count += 1
                        if added_count >= 2:
                            kicked = True
                            break
                        else:
                            A.clear()
                            A.extend(old_A)
                            B.clear()
                            B.extend(old_B)
                            sums_A, sums_B, diffs_A, diffs_B = rebuild(A, B)
                if not kicked:
                    break
        
        return A, B, sums_A, sums_B, diffs_A, diffs_B

    best_score = 0
    best_sets = ([1], [2]) # Fallback
    
    # Randomized Greedy with restarts
    for restart in range(500): # Max restarts, but will break on time
        if elapsed() > deadline - 4.0:
            break
            
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        # Shuffle candidate elements
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if elapsed() > deadline - 4.0:
                break
                
            # Determine priority: try to add to the smaller set first
            # If sizes are equal, pick randomly
            if len(A) < len(B):
                order = [0, 1]
            elif len(B) < len(A):
                order = [1, 0]
            else:
                order = random.sample([0, 1], 2)
            
            added = False
            for side in order:
                if side == 0:
                    if len(A) >= 64:
                        continue
                    if can_add_to_set(x, A, sums_A, diffs_A, diffs_B):
                        add_to_set(x, A, sums_A, diffs_A)
                        added = True
                        break
                else:
                    if len(B) >= 64:
                        continue
                    if can_add_to_set(x, B, sums_B, diffs_B, diffs_A):
                        add_to_set(x, B, sums_B, diffs_B)
                        added = True
                        break
            
        # Local search polish: try to add skipped elements
        if A and B:
            A, B, sums_A, sums_B, diffs_A, diffs_B = local_search_polish(
                A, B, sums_A, sums_B, diffs_A, diffs_B
            )
            
            if is_valid_solution(A, B):
                score = len(A) + len(B)
                if score > best_score:
                    best_score = score
                    best_sets = (A[:], B[:])
    
    return list(best_sets)
