import sys

N = 256

def construct():
    def is_sidon(A):
        sums = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def is_maximal(A, N):
        if not is_sidon(A):
            return False
        sums = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums.add(A[i] + A[j])
        
        for x in range(1, N + 1):
            if x in A:
                continue
            new_sums = [x + a for a in A] + [2 * x]
            valid = True
            new_sum_set = set(new_sums)
            if len(new_sum_set) != len(new_sums):
                valid = False
            else:
                for s in new_sums:
                    if s in sums:
                        valid = False
                        break
            if valid:
                return False
        return True

    # Approach: Deterministic Greedy with Max-Sum-Coverage Heuristic
    # We pick the element that adds the most new unique sums to the set.
    # This tends to create a "dense" sum structure, which blocks more candidates.
    
    A = []
    sums = set()
    
    while True:
        best_x = -1
        best_new_count = -1
        
        # Iterate through all candidates
        for x in range(1, N + 1):
            if x in A:
                continue
            
            new_sums_list = [x + a for a in A] + [2 * x]
            
            # Check internal collision
            if len(set(new_sums_list)) != len(new_sums_list):
                continue
            
            # Check collision with existing sums
            can_add = True
            for s in new_sums_list:
                if s in sums:
                    can_add = False
                    break
            
            if not can_add:
                continue
            
            # Count new sums
            new_sums_set = set(new_sums_list)
            added = len(new_sums_set - sums)
            
            # Tie-breaking: Prefer larger x to spread sums higher?
            # Or smaller x?
            # Let's try preferring larger x to cover the upper range first.
            if added > best_new_count or (added == best_new_count and x > best_x):
                best_new_count = added
                best_x = x
        
        if best_x == -1:
            # No more elements can be added
            break
        
        A.append(best_x)
        # Update sums
        for a in A[:-1]:
            sums.add(best_x + a)
        sums.add(2 * best_x)
        
    # Verify maximality (should be true by construction, but safe to check)
    if is_maximal(A, N):
        return A
    else:
        # Fallback: Random Greedy if something went wrong
        import random
        for _ in range(100):
            A_rand = []
            sums_rand = set()
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            for x in candidates:
                new_sums = [x + a for a in A_rand] + [2 * x]
                if len(set(new_sums)) == len(new_sums) and not sums_rand.intersection(new_sums):
                    A_rand.append(x)
                    sums_rand.update(new_sums)
            if is_maximal(A_rand, N):
                return A_rand
        return A

