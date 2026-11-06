import random
import time

N = 256

def construct():
    # Helper to check if a set is Sidon
    def is_sidon(A):
        sums = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    # Helper to check if A is maximal Sidon in [1, N]
    def is_maximal(A):
        if not is_sidon(A):
            return False
        sums = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums.add(A[i] + A[j])
        
        for x in range(1, N + 1):
            if x in A:
                continue
            # Check if x can be added
            # New sums: 2*x and x+a for a in A
            # Collision if 2*x in sums OR (x+a1 == a2+a3)
            
            # Fast check:
            if 2 * x in sums:
                continue # Blocked by diagonal sum collision with existing pair
            
            blocked = False
            for a in A:
                s = x + a
                if s in sums:
                    # s = a + x. If s is in sums, it means s = b + c for b,c in A.
                    # If b == c, then s=2b. x+a=2b => x=2b-a.
                    # If b != c, then x+a=b+c => x=b+c-a.
                    # In either case, adding x creates a collision.
                    blocked = True
                    break
            if blocked:
                continue # x is blocked, cannot be added
            
            # If we reach here, x can be added
            return False
        return True

    def get_blocked_count(A):
        """
        Returns the number of elements in [1, N] \ A that are blocked.
        Used to guide search towards more "covering" sets.
        """
        if not is_sidon(A):
            return 0
        sums = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums.add(A[i] + A[j])
        
        count = 0
        for x in range(1, N + 1):
            if x in A:
                continue
            if 2 * x in sums:
                count += 1
                continue
            for a in A:
                if x + a in sums:
                    count += 1
                    break
        return count

    best_set = None
    best_size = float('inf')
    
    start_time = time.time()
    timeout = 28.0 # Leave 2s buffer
    
    # Phase 1: Find a maximal Sidon set using Local Search
    # Initialize with a random set
    for restart in range(100):
        if time.time() - start_time > timeout:
            break
            
        # Start with a small random set
        size_init = random.randint(10, 20)
        A = set(random.sample(range(1, N + 1), size_init))
        
        # Repair to Sidon
        changed = True
        while changed:
            changed = False
            A_list = list(A)
            for i in range(len(A_list)):
                for j in range(i + 1, len(A_list)):
                    # Check collision between A_list[i], A_list[j] and others?
                    # Simpler: just check whole set
                    pass
        
        # Efficient repair: remove one element from any conflicting pair
        A_list = list(A)
        while True:
            # Find a conflict
            conflict_found = False
            sums = {}
            for i in range(len(A_list)):
                for j in range(i, len(A_list)):
                    s = A_list[i] + A_list[j]
                    if s in sums:
                        # Conflict between (sums[s]) and (A_list[i], A_list[j])
                        # Remove one of the elements
                        idx_to_remove = sums[s][0] # Remove first index of previous pair
                        # Actually, let's just remove A_list[i] or A_list[j]
                        A_list.pop(i)
                        conflict_found = True
                        break
                    else:
                        sums[s] = (i, j)
                if conflict_found:
                    break
            if not conflict_found:
                break
        
        A = set(A_list)
        
        # Local Search to make it Maximal
        # We want to increase "blocked count" until it covers all non-members
        current_blocked = get_blocked_count(A)
        max_iterations = 500
        
        for _ in range(max_iterations):
            if time.time() - start_time > timeout:
                break
                
            if is_maximal(A):
                break
            
            # Try a swap
            a_list = list(A)
            if not a_list:
                continue
            a_out = random.choice(a_list)
            # Pick a candidate to add
            # Prefer candidates that are currently "unblocked" or random
            candidates = [x for x in range(1, N + 1) if x not in A]
            if not candidates:
                break
            a_in = random.choice(candidates)
            
            A_new = A - {a_out} | {a_in}
            
            if is_sidon(A_new):
                new_blocked = get_blocked_count(A_new)
                # Accept if it blocks more, or with some probability if equal
                if new_blocked > current_blocked or (new_blocked == current_blocked and random.random() < 0.1):
                    A = A_new
                    current_blocked = new_blocked
        
        # Phase 2: Pruning
        # Try to remove elements while maintaining maximality
        changed = True
        while changed:
            changed = False
            for a in list(A):
                if time.time() - start_time > timeout:
                    break
                A_test = A - {a}
                if is_maximal(A_test):
                    A = A_test
                    changed = True
                    break # Restart loop as A changed
            
        if is_maximal(A):
            if len(A) < best_size:
                best_size = len(A)
                best_set = list(A)
                
    # Fallback: Simple Greedy
    if best_set is None:
        A = []
        sums = set()
        for x in range(1, N + 1):
            new_sums = [x + a for a in A] + [2 * x]
            if len(set(new_sums)) == len(new_sums) and not sums.intersection(new_sums):
                A.append(x)
                sums.update(new_sums)
        # Prune this greedy set too
        changed = True
        while changed:
            changed = False
            for a in list(A):
                A_test = A - {a}
                if is_maximal(A_test):
                    A = A_test
                    changed = True
                    break
        best_set = A
        best_size = len(A)
        
    return best_set
