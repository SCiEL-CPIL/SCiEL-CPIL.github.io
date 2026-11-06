import random
import time

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

    def get_sums(A):
        sums = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums.add(A[i] + A[j])
        return sums

    def is_maximal(A, N):
        if not is_sidon(A):
            return False
        A_set = set(A)
        sums = get_sums(A)
        
        for x in range(1, N + 1):
            if x in A_set:
                continue
            # Check if x can be added to A
            # New sums: x + a for a in A, and 2*x
            # Collision if:
            # 1. 2*x in sums
            # 2. x + a1 == x + a2 (impossible for distinct a)
            # 3. x + a in sums for some a in A
            # 4. x + a1 == 2*x => a1 == x (impossible as x not in A)
            # 5. x + a1 == x + a2 (no)
            # Wait, collision between new sums:
            # x + a1 == x + a2 => a1 == a2
            # x + a1 == 2*x => a1 == x (x not in A)
            # So internal new sums are distinct and don't collide with each other.
            # We just need to check against existing sums.
            
            can_add = True
            if 2 * x in sums:
                can_add = False
            else:
                for a in A:
                    if (x + a) in sums:
                        can_add = False
                        break
            
            if can_add:
                return False
        return True

    def generate_random_maximal():
        A = []
        sums = set()
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            # Check if x can be added
            new_sums = [x + a for a in A] + [2 * x]
            if len(set(new_sums)) != len(new_sums):
                continue
            if any(s in sums for s in new_sums):
                continue
            
            A.append(x)
            sums.update(new_sums)
        
        # The resulting A is Sidon. Is it maximal?
        # Greedy construction usually results in a maximal set if we iterate through all candidates?
        # No, if we skip x because it conflicts, it might not conflict later if we removed some other element?
        # But we are adding, not removing.
        # If x was skipped, it conflicted with current A.
        # Later A grows. So if it conflicted before, it still conflicts?
        # Not necessarily. The conflict was with some a in A_old.
        # If we add more elements, the set of sums grows.
        # So if x conflicted with A_old, it conflicts with A_new (superset of sums).
        # So yes, the greedy process produces a maximal Sidon set.
        return A

    def try_remove(A, N):
        """
        Tries to remove elements from A while keeping it maximal.
        Returns the reduced set.
        """
        A = list(A)
        improved = True
        while improved:
            improved = False
            # Shuffle to avoid bias
            indices = list(range(len(A)))
            random.shuffle(indices)
            
            for idx in indices:
                x = A[idx]
                A_trial = [a for i, a in enumerate(A) if i != idx]
                
                if is_maximal(A_trial, N):
                    A = A_trial
                    improved = True
                    break # Restart loop with new A
            
        return A

    best_set = None
    best_size = float('inf')
    
    start_time = time.time()
    timeout = 25.0 # Leave 5s buffer
    
    # Run multiple iterations of: Generate -> Reduce
    while time.time() - start_time < timeout:
        A = generate_random_maximal()
        A_reduced = try_remove(A, N)
        
        if len(A_reduced) < best_size:
            best_size = len(A_reduced)
            best_set = A_reduced[:]
            
    if best_set is None:
        # Fallback
        A = generate_random_maximal()
        best_set = A

    return best_set
