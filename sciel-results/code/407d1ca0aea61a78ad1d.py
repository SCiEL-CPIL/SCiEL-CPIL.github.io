import random
import time

N = 256
MAX_SUM = 2 * N

def construct():
    start_time = time.time()
    deadline = start_time + 25.0

    # Bitmask helpers for sum collision checks
    # Sums range from 2 to 512. We use bits 0..512.
    # Bit s is 1 if sum s is used.
    
    def sums_to_mask(A):
        mask = 0
        for i in range(len(A)):
            for j in range(i, len(A)):
                mask |= 1 << (A[i] + A[j])
        return mask

    def is_sidon_compatible_mask(x, A, mask):
        # Check 2x
        if (mask >> (2 * x)) & 1:
            return False
        # Check x + y
        for y in A:
            if (mask >> (x + y)) & 1:
                return False
        return True

    def get_newly_blocked_mask(x, A, mask, unblocked):
        new_mask = mask | (1 << (2 * x))
        for y in A:
            new_mask |= 1 << (x + y)
        
        newly_blocked = []
        for z in unblocked:
            if z == x:
                continue
            # Check if z is blocked by new_mask
            # z is blocked if 2z is in new_mask OR (z+y is in new_mask for some y in A) OR (z+x is in new_mask)
            # Note: The condition for z to be *unblockable* is that 2z is NOT in new_mask 
            # AND (z+y) is NOT in new_mask for all y in A.
            # If it is compatible, it is NOT blocked.
            
            # Check 2z
            if (new_mask >> (2 * z)) & 1:
                newly_blocked.append(z)
                continue
            
            blocked = False
            for y in A:
                if (new_mask >> (z + y)) & 1:
                    blocked = True
                    break
            if not blocked and (new_mask >> (z + x)) & 1:
                blocked = True
                
            if blocked:
                newly_blocked.append(z)
        
        return newly_blocked

    def is_maximal_mask(A):
        """Check if A is inclusion-maximal Sidon in [1,N] using bitmask."""
        mask = sums_to_mask(A)
        A_set = set(A)
        for x in range(1, N + 1):
            if x in A_set:
                continue
            # Check if x can be added
            if (mask >> (2 * x)) & 1:
                continue
            compatible = True
            for y in A:
                if (mask >> (x + y)) & 1:
                    compatible = False
                    break
            if compatible:
                return False
        return True

    def greedy_max_coverage(init_A=None, randomize_ties=False, bias=0.0, size_penalty=0.0):
        if init_A is None:
            A = []
        else:
            A = list(init_A)
        
        mask = sums_to_mask(A)
        A_set = set(A)
        
        unblocked = set(range(1, N + 1)) - A_set
        
        while unblocked:
            best_x = None
            best_score = -1
            best_newly_blocked = []
            
            for x in range(1, N + 1):
                if x in A_set:
                    continue
                if not is_sidon_compatible_mask(x, A, mask):
                    continue
                
                newly_blocked = get_newly_blocked_mask(x, A, mask, unblocked)
                score = len(newly_blocked)
                
                if bias != 0.0:
                    if bias > 0:
                        score -= x * bias
                    else:
                        score += x * abs(bias)
                
                if size_penalty != 0.0:
                    score -= size_penalty * (len(A) + 1)
                
                if randomize_ties:
                    score += random.uniform(0, 0.5)
                
                if score > best_score:
                    best_score = score
                    best_x = x
                    best_newly_blocked = newly_blocked
            
            if best_x is None:
                break

            A.append(best_x)
            A_set.add(best_x)
            mask |= 1 << (2 * best_x)
            for y in A:
                if y != best_x:
                    mask |= 1 << (best_x + y)
            
            for z in best_newly_blocked:
                if z in unblocked:
                    unblocked.remove(z)
            
            if best_x in unblocked:
                unblocked.remove(best_x)

        return A

    def local_search(A, max_iters=100):
        """Try to reduce |A| by removing or swapping elements while preserving Sidon + maximality."""
        A = list(A)
        best_A = list(A)
        best_size = len(A)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            A_set = set(A)
            
            # Try removing each element
            for i, elem in enumerate(A):
                candidate = A[:i] + A[i+1:]
                if len(candidate) == 0:
                    continue
                
                # Check if candidate is maximal
                # If it is, we can reduce size
                if is_maximal_mask(candidate):
                    A = candidate
                    best_A = list(A)
                    best_size = len(A)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swapping: remove elem, add another
            # We only do this if removing elem makes it non-maximal, 
            # and we want to find a replacement that keeps it maximal (size stays same, but might enable further removals later? 
            # Actually, swapping doesn't reduce size immediately. 
            # However, it might lead to a state where a removal is possible.
            # Given time constraints, we prioritize removals. 
            # But sometimes a swap is needed to "unlock" a removal.
            
            # Let's try swaps only if we are stuck and have time.
            if time.time() < deadline - 2.0:
                for i, elem in enumerate(A):
                    candidate = A[:i] + A[i+1:]
                    cand_set = set(candidate)
                    
                    # Quick check: if candidate is maximal, we would have caught it above.
                    # So candidate is NOT maximal.
                    
                    # Try to add x to candidate to make it maximal again?
                    # If we can add x such that candidate U {x} is maximal, 
                    # then we have a set of same size. 
                    # Does this help? Only if it allows a future removal.
                    # This is a deep search. Let's stick to removals primarily for speed.
                    
                    # Alternative: Try to find a x such that candidate U {x} is Sidon and maximal.
                    # If found, swap.
                    for x in range(1, N + 1):
                        if x in cand_set:
                            continue
                        if not is_sidon_compatible_mask(x, candidate, sums_to_mask(candidate)):
                            continue
                        new_A = candidate + [x]
                        if is_maximal_mask(new_A):
                            A = new_A
                            best_A = list(A)
                            # Size didn't change, but structure did.
                            # We mark improved to restart the loop and try removals again on the new structure.
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best_A

    # Initial deterministic run
    best_set = greedy_max_coverage()
    best_size = len(best_set)
    
    # Local search on initial
    improved = local_search(best_set, max_iters=200)
    if len(improved) < best_size:
        best_size = len(improved)
        best_set = improved
    
    # Randomized restarts
    while time.time() < deadline:
        r = random.random()
        if r < 0.3:
            current_set = greedy_max_coverage(randomize_ties=True)
        elif r < 0.5:
            current_set = greedy_max_coverage(randomize_ties=True, bias=0.02)
        elif r < 0.7:
            current_set = greedy_max_coverage(randomize_ties=True, bias=-0.02)
        elif r < 0.85:
            current_set = greedy_max_coverage(randomize_ties=True, size_penalty=0.5)
        else:
            init_x = random.randint(1, N)
            current_set = greedy_max_coverage(init_A=[init_x], randomize_ties=True)
        
        # Local search
        improved = local_search(current_set, max_iters=100)
        if len(improved) < best_size:
            best_size = len(improved)
            best_set = improved

    return best_set
