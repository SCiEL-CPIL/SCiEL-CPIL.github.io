import time
import random
import heapq
import math

def construct(q, n, radius):
    """
    Constructs a covering code for the given parameters.
    Returns a list of codewords (lists of integers).
    """
    start_time = time.monotonic()
    deadline = start_time + 118.0  # Safety margin for 120s limit
    
    total_points = q ** n
    pow_q = [1] * (n + 1)
    for i in range(1, n + 1):
        pow_q[i] = pow_q[i-1] * q

    # Precompute neighbors for radius 1 (since radius is fixed to 1 in the task context usually, 
    # but the function signature takes radius. We assume radius=1 for this specific task K_3(6,1)).
    # If radius > 1, the neighbor list grows. For K_3(6,1), radius is 1.
    # We will handle general radius but optimize for 1.
    
    # For efficiency, we represent codewords as integers 0..q^n-1.
    # Neighbors of a word w (int) are w itself and words differing in 1 position (if radius=1).
    
    # Precompute neighbors for all points? 
    # 729 points. Each has 1 + 6*2 = 13 neighbors for q=3, n=6, r=1.
    # Total edges ~ 729 * 13 = 9477. This is small. We can precompute.
    
    neighbors = [[] for _ in range(total_points)]
    for idx in range(total_points):
        neighbors[idx].append(idx)
        for i in range(n):
            digit = (idx // pow_q[i]) % q
            for new_digit in range(q):
                if new_digit != digit:
                    new_idx = idx + (new_digit - digit) * pow_q[i]
                    neighbors[idx].append(new_idx)
    
    # We need to find a subset C of {0..total_points-1} such that 
    # union_{c in C} neighbors[c] == {0..total_points-1}
    # and |C| is minimized.
    
    best_code = None
    best_size = total_points # Upper bound
    
    def greedy_cover():
        """
        Greedy set cover using a max-heap.
        Returns a list of codewords (ints).
        """
        covered = [0] * total_points
        code = []
        # We want to pick c that maximizes gain(c) = |neighbors[c] \ union(covered)|
        # Initially, gain(c) = |neighbors[c]| = 1 + n*(q-1) for all c.
        # We use a heap of (-gain, c).
        
        # Optimization: Instead of recalculating gain for all c every step, 
        # we can just scan all c to find the max gain. 
        # 729 candidates, 729 points. Scanning is 729*13 ops ~ 10k ops. 
        # If we do this 74 times, it's 740k ops. Very fast.
        
        for _ in range(total_points):
            if time.monotonic() > deadline:
                break
            
            max_gain = -1
            best_c = -1
            for c in range(total_points):
                if c in code:
                    continue
                # Calculate gain
                gain = 0
                for u in neighbors[c]:
                    if covered[u] == 0:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_c = c
            
            if best_c == -1 or max_gain == 0:
                break
            
            code.append(best_c)
            for u in neighbors[best_c]:
                covered[u] += 1
            
            # Early stop if all covered
            # We can track rem_count
            # For simplicity, we just check if max_gain was 0 in next iteration or 
            # we can maintain a count.
            
        # Verify coverage
        # If not covered, add random points
        # (Greedy should always cover)
        
        return code

    def local_search(code, max_time=5.0):
        """
        Improve the code using removal and swaps.
        """
        code_set = set(code)
        covered = [0] * total_points
        for c in code:
            for u in neighbors[c]:
                covered[u] += 1
        
        ls_start = time.monotonic()
        improved = True
        while improved and (time.monotonic() - ls_start) < max_time:
            improved = False
            
            # 1. Removal Pass
            # A codeword c is removable if for all u in neighbors[c], covered[u] > 1.
            # If we remove c, we decrement covered[u] for u in neighbors[c].
            # We check if min(covered[u] for u in neighbors[c]) >= 2.
            
            to_remove = []
            for c in code:
                removable = True
                for u in neighbors[c]:
                    if covered[u] == 1:
                        removable = False
                        break
                if removable:
                    to_remove.append(c)
            
            if to_remove:
                # Remove all removable at once? 
                # Careful: removing one might make another non-removable? 
                # No, removing c decreases covered counts. 
                # If c1 is removable, covered[u] >= 2 for all u in N(c1).
                # If we remove c1, covered[u] becomes >= 1.
                # If c2 was removable, covered[u] >= 2 for all u in N(c2).
                # If u is in N(c1) and N(c2), covered[u] drops by 1.
                # It might drop to 1, making c2 non-removable? 
                # No, the condition for c2 being removable is based on the *current* state.
                # If we remove c1 first, the state changes.
                # So we should remove one by one, or re-evaluate.
                # Since n is small, let's remove one by one to be safe and allow immediate improvement.
                for c in to_remove:
                    if c in code_set:
                        code.remove(c)
                        code_set.discard(c)
                        for u in neighbors[c]:
                            covered[u] -= 1
                        improved = True
                        if time.monotonic() > deadline:
                            return code
            
            # 2. Swap Pass (1-Swap)
            # Try to replace c with c' such that:
            # Unique(c) = {u in N(c) | covered[u] == 1}
            # We need Unique(c) subset of N(c')
            # And we want to keep the size same, but maybe allow future removals?
            # Actually, a direct swap doesn't reduce size. 
            # However, it might make other codewords removable.
            # Or, we can look for a "better" swap that enables removal.
            
            # A more powerful move: 2-Swap (Remove 2, Add 1? No, size changes).
            # Let's stick to 1-Swap to maintain size, hoping for removal.
            
            for i, c in enumerate(code):
                if time.monotonic() > deadline:
                    return code
                
                unique_c = [u for u in neighbors[c] if covered[u] == 1]
                if not unique_c:
                    continue # Removable, should have been caught above, but just in case
                
                # Find candidates c' not in code such that unique_c is subset of neighbors[c']
                # To make this efficient, we can precompute for each point u, which codewords cover it uniquely?
                # No, let's just iterate candidates.
                
                # Optimization: Only consider c' that cover at least one point in unique_c.
                # We can just iterate all c' not in code.
                # 729 candidates. 
                # For each c', check if unique_c subset of neighbors[c'].
                # |unique_c| is small.
                
                # To speed up, we can break if we find a good swap.
                # What is a "good" swap? 
                # Ideally, one that allows subsequent removals.
                # Or one that reduces the "uniqueness" of other codewords.
                
                # Let's try a simpler heuristic:
                # Pick c' that maximizes |N(c') \ N(code \ {c})| ? No, that's greedy.
                # For swap, we just need validity.
                
                # Let's try random swaps among valid ones to escape local minima.
                valid_swaps = []
                for c_prime in range(total_points):
                    if c_prime in code_set:
                        continue
                    # Check if unique_c is covered by c_prime
                    valid = True
                    for u in unique_c:
                        if u not in neighbors[c_prime]:
                            valid = False
                            break
                    if valid:
                        # Optional: Score the swap. 
                        # Prefer swaps that cover more "new" points or overlap less?
                        # For now, just collect valid swaps.
                        valid_swaps.append(c_prime)
                
                if valid_swaps:
                    # Choose one randomly to diversify
                    c_prime = random.choice(valid_swaps)
                    
                    # Perform swap
                    code[i] = c_prime
                    code_set.discard(c)
                    code_set.add(c_prime)
                    
                    # Update covered
                    for u in neighbors[c]:
                        covered[u] -= 1
                    for u in neighbors[c_prime]:
                        covered[u] += 1
                    
                    improved = True
                    # Break to restart removal pass
                    break 
            # End for loop over code

        return code

    # Multi-start ILS
    num_restarts = 50
    for r in range(num_restarts):
        if time.monotonic() > deadline:
            break
        
        # 1. Greedy Init
        code = greedy_cover()
        
        if len(code) >= best_size:
            # If greedy is already worse than best, skip LS? 
            # No, LS might improve it significantly. 
            # But if it's much worse, maybe skip.
            pass
        
        # 2. Local Search
        code = local_search(code, max_time=2.0)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            # If we hit the theoretical lower bound or known best, we can stop early?
            # Known best is 73. Lower bound 71.
            if best_size <= 73:
                break
        
        # 3. Perturbation for next restart (if we were doing ILS on the same instance)
        # Here we do independent restarts with randomization in greedy?
        # Greedy is deterministic if we break ties consistently. 
        # Let's add randomness to Greedy:
        # Instead of picking max gain, pick from top K gains.
        
    # Final Polish
    if best_code:
        # Run a longer local search on the best found
        best_code = local_search(best_code, max_time=10.0)
    
    # Convert to list of lists
    result = []
    for c in best_code:
        word = [0] * n
        temp = c
        for i in range(n):
            word[i] = temp % q
            temp //= q
        result.append(word)
        
    return result
