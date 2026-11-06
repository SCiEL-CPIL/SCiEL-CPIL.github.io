import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 180
    
    # Fast validity check
    def is_valid(S):
        S_set = set(S)
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i+1, n):
                b = S_list[j]
                # c = a*b/(b-a)
                den = b - a
                if den == 0:
                    continue
                num = a * b
                if num % den == 0:
                    c = num // den
                    if c != a and c != b and c in S_set and c <= N:
                        return False
        return True

    # Step 1: Find the smallest K such that {K, ..., 180} is valid
    # We can binary search or linear scan. Linear scan from 180 down is safe.
    # Actually, we can just try to build the largest suffix.
    # Start with K=180. Decrease K while valid.
    K = 180
    current_suffix = set(range(K, N + 1))
    while K > 1:
        new_K = K - 1
        # Check if {new_K, ..., 180} is valid
        # Optimization: only check triples involving new_K
        # If new_K is a, check b,c in suffix.
        # If new_K is b, check a,c in suffix.
        # If new_K is c, check a,b in suffix.
        
        # Since a < b < c is not strictly true, but a is the smallest.
        # In 1/a = 1/b + 1/c, a is always the smallest.
        # So if we add new_K, it can only be 'a'.
        # So we only need to check if there exist b, c in {new_K+1, ..., 180}
        # such that 1/new_K = 1/b + 1/c.
        
        valid_add = True
        a = new_K
        for b in range(a + 1, N + 1):
            # c = a*b/(b-a)
            den = b - a
            num = a * b
            if num % den == 0:
                c = num // den
                if c != b and c in current_suffix:
                    valid_add = False
                    break
            # Also check if c could be in the new set but not in current_suffix?
            # No, we are checking if the NEW set {new_K, ..., 180} is valid.
            # The elements are new_K and {new_K+1, ..., 180}.
            # current_suffix is {K, ..., 180} which is {new_K+1, ..., 180}.
            # So checking against current_suffix is correct for b,c.
            # We also need to ensure c != new_K.
            # Since c > a (harmonic mean property? No, c can be < a?
            # 1/a = 1/b + 1/c. If b,c > a, then 1/b + 1/c < 2/a.
            # It is possible for c < a?
            # If c < a, then 1/c > 1/a, so 1/b < 0, impossible.
            # So c > a and b > a.
            # So c cannot be new_K.
        
        if valid_add:
            K = new_K
            current_suffix.add(K)
        else:
            break
            
    best_set = set(current_suffix)
    
    # Step 2: Greedily add remaining elements from 1 to K-1
    for candidate in range(1, K):
        if time.time() > deadline:
            break
        new_set = best_set | {candidate}
        if is_valid(new_set):
            best_set = new_set
            
    # Step 3: Local Search / Swaps
    # Try to improve by swapping or adding
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add any missing element
        for candidate in range(1, N + 1):
            if time.time() > deadline:
                break
            if candidate in best_set:
                continue
            new_set = best_set | {candidate}
            if is_valid(new_set):
                best_set = new_set
                improved = True
                break
                
        # If no simple add worked, try swaps
        # This might be slow. Limit iterations.
        if not improved:
            # Try a few random swaps or systematic
            # Systematic: for each x in best_set, for each y not in best_set
            # Swap x,y and check validity.
            # This is O(N^2 * ValidityCheck). Might be too slow.
            # We'll skip complex swaps if time is tight, but the greedy add is usually strong.
            pass
            
    return list(best_set)
# EVOLVE-BLOCK-END
