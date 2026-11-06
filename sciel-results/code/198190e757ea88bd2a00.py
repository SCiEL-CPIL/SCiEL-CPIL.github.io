"""
Program for large integer sets avoiding equal-gcd triples in [1,128].
Strategy: Bitmask-based local search with precomputed conflict masks.
"""

def construct():
    import time
    import random
    from math import gcd

    DEADLINE = 85.0  # Safety margin from 90s limit
    start_time = time.time()

    N = 128
    # Precompute gcd matrix
    # Using 0-indexed internally for bitmasks
    gcd_mat = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            gcd_mat[i][j] = gcd(i + 1, j + 1)

    # Precompute conflict masks
    # conflict[x][a] is a bitmask of b's such that {a, b, x} is a forbidden triple.
    # Condition: gcd(a,b) == gcd(a,x) == gcd(b,x)
    # Note: a, b, x must be distinct.
    # We will store this as a list of N integers, each an N-bit integer.
    # conflict[x][a] has bit b set if b is a "bad partner" for a when x is added.
    
    conflict = [[0] * N for _ in range(N)]
    
    for x in range(N):
        for a in range(N):
            if a == x:
                continue
            g_ax = gcd_mat[a][x]
            bad_b_mask = 0
            for b in range(N):
                if b == x or b == a:
                    continue
                g_ab = gcd_mat[a][b]
                g_bx = gcd_mat[b][x]
                if g_ax == g_ab == g_bx:
                    bad_b_mask |= (1 << b)
            conflict[x][a] = bad_b_mask

    def can_add_bitmask(S_mask, x):
        """
        Check if x can be added to set represented by S_mask.
        S_mask is an integer with bits set for elements in S.
        x is 0-indexed.
        """
        # Iterate through all a in S
        # We need to check if there exists a in S such that
        # conflict[x][a] intersects with (S_mask without a).
        
        # Optimization: If S is empty, return True.
        if S_mask == 0:
            return True
        
        # We can iterate through set bits in S_mask
        temp_mask = S_mask
        while temp_mask:
            # Get lowest set bit
            lsb = temp_mask & -temp_mask
            a = lsb.bit_length() - 1
            
            # Check if any b in S (excluding a) is in conflict[x][a]
            # S without a: S_mask & ~(1 << a)
            # Intersection: conflict[x][a] & (S_mask & ~(1 << a))
            if conflict[x][a] & (S_mask & ~(1 << a)):
                return False
            
            temp_mask ^= lsb
        return True

    def get_list_from_mask(S_mask):
        """Convert bitmask to list of 1-indexed values for final output."""
        res = []
        for i in range(N):
            if S_mask & (1 << i):
                res.append(i + 1)
        return res

    def is_valid_list(lst):
        """Final validation check."""
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                for k in range(j + 1, len(lst)):
                    a, b, c = lst[i], lst[j], lst[k]
                    if gcd(a, b) == gcd(a, c) == gcd(b, c):
                        return False
        return True

    best_result_mask = 0
    best_size = 0
    rng = random.Random(42)

    # Initial greedy to get a decent start
    # Try adding elements in random order
    for _ in range(5):
        order = list(range(N))
        rng.shuffle(order)
        current_mask = 0
        for x in order:
            if can_add_bitmask(current_mask, x):
                current_mask |= (1 << x)
        
        if bin(current_mask).count('1') > best_size:
            best_size = bin(current_mask).count('1')
            best_result_mask = current_mask

    # Local Search: Hill Climbing with Bitmasks
    # Move: Remove one element, try to add another (or multiple).
    # Or: Remove one, then greedily add as many as possible.
    
    current_mask = best_result_mask
    current_size = best_size

    no_improve_count = 0
    
    while time.time() < DEADLINE:
        if no_improve_count > 50:
            # Restart with a new random perturbation
            # Keep the best 80% and randomize the rest?
            # Or just restart from a new random greedy
            order = list(range(N))
            rng.shuffle(order)
            current_mask = 0
            for x in order:
                if can_add_bitmask(current_mask, x):
                    current_mask |= (1 << x)
            current_size = bin(current_mask).count('1')
            no_improve_count = 0
            continue

        # Try a swap: remove a random element from current set
        if current_size == 0:
            break
            
        set_bits = []
        temp = current_mask
        while temp:
            lsb = temp & -temp
            set_bits.append(lsb.bit_length() - 1)
            temp ^= lsb
        
        # Pick a random element to remove
        a_to_remove = rng.choice(set_bits)
        new_mask = current_mask & ~(1 << a_to_remove)
        
        # Try to add any element not in new_mask
        # We can do a greedy pass on the remaining elements
        # To speed up, just try to add elements in random order or sorted order
        # Let's try to add elements in a random order to escape local optima
        candidates = [x for x in range(N) if not (new_mask & (1 << x))]
        rng.shuffle(candidates)
        
        improved = False
        for x in candidates:
            if can_add_bitmask(new_mask, x):
                new_mask |= (1 << x)
                improved = True
        
        new_size = bin(new_mask).count('1')
        
        if new_size > current_size:
            current_mask = new_mask
            current_size = new_size
            no_improve_count = 0
            
            if current_size > best_size:
                best_size = current_size
                best_result_mask = current_mask
        else:
            no_improve_count += 1

    # Final Result
    result = get_list_from_mask(best_result_mask)
    
    # Fallback if something went wrong
    if not result:
        result = [1]
        
    # Final Validation
    if not is_valid_list(result):
        # If invalid, fall back to a simple known valid set: all powers of 2
        # {1, 2, 4, 8, 16, 32, 64, 128} is valid?
        # gcd(1,2)=1, gcd(1,4)=1, gcd(2,4)=2. OK.
        # gcd(2,4)=2, gcd(2,8)=2, gcd(4,8)=4. OK.
        # Generally powers of 2 are safe.
        # But we can do better: all even numbers except those forming triples?
        # Let's just use a small valid set to ensure score > 0
        result = [1, 2, 4, 8, 16, 32, 64, 128]
        if not is_valid_list(result):
            result = [1]

    return result

if __name__ == "__main__":
    res = construct()
    print(f"Final size: {len(res)}")
    print(res)
