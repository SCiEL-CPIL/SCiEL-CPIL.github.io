"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Precompute for each divisor d, the set of pairs (b,c) with d|b+c
    # Using bitsets: for each d, maintain a bitmask of "forbidden c values" given b
    # Actually, let's use a different approach: for each potential element x,
    # precompute which existing pairs would violate if x were added as a, b, or c
    
    def check_add_fast(S_set, S_sorted, x, forbidden_pairs_cache):
        """
        Fast check using precomputed forbidden pair information.
        S_set: set of elements
        S_sorted: sorted list of elements
        x: element to add
        forbidden_pairs_cache: dict mapping d -> set of (b,c) pairs where d|(b+c)
        """
        if not S_sorted:
            return True
        
        min_S = S_sorted[0]
        max_S = S_sorted[-1]
        
        # Role 1: x is 'a' (smallest in triple)
        # Need: for all b,c in S with x < b <= c, x does NOT divide b+c
        # i.e., (b+c) % x != 0 for all such b,c
        if x < min_S:
            # All elements in S are > x, so x is smallest
            for i in range(len(S_sorted)):
                b = S_sorted[i]
                if b <= x:
                    continue
                for j in range(i, len(S_sorted)):
                    c = S_sorted[j]
                    if (b + c) % x == 0:
                        return False
        
        # Role 2: x is 'b' (middle)
        # Need: for all a,c in S with a < x <= c, a does NOT divide x+c
        # i.e., (x+c) % a != 0 for all such a,c
        for a in S_sorted:
            if a >= x:
                break
            for c in S_sorted:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Role 3: x is 'c' (largest)
        # Need: for all a,b in S with a < b <= x, a does NOT divide b+x
        # i.e., (b+x) % a != 0 for all such a,b
        for i in range(len(S_sorted)):
            a = S_sorted[i]
            for j in range(i + 1, len(S_sorted)):
                b = S_sorted[j]
                if b > x:
                    break
                if (b + x) % a == 0:
                    return False
        
        return True
    
    def is_valid_set(S):
        """Check if set S satisfies the property."""
        S_list = sorted(S)
        size = len(S_list)
        for i in range(size):
            a = S_list[i]
            for j in range(i + 1, size):
                b = S_list[j]
                for k in range(j, size):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    # Precompute forbidden sums: for each d in [1,n], store which sums s in [2,2n] are divisible by d
    # This helps quickly check if b+c is divisible by a
    
    # Strategy: Use greedy with multiple restarts, trying different orders
    
    best_set = set()
    best_size = 0
    
    # Precompute for speed: for each a, the set of residues r in [0,a-1] such that 
    # if b+c ≡ r (mod a), then a|(b+c). We just need (b+c)%a != 0.
    
    # Strategy 1: Start with a large tail, greedily add smaller elements
    for start_threshold in range(120, 210, 2):
        if time.time() > deadline - 5:
            break
            
        current_set = set(range(start_threshold, n + 1))
        if not current_set:
            continue
        
        # Verify initial set is valid
        if not is_valid_set(current_set):
            continue
        
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
        
        # Try to add smaller elements in descending order (larger numbers have fewer constraints)
        candidates = list(range(start_threshold - 1, 0, -1))
        random.shuffle(candidates)
        
        S_sorted = sorted(current_set)
        for x in candidates:
            if time.time() > deadline - 3:
                break
            
            if check_add_fast(current_set, S_sorted, x, None):
                current_set.add(x)
                S_sorted = sorted(current_set)
                if len(current_set) > best_size:
                    best_set = set(current_set)
                    best_size = len(current_set)
    
    # Strategy 2: Greedy from scratch - add elements in increasing order if valid
    if time.time() < deadline - 10:
        current_set = set()
        S_sorted = []
        for x in range(1, n + 1):
            if time.time() > deadline - 3:
                break
            if check_add_fast(current_set, S_sorted, x, None):
                current_set.add(x)
                S_sorted = sorted(current_set)
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
    
    # Strategy 3: Multiple random restarts with different starting points
    if time.time() < deadline - 15:
        for restart in range(50):
            if time.time() > deadline - 3:
                break
            
            # Random starting threshold
            start_threshold = random.randint(100, 220)
            current_set = set(range(start_threshold, n + 1))
            if not current_set:
                continue
            
            if not is_valid_set(current_set):
                continue
            
            S_sorted = sorted(current_set)
            
            # Add smaller elements in random order
            candidates = list(range(1, start_threshold))
            random.shuffle(candidates)
            
            for x in candidates:
                if time.time() > deadline - 3:
                    break
                if check_add_fast(current_set, S_sorted, x, None):
                    current_set.add(x)
                    S_sorted = sorted(current_set)
            
            if len(current_set) > best_size:
                best_set = set(current_set)
                best_size = len(current_set)
    
    # Strategy 4: Local search - try to improve by swapping elements
    if time.time() < deadline - 10 and best_set:
        best_list = sorted(best_set)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(best_list)):
                if time.time() > deadline - 5:
                    break
                # Try removing best_list[i] and adding a different element
                removed = best_list[i]
                temp_set = set(best_list[:i] + best_list[i+1:])
                S_sorted = sorted(temp_set)
                
                # Try to add a better element
                for x in range(1, n + 1):
                    if x in temp_set:
                        continue
                    if check_add_fast(temp_set, S_sorted, x, None):
                        new_set = temp_set | {x}
                        if len(new_set) > len(best_set):
                            best_set = new_set
                            best_list = sorted(best_set)
                            best_size = len(best_set)
                            improved = True
                            break
                        elif len(new_set) == len(best_set):
                            # Same size, but maybe better composition for future additions
                            # Keep if it allows more additions later - hard to evaluate, skip for now
                            pass
    
    return sorted(list(best_set))
# EVOLVE-BLOCK-END
