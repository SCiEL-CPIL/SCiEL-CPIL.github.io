import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 115  # Safety margin for 120s limit
    
    N = 1000
    max_sum = 2 * N

    def greedy_randomized():
        S = []
        S_bits = 0
        counts = np.zeros(max_sum + 1, dtype=np.int8)
        C2_bits = 0
        
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            # Check validity
            # 1. Check sums x+y for y in S
            new_sums_bits = S_bits << x
            if new_sums_bits & C2_bits:
                continue
            
            # 2. Check sum 2x
            if counts[2 * x] >= 2:
                continue
            
            # Valid, add x
            S.append(x)
            S_bits |= (1 << x)
            
            # Update counts
            for y in S:
                s = x + y
                counts[s] += 1
                if counts[s] == 2:
                    C2_bits |= (1 << s)
                    
        return S, S_bits, counts, C2_bits

    def try_add(S, S_bits, counts, C2_bits, x):
        """Try to add element x to the set. Returns True if successful."""
        # Check validity
        new_sums_bits = S_bits << x
        if new_sums_bits & C2_bits:
            return False
        
        if counts[2 * x] >= 2:
            return False
        
        # Valid, add x
        S.append(x)
        S_bits |= (1 << x)
        
        # Update counts
        for y in S:
            s = x + y
            counts[s] += 1
            if counts[s] == 2:
                C2_bits |= (1 << s)
        
        return True

    def kick_and_refill(S, S_bits, counts, C2_bits, kick_size=2):
        """Remove random elements and greedily refill."""
        if len(S) <= kick_size:
            return S, S_bits, counts, C2_bits
        
        S = list(S)
        S_bits = S_bits
        counts = counts.copy()
        C2_bits = C2_bits
        
        # Remove kick_size random elements
        to_remove = random.sample(S, min(kick_size, len(S)))
        S_set = set(S)
        for x in to_remove:
            S.remove(x)
            S_set.discard(x)
            S_bits &= ~(1 << x)
            
            # Decrement counts for all pairs involving x
            for y in S:
                s = x + y
                counts[s] -= 1
            counts[2 * x] -= 1
        
        # Rebuild S_bits and C2_bits from scratch for correctness
        S_bits = 0
        C2_bits = 0
        for x in S:
            S_bits |= (1 << x)
        
        # Rebuild counts
        counts = np.zeros(max_sum + 1, dtype=np.int8)
        for i, x in enumerate(S):
            for y in S[i:]:
                s = x + y
                counts[s] += 1
                if counts[s] == 2:
                    C2_bits |= (1 << s)
        
        # Greedily refill
        remaining = [x for x in range(1, N + 1) if x not in S_set]
        random.shuffle(remaining)
        
        for x in remaining:
            if try_add(S, S_bits, counts, C2_bits, x):
                S_set.add(x)
        
        return S, S_bits, counts, C2_bits

    def local_search(S, S_bits, counts, C2_bits, max_swaps=50):
        """Try swap-based local search."""
        S = list(S)
        S_bits = S_bits
        counts = counts.copy()
        C2_bits = C2_bits
        
        S_set = set(S)
        improved = True
        swap_count = 0
        
        while improved and swap_count < max_swaps:
            improved = False
            # Try swapping each element with a random candidate
            for i in range(len(S)):
                if swap_count >= max_swaps:
                    break
                
                x = S[i]
                # Try removing x
                S.remove(x)
                S_set.discard(x)
                S_bits &= ~(1 << x)
                
                # Decrement counts for all pairs involving x
                for y in S:
                    s = x + y
                    counts[s] -= 1
                counts[2 * x] -= 1
                
                # Try adding random candidates
                candidates = [c for c in range(1, N + 1) if c not in S_set]
                random.shuffle(candidates)
                
                added = False
                for c in candidates:
                    # Check if we can add c
                    new_sums_bits = S_bits << c
                    if new_sums_bits & C2_bits:
                        continue
                    if counts[2 * c] >= 2:
                        continue
                    
                    # Add c
                    S.append(c)
                    S_set.add(c)
                    S_bits |= (1 << c)
                    for y in S:
                        s = c + y
                        counts[s] += 1
                        if counts[s] == 2:
                            C2_bits |= (1 << s)
                    
                    added = True
                    improved = True
                    swap_count += 1
                    break
                
                if not added:
                    # Restore x
                    S.append(x)
                    S_set.add(x)
                    S_bits |= (1 << x)
                    for y in S:
                        s = x + y
                        counts[s] += 1
                        if counts[s] == 2:
                            C2_bits |= (1 << s)
                    counts[2 * x] += 1
        
        return S, S_bits, counts, C2_bits

    best_S, best_bits, best_counts, best_C2 = greedy_randomized()
    best_size = len(best_S)
    
    # Run multiple greedy attempts and local search improvements
    while time.time() - start_time < time_limit:
        # Try a new greedy run
        S, bits, counts, C2 = greedy_randomized()
        
        # Try local search on this solution
        S_ls, bits_ls, counts_ls, C2_ls = local_search(S, bits, counts, C2, max_swaps=20)
        
        if len(S_ls) > best_size:
            best_S = S_ls
            best_size = len(S_ls)
            best_bits = bits_ls
            best_counts = counts_ls
            best_C2 = C2_ls
        
        # Also try kick-and-refill on the best solution
        if time.time() - start_time < time_limit - 5:
            S_kick, bits_kick, counts_kick, C2_kick = kick_and_refill(
                list(best_S), best_bits, best_counts.copy(), best_C2, kick_size=2
            )
            if len(S_kick) > best_size:
                best_S = S_kick
                best_size = len(S_kick)
                best_bits = bits_kick
                best_counts = counts_kick
                best_C2 = C2_kick
    
    return best_S

# EVOLVE-BLOCK-END
