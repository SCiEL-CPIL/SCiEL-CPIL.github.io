import random
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin
    
    def is_pairwise_coprime(vals):
        """Check if all distinct values in vals are pairwise coprime."""
        unique_vals = list(set(vals))
        n = len(unique_vals)
        if n <= 1:
            return True
        for i in range(n):
            for j in range(i + 1, n):
                if math.gcd(unique_vals[i], unique_vals[j]) != 1:
                    return False
        return True

    best_size = 0
    best_A = []
    best_B = []
    
    # Precompute primes up to 256 for faster gcd checks? 
    # math.gcd is fast enough.
    
    # Strategy: Random search + Local Search
    # Start with random A, B.
    # Try to swap elements in A or B to increase the count of coprime sums.
    # Actually, the constraint is hard: ALL distinct sums must be coprime.
    # So we just want to find ANY valid set with max size.
    # Max size is 9.
    
    # Let's try to find a set of size 9 first.
    # If not found, try 8, 7, etc.
    
    # Random search
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Generate random A and B
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        
        sums = [a + b for a in A for b in B]
        unique_sums = list(set(sums))
        
        if len(unique_sums) > best_size:
            if is_pairwise_coprime(unique_sums):
                best_size = len(unique_sums)
                best_A = A
                best_B = B
                if best_size == 9:
                    break # Can't do better
        
        # Local search: try to modify A or B
        # This part is optional if random search is fast enough, but let's add it.
        # Actually, random search might be slow. Let's do a structured search.
        
        # Structured search over differences
        # d1, d2, e1, e2
        # We can limit the range of differences to, say, 1..50
        # and check for valid S0.
        
        # Let's do a hybrid: if random search fails, do structured.
        # But random search is simple. Let's stick to it but make it smarter.
        # Sort A and B to ensure we cover the space.
        
        # Better: Iterate over all A, and for each A, do a few random B's.
        # Or iterate over all pairs of (A, B) from a precomputed list of "good" candidates?
        
        # Let's try iterating over all A (341k) and for each A, 
        # check a subset of B's.
        # This might be too slow in Python.
        
        # Let's try the difference method.
        # Pick random d1, d2, e1, e2 in [1, 100]
        d1 = random.randint(1, 100)
        d2 = random.randint(1, 100)
        e1 = random.randint(1, 100)
        e2 = random.randint(1, 100)
        
        # Offsets
        k00 = 0
        k10 = d1
        k20 = d1 + d2
        k01 = e1
        k11 = d1 + e1
        k21 = d1 + d2 + e1
        k02 = e1 + e2
        k12 = d1 + e1 + e2
        k22 = d1 + d2 + e1 + e2
        
        ks = [k00, k10, k20, k01, k11, k21, k02, k12, k22]
        unique_ks = list(set(ks))
        
        # Check if offsets are distinct (i.e., sums are distinct)
        if len(unique_ks) < 9:
            continue
            
        # Find S0 in [2, 256] such that S0 + k_i are pairwise coprime
        found_S0 = -1
        for S0 in range(2, 257):
            vals = [S0 + k for k in unique_ks]
            if is_pairwise_coprime(vals):
                found_S0 = S0
                break
        
        if found_S0 != -1:
            # Reconstruct A and B
            # S0 = A0 + B0
            # We need A0, B0 such that A = [A0, A0+d1, A0+d1+d2] in [1, 128]
            # and B = [B0, B0+e1, B0+e1+e2] in [1, 128]
            
            # Min A0 = 1, Max A0 = 128 - (d1 + d2)
            # Min B0 = 1, Max B0 = 128 - (e1 + e2)
            
            min_A0 = 1
            max_A0 = 128 - (d1 + d2)
            min_B0 = 1
            max_B0 = 128 - (e1 + e2)
            
            if max_A0 < min_A0 or max_B0 < min_B0:
                continue
                
            # We need A0 + B0 = found_S0
            # So B0 = found_S0 - A0
            # We need min_B0 <= found_S0 - A0 <= max_B0
            # => found_S0 - max_B0 <= A0 <= found_S0 - min_B0
            
            low_A0 = max(min_A0, found_S0 - max_B0)
            high_A0 = min(max_A0, found_S0 - min_B0)
            
            if low_A0 <= high_A0:
                A0 = low_A0 # Pick any valid
                B0 = found_S0 - A0
                
                A = [A0, A0 + d1, A0 + d1 + d2]
                B = [B0, B0 + e1, B0 + e1 + e2]
                
                # Verify
                sums = [a + b for a in A for b in B]
                unique_sums = list(set(sums))
                if is_pairwise_coprime(unique_sums):
                    if len(unique_sums) > best_size:
                        best_size = len(unique_sums)
                        best_A = A
                        best_B = B
                        if best_size == 9:
                            break

    return [best_A, best_B]
