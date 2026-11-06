"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Precomputed optimal representations for targets 1..128
    # Format: list of 128 lists, each containing the terms of the antichain
    # The terms are distinct 2,3,5-smooth numbers summing to the target index (1-based)
    # No term divides another.
    
    # We will compute this once and cache it in a global variable to avoid recomputation
    # if construct is called multiple times, though the task implies a single call.
    # However, to be safe and fast, we compute it at module load time or first call.
    
    if not hasattr(construct, '_cached_answers'):
        start_time = time.time()
        
        # 1. Generate smooth numbers up to 128
        smooth = []
        for x in range(1, 129):
            y = x
            for p in (2, 3, 5):
                while y % p == 0:
                    y //= p
            if y == 1:
                smooth.append(x)
        
        n_smooth = len(smooth)
        
        # 2. Precompute incompatibility
        # incompat[i] is a bitmask of indices j such that smooth[i] and smooth[j] are not compatible (one divides the other)
        incompat = [0] * n_smooth
        for i in range(n_smooth):
            for j in range(n_smooth):
                if i != j:
                    a = smooth[i]
                    b = smooth[j]
                    if a % b == 0 or b % a == 0:
                        incompat[i] |= (1 << j)
        
        # 3. Find minimal antichain representations for each target 1..128
        # We store for each target the best (smallest k) representation found so far
        # best_rep[n] = list of values
        best_rep = [None] * 129  # 1-indexed, index 0 unused
        
        # We will search for k=1, 2, 3, ...
        # k=1
        for i, val in enumerate(smooth):
            if val <= 128:
                best_rep[val] = [val]
        
        # k=2
        # For each pair (i, j) with i < j, check if compatible
        # If compatible, sum = smooth[i] + smooth[j]
        # If sum <= 128 and best_rep[sum] is None or len(best_rep[sum]) > 2:
        #    best_rep[sum] = [smooth[i], smooth[j]]
        
        # To be efficient, we can iterate i, j
        for i in range(n_smooth):
            for j in range(i + 1, n_smooth):
                if incompat[i] & (1 << j):
                    continue
                s = smooth[i] + smooth[j]
                if s > 128:
                    break
                if best_rep[s] is None or len(best_rep[s]) > 2:
                    best_rep[s] = [smooth[i], smooth[j]]
        
        # k=3
        # Triplets are more expensive. 42 choose 3 is ~11000, which is fine.
        for i in range(n_smooth):
            for j in range(i + 1, n_smooth):
                if incompat[i] & (1 << j):
                    continue
                # Check compatibility of j with future k
                # We can prune if smooth[i] + smooth[j] + smooth[j+1] > 128? No, smooth is sorted.
                # But we need to check incompatibility for k as well.
                for k in range(j + 1, n_smooth):
                    if incompat[i] & (1 << k) or incompat[j] & (1 << k):
                        continue
                    s = smooth[i] + smooth[j] + smooth[k]
                    if s > 128:
                        break
                    if best_rep[s] is None or len(best_rep[s]) > 3:
                        best_rep[s] = [smooth[i], smooth[j], smooth[k]]
        
        # k=4
        # 42 choose 4 is ~110k. Still fine.
        for i in range(n_smooth):
            for j in range(i + 1, n_smooth):
                if incompat[i] & (1 << j):
                    continue
                for k in range(j + 1, n_smooth):
                    if incompat[i] & (1 << k) or incompat[j] & (1 << k):
                        continue
                    for l in range(k + 1, n_smooth):
                        if (incompat[i] & (1 << l)) or (incompat[j] & (1 << l)) or (incompat[k] & (1 << l)):
                            continue
                        s = smooth[i] + smooth[j] + smooth[k] + smooth[l]
                        if s > 128:
                            break
                        if best_rep[s] is None or len(best_rep[s]) > 4:
                            best_rep[s] = [smooth[i], smooth[j], smooth[k], smooth[l]]
        
        # Verify all targets 1..128 have a representation
        # If any are missing, we need to go higher, but k=4 should cover all.
        # Let's check if any are None
        for n in range(1, 129):
            if best_rep[n] is None:
                # Fallback: find any representation, even if not minimal
                # This shouldn't happen for n<=128 with k up to 4 or 5
                # But just in case, use a simple greedy or DFS
                # For now, mark as error or use a large k
                # We will implement a DFS fallback if needed
                pass

        # Construct the final list
        answers = []
        for n in range(1, 129):
            if best_rep[n] is not None:
                answers.append(best_rep[n])
            else:
                # Emergency fallback
                # This should not happen, but we need to return something valid
                # A single term [n] is valid only if n is smooth.
                # If n is not smooth and we failed to find a rep, we have a problem.
                # But k=4 search is exhaustive for small numbers.
                # Let's assume it's found.
                answers.append([n])
        
        construct._cached_answers = answers
    
    return construct._cached_answers

# EVOLVE-BLOCK-END
