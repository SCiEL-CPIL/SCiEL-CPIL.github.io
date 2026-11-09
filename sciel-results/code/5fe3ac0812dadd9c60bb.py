import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    N = 512
    start_time = time.time()
    time_limit = 55.0  # Safe margin for 60s limit

    # 1. Precompute Conflict Masks
    # masks[x] is a list of integers. Each integer is a bitmask of 3 elements.
    # If (S & mask) == mask, then x CANNOT be added to S.
    masks = [[] for _ in range(N)]
    
    # Generate all 4-APs
    for d in range(1, (N // 3) + 1):
        for a in range(1, N - 3 * d + 1):
            # 1-indexed: a, a+d, a+2d, a+3d
            # 0-indexed:
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            
            # For i0, the other 3 are i1, i2, i3
            masks[i0].append((1 << i1) | (1 << i2) | (1 << i3))
            # For i1, the other 3 are i0, i2, i3
            masks[i1].append((1 << i0) | (1 << i2) | (1 << i3))
            # For i2, the other 3 are i0, i1, i3
            masks[i2].append((1 << i0) | (1 << i1) | (1 << i3))
            # For i3, the other 3 are i0, i1, i2
            masks[i3].append((1 << i0) | (1 << i1) | (1 << i2))

    def can_add(x, S):
        """
        Check if x (0-indexed) can be added to S (integer bitmask).
        """
        for m in masks[x]:
            if (S & m) == m:
                return False
        return True

    def get_size(S):
        return bin(S).count("1")

    def count_conflicts(x, S):
        """
        Count how many masks for x are fully present in S.
        This is a measure of 'how bad' adding x would be.
        """
        count = 0
        for m in masks[x]:
            if (S & m) == m:
                count += 1
        return count

    def greedy_construct(seed):
        rng = random.Random(seed)
        S = 0
        elements = list(range(N))
        rng.shuffle(elements)
        
        for x in elements:
            if can_add(x, S):
                S |= (1 << x)
        return S

    def local_search(S, time_budget, max_passes=10):
        """
        Local search with size-improving and conflict-reducing moves.
        """
        current_size = get_size(S)
        best_size = current_size
        best_S = S
        
        for _ in range(max_passes):
            if time.time() - start_time > time_budget:
                break
                
            improved = False
            # 1. Try to add elements (size increase)
            for u in range(N):
                if S & (1 << u):
                    continue
                if can_add(u, S):
                    S |= (1 << u)
                    current_size += 1
                    best_size = current_size
                    best_S = S
                    improved = True
                    if time.time() - start_time > time_budget:
                        return best_S
            
            if improved:
                continue # Restart pass as S changed significantly
            
            # 2. Try swaps (size neutral, but potentially conflict-reducing)
            # We look for a swap (u, v) such that u is not in S, v is in S,
            # and u can be added to S \ {v}.
            # To avoid O(N^2) per pass which is too slow, we limit candidates.
            # However, N=512, so N^2 is ~260k. With masks check, it might be okay for a few passes.
            # Let's try a randomized subset of swaps if deterministic is too slow?
            # Actually, deterministic sweep is safer for reproducibility.
            
            # Optimization: Only consider u that has conflicts?
            # If u has no conflicts, we would have added it in step 1.
            # So all u here have conflicts.
            
            found_swap = False
            # To speed up, we can iterate u in random order?
            # Let's stick to deterministic for now, but break early if no swap found after full scan.
            
            for u in range(N):
                if S & (1 << u):
                    continue
                
                # Check if u can be added to S \ {v} for some v in S
                # This requires checking all v in S? That's O(N) per u -> O(N^2).
                # Total per pass O(N^2 * avg_masks).
                # avg_masks is ~ N/3 * something? No, number of APs containing x.
                # Max APs containing x is ~ N/3. So ~170 masks.
                # 512 * 512 * 170 ~ 44 million operations per pass. Too slow in Python.
                
                # We need a faster way to find a valid swap.
                # Idea: For each u, find the 'most blocking' v?
                # Or, just try a few random v for each u?
                
                # Let's try a heuristic: 
                # For each u, pick a random v in S and check if swap works.
                # If it works, do it.
                # This is O(N * K) where K is number of random tries per u.
                
                if time.time() - start_time > time_budget:
                    break
                
                # Try up to 5 random v's for this u
                S_list = [i for i in range(N) if S & (1 << i)]
                if not S_list:
                    continue
                
                for _ in range(5):
                    v = random.choice(S_list)
                    # Check if u can be added to S without v
                    S_temp = S & ~(1 << v)
                    if can_add(u, S_temp):
                        # Swap!
                        S = S_temp | (1 << u)
                        found_swap = True
                        improved = True
                        break
                if improved:
                    # Restart the inner u loop? Or continue?
                    # Since size is same, continuing is okay, but restarting might be better 
                    # to catch new opportunities. 
                    # However, restarts are expensive. 
                    # Let's just break out of the u loop and restart the pass.
                    break
            
            if not improved:
                break # Local optimum

        return best_S

    def perturb(S, k, rng):
        """
        Remove k random elements from S.
        """
        S_list = [i for i in range(N) if S & (1 << i)]
        if len(S_list) <= k:
            return 0
        to_remove = rng.sample(S_list, k)
        for x in to_remove:
            S &= ~(1 << x)
        return S

    # Main
    best_S = 0
    best_size = 0
    rng = random.Random(42)
    
    # Try a few different greedy seeds
    seeds = [0, 1, 42, 123, 999]
    
    for seed in seeds:
        if time.time() - start_time > time_limit - 5:
            break
            
        S = greedy_construct(seed)
        size = get_size(S)
        
        # Run local search
        S = local_search(S, time_limit - 2)
        size = get_size(S)
        
        if size > best_size:
            best_size = size
            best_S = S
            
        # Iterated Local Search: Perturb and search again
        # We do this a few times if time permits
        for _ in range(5):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Perturb
            S_pert = perturb(best_S, 10, rng)
            
            # Local Search
            S_pert = local_search(S_pert, time_limit - 2)
            size_pert = get_size(S_pert)
            
            if size_pert > best_size:
                best_size = size_pert
                best_S = S_pert
            # If not improved, we can still keep best_S

    # Final greedy sweep on the best set found
    for x in range(N):
        if not (best_S & (1 << x)) and can_add(x, best_S):
            best_S |= (1 << x)
            best_size += 1

    # Convert to list of 1-indexed integers
    result = [i + 1 for i in range(N) if best_S & (1 << i)]
    return result

# EVOLVE-BLOCK-END
