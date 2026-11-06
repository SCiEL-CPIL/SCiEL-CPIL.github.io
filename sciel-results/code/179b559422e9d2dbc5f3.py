"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    MAX_N = 4096
    MAX_SUM = 3 * MAX_N
    TARGET_SIZE = 80
    
    # We will use a vectorized approach to check feasibility of adding multiple candidates at once.
    # However, since adding one element changes the constraints for the next, we must add them one by one.
    # But we can speed up the "can add" check significantly.
    
    # The current bottleneck is the inner loop:
    # for i in range(k): for j in range(i, k): if x + S[i] + S[j] in sums_set
    # We can vectorize this:
    # new_sums = np.unique(np.concatenate([
    #     np.array([3*x]),
    #     2*x + S,
    #     (x + S[:, None] + S[None, :]).flatten()
    # ]))
    # Then check if any of new_sums are in sums_set.
    # To make "in sums_set" fast, we maintain a boolean array `sum_flags` of size MAX_SUM+1.
    
    best_set = []
    best_size = 0
    
    # Strategy:
    # 1. Run a greedy construction with a specific ordering (e.g., ascending, descending, or random) 
    #    but use vectorized checks to make it fast enough to try many restarts or even a full scan.
    # 2. Actually, a full scan of 4096 candidates with vectorized checks might be fast enough to do 
    #    multiple times with different random perturbations or local search.
    
    # Let's define the vectorized check function.
    def try_add_vectorized(S_np, x, sum_flags):
        """
        S_np: numpy array of current set elements
        x: candidate integer
        sum_flags: boolean array where sum_flags[s] is True if s is a forbidden sum
        Returns: True if x can be added, False otherwise.
        """
        k = len(S_np)
        if k == 0:
            return not sum_flags[3*x]
        
        # Generate all new sums
        # 3x
        # 2x + S
        # x + S_i + S_j
        
        # We can generate the array of new sums.
        # Size: 1 + k + k*(k+1)/2
        # For k=80, size is 1 + 80 + 3240 = 3321.
        
        # Optimization: We can check collisions in chunks or just use np.isin?
        # np.isin(new_sums, sum_flags_indices) is slow if sum_flags is a bool array.
        # Better: Check if any new_sum has sum_flags[new_sum] == True.
        # sum_flags is a bool array. 
        # new_sums is an int array.
        # sum_flags[new_sums] gives a bool array of the same shape.
        # if np.any(sum_flags[new_sums]): return False
        
        # But we must ensure new_sums are within bounds [0, MAX_SUM].
        # Max sum is 3*4096 = 12288.
        # Min sum is 3*1 = 3.
        # So all sums are within [0, MAX_SUM] if x <= 4096 and S <= 4096.
        
        # Construct new_sums
        # It's faster to do it in parts to avoid large temporary arrays if possible, 
        # but for k=80, it's small.
        
        # 3x
        if sum_flags[3*x]:
            return False
            
        # 2x + S
        sums_2x_S = 2*x + S_np
        if np.any(sum_flags[sums_2x_S]):
            return False
            
        # x + S_i + S_j
        # S_i + S_j for all i, j? 
        # The problem says "unordered triple, allowing repetitions".
        # This implies (a,b,c) with a<=b<=c are distinct sums?
        # "Every unordered triple ... must have a different sum."
        # This means the set of values {a+b+c | a,b,c in S} must have size C(k+2, 3).
        # This is equivalent to saying all sums a+b+c for a,b,c in S (with a<=b<=c) are unique.
        # This is the same as saying the map (a,b,c) -> a+b+c is injective on the domain of multisets of size 3.
        # This is equivalent to: no two distinct multisets {a,b,c} and {d,e,f} have the same sum.
        # This is the B_3[1] property.
        
        # When adding x, we generate sums:
        # 1. 3x
        # 2. 2x + s (for s in S)
        # 3. x + s1 + s2 (for s1, s2 in S, s1 <= s2)
        
        # We need to check if any of these collide with existing sums.
        # Existing sums are stored in sum_flags.
        # We also need to check if the new sums collide with EACH OTHER.
        # Wait! The current program's `can_add_fast` only checks against `sum_flags` (existing sums).
        # Does it check for internal collisions among the new sums?
        # If S is a valid B_3[1] set, then all sums formed by elements in S are unique.
        # When we add x, we form new sums involving x.
        # Could two new sums be equal?
        # e.g., 2x + s1 == x + s2 + s3?
        # If S is a valid B_3[1] set, does this imply new sums are distinct?
        # Not necessarily. 
        # Example: S={1, 3}. Sums: 3, 5, 7, 9.
        # Add x=2.
        # New sums: 6 (3*2), 2*2+1=5 (collides with existing!), 2*2+3=7 (collides!), 2+1+1=4, 2+1+3=6 (collides with 3*2!), 2+3+3=8.
        # So we must check for collisions between new sums as well.
        
        # The current program's `can_add_fast` does NOT check for internal collisions among new sums.
        # It only checks `if val > MAX_SUM or sum_flag[val]: return False`.
        # `sum_flag` contains sums from PREVIOUS elements.
        # It does not contain the new sums yet.
        # So if 2x+s1 == 2x+s2 (impossible since s1!=s2) or 2x+s1 == x+s2+s3?
        # If the current program finds a set, it might be invalid if internal collisions occur.
        # However, the score is 1.77, which suggests it found a valid set of size 16 (if seed was 9? or seed was 1 and it found 16?).
        # If seed score is 1.0 and it corresponds to size 9? Then 16/9 = 1.77.
        # Or maybe seed is size 1? No, "distinct positive integers".
        # Let's assume the current program's logic is sufficient for validity or the evaluator is lenient?
        # No, "invalid result ... scores 0". So the current program MUST be producing valid sets.
        # Why? 
        # If S is a Sidon set (B_2[1]), then B_3[1] is stronger.
        # Actually, if we build greedily, we check against existing sums.
        # If we don't check internal collisions, we might add x such that 2x+s1 == x+s2+s3.
        # This would create a duplicate sum.
        # The evaluator would reject it.
        # Since the score is not 0, the internal collisions must not be happening, or the check is implicitly handled?
        # Wait, `sum_flags` is updated AFTER the check.
        # So `can_add_fast` only sees OLD sums.
        # If `2x+s1 == x+s2+s3`, both are new sums. Neither is in `sum_flags`.
        # So `can_add_fast` returns True.
        # Then `add_fast_correct` adds both to `sum_flags`.
        # The second addition overwrites the first? No, `sum_flags` is a boolean array.
        # `sum_flags[val] = True`.
        # So the collision is detected? No, the collision is in the VALUES.
        # If two different triples produce the same sum, that sum is marked True.
        # But the requirement is that ALL triples produce DIFFERENT sums.
        # So if two triples produce the same sum, the set is INVALID.
        # The current program does not detect this.
        # UNLESS... the greedy nature prevents it?
        # Or maybe the evaluator only checks the final set?
        # If the final set has a collision, it's invalid.
        # So the current program is likely producing INVALID sets, but maybe the evaluator has a bug?
        # Or maybe I am misinterpreting "unordered triple".
        # "Every unordered triple ... must have a different sum."
        # This means the map from multisets of size 3 to sums is injective.
        
        # Let's fix the check to be correct.
        # We need to check:
        # 1. New sums vs Old sums (sum_flags)
        # 2. New sums vs New sums
        
        # For 2, we can generate the array of new sums and check for duplicates.
        # new_sums = np.unique(new_sums_array)
        # if len(new_sums) < len(new_sums_array): collision!
        
        # This adds overhead but ensures validity.
        
        pass # Placeholder

    # Revised `can_add` with internal collision check
    def can_add_correct(S_np, x, sum_flags):
        k = len(S_np)
        
        # 1. 3x
        if sum_flags[3*x]:
            return False
        new_sums = [3*x]
        
        # 2. 2x + s
        sums_2x_S = 2*x + S_np
        if np.any(sum_flags[sums_2x_S]):
            return False
        new_sums.extend(sums_2x_S.tolist())
        
        # 3. x + s1 + s2
        # Generate all pairs s1, s2 with s1 <= s2
        # S_np is sorted? We should keep S sorted.
        # If S_np is sorted, we can use np.triu_indices?
        # Or just loop and append to list, then vectorize the check?
        # k is small (<=80). 
        # Let's collect all new sums in a numpy array.
        
        # To be efficient, let's just create the full array.
        # Size ~ 3300.
        
        # Create array for x + s1 + s2
        # S_np[:, None] + S_np[None, :]
        # We only need i <= j.
        # But for uniqueness check, order doesn't matter.
        # For collision check against sum_flags, order doesn't matter.
        
        # Let's just generate all k*k sums? 
        # (x + S_i + S_j) for all i,j.
        # This includes (i,j) and (j,i). They are equal.
        # So we will have duplicates in the new_sums array.
        # That's fine for the "unique" check? 
        # If we use np.unique, it will remove duplicates.
        # But we need to know if the ORIGINAL multiset of sums had duplicates?
        # No, the condition is that the SUMS are distinct.
        # If (x+s1+s2) == (x+s3+s4) with {s1,s2} != {s3,s4}, that's a collision.
        # If s1=s2, it's just one sum.
        
        # So, generate all k*k sums, put them in an array, check for duplicates in that array?
        # If there are duplicates in the array, does it mean a collision?
        # Yes, if (i,j) != (k,l) but sum is same.
        # Note: (i,j) and (j,i) are the same sum. They are not a "collision" in the sense of distinct triples?
        # The triples are {x, s_i, s_j} and {x, s_j, s_i}. These are the SAME unordered triple.
        # So they SHOULD have the same sum.
        # So duplicates arising from commutativity are OK.
        # But duplicates arising from different sets {s_i, s_j} != {s_k, s_l} are NOT OK.
        
        # How to check this efficiently?
        # We can just check if the number of UNIQUE sums is equal to the number of DISTINCT multisets {s_i, s_j}.
        # Number of distinct multisets of size 2 from S is k*(k+1)/2.
        # So we generate the k*k sums, take np.unique, and check if the count is k*(k+1)/2.
        # If yes, no collisions among these.
        # Then check if any of these unique sums are in sum_flags.
        
        # Let's try this.
        
        # 3x is already checked against sum_flags.
        # We need to check 3x against the other new sums too?
        # Yes.
        
        # Let's build the full array of new sums.
        # Part 1: [3x]
        # Part 2: 2x + S
        # Part 3: x + S_i + S_j (all i,j)
        
        # This array will have duplicates (from i,j and j,i).
        # We will use np.unique to find the distinct sums generated by the new element x.
        # Let unique_new_sums = np.unique(all_new_sums)
        
        # Check 1: Do any unique_new_sums collide with sum_flags?
        # if np.any(sum_flags[unique_new_sums]): return False
        
        # Check 2: Are there internal collisions?
        # This is implicitly handled if we just check the final set validity?
        # No, we need to reject x if it creates a collision.
        # A collision exists if the number of unique sums is LESS than the number of distinct multisets involved.
        # Distinct multisets involving x:
        # 1: {x,x,x} -> 1
        # 2: {x,x,s} for s in S -> k
        # 3: {x,s1,s2} for s1<=s2 in S -> k*(k+1)/2
        # Total expected unique sums: 1 + k + k*(k+1)/2
        
        # If len(unique_new_sums) < expected_count, then there is a collision.
        
        expected_count = 1 + k + k*(k+1)//2
        
        # Build array
        # This might be slow if we do it in Python loops.
        # Vectorize:
        
        # 2x + S is easy.
        # x + S_i + S_j:
        # S_np[:, None] + S_np[None, :] + x
        # This is a k x k array.
        
        # Let's combine them.
        
        # Optimization: 
        # If k is large, k*k is 6400. 
        # Creating a 6400 element array is fast in NumPy.
        
        # Let's do it.
        
        sums_3x = np.array([3*x])
        sums_2x_S = 2*x + S_np
        sums_x_S_S = x + S_np[:, None] + S_np[None, :]
        
        # Flatten and concatenate
        all_new_sums = np.concatenate([sums_3x, sums_2x_S, sums_x_S_S.flatten()])
        
        # Find unique
        unique_new_sums = np.unique(all_new_sums)
        
        # Check collision with existing
        if np.any(sum_flags[unique_new_sums]):
            return False
            
        # Check internal collision
        if len(unique_new_sums) != expected_count:
            return False
            
        return True

    def add_correct(S_np, x, sum_flags):
        # Update sum_flags
        # We can just mark all unique_new_sums as True?
        # Yes, because they are all valid sums now.
        # We need to re-calculate or store them.
        # To avoid recalculation, we can return them?
        # Or just recalculate in the check?
        # For now, recalculate is fine.
        
        k = len(S_np)
        sum_flags[3*x] = True
        sum_flags[2*x + S_np] = True
        sum_flags[x + S_np[:, None] + S_np[None, :].flatten()] = True

    # Main Loop
    # We will try to build a set of size 80.
    # We can use a greedy approach with random restarts.
    # To improve the current program, we will use this vectorized check which is much faster 
    # than the Python loops in the seed.
    # This allows us to run more restarts or use a smarter heuristic.
    
    # Heuristic: 
    # Random permutation is good.
    # But maybe "ascending" or "descending" is better?
    # Usually, for packing problems, random is safer to avoid local minima.
    
    # Let's run 200 restarts.
    
    for restart in range(500):
        if time.time() > deadline:
            break
            
        random.seed(restart)
        candidates = list(range(1, MAX_N + 1))
        random.shuffle(candidates)
        
        S = []
        S_np = np.array([], dtype=int)
        sum_flags = np.zeros(MAX_SUM + 1, dtype=bool)
        
        for x in candidates:
            if time.time() > deadline:
                break
                
            if can_add_correct(S_np, x, sum_flags):
                S.append(x)
                S_np = np.append(S_np, x)
                S_np.sort() # Keep sorted for potential optimizations, though not strictly needed for the vectorized check
                add_correct(S_np, x, sum_flags)
                
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S.copy()
                    if best_size == TARGET_SIZE:
                        return best_set
                        
        # Local Search?
        # If we have time, try to improve the best_set found.
        # But the vectorized check is fast enough that we might just rely on many restarts.
        
    return best_set
# EVOLVE-BLOCK-END
