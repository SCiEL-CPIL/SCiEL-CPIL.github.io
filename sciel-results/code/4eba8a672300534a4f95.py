"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 60s limit
    
    # 1. Generate all 2,3,5-smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth.append(x)
    
    # Sort smooth numbers in descending order to help pruning or just ascending for DP
    # For DP, order doesn't strictly matter, but let's keep them sorted for consistency
    smooth.sort()
    n_smooth = len(smooth)
    
    # Precompute divisibility relations
    # div_mask[i] is a bitmask of all indices j such that smooth[j] divides smooth[i]
    # We need to ensure that if we pick i, we cannot pick any j in div_mask[i] \ {i}
    # Actually, the condition is "no chosen term may divide another".
    # So if we pick i, we cannot pick any j where smooth[j] | smooth[i] OR smooth[i] | smooth[j].
    # Let's precompute a conflict_mask[i] which contains all j != i such that 
    # smooth[i] divides smooth[j] OR smooth[j] divides smooth[i].
    
    conflict_mask = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i == j:
                continue
            if smooth[i] % smooth[j] == 0 or smooth[j] % smooth[i] == 0:
                conflict_mask[i] |= (1 << j)
    
    # DP State:
    # We want to represent each target n (1..128) as a sum of a subset of smooth numbers
    # such that the subset is an antichain (no conflicts) and the size is minimized.
    # 
    # Since n is small (<=128), we can use a DP over the target sum.
    # However, the antichain constraint depends on the *set* of chosen numbers, not just the sum.
    # The number of smooth numbers is small (approx 50).
    # Let's count them:
    # 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48, 50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100, 108, 120, 125, 128
    # Let's list them properly in code logic.
    
    # A standard DP on sum won't work because "sum=10" could be {10} or {2,3,5} (conflict!) or {4,6} (conflict).
    # We need to know *which* numbers were used to check conflicts for future additions.
    # 
    # Alternative approach: Since the max target is 128 and the max number of terms is small (likely < 10 for minimal),
    # and the pool of smooth numbers is small (~37), we can use a recursive search with memoization or
    # a BFS/DP over (target_sum, used_mask).
    # used_mask has 2^37 states, which is too large.
    #
    # However, note that the numbers are distinct.
    # Let's try a different DP:
    # dp[sum][mask_of_conflicts_blocked]? No.
    #
    # Let's look at the structure. The smooth numbers are highly composite.
    # Most small numbers divide many others.
    # If we pick 1, we can't pick anything else.
    # If we pick 2, we can't pick 4, 8, ... or 1.
    #
    # Let's try a recursive backtracking with strong pruning for each target, but optimized.
    # Or, better: Pre-compute the minimal representation for each sum using a specific algorithm.
    #
    # Since N is small (128), we can iterate through all possible *antichains* of smooth numbers?
    # No, that's exponential.
    #
    # Let's use a DP where the state is (current_sum, last_index_considered) is not enough.
    #
    # Let's try this:
    # For each target N, find the minimum k such that there exists an antichain of size k summing to N.
    # We can do this by iterating k from 1 to 10 (max likely needed).
    # For a fixed k, we can use a recursive search that picks k numbers from the smooth list.
    # To speed this up, we can use the fact that we are summing to N.
    #
    # Actually, there is a simpler observation:
    # The set of smooth numbers is small. Let's call it S.
    # We can precompute all valid antichains?
    # Number of antichains in the divisibility poset of smooth numbers <= 128.
    # This might be manageable.
    # Let's generate all antichains of S.
    # An antichain is a subset A of S such that for all x,y in A, x !| y and y !| x.
    # We can generate these using a recursive backtracking over the sorted list of smooth numbers.
    # When we pick a number x, we must exclude all numbers that conflict with x.
    
    # Let's try generating all antichains and their sums.
    # If the number of antichains is small enough, we can just store the best (min size) for each sum.
    
    antichains = []
    
    def generate_antichains(index, current_set, blocked_mask):
        # current_set is a bitmask of chosen numbers
        # blocked_mask is a bitmask of numbers that cannot be chosen (conflict with current_set)
        # We iterate through indices >= index
        # Optimization: if current_set is not empty, it's an antichain. Record it.
        if current_set != 0:
            # Calculate sum and size
            s = 0
            cnt = 0
            temp_mask = current_set
            while temp_mask:
                lsb = temp_mask & -temp_mask
                idx = lsb.bit_length() - 1
                s += smooth[idx]
                cnt += 1
                temp_mask ^= lsb
            antichains.append((s, cnt, current_set))
            
        for i in range(index, n_smooth):
            if time.time() > deadline:
                return
            # If i is blocked, skip
            if blocked_mask & (1 << i):
                continue
            # Choose i
            new_set = current_set | (1 << i)
            new_blocked = blocked_mask | (1 << i) | conflict_mask[i]
            generate_antichains(i + 1, new_set, new_blocked)
            # Pruning: if we don't choose i, we continue.
            # This generates all subsets that are antichains.
            
    # This might be too slow if there are many antichains.
    # Let's check the size. The poset is small.
    # Let's run it.
    
    # To avoid recursion depth issues or overhead, we can also do a BFS/DP if we define state differently,
    # but generating antichains is a valid "different strategy" (enumeration of valid configurations vs search for sum).
    
    # Let's add a limit on the number of antichains or time.
    # If it's too slow, we fall back to a smarter search.
    
    # Actually, let's just run the generation.
    generate_antichains(0, 0, 0)
    
    # Now, for each target 1..128, find the antichain with the minimum count that sums to it.
    # If multiple, any is fine.
    # We need to return the list of terms, not just the mask.
    
    # best_for_sum[sum] = (min_count, mask)
    best_for_sum = [None] * 129
    
    for s, cnt, mask in antichains:
        if s <= 128:
            if best_for_sum[s] is None or cnt < best_for_sum[s][0]:
                best_for_sum[s] = (cnt, mask)
                
    # Now construct the answers
    answers = []
    for n in range(1, 129):
        if best_for_sum[n] is not None:
            cnt, mask = best_for_sum[n]
            terms = []
            temp_mask = mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                idx = lsb.bit_length() - 1
                terms.append(smooth[idx])
                temp_mask ^= lsb
            # The problem requires distinct terms. The mask ensures distinct indices.
            # The problem requires 1..30 terms.
            # The problem requires no term divides another. The antichain property ensures this.
            # The problem requires summing to target. We filtered by sum.
            answers.append(terms)
        else:
            # Fallback: This should not happen for 1..128 with smooth numbers,
            # but just in case, use the seed logic (brute force) for this specific n.
            # Or just return an empty list? No, must be valid.
            # Let's do a quick brute force for this n if DP failed.
            found = None
            for k in range(1, 11):
                # Use itertools.combinations
                from itertools import combinations
                candidates = [x for x in smooth if x <= n]
                for combo in combinations(candidates, k):
                    if sum(combo) == n:
                        # Check antichain
                        is_anti = True
                        for a in range(len(combo)):
                            for b in range(a+1, len(combo)):
                                if combo[a] % combo[b] == 0 or combo[b] % combo[a] == 0:
                                    is_anti = False
                                    break
                            if not is_anti:
                                break
                        if is_anti:
                            found = list(combo)
                            break
                if found:
                    break
            answers.append(found if found else [n] if n in smooth else [])
            
    return answers
# EVOLVE-BLOCK-END
