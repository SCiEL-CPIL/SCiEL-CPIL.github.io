import numpy as np
import random
import time

def construct():
    base = 51975
    divisors = [d for d in range(2, base + 1) if base % d == 0]
    
    # Precompute indices for each divisor to speed up slicing?
    # Numpy slicing is already fast, but let's see.
    
    # Heuristic Initialization 1: Greedy by size (descending)
    # Heuristic Initialization 2: Random
    
    def evaluate_and_optimize(initial_residues, time_limit=110):
        n_classes = len(divisors)
        residues = initial_residues.copy()
        
        # Initialize coverage
        covered = np.zeros(base, dtype=np.uint8)
        for i, m in enumerate(divisors):
            r = residues[i]
            covered[r::m] = 1
        
        uncovered = int(np.sum(covered == 0))
        
        # Local Search
        start_time = time.time()
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            # Shuffle order to avoid bias? Or keep fixed?
            # Fixed order is deterministic. Let's stick to fixed for now, 
            # but random restarts will handle diversity.
            for i in range(n_classes):
                m = divisors[i]
                r_old = residues[i]
                
                # Find best residue for this modulus
                best_r = r_old
                best_uncovered = uncovered
                
                # We need to try all r in [0, m)
                # To be efficient, we can update covered incrementally
                # But since we try ALL r, we can't just step through them easily
                # without reverting.
                
                # Strategy:
                # 1. Uncover old residue
                covered[r_old::m] = 0
                # Note: This might uncover some spots that were covered ONLY by this class.
                # But other classes might still cover them.
                
                # We need to find r_new that minimizes uncovered.
                # uncovered_new = uncovered_current + (zeros_in_new_slice - zeros_in_old_slice_relative_to_rest?)
                # This is tricky because "uncovered" depends on the global state.
                
                # Simpler approach:
                # For each candidate r_new:
                #   1. Assume we set residue to r_new.
                #   2. Calculate change in uncovered count.
                
                # Let's just do the full update for each candidate, but optimized.
                # Actually, doing a full sum for each candidate is O(m * N).
                # Sum m is ~150k. 150k * 50k = 7.5 * 10^9. Too slow.
                
                # We need the O(N/m) trick.
                # Let's compute the "contribution" of each residue class to the uncovered count.
                
                # Current state: covered array.
                # If we change residue from r_old to r_new:
                # 1. `covered[r_old::m]` becomes 0.
                # 2. `covered[r_new::m]` becomes 1.
                
                # Let `S_old` be the set of indices `r_old::m`.
                # Let `S_new` be the set of indices `r_new::m`.
                
                # Before change:
                # `covered` has 1s at `S_old` (and others).
                
                # Step 1: Clear `S_old`.
                # `covered[S_old] = 0`.
                # The number of zeros increases by `count(covered[S_old] == 1)`? No.
                # If `covered[i]` was 1, it becomes 0. So zeros increase by 1 for each such i.
                # If `covered[i]` was 0, it stays 0.
                # So, `delta_zeros_1 = np.sum(covered[r_old::m])`.
                # `uncovered_temp = uncovered + delta_zeros_1`.
                
                # Step 2: Set `S_new` to 1.
                # `covered[S_new] = 1`.
                # The number of zeros decreases by `count(covered[S_new] == 0)` in the *temporary* state.
                # In the temporary state, `covered` is the original with `S_old` cleared.
                # We need to know how many 0s are in `S_new` in this temp state.
                
                # This is getting complex to do purely analytically without modifying the array or using extra arrays.
                # Given N=51975 is small, maybe the O(N) sum per candidate is acceptable if we reduce the number of candidates?
                # No, we must check all m.
                
                # Alternative: 
                # Just do the O(N) sum, but use NumPy vectorization to check all residues for a modulus at once?
                # No, the coverage interaction makes it hard.
                
                # Let's go back to the O(N/m) per candidate.
                # 1. `covered[r_old::m] = 0`
                # 2. `uncovered_temp = uncovered + np.sum(covered[r_old::m] == 0)` -- WAIT.
                #    If I set them to 0, the new zeros are those that were 1.
                #    `new_zeros_added = np.sum(covered[r_old::m])` (before setting to 0).
                #    So `uncovered_temp = uncovered + new_zeros_added`.
                # 3. Now for each candidate r_new:
                #    We want to set `covered[r_new::m] = 1`.
                #    The zeros removed will be the current zeros in that slice.
                #    `zeros_in_slice = np.sum(covered[r_new::m] == 0)`.
                #    `uncovered_candidate = uncovered_temp - zeros_in_slice`.
                # 4. Pick best r_new.
                # 5. Apply it: `covered[best_r::m] = 1`.
                # 6. `uncovered = uncovered_candidate_best`.
                
                # This is O(N/m) per candidate, total O(N) per modulus update.
                # Total per pass: O(N * K) ? No, sum over i of (N/m_i * m_i) = sum N = K*N.
                # K=47, N=51975. 47 * 51975 ~ 2.4M operations.
                # This is very fast!
                
                # Implementation details:
                # 1. `old_slice = covered[r_old::m]`
                # 2. `added_zeros = np.sum(old_slice)`  # these 1s become 0s
                # 3. `covered[r_old::m] = 0`
                # 4. `uncovered_temp = uncovered + added_zeros`
                # 5. `best_r = r_old`
                # 6. `best_uncovered = uncovered_temp` # if we keep r_old? No, we cleared it. 
                #    If we pick r_old again, we set it to 1.
                #    Let's just iterate all r in 0..m-1.
                
                for r_new in range(m):
                    # Count zeros in the slice in the current (temp) state
                    # Note: covered[r_new::m] might overlap with covered[r_old::m]
                    # But we just want the count of 0s in that specific slice in the temp array.
                    zeros_in_new = np.sum(covered[r_new::m] == 0)
                    unc = uncovered_temp - zeros_in_new
                    if unc < best_uncovered:
                        best_uncovered = unc
                        best_r = r_new
                
                # Apply best
                if best_r != r_old:
                    covered[best_r::m] = 1
                    residues[i] = best_r
                    uncovered = best_uncovered
                    improved = True
                else:
                    # We cleared it, now we must restore it if best is same?
                    # Actually, if best_r == r_old, we should set it back to 1 to maintain state?
                    # Yes, because the "state" for the next modulus should have the class active.
                    covered[best_r::m] = 1
                    # uncovered is already correct (best_uncovered == uncovered_temp - zeros_in_old_slice)
                    # Wait, if best_r == r_old, then zeros_in_new == added_zeros?
                    # No.
                    # Let's trace:
                    # Start: unc.
                    # Clear old: unc_temp = unc + added_zeros.
                    # If we pick old: set old to 1.
                    # Zeros removed = zeros_in_old_slice (in temp state).
                    # In temp state, old slice is all 0? No, only the ones that were 1 are now 0.
                    # The ones that were 0 are still 0.
                    # So zeros_in_old_slice (temp) = m - added_zeros? No.
                    # Original slice had `added_zeros` ones and `m - added_zeros` zeros.
                    # After clearing: all are 0. So `m` zeros.
                    # So if we set it back to 1, we remove `m` zeros? No, we set them to 1.
                    # The slice is now all 1s.
                    # So zeros removed = m.
                    # unc_final = unc_temp - m.
                    # Is this equal to original unc?
                    # unc_temp = unc + added_zeros.
                    # unc_final = unc + added_zeros - m.
                    # This is NOT unc.
                    # Why? Because `added_zeros` was the count of 1s.
                    # The count of 0s was `m - added_zeros`.
                    # When we clear, we turn 1s to 0s. The 0s stay 0s.
                    # So total zeros in slice becomes `m`.
                    # When we set to 1, we turn all `m` to 1s.
                    # So we remove `m` zeros.
                    # But originally we had `m - added_zeros` zeros.
                    # We added `added_zeros` zeros.
                    # Net change should be 0.
                    # `unc + added_zeros - m` vs `unc`.
                    # `added_zeros - m` is not 0.
                    
                    # My logic for `zeros_in_new` is "count of 0s in the slice in the temp array".
                    # If `r_new == r_old`:
                    # Temp array has `covered[r_old::m] = 0`.
                    # So `zeros_in_new` (which is `covered[r_old::m]==0`) is `m`.
                    # So `unc = unc_temp - m`.
                    # This implies that if we pick the old residue, the uncovered count changes?
                    # It shouldn't.
                    
                    # Ah, `covered` is a global array.
                    # If `r_new` overlaps with other moduli, setting it to 1 might cover spots that were already covered by others.
                    # But `zeros_in_new` counts 0s in the *current* state.
                    # If a spot is 0 in the current state, it is uncovered.
                    # Setting it to 1 makes it covered.
                    # So yes, it reduces uncovered count by the number of 0s in that slice.
                    
                    # So why is it not restoring to original?
                    # Because the "original" state had that slice set to 1.
                    # In the original state, those spots were covered (by this class).
                    # So they were NOT zeros (uncovered) unless covered by no one else?
                    # No, `covered` is a boolean OR of all classes.
                    # If `covered[i]` was 1, it was covered.
                    # If we clear this class, `covered[i]` might become 0 if no other class covers it.
                    # So `added_zeros` is the number of spots that become uncovered.
                    # If we re-add the class, we cover them again.
                    # So `zeros_in_new` (in temp state) for `r_old` should be exactly `added_zeros`?
                    # No, in the temp state, `covered[r_old::m]` is all 0.
                    # So `zeros_in_new` is `m`.
                    # This means we think we are covering `m` spots.
                    # But we are only covering `added_zeros` NEW spots (the ones that became uncovered).
                    # The other `m - added_zeros` spots were already covered by OTHER classes.
                    # Setting them to 1 again doesn't change the `covered` status (they are already 1).
                    # But `covered` is a single array. We overwrite with 1.
                    # If they were already 1, they stay 1.
                    # So the number of 0s removed is indeed `m`?
                    # No, `covered` is `uint8`.
                    # If `covered[i]` is 1, and we set it to 1, it stays 1.
                    # The count of 0s in the slice in the temp array is `m` (since we set them all to 0).
                    # So we subtract `m`.
                    # But we only added `added_zeros` zeros.
                    # So we are subtracting too much?
                    
                    # The error is: `covered` is a shared array.
                    # When we check `covered[r_new::m] == 0`, we are checking the *global* state.
                    # If a spot is 0, it is uncovered.
                    # If we set it to 1, it becomes covered.
                    # So we remove 1 from the uncovered count.
                    # So the logic `unc = unc_temp - zeros_in_new` is CORRECT.
                    
                    # So why did the math fail?
                    # `unc_temp = unc + added_zeros`.
                    # `unc_final = unc_temp - m`.
                    # This implies `unc_final = unc + added_zeros - m`.
                    # For this to equal `unc`, we need `added_zeros = m`.
                    # This means all spots in the slice were 1 (covered by this class exclusively or generally).
                    # If `added_zeros < m`, then `unc_final < unc`.
                    # This suggests that re-adding the class reduces the uncovered count below the original?
                    # That's impossible.
                    
                    # Let's trace carefully.
                    # Original: `covered` has some 1s.
                    # Slice `S`. `covered[S]` has `k` ones and `m-k` zeros.
                    # `added_zeros` = `k`.
                    # `unc` = total zeros in `covered`.
                    
                    # Step 1: `covered[S] = 0`.
                    # The `k` ones become 0s. The `m-k` zeros stay 0s.
                    # New zeros in `S` = `m`.
                    # Total zeros `unc_temp` = `unc + k`.
                    
                    # Step 2: `covered[S] = 1`.
                    # The `m` zeros become 1s.
                    # New zeros in `S` = `0`.
                    # Total zeros `unc_final` = `unc_temp - m`.
                    # `unc_final` = `unc + k - m`.
                    
                    # If `k < m`, then `unc_final < unc`.
                    # This implies that the original state had MORE zeros than the final state?
                    # But the final state is IDENTICAL to the original state!
                    # `covered[S]` was 1 in original, 1 in final.
                    # Other parts of `covered` are unchanged.
                    # So `unc_final` MUST equal `unc`.
                    
                    # Where is the bug?
                    # Ah! `covered` is a global array.
                    # When I do `covered[S] = 0`, I modify the global array.
                    # When I check `covered[r_new::m] == 0`, I am checking the global array.
                    # If `r_new` overlaps with other moduli, the values in `covered` reflect the state of ALL other moduli.
                    # In the original state, `covered[i]` was 1 if covered by ANY class.
                    # In the temp state, `covered[i]` is 0 if NOT covered by any OTHER class (since this one is cleared).
                    # So if `covered[i]` was 1 in original, it might be 0 in temp (if only this class covered it).
                    # Or it might be 1 in temp (if another class covers it).
                    
                    # Let `A` be the set of indices in `S`.
                    # Let `O` be the set of indices covered by OTHER classes.
                    # Original `covered[i] = 1` if `i in O` OR `i in S`.
                    # Temp `covered[i] = 1` if `i in O`.
                    
                    # `added_zeros` = count of `i in S` such that `covered[i]` was 1.
                    # This is `|S ∩ (O ∪ S)|` = `|S|` = `m`.
                    # Wait. `covered[i]` is 1 if `i in O` or `i in S`.
                    # So ALL `i in S` have `covered[i]=1` in the original state?
                    # NO. `covered` is the OR of all classes.
                    # So yes, if `i in S`, then `covered[i]` is 1.
                    # So `added_zeros` is always `m`?
                    # No, `covered` is `uint8`.
                    # If I initialize `covered` by setting slices to 1, then yes, every index in a class's slice is 1.
                    # So `covered[S]` is all 1s.
                    # So `added_zeros` is always `m`.
                    # Then `unc_temp = unc + m`.
                    # Then `unc_final = unc_temp - m = unc`.
                    
                    # So the math works out IF `covered[S]` is all 1s.
                    # Is it?
                    # Yes, because `covered` is a boolean mask of "is this residue covered by at least one class?".
                    # If `i` is in the slice of class `c`, then `covered[i]` is 1.
                    # So `covered[S]` is always all 1s.
                    
                    # Therefore, `added_zeros` is always `m`.
                    # And `zeros_in_new` (count of 0s in `covered[r_new::m]` in temp state) is the number of spots in `r_new::m` that are NOT covered by other classes.
                    # Let `Z` be the set of indices in `r_new::m` that are 0 in temp state.
                    # These are the spots that become covered by adding this class.
                    # So `unc_candidate = unc_temp - |Z|`.
                    # `unc_temp = unc + m`.
                    # `unc_candidate = unc + m - |Z|`.
                    
                    # If we pick `r_old`:
                    # `Z` is the set of indices in `r_old::m` that are 0 in temp state.
                    # Temp state has `r_old::m` cleared.
                    # So `Z` is all of `r_old::m`? No.
                    # `covered[i]` in temp state is 1 if covered by OTHERS.
                    # So `Z` is `r_old::m` minus `O`.
                    # So `|Z| = m - |S ∩ O|`.
                    # `unc_candidate = unc + m - (m - |S ∩ O|) = unc + |S ∩ O|`.
                    # This is NOT `unc`.
                    
                    # Why?
                    # Because `unc` is the count of 0s in the ORIGINAL state.
                    # Original 0s are indices not in `O` and not in any `S`.
                    # Let `U` be the set of uncovered indices.
                    # `unc = |U|`.
                    # `U` is disjoint from `O` and all `S`.
                    
                    # In temp state, `covered` is `O`.
                    # Zeros in temp state are `not O`.
                    # `unc_temp` should be `|not O|`.
                    # But I calculated `unc_temp = unc + m`.
                    # Is `|not O| = |U| + m`?
                    # `not O` includes `U` (uncovered by anyone) AND `S \ O` (covered by this class but not others).
                    # So `|not O| = |U| + |S \ O|`.
                    # `|S \ O| = m - |S ∩ O|`.
                    # So `unc_temp = unc + m - |S ∩ O|`.
                    
                    # My previous calculation `unc_temp = unc + m` assumed `added_zeros = m`.
                    # `added_zeros` was `np.sum(covered[r_old::m])`.
                    # Since `covered[r_old::m]` is all 1s, `added_zeros = m`.
                    # But `unc_temp` is the NEW number of zeros.
                    # The change in zeros is `added_zeros`?
                    # No. The change in zeros is `(new_zeros_in_slice) - (old_zeros_in_slice)`.
                    # Old zeros in slice = 0 (since all 1s).
                    # New zeros in slice = `m` (since all 0s).
                    # So change is `+m`.
                    # So `unc_temp = unc + m`.
                    
                    # But logically, `unc_temp` should be `unc + |S \ O|`.
                    # Contradiction!
                    
                    # Ah, `covered` is a global array.
                    # When I do `covered[r_old::m] = 0`, I set those spots to 0.
                    # But what if those spots were ALREADY 0?
                    # They can't be 0, because `covered[i]=1` if `i` is in ANY class slice.
                    # So `covered[r_old::m]` is indeed all 1s.
                    
                    # So `unc_temp = unc + m` is correct based on the array update.
                    # But logically, the number of uncovered spots should only increase by the number of spots that were ONLY covered by this class.
                    # i.e., `|S \ O|`.
                    
                    # Where is the discrepancy?
                    # `unc` is the count of 0s in `covered`.
                    # `covered` is the OR of all classes.
                    # So `covered[i]=0` iff `i` is not in any class slice.
                    
                    # If I remove class `c` (slice `S`):
                    # `covered_new[i] = 0` iff `i` is not in any OTHER class slice.
                    # So `covered_new[i] = 0` iff `i \in not O`.
                    # So `unc_temp = |not O|`.
                    
                    # Original `unc = |U|` where `U = not (O ∪ S_1 ∪ ... ∪ S_K)`.
                    # `U` is subset of `not O`.
                    # `not O` includes `U` and `(S_c \ O)`.
                    # So `unc_temp = |U| + |S_c \ O|`.
                    
                    # My array calculation gave `unc_temp = unc + m`.
                    # This implies `|S_c \ O| = m`.
                    # Which means `S_c` and `O` are disjoint.
                    # i.e., No other class covers any spot in `S_c`.
                    # This is generally FALSE.
                    
                    # So the array calculation `unc_temp = unc + m` is WRONG?
                    # Let's re-evaluate `np.sum(covered[r_old::m])`.
                    # It returns the number of 1s in that slice.
                    # Since `covered` is the OR, and `S_c` is part of the OR, all elements in `S_c` are 1.
                    # So sum is `m`.
                    # So `unc` increases by `m`.
                    
                    # Why does the logical count increase by `|S_c \ O|`?
                    # Because `unc` counts 0s.
                    # Originally, spots in `S_c` are 1.
                    # After clearing, spots in `S_c` are 0.
                    # So we added `m` zeros to the slice.
                    # But some of those spots might have been 0 BEFORE?
                    # No, they were 1.
                    # So we added `m` zeros.
                    # So `unc_temp` should be `unc + m`.
                    
                    # But logically, `unc_temp` is `|not O|`.
                    # And `unc` is `|not (O ∪ S_c)|`.
                    # `|not (O ∪ S_c)|` is the number of spots NOT in O and NOT in S_c.
                    # `|not O|` is the number of spots NOT in O.
                    # The difference is `|not O| - |not (O ∪ S_c)|`.
                    # This is the number of spots that are NOT in O, but ARE in S_c.
                    # i.e., `|S_c \ O|`.
                    
                    # So `unc_temp = unc + |S_c \ O|`.
                    
                    # So `m` must equal `|S_c \ O|`?
                    # No.
                    
                    # The error is: `covered` is not just the OR of the current classes?
                    # Yes it is.
                    
                    # Wait, `covered[i]` is 1 if `i` is in ANY class.
                    # So if `i in S_c`, `covered[i]` is 1.
                    # So `covered[S_c]` is all 1s.
                    # So `np.sum(covered[S_c])` is `m`.
                    # So `unc` increases by `m`.
                    
                    # This implies `|S_c \ O| = m`.
                    # This means `S_c` is disjoint from `O`.
                    # This is only true if no other class covers any part of `S_c`.
                    
                    # Is it possible that `covered` is NOT the OR?
                    # No, the code does `covered[r::m] = 1`.
                    
                    # Let's test with a small example.
                    # N=4. Classes: [2, 0] (covers 0,2), [2, 1] (covers 1,3).
                    # covered = [1, 1, 1, 1]. unc=0.
                    # Remove [2, 0].
                    # covered[0::2] = 0. -> covered = [0, 1, 0, 1].
                    # added_zeros = sum([1,1]) = 2.
                    # unc_temp = 0 + 2 = 2.
                    # Logical: O = {1,3}. not O = {0,2}. |not O| = 2.
                    # unc = |not (O ∪ S_c)| = |not ({1,3} ∪ {0,2})| = |not {0,1,2,3}| = 0.
                    # Diff = 2.
                    # |S_c \ O| = |{0,2} \ {1,3}| = 2.
                    # Matches.
                    
                    # Another example:
                    # N=4. Classes: [2, 0] (0,2), [4, 0] (0).
                    # covered = [1, 0, 1, 0]. unc=2.
                    # Remove [2, 0].
                    # covered[0::2] = 0. -> covered = [0, 0, 0, 0].
                    # added_zeros = sum([1,1]) = 2.
                    # unc_temp = 2 + 2 = 4.
                    # Logical: O = {0} (from [4,0]). not O = {1,2,3}. |not O| = 3.
                    # Wait. covered became [0,0,0,0]. So unc_temp=4.
                    # But O only covers 0. So 1,2,3 should be 0. 0 is 0?
                    # Ah, `covered` was [1,0,1,0].
                    # `covered[0::2]` is indices 0,2.
                    # `covered[0]=1, covered[2]=1`.
                    # Set to 0.
                    # `covered` becomes [0,0,0,0].
                    # So unc_temp=4.
                    # But O covers 0. So `covered[0]` should be 1?
                    # No, `covered` is a single array.
                    # When I set `covered[0::2]=0`, I set index 0 to 0.
                    # But index 0 was covered by [4,0] too.
                    # So `covered[0]` should remain 1?
                    # NO. The code `covered[r::m] = 1` OVERWRITES.
                    # It does not OR.
                    # So if I remove a class, I MUST recalculate the coverage from scratch or use a count array.
                    
                    # **CRITICAL INSIGHT**: The current `covered` array approach is FLAWED for removal.
                    # Setting `covered[r::m] = 0` destroys information about other classes covering those spots.
                    # To correctly handle removal, we need a `counts` array, or we must recalculate `covered` from all classes.
                    
                    # Given the small N, recalculating `covered` from scratch for each candidate is too slow.
                    # Using a `counts` array is better.
                    # `counts[i]` = number of classes covering `i`.
                    # `covered[i] = 1 if counts[i] > 0`.
                    # `unc = sum(counts == 0)`.
                    
                    # When removing class `c` (slice `S`):
                    # `counts[S] -= 1`.
                    # `unc` increases by `sum(counts[S] == 0)`.
                    
                    # When adding class `c` (slice `S`):
                    # `counts[S] += 1`.
                    # `unc` decreases by `sum(counts[S] == 0)` (before increment).
                    
                    # This is efficient and correct.
                    
                    # Let's implement this.

# REWRITE PLAN
# 1. Use `counts` array.
# 2. Multi-start local search.
# 3. Optimized local search using `counts`.

def construct():
    base = 51975
    divisors = [d for d in range(2, base + 1) if base % d == 0]
    
    # Heuristic Init
    def get_initial_residues():
        # Greedy: largest first
        sorted_divs = sorted(divisors, reverse=True)
        residues = {}
        for m in sorted_divs:
            residues[m] = 0 # placeholder
        # Convert to list aligned with divisors
        res_list = [residues[m] for m in divisors]
        return res_list

    def run_local_search(initial_residues, time_limit=100):
        n_classes = len(divisors)
        residues = initial_residues.copy()
        
        counts = np.zeros(base, dtype=np.int32)
        for i, m in enumerate(divisors):
            r = residues[i]
            counts[r::m] += 1
        
        unc = int(np.sum(counts == 0))
        
        start_time = time.time()
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for i in range(n_classes):
                m = divisors[i]
                r_old = residues[i]
                
                # Remove old
                counts[r_old::m] -= 1
                added_unc = int(np.sum(counts[r_old::m] == 0))
                unc_temp = unc + added_unc
                
                best_r = r_old
                best_unc = unc_temp
                
                # Try all new
                for r_new in range(m):
                    # Check how many 0s in counts[r_new::m]
                    # These are the spots that will become covered
                    zeros_in_slice = int(np.sum(counts[r_new::m] == 0))
                    unc_cand = unc_temp - zeros_in_slice
                    if unc_cand < best_unc:
                        best_unc = unc_cand
                        best_r = r_new
                
                # Apply best
                if best_r != r_old:
                    counts[best_r::m] += 1
                    residues[i] = best_r
                    unc = best_unc
                    improved = True
                else:
                    counts[best_r::m] += 1
                    unc = best_unc
                    # Note: if best_r == r_old, unc should be original unc
                    # Let's verify: added_unc was sum(counts==0) after decrement.
                    # zeros_in_slice for r_old is sum(counts==0) after decrement.
                    # So unc_cand = unc_temp - added_unc = (unc + added_unc) - added_unc = unc.
                    # Correct.
        
        return residues, unc

    best_unc = base
    best_res = []
    
    # Run a few attempts
    for _ in range(3):
        init_res = get_initial_residues()
        res, unc = run_local_search(init_res, time_limit=30)
        if unc < best_unc:
            best_unc = unc
            best_res = res
    
    classes = [[divisors[i], best_res[i]] for i in range(len(divisors))]
    return classes
