"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill

    base = 51975

    # All odd divisors of 51975 greater than 1
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    divisors.sort()  # INCREASING order
    num_moduli = len(divisors)

    best_score = base
    best_classes = None

    # Precompute indices for each divisor for faster access if needed, 
    # but slicing is usually fast enough. 
    # However, for incremental updates, we need to be careful.

    def update_score_incremental(counts, uncovered, m, old_r, new_r):
        """
        Update counts and uncovered score when changing residue of modulus m
        from old_r to new_r.
        Returns new uncovered count.
        """
        if old_r == new_r:
            return uncovered

        # Indices affected
        old_idx = np.arange(old_r, base, m)
        new_idx = np.arange(new_r, base, m)

        # We need to handle the union of these sets.
        # To avoid O(N) set operations, we can just update the specific indices.
        # But we must be careful not to double count if there is overlap.
        
        # Simpler approach: 
        # 1. Decrement old, track changes
        # 2. Increment new, track changes
        # But overlap makes this tricky.
        
        # Alternative: Just recompute uncovered for the affected indices?
        # No, uncovered is global.
        
        # Let's stick to the previous method but optimize the cost.
        # The previous method did:
        # counts[old::m] -= 1
        # counts[new::m] += 1
        # score = count_nonzero(counts == 0)
        
        # This is O(N) for the count_nonzero.
        # Can we do better?
        # The change in uncovered is:
        # +1 for each i in old_idx \ new_idx where counts[i] becomes 0
        # -1 for each i in new_idx \ old_idx where counts[i] was 0
        
        # Let's find the symmetric difference.
        # Since m is a divisor, the sets are arithmetic progressions.
        # If old_r == new_r, diff is empty.
        # Otherwise, they are disjoint? No.
        # Example: m=3, old=0, new=1. 0,3,6... vs 1,4,7... Disjoint.
        # Example: m=3, old=0, new=3 (not allowed, r<m).
        # Since 0 <= r < m, old_r != new_r implies the sets are disjoint?
        # Let's check.
        # i = old_r + k*m = new_r + j*m
        # old_r - new_r = (j-k)*m
        # |old_r - new_r| < m, so (j-k) must be 0, implying old_r == new_r.
        # So YES, if old_r != new_r, the sets are DISJOINT.
        
        # This simplifies things greatly!
        
        # 1. Process old_idx
        # For i in old_idx:
        #   counts[i] -= 1
        #   if counts[i] == 0: uncovered += 1
        
        # 2. Process new_idx
        # For i in new_idx:
        #   if counts[i] == 0: uncovered -= 1
        #   counts[i] += 1
        
        # We can vectorize this.
        
        # Step 1: Decrement old
        old_counts = counts[old_idx].copy() # Avoid modifying view? No, we modify counts directly.
        # But we need to know which ones became 0.
        # counts[old_idx] -= 1
        # new_zero_mask = (counts[old_idx] == 0)
        # uncovered += sum(new_zero_mask)
        
        # Step 2: Increment new
        # new_zero_mask = (counts[new_idx] == 0)
        # uncovered -= sum(new_zero_mask)
        # counts[new_idx] += 1
        
        # Let's implement this.
        
        # Decrement old
        counts[old_idx] -= 1
        # Count how many became zero
        # Note: counts[old_idx] is a view.
        became_zero = np.count_nonzero(counts[old_idx] == 0)
        uncovered += became_zero
        
        # Increment new
        was_zero = np.count_nonzero(counts[new_idx] == 0)
        uncovered -= was_zero
        counts[new_idx] += 1
        
        return uncovered

    def greedy_initialize(order, rng):
        """Greedy initialization with a given order of moduli."""
        counts = np.zeros(base, dtype=np.int8)
        residues = [0] * num_moduli
        uncovered = base

        for idx in order:
            m = divisors[idx]
            best_r = 0
            best_gain = -1 # We want to minimize uncovered, so maximize gain (reduction in uncovered)
            
            # For small moduli, checking all residues is expensive if we do O(N) each time.
            # But we can estimate gain.
            # Gain = number of currently uncovered indices in r::m
            
            if m <= 50:
                # Check all residues
                for r in range(m):
                    # Count uncovered in r::m
                    # This is O(N/m). Sum over r is O(N).
                    # So total for this modulus is O(N).
                    # For small m, this is acceptable if we don't do it too often.
                    gain = np.count_nonzero(counts[r::m] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
            else:
                # Sample residues
                # We want to find a residue that covers many uncovered spots.
                # Random sampling is okay.
                cands = set()
                cands.add(0)
                cands.add(rng.randint(0, m - 1))
                # Add a few more
                for _ in range(20):
                    cands.add(rng.randint(0, m - 1))
                
                for r in cands:
                    gain = np.count_nonzero(counts[r::m] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r

            residues[idx] = best_r
            # Update counts and uncovered
            counts[best_r::m] += 1
            # We need to update uncovered.
            # The indices covered are best_r::m.
            # Those that were 0 are now 1, so uncovered decreases by gain.
            # But wait, `gain` was calculated BEFORE the update.
            # So uncovered -= gain.
            uncovered -= best_gain
            # However, this assumes no double counting issues, which is fine for initialization.
            # But `counts` might have been > 0 for some, so gain is exactly the number of 0s.
            # So this is correct.

        return residues, counts, uncovered

    def local_search(residues, counts, uncovered, rng, time_limit):
        """Simulated annealing local search."""
        current_residues = residues[:]
        current_counts = counts.copy()
        current_uncovered = uncovered

        temperature = 5.0
        cooling = 0.999
        min_temp = 0.1
        best_local_uncovered = current_uncovered
        best_local_residues = current_residues[:]

        start_ls = time.time()

        while time.time() - start_ls < time_limit and temperature > min_temp:
            # Pick a random modulus index
            # Bias towards smaller moduli? They have more impact.
            i = rng.randint(0, num_moduli - 1)
            m = divisors[i]
            old_r = current_residues[i]
            new_r = rng.randint(0, m - 1)

            if new_r == old_r:
                continue

            # Calculate new uncovered
            new_uncovered = update_score_incremental(current_counts, current_uncovered, m, old_r, new_r)
            delta = new_uncovered - current_uncovered

            if delta < 0 or rng.random() < np.exp(-delta / temperature):
                current_residues[i] = new_r
                current_uncovered = new_uncovered
                if current_uncovered < best_local_uncovered:
                    best_local_uncovered = current_uncovered
                    best_local_residues = current_residues[:]
            else:
                # Revert
                # Update back
                update_score_incremental(current_counts, current_uncovered, m, new_r, old_r)
                # Note: update_score_incremental updates counts in place.
                # We passed current_uncovered, but it returns the new value.
                # We need to restore current_uncovered to the old value.
                # The function modifies counts. We need to ensure counts are restored.
                # The function does:
                # counts[old] -= 1, counts[new] += 1
                # To revert:
                # counts[new] -= 1, counts[old] += 1
                # Which is exactly update_score_incremental(counts, _, m, new_r, old_r)
                # BUT, we also need to restore the `current_uncovered` variable.
                # The function returns the new uncovered.
                # Let's just recompute or store the old value.
                
                # Actually, my update_score_incremental modifies counts and returns new uncovered.
                # It doesn't restore the old uncovered variable in the caller scope automatically.
                # In the 'accept' branch, I did current_uncovered = new_uncovered.
                # In the 'reject' branch, I called update... which modified counts.
                # I need to set current_uncovered back to the old value.
                # Let's refactor update to not rely on side effects for the score variable, or just be careful.
                
                # Let's re-read my update function.
                # It takes uncovered, modifies counts, returns new uncovered.
                # So in reject case:
                # I called update_score_incremental(..., new_r, old_r)
                # This restored counts to original state.
                # And it returned the original uncovered (since it's symmetric).
                # So I should assign: current_uncovered = update_score_incremental(...)
                
                current_uncovered = update_score_incremental(current_counts, current_uncovered, m, new_r, old_r)

            temperature *= cooling

        return best_local_residues, best_local_uncovered

    def hill_climb(residues, counts, uncovered, rng, time_limit):
        """Deterministic hill climbing: try each modulus in sequence, pick best residue."""
        current_residues = residues[:]
        current_counts = counts.copy()
        current_uncovered = uncovered
        improved = True
        start_hc = time.time()

        while improved and time.time() - start_hc < time_limit:
            improved = False
            for i in range(num_moduli):
                if time.time() - start_hc > time_limit:
                    break
                m = divisors[i]
                old_r = current_residues[i]

                # Try to find a better residue for this modulus
                # We want to minimize uncovered.
                # Remove current contribution
                # We can't just "remove" and re-add easily without tracking.
                # Let's use the incremental update logic.
                
                # Temporarily remove old_r
                # We need to find best new_r.
                # Let's evaluate candidates.
                
                best_r = old_r
                best_uncovered = current_uncovered
                
                # Candidate residues
                if m <= 20:
                    candidates = range(m)
                else:
                    # Sample
                    candidates = set()
                    candidates.add(old_r)
                    for _ in range(30):
                        candidates.add(rng.randint(0, m - 1))
                    candidates = list(candidates)
                
                # We need to evaluate the score if we switch to each candidate.
                # To do this efficiently, we can:
                # 1. Remove old_r (update counts, uncovered)
                # 2. For each candidate r:
                #    a. Add r (update counts, uncovered)
                #    b. Record score
                #    c. Remove r (revert counts, uncovered)
                # 3. Restore old_r (update counts, uncovered)
                
                # Step 1: Remove old_r
                u_without_old = update_score_incremental(current_counts, current_uncovered, m, old_r, old_r) # No-op? 
                # My function requires old != new. 
                # Let's make a helper or handle it.
                # If old == new, no change.
                # But we want to "remove" it. 
                # We can simulate removal by setting it to a dummy value? No.
                # Let's just do the update manually for removal.
                
                # Removal of old_r:
                # counts[old_r::m] -= 1
                # uncovered += count_nonzero(counts[old_r::m] == 0) # Before decrement?
                # No, after decrement, if it becomes 0, it was 1.
                # So:
                idx_old = np.arange(old_r, base, m)
                current_counts[idx_old] -= 1
                became_zero = np.count_nonzero(current_counts[idx_old] == 0)
                u_without_old = current_uncovered + became_zero
                
                # Now try candidates
                for r2 in candidates:
                    if r2 == old_r:
                        continue
                    
                    # Add r2
                    idx_new = np.arange(r2, base, m)
                    was_zero = np.count_nonzero(current_counts[idx_new] == 0)
                    u_with_new = u_without_old - was_zero
                    current_counts[idx_new] += 1
                    
                    if u_with_new < best_uncovered:
                        best_uncovered = u_with_new
                        best_r = r2
                        
                    # Remove r2 (revert)
                    current_counts[idx_new] -= 1
                    # uncovered goes back to u_without_old
                    # We don't need to update the variable u_without_old, it's constant in this loop.
                
                # Restore old_r
                current_counts[idx_old] += 1
                # uncovered goes back to current_uncovered
                # We don't need to update variable.
                
                if best_r != old_r:
                    # Apply the best change
                    # We already know the best_r and best_uncovered.
                    # We need to update counts and current_uncovered.
                    # Current state: counts has old_r removed, no new_r added.
                    # We want: counts has old_r removed, best_r added.
                    
                    idx_best = np.arange(best_r, base, m)
                    was_zero_best = np.count_nonzero(current_counts[idx_best] == 0)
                    current_uncovered = u_without_old - was_zero_best
                    current_counts[idx_best] += 1
                    
                    current_residues[i] = best_r
                    improved = True

        return current_residues, current_uncovered

    def remove_redundant(classes):
        """Remove classes that don't contribute to coverage."""
        if not classes:
            return classes
        counts = np.zeros(base, dtype=np.int8)
        for m, r in classes:
            counts[r::m] += 1
        cur_uncov = int(np.count_nonzero(counts == 0))

        new_classes = []
        for cls in classes:
            m, r = cls
            counts[r::m] -= 1
            unc_without = int(np.count_nonzero(counts == 0))
            if unc_without <= cur_uncov:
                # This class is redundant
                pass
            else:
                new_classes.append(cls)
                counts[r::m] += 1
                cur_uncov = unc_without

        return new_classes

    rng = random.Random(42)

    # Multiple restarts
    # Strategy: Small to Large is usually good for covering.
    orders = [
        list(range(num_moduli)),
        list(range(num_moduli - 1, -1, -1)),
        rng.sample(range(num_moduli), num_moduli)
    ]

    for restart in range(5):
        if time.time() > deadline - 10:
            break

        order = orders[restart % len(orders)]
        residues, counts, uncovered = greedy_initialize(order, rng)

        if uncovered < best_score:
            best_score = uncovered
            best_classes = [[divisors[i], residues[i]] for i in range(num_moduli)]

        # Local search
        if time.time() > deadline - 15:
            break
        ls_time = min(10.0, deadline - 15 - time.time())
        if ls_time > 1.0:
            best_residues, best_ls_uncovered = local_search(residues, counts, uncovered, rng, ls_time)
            if best_ls_uncovered < best_score:
                best_score = best_ls_uncovered
                best_classes = [[divisors[i], best_residues[i]] for i in range(num_moduli)]
            # Use best_residues for next step? Or keep original?
            # Let's use the best found so far for hill climb.
            current_res = best_residues if best_ls_uncovered < uncovered else residues
            current_unc = best_ls_uncovered if best_ls_uncovered < uncovered else uncovered
        else:
            current_res = residues
            current_unc = uncovered

        # Hill climb
        if time.time() > deadline - 10:
            break
        hc_time = min(10.0, deadline - 10 - time.time())
        if hc_time > 1.0:
            hc_residues, hc_uncovered = hill_climb(current_res, counts, current_unc, rng, hc_time)
            if hc_uncovered < best_score:
                best_score = hc_uncovered
                best_classes = [[divisors[i], hc_residues[i]] for i in range(num_moduli)]

    # Final polish
    if best_classes is not None and time.time() < deadline - 5:
        best_classes = remove_redundant(best_classes)
        counts = np.zeros(base, dtype=np.int8)
        for m, r in best_classes:
            counts[r::m] += 1
        best_score = int(np.count_nonzero(counts == 0))

    return best_classes
# EVOLVE-BLOCK-END
