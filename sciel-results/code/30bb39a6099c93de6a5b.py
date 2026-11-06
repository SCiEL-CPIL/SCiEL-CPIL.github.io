import numpy as np
import time
import random

def construct():
    base = 51975
    start_time = time.time()
    
    # Get all divisors > 1 of base
    divisors = [d for d in range(2, base + 1) if base % d == 0]
    divisors.sort()
    
    # Precompute coverage masks for each (m, r) is too expensive (51975 * 51975).
    # Instead, we compute on the fly or use slicing.
    # For local search, we need to evaluate the impact of changes quickly.
    
    # Strategy:
    # 1. Greedy construction: Start with empty set. Repeatedly add (m, r) that covers most uncovered residues.
    # 2. Local Search: Try to swap (m, r) in the set with (m', r') not in the set, or change r to r' for same m.
    #    Accept if uncovered count decreases.
    
    def count_uncovered(covered):
        return (~covered).sum()
    
    def evaluate(classes):
        covered = np.zeros(base, dtype=bool)
        for m, r in classes:
            covered[r::m] = True
        return count_uncovered(covered)
    
    best_uncovered = base
    best_classes = []
    
    # Helper to get coverage mask for a single class
    def get_mask(m, r):
        mask = np.zeros(base, dtype=bool)
        mask[r::m] = True
        return mask
    
    # Greedy Phase
    # We will try to build a set of classes.
    # To speed up, we can restrict the search space of moduli initially or use all.
    # There are 3^3 * 5^2 * 7 * 11 = 27 * 25 * 7 * 11 = 51975 divisors.
    # Actually, number of divisors is (3+1)*(2+1)*(1+1)*(1+1) = 4*3*2*2 = 48 divisors.
    # Wait, 51975 = 3^3 * 5^2 * 7^1 * 11^1.
    # Number of divisors is (3+1)*(2+1)*(1+1)*(1+1) = 4 * 3 * 2 * 2 = 48.
    # This is a very small number of moduli!
    # We can use all 47 divisors (excluding 1) if we want.
    # But we must use DISTINCT moduli.
    
    available_moduli = [d for d in divisors if d > 1]
    
    # Greedy Initialization
    covered = np.zeros(base, dtype=bool)
    current_classes = []
    used_moduli = set()
    
    # We can pick any subset of moduli. Let's start with a greedy selection.
    # Since there are only 47 moduli, we can try to include all of them if it helps.
    # But maybe smaller moduli are better?
    # Let's try a greedy algorithm that adds the best (m, r) pair from all available moduli.
    
    for _ in range(len(available_moduli)):
        best_gain = -1
        best_m = None
        best_r = None
        
        # To speed up, we might not check all residues for all moduli every step.
        # But 47 moduli * avg 100 residues = 4700 checks. Each check is O(N/m).
        # Total operations ~ sum(N/m). This is manageable.
        
        for m in available_moduli:
            if m in used_moduli:
                continue
            
            # Find best r for this m
            # We want to maximize number of newly covered residues.
            # uncovered_mask = ~covered
            # For each r, gain = sum(uncovered_mask[r::m])
            
            # Optimization: If m is large, r::m is small.
            # We can iterate r.
            max_gain_for_m = -1
            best_r_for_m = 0
            
            # If m is small, we check all r. If m is large, we check all r.
            # Since max m is 51975, and min m is 3.
            # Sum of m for all divisors is not too large.
            # Let's just loop.
            for r in range(m):
                # Gain is number of True in ~covered[r::m]
                # This is equivalent to len(uncovered_indices_in_slice)
                # Using numpy: (~covered[r::m]).sum()
                gain = (~covered[r::m]).sum()
                if gain > max_gain_for_m:
                    max_gain_for_m = gain
                    best_r_for_m = r
            
            if max_gain_for_m > best_gain:
                best_gain = max_gain_for_m
                best_m = m
                best_r = best_r_for_m
        
        if best_m is None or best_gain <= 0:
            break
        
        # Add the best class
        current_classes.append([best_m, best_r])
        used_moduli.add(best_m)
        covered[best_r::best_m] = True
        
        if time.time() - start_time > 100:
            break
    
    current_uncovered = count_uncovered(covered)
    if current_uncovered < best_uncovered:
        best_uncovered = current_uncovered
        best_classes = current_classes.copy()
    
    # Local Search Phase
    # We have a set of classes. We try to improve it.
    # Moves:
    # 1. Change residue r of an existing modulus m to r'.
    # 2. Replace an existing modulus m with a new modulus m' (and best residue r').
    # 3. Add a new modulus m' (if not used) with best residue r'. (Only if we allow size to grow, but greedy already added as many as possible with positive gain. Maybe we missed some due to local optimum in greedy).
    
    # Let's define a function to get the state (covered, uncovered_count, classes, used_moduli)
    # and perform moves.
    
    time_limit = 110.0
    
    while time.time() < time_limit:
        improved = False
        
        # Try Move 1: Change residue
        for i, (m, r) in enumerate(current_classes):
            # Try a few random residues or all if m is small
            # If m is small, try all. If m is large, try random subset or best few.
            # To be safe and fast, let's try all if m < 100, else random 10.
            if m < 100:
                candidates = range(m)
            else:
                candidates = random.sample(range(m), min(20, m))
            
            for new_r in candidates:
                if new_r == r:
                    continue
                
                # Calculate delta
                # Old contribution: covered[r::m]
                # New contribution: covered[new_r::m]
                # We need to know how many residues are covered ONLY by this class.
                # This is complex. Easier: simulate the change.
                
                # Save state
                old_covered = covered.copy()
                old_count = current_uncovered
                
                # Apply change
                # Remove old coverage? No, just update the specific slice?
                # It's hard to "remove" coverage because other classes might cover those spots.
                # So we must recompute the whole covered array or use a count array.
                
                # Let's use a count array approach for speed in local search?
                # count[x] = number of classes covering x.
                # If count[x] > 0, it's covered.
                # This allows easy add/remove.
                
                # Let's restart local search with count array.
                # This is a significant change. Let's break and restart logic with count array.
                pass
            
        # Break out to restart with better data structure
        break
        
    # Restart Local Search with Count Array for efficiency
    # Initialize count array from best_classes
    count_arr = np.zeros(base, dtype=int)
    for m, r in best_classes:
        count_arr[r::m] += 1
    
    current_uncovered = (count_arr == 0).sum()
    current_classes = best_classes.copy()
    used_moduli = set(m for m, r in current_classes)
    
    # Precompute masks for all possible (m, r) is too big.
    # We will compute delta on the fly.
    
    def get_delta_add(m, r):
        # How many new residues are covered?
        # These are residues x where count_arr[x] == 0 and x % m == r.
        # mask = (count_arr == 0) & (indices % m == r)
        # We can use slicing.
        # indices = np.arange(base)
        # But slicing is faster.
        # uncovered_mask = (count_arr == 0)
        # gain = uncovered_mask[r::m].sum()
        return (count_arr[r::m] == 0).sum()
        
    def get_delta_remove(m, r):
        # How many residues become uncovered?
        # These are residues x where count_arr[x] == 1 and x % m == r.
        # loss = (count_arr[r::m] == 1).sum()
        return (count_arr[r::m] == 1).sum()
        
    def apply_add(m, r):
        count_arr[r::m] += 1
        
    def apply_remove(m, r):
        count_arr[r::m] -= 1
        
    # Local Search Loop
    no_improve_count = 0
    
    while time.time() < time_limit and no_improve_count < 10:
        no_improve_count += 1
        improved = False
        
        # 1. Try to change residue of existing classes
        for i, (m, r) in enumerate(current_classes):
            # Try all residues if m is small, else sample
            if m <= 50:
                res_list = range(m)
            else:
                res_list = random.sample(range(m), min(50, m))
                
            for new_r in res_list:
                if new_r == r:
                    continue
                    
                # Delta = (gain from new_r) - (loss from old_r)
                # But we must be careful: the gain from new_r is calculated assuming old_r is still there?
                # No, we are replacing.
                # Loss: residues that become uncovered if we remove old_r.
                #   These are x where count[x]==1 and x% m == r.
                # Gain: residues that become covered if we add new_r.
                #   These are x where count[x]==0 and x% m == new_r.
                #   BUT, if x is in both sets (overlap), the net effect is different.
                #   Actually, if we remove old and add new:
                #   New count[x] = old count[x] - 1 (if x% m == r) + 1 (if x% m == new_r)
                #   
                #   Let's just simulate on a copy or calculate carefully.
                #   Simulation is safer.
                
                # To avoid copying the whole array, let's calculate:
                # loss = sum(1 for x in old_slice if count[x]==1)
                # gain = sum(1 for x in new_slice if count[x]==0)
                #   Wait, if x is in old_slice and count[x]==1, it becomes 0 (uncovered) -> LOSS.
                #   If x is in new_slice and count[x]==0, it becomes 1 (covered) -> GAIN.
                #   If x is in BOTH slices (overlap):
                #     If count[x]==1: remove -> 0, add -> 1. Net 0.
                #     If count[x]==0: remove (no effect as it wasn't covered by this class? No, count[x] is total. If count[x]==0, it wasn't covered by this class, so removing this class doesn't change it? Wait.
                #     The class (m,r) contributes 1 to count[x] if x%m==r.
                #     If x%m==r and x%m==new_r, then r==new_r mod m. Since 0<=r,new_r<m, r==new_r.
                #     So slices are disjoint if r != new_r.
                #     So no overlap.
                
                loss = (count_arr[r::m] == 1).sum()
                gain = (count_arr[new_r::m] == 0).sum()
                
                if gain > loss:
                    # Apply move
                    apply_remove(m, r)
                    apply_add(m, new_r)
                    current_classes[i][1] = new_r
                    current_uncovered += loss - gain
                    improved = True
                    no_improve_count = 0
                    break # Restart loop for this class? Or continue?
                    # Let's restart the inner loop to be safe with counts.
                    
            if improved:
                break
                
        if improved:
            continue
            
        # 2. Try to replace a class with a new one
        # Pick a random class to remove, and a random unused modulus to add.
        if len(current_classes) > 0:
            # Try to remove a class that causes high loss? Or just random.
            # Let's try to find a swap.
            # We'll try a few random swaps.
            for _ in range(100):
                if time.time() > time_limit:
                    break
                    
                # Pick random index to remove
                idx = random.randint(0, len(current_classes) - 1)
                m_old, r_old = current_classes[idx]
                
                # Pick random unused modulus
                unused_mods = [m for m in available_moduli if m not in used_moduli]
                if not unused_mods:
                    break
                    
                m_new = random.choice(unused_mods)
                
                # Find best r for m_new
                # We want to maximize gain.
                # gain = (count_arr[r::m_new] == 0).sum()
                # We can just find the r with max gain.
                best_r_new = 0
                max_gain = -1
                if m_new <= 50:
                    res_list = range(m_new)
                else:
                    res_list = random.sample(range(m_new), min(50, m_new))
                    
                for r in res_list:
                    gain = (count_arr[r::m_new] == 0).sum()
                    if gain > max_gain:
                        max_gain = gain
                        best_r_new = r
                        
                loss = (count_arr[r_old::m_old] == 1).sum()
                
                if max_gain > loss:
                    # Apply swap
                    apply_remove(m_old, r_old)
                    apply_add(m_new, best_r_new)
                    current_classes[idx] = [m_new, best_r_new]
                    used_moduli.remove(m_old)
                    used_moduli.add(m_new)
                    current_uncovered += loss - max_gain
                    improved = True
                    no_improve_count = 0
                    break
                    
            if improved:
                continue

        # 3. Try to add a new class (if we haven't used all moduli)
        # This might help if greedy stopped too early.
        unused_mods = [m for m in available_moduli if m not in used_moduli]
        if unused_mods:
            # Find the best (m, r) to add
            best_gain_add = 0
            best_m_add = None
            best_r_add = None
            
            # Sample some unused moduli
            sample_mods = random.sample(unused_mods, min(20, len(unused_mods)))
            for m in sample_mods:
                if m <= 50:
                    res_list = range(m)
                else:
                    res_list = random.sample(range(m), min(50, m))
                    
                for r in res_list:
                    gain = (count_arr[r::m] == 0).sum()
                    if gain > best_gain_add:
                        best_gain_add = gain
                        best_m_add = m
                        best_r_add = r
                        
            if best_m_add is not None and best_gain_add > 0:
                # Add it
                apply_add(best_m_add, best_r_add)
                current_classes.append([best_m_add, best_r_add])
                used_moduli.add(best_m_add)
                current_uncovered -= best_gain_add
                improved = True
                no_improve_count = 0

        if not improved:
            # Random restart or perturbation?
            # Let's try a random perturbation: change a random residue to a random value.
            if len(current_classes) > 0 and random.random() < 0.1:
                idx = random.randint(0, len(current_classes) - 1)
                m, r = current_classes[idx]
                new_r = random.randint(0, m - 1)
                if new_r != r:
                    loss = (count_arr[r::m] == 1).sum()
                    gain = (count_arr[new_r::m] == 0).sum()
                    # Accept even if worse? Simulated Annealing style.
                    # For now, only accept if better or equal to avoid noise.
                    # But we are stuck, so maybe accept worse with low prob?
                    # Let's just accept if better.
                    if gain >= loss:
                        apply_remove(m, r)
                        apply_add(m, new_r)
                        current_classes[idx][1] = new_r
                        current_uncovered += loss - gain
                        no_improve_count = 0

    # Final check
    if current_uncovered < best_uncovered:
        best_uncovered = current_uncovered
        best_classes = current_classes.copy()
        
    # Ensure we return a valid list
    if not best_classes:
        best_classes = [[3, 0]]
        
    return best_classes
