import time
import math
import random

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Generate all divisors of 51975 > 1
    divisors = []
    for i in range(1, int(math.isqrt(base)) + 1):
        if base % i == 0:
            if i > 1:
                divisors.append(i)
            other = base // i
            if other > 1 and other != i:
                divisors.append(other)
    divisors.sort()
    
    # 2. Precompute bitmasks for all classes (m, r)
    # mask[m][r] is an integer where bit x is 1 if x % m == r
    masks = {}
    for m in divisors:
        masks[m] = [0] * m
        for r in range(m):
            mask = 0
            # Iterate through residues
            # x = r, r+m, r+2m, ... < base
            x = r
            while x < base:
                mask |= (1 << x)
                x += m
            masks[m][r] = mask
            
    # List of all possible classes as (m, r, mask)
    all_classes = []
    for m in divisors:
        for r in range(m):
            all_classes.append((m, r, masks[m][r]))
            
    # Map from (m, r) to index in all_classes for quick lookup? 
    # Actually, we just need to track which moduli are used.
    
    def calculate_uncovered_count(mask_union):
        return base - mask_union.bit_count()
        
    def greedy_cover(time_limit):
        """
        Greedy set cover.
        Returns (classes_list, union_mask)
        """
        uncovered = (1 << base) - 1
        used_moduli = set()
        classes = []
        union_mask = 0
        
        # To speed up greedy, we can pre-sort or just scan.
        # Scanning all classes every time is O(N_classes * iterations).
        # N_classes is sum of divisors. 
        # Sum of divisors of 51975 is roughly manageable.
        # Let's estimate sum of divisors.
        # 51975 = 3^3 * 5^2 * 7 * 11
        # Sigma(n) = Sigma(3^3)*Sigma(5^2)*Sigma(7)*Sigma(11)
        # Sigma(27) = 1+3+9+27 = 40
        # Sigma(25) = 1+5+25 = 31
        # Sigma(7) = 8
        # Sigma(11) = 12
        # Product = 40 * 31 * 8 * 12 = 12480 classes.
        # This is small enough to scan in a tight loop if optimized.
        
        while time.time() < time_limit:
            best_gain = 0
            best_m = -1
            best_r = -1
            best_mask = 0
            
            # Iterate through available moduli
            for m in divisors:
                if m in used_moduli:
                    continue
                # For each residue of m, calculate gain
                # gain = popcount(uncovered & mask[m][r])
                # We can optimize by checking if any residue has gain > 0
                for r in range(m):
                    inter = uncovered & masks[m][r]
                    gain = inter.bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_m = m
                        best_r = r
                        best_mask = masks[m][r]
                        
            if best_gain == 0:
                break
                
            classes.append([best_m, best_r])
            used_moduli.add(best_m)
            uncovered &= ~best_mask
            union_mask |= best_mask
            
        return classes, union_mask, uncovered, used_moduli

    def local_search(classes, union_mask, uncovered, used_moduli, time_limit):
        """
        Local search to improve the solution.
        Moves: Add, Remove, Swap.
        """
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Phase 1: Try to Remove any class that contributes 0 unique coverage
            # A class (m, r) is redundant if its mask is a subset of the union of OTHER classes.
            # Equivalent: (mask & ~union_mask_others) == 0
            # Or: (mask & uncovered) == 0 is NOT the condition for redundancy.
            # Redundancy: Removing it doesn't change the union.
            # i.e., union_mask_without_this == union_mask
            # i.e., (union_mask & ~mask) == union_mask_without_this
            # So if (mask & ~union_mask) == 0? No.
            # If mask is subset of (union_mask XOR mask)? No.
            # If we remove mask, new_union = union_mask & ~mask.
            # If new_union == union_mask, then mask was redundant.
            # This happens if mask & ~union_mask == 0? No, that's always true if mask is in union.
            # The condition is: mask is covered by the OTHER classes.
            # Let U be union of all classes.
            # Let U' be union of all classes except current.
            # If U == U', current is redundant.
            # U' = U & ~mask_current.
            # So if U == (U & ~mask_current), then mask_current & ~U == 0? No.
            # U & ~mask_current == U  <=>  mask_current & ~U == 0? No.
            # U & ~mask_current == U  <=>  (U & mask_current) is subset of U (always true) 
            # AND U has no bits that are ONLY in mask_current.
            # i.e., (mask_current & ~U') == 0.
            # But U' = U & ~mask_current.
            # So we need to check if mask_current has any bits that are not in U & ~mask_current.
            # This is equivalent to checking if mask_current is a subset of U & ~mask_current? No.
            
            # Simpler check:
            # Calculate unique bits of current class: unique = mask & ~ (union of others)
            # Union of others = union_mask & ~mask
            # unique = mask & ~(union_mask & ~mask) = mask & (~union_mask | mask) = mask & ~union_mask? No.
            # ~(A & ~B) = ~A | B.
            # unique = mask & (~union_mask | mask) = (mask & ~union_mask) | (mask & mask) = (mask & ~union_mask) | mask = mask.
            # This logic is circular.
            
            # Let's just compute:
            # other_union = union_mask & ~mask
            # if other_union == union_mask: redundant.
            
            for i in range(len(classes)):
                m, r = classes[i]
                mask = masks[m][r]
                other_union = union_mask & ~mask
                if other_union == union_mask:
                    # Redundant
                    classes.pop(i)
                    used_moduli.discard(m)
                    union_mask = other_union
                    uncovered = ~union_mask & ((1 << base) - 1)
                    improved = True
                    break # Restart loop after modification
            
            if improved:
                continue
                
            # Phase 2: Try to Add a new class
            best_add_gain = 0
            best_add_m = -1
            best_add_r = -1
            best_add_mask = 0
            
            for m in divisors:
                if m in used_moduli:
                    continue
                for r in range(m):
                    mask = masks[m][r]
                    gain = (uncovered & mask).bit_count()
                    if gain > best_add_gain:
                        best_add_gain = gain
                        best_add_m = m
                        best_add_r = r
                        best_add_mask = mask
                        
            if best_add_gain > 0:
                classes.append([best_add_m, best_add_r])
                used_moduli.add(best_add_m)
                union_mask |= best_add_mask
                uncovered &= ~best_add_mask
                improved = True
                continue
            
            # Phase 3: Try to Swap
            # Replace class i with class j (j unused)
            # Gain from adding j: G_j = popcount(uncovered & mask_j)
            # Loss from removing i: L_i = popcount(unique_i) where unique_i = mask_i & ~ (union_mask & ~mask_i)
            # Net gain = G_j - L_i
            # We want Net gain > 0.
            
            # Precompute losses for all current classes
            losses = []
            for i in range(len(classes)):
                m, r = classes[i]
                mask = masks[m][r]
                other_union = union_mask & ~mask
                unique_bits = mask & ~other_union
                loss = unique_bits.bit_count()
                losses.append((loss, i, m, r))
                
            # Sort by loss ascending (try removing low loss ones first)
            losses.sort(key=lambda x: x[0])
            
            # Precompute gains for all unused classes
            gains = []
            for m in divisors:
                if m in used_moduli:
                    continue
                for r in range(m):
                    mask = masks[m][r]
                    gain = (uncovered & mask).bit_count()
                    if gain > 0:
                        gains.append((gain, m, r, mask))
                        
            # Sort by gain descending
            gains.sort(key=lambda x: x[0], reverse=True)
            
            # Try combinations of top gains and top (low) losses
            # Limit iterations to avoid timeout
            found_swap = False
            for g_val, m_new, r_new, mask_new in gains[:20]:
                for l_val, i_old, m_old, r_old in losses[:20]:
                    if g_val > l_val:
                        # Perform swap
                        # Remove old
                        classes.pop(i_old)
                        used_moduli.discard(m_old)
                        
                        # Recalculate union without old
                        mask_old = masks[m_old][r_old]
                        # Note: union_mask still contains mask_old.
                        # We need to remove it.
                        # union_mask_new = union_mask & ~mask_old
                        # But other classes might cover parts of mask_old.
                        # So we just do:
                        union_mask = union_mask & ~mask_old
                        
                        # Add new
                        classes.append([m_new, r_new])
                        used_moduli.add(m_new)
                        union_mask |= mask_new
                        
                        uncovered = ~union_mask & ((1 << base) - 1)
                        
                        improved = True
                        found_swap = True
                        break
                if found_swap:
                    break
            
        return classes, union_mask, uncovered, used_moduli

    best_classes = []
    best_uncovered_count = base
    best_union = 0
    
    # Multi-start strategy
    # 1. Standard Greedy
    classes1, union1, unc1, used1 = greedy_cover(deadline - 20)
    classes1, union1, unc1, used1 = local_search(classes1, union1, unc1, used1, deadline - 5)
    unc_count1 = base - union1.bit_count()
    if unc_count1 < best_uncovered_count:
        best_uncovered_count = unc_count1
        best_classes = [c[:] for c in classes1]
        best_union = union1
        
    if time.time() > deadline - 10:
        return best_classes
        
    # 2. Greedy with random tie-breaking / perturbation
    # We can simulate different orders by shuffling divisors or adding random noise to gains.
    # Simpler: Run greedy, then perturb the result and local search.
    
    for attempt in range(5):
        if time.time() > deadline - 2:
            break
            
        # Start from best known
        current_classes = [c[:] for c in best_classes]
        
        # Rebuild state
        current_union = 0
        current_used = set()
        for m, r in current_classes:
            current_union |= masks[m][r]
            current_used.add(m)
        current_uncovered = ~current_union & ((1 << base) - 1)
        
        # Perturb: Remove 1-2 random classes
        if current_classes:
            num_remove = random.randint(1, 2)
            indices_to_remove = random.sample(range(len(current_classes)), min(num_remove, len(current_classes)))
            for idx in sorted(indices_to_remove, reverse=True):
                m, r = current_classes[idx]
                mask = masks[m][r]
                current_union &= ~mask
                current_used.discard(m)
                current_classes.pop(idx)
            current_uncovered = ~current_union & ((1 << base) - 1)
            
        # Run local search
        current_classes, current_union, current_uncovered, current_used = local_search(
            current_classes, current_union, current_uncovered, current_used, deadline - 1
        )
        
        unc_count = base - current_union.bit_count()
        if unc_count < best_uncovered_count:
            best_uncovered_count = unc_count
            best_classes = [c[:] for c in current_classes]
            best_union = current_union

    return best_classes

# Quick sanity check / test if run directly
if __name__ == "__main__":
    import time
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time: {end - start:.2f}s")
    print(f"Classes: {len(res)}")
    # Calculate score
    base = 51975
    union = 0
    for m, r in res:
        # Recreate mask for check
        mask = 0
        x = r
        while x < base:
            mask |= (1 << x)
            x += m
        union |= mask
    uncovered = base - union.bit_count()
    print(f"Uncovered: {uncovered}")
