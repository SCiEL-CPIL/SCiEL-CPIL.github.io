import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    denominators = list(range(2, 49))
    n = len(denominators)
    recips_float = [1.0 / d for d in denominators]
    recips_frac = [Fraction(1, d) for d in denominators]

    best_deficit = None
    best_indices = None
    best_sum_float = 0.0

    def update_best(indices, s_frac):
        nonlocal best_deficit, best_indices, best_sum_float
        if s_frac >= 1:
            return
        deficit = 1 - s_frac
        if best_deficit is None or deficit < best_deficit:
            best_deficit = deficit
            best_indices = list(indices)
            best_sum_float = float(s_frac)

    # Phase 1: Greedy construction to get a good initial solution
    # Sort by reciprocal descending (largest reciprocals first = smallest denominators)
    # Try multiple strategies: include/exclude large denominators
    for strategy in range(3):
        if time.time() > deadline:
            break
        # Strategy 0: Include all denominators, try to remove
        # Strategy 1: Start with denominators 2-24, add from 25-48
        # Strategy 2: Random-ish but deterministic greedy
        
        if strategy == 0:
            # Start with all, remove to get below 1 with minimal deficit
            mask = (1 << n) - 1
            s_float = sum(recips_float)
            # Remove smallest reciprocals first (largest denominators) until sum < 1
            for i in range(n - 1, -1, -1):
                if s_float - recips_float[i] < 1.0:
                    break
                s_float -= recips_float[i]
                mask ^= (1 << i)
            # Now try to add back smaller reciprocals to get closer to 1
            improved = True
            while improved:
                improved = False
                s_frac = Fraction(0)
                for i in range(n):
                    if (mask >> i) & 1:
                        s_frac += recips_frac[i]
                if s_frac >= 1:
                    break
                deficit = 1 - s_frac
                # Try adding each excluded element
                for i in range(n):
                    if not ((mask >> i) & 1):
                        new_s = s_frac + recips_frac[i]
                        if new_s < 1:
                            new_deficit = 1 - new_s
                            if new_deficit < deficit:
                                mask ^= (1 << i)
                                s_frac = new_s
                                deficit = new_deficit
                                improved = True
                                break
            if best_deficit is None:
                s_frac = Fraction(0)
                for i in range(n):
                    if (mask >> i) & 1:
                        s_frac += recips_frac[i]
                update_best([i for i in range(n) if (mask >> i) & 1], s_frac)
            else:
                s_frac = Fraction(0)
                for i in range(n):
                    if (mask >> i) & 1:
                        s_frac += recips_frac[i]
                if s_frac < 1:
                    d = 1 - s_frac
                    if d < best_deficit:
                        best_deficit = d
                        best_indices = [i for i in range(n) if (mask >> i) & 1]
                        best_sum_float = float(s_frac)

        elif strategy == 1:
            # Greedy: pick elements that minimize deficit
            # Start with nothing, add elements that bring us closer to 1
            mask = 0
            s_frac = Fraction(0)
            # Sort candidates by how much they help (reciprocal value)
            # But be smart: prefer elements that don't overshoot
            for i in range(n - 1, -1, -1):  # From largest denominator to smallest
                if s_frac + recips_frac[i] < 1:
                    mask |= (1 << i)
                    s_frac += recips_frac[i]
            # Now try to improve by swapping
            improved = True
            while improved:
                improved = False
                s_frac = Fraction(0)
                for i in range(n):
                    if (mask >> i) & 1:
                        s_frac += recips_frac[i]
                if s_frac >= 1:
                    break
                deficit = 1 - s_frac
                # Try adding
                for i in range(n):
                    if not ((mask >> i) & 1):
                        if s_frac + recips_frac[i] < 1:
                            new_deficit = 1 - (s_frac + recips_frac[i])
                            if new_deficit < deficit:
                                mask |= (1 << i)
                                s_frac += recips_frac[i]
                                deficit = new_deficit
                                improved = True
                                break
                if improved:
                    continue
                # Try removing one and adding another
                for i in range(n):
                    if (mask >> i) & 1:
                        temp_s = s_frac - recips_frac[i]
                        for j in range(n):
                            if not ((mask >> j) & 1) and j != i:
                                if temp_s + recips_frac[j] < 1:
                                    new_deficit = 1 - (temp_s + recips_frac[j])
                                    if new_deficit < deficit:
                                        mask &= ~(1 << i)
                                        mask |= (1 << j)
                                        s_frac = temp_s + recips_frac[j]
                                        deficit = new_deficit
                                        improved = True
                                        break
                        if improved:
                            break
            if best_deficit is None:
                s_frac = Fraction(0)
                for i in range(n):
                    if (mask >> i) & 1:
                        s_frac += recips_frac[i]
                update_best([i for i in range(n) if (mask >> i) & 1], s_frac)
            else:
                s_frac = Fraction(0)
                for i in range(n):
                    if (mask >> i) & 1:
                        s_frac += recips_frac[i]
                if s_frac < 1:
                    d = 1 - s_frac
                    if d < best_deficit:
                        best_deficit = d
                        best_indices = [i for i in range(n) if (mask >> i) & 1]
                        best_sum_float = float(s_frac)

    # Phase 2: Local search / hill climbing with multiple restarts
    if best_indices is not None:
        current_indices = list(best_indices)
    else:
        current_indices = [0]  # denominator 2, sum=0.5
    
    # Compute current sum
    current_sum_frac = Fraction(0)
    for i in current_indices:
        current_sum_frac += recips_frac[i]
    
    current_deficit = 1 - current_sum_frac if current_sum_frac < 1 else None
    
    # Local search: try adding, removing, swapping
    max_iterations = 5000
    iteration = 0
    while time.time() < deadline and iteration < max_iterations:
        iteration += 1
        
        if current_deficit is None or current_deficit <= 0:
            break
        
        current_set = set(current_indices)
        improved = False
        
        # Try adding one element
        for i in range(n):
            if i not in current_set:
                new_sum = current_sum_frac + recips_frac[i]
                if new_sum < 1:
                    new_deficit = 1 - new_sum
                    if new_deficit < current_deficit:
                        current_indices.append(i)
                        current_set.add(i)
                        current_sum_frac = new_sum
                        current_deficit = new_deficit
                        improved = True
                        break
        
        if improved:
            continue
        
        # Try removing one element (to make room for better ones)
        removed_any = False
        for i in list(current_indices):
            if len(current_indices) <= 1:
                break
            new_sum = current_sum_frac - recips_frac[i]
            # After removal, try to add back something better
            current_set.remove(i)
            current_indices.remove(i)
            current_sum_frac = new_sum
            
            # Try to add the best element
            best_add = None
            best_add_deficit = None
            for j in range(n):
                if j not in current_set:
                    new_s2 = new_sum + recips_frac[j]
                    if new_s2 < 1:
                        d2 = 1 - new_s2
                        if best_add_deficit is None or d2 < best_add_deficit:
                            best_add_deficit = d2
                            best_add = j
            
            if best_add is not None and best_add_deficit is not None and best_add_deficit < current_deficit:
                current_indices.append(best_add)
                current_set.add(best_add)
                current_sum_frac = new_sum + recips_frac[best_add]
                current_deficit = best_add_deficit
                improved = True
            else:
                # Restore
                current_indices.append(i)
                current_set.add(i)
                current_sum_frac = new_sum + recips_frac[i]
            removed_any = True
            break
        
        if improved:
            continue
        
        # Try swap: remove one, add one
        swap_improved = False
        for i in list(current_indices):
            if len(current_indices) <= 1:
                break
            current_set.remove(i)
            new_sum = current_sum_frac - recips_frac[i]
            
            for j in range(n):
                if j not in current_set:
                    new_s = new_sum + recips_frac[j]
                    if new_s < 1:
                        new_d = 1 - new_s
                        if new_d < current_deficit:
                            current_indices.remove(i)
                            current_indices.append(j)
                            current_set.add(j)
                            current_sum_frac = new_s
                            current_deficit = new_d
                            swap_improved = True
                            break
                    current_set.add(i)
            if swap_improved:
                break
            else:
                current_set.add(i)
        
        if swap_improved:
            continue
        
        # No improvement found; break or do a random-ish perturbation
        break
    
    # Update best if we found something better
    if current_deficit is not None and current_deficit < 1:
        if best_deficit is None or current_deficit < best_deficit:
            best_deficit = current_deficit
            best_indices = current_indices
            best_sum_float = float(current_sum_frac)

    # Phase 3: More systematic local search with multiple perturbation restarts
    if best_indices is not None and time.time() < deadline:
        base_indices = list(best_indices)
        base_sum = Fraction(0)
        for i in base_indices:
            base_sum += recips_frac[i]
        base_deficit = 1 - base_sum
        
        # Try perturbation: remove k random elements, then greedily add back
        import random
        rng = random.Random(42)
        
        for restart in range(50):
            if time.time() > deadline - 2:
                break
            
            perturbed = list(base_indices)
            if len(perturbed) > 2:
                # Remove 1-3 random elements
                k = min(rng.randint(1, 3), len(perturbed) - 1)
                to_remove = rng.sample(perturbed, k)
                pset = set(perturbed)
                psum = base_sum
                for i in to_remove:
                    psum -= recips_frac[i]
                    pset.remove(i)
                perturbed = list(pset)
            else:
                pset = set(perturbed)
                psum = base_sum
            
            # Greedily add elements to minimize deficit
            improved = True
            while improved and time.time() < deadline - 1:
                improved = False
                best_add_idx = None
                best_add_def = None
                for i in range(n):
                    if i not in pset:
                        new_s = psum + recips_frac[i]
                        if new_s < 1:
                            d = 1 - new_s
                            if best_add_def is None or d < best_add_def:
                                best_add_def = d
                                best_add_idx = i
                if best_add_idx is not None and best_add_def < (1 - psum if psum < 1 else float('inf')):
                    pset.add(best_add_idx)
                    psum += recips_frac[best_add_idx]
                    improved = True
            
            if psum < 1:
                d = 1 - psum
                if best_deficit is None or d < best_deficit:
                    best_deficit = d
                    best_indices = list(pset)
                    best_sum_float = float(psum)

    # Phase 4: Targeted search for small combinations that might be very close to 1
    # Focus on subsets where sum is very close to 1
    if time.time() < deadline - 5:
        # Check specific known-good patterns
        # The sum 1/2 + 1/3 + 1/7 + 1/42 = 1 exactly, so we need slightly less
        # Try: all of 2-48 except a carefully chosen set
        
        # Strategy: start with a known near-miss and adjust
        # 1/2 + 1/3 + 1/6 = 1, so exclude 6 and adjust
        # Let's try various exclusions from the full set
        
        full_sum = sum(recips_float)
        # full sum is about 3.8, so we need to exclude a lot
        
        # Better: work with the current best and try fine-tuning
        if best_indices is not None:
            bset = set(best_indices)
            bsum = Fraction(0)
            for i in best_indices:
                bsum += recips_frac[i]
            
            # Try each swap more carefully
            for trial in range(200):
                if time.time() > deadline - 1:
                    break
                if not bset or bsum >= 1:
                    break
                cur_d = 1 - bsum
                found = False
                for i in list(bset):
                    if len(bset) <= 1:
                        break
                    for j in range(n):
                        if j not in bset:
                            new_s = bsum - recips_frac[i] + recips_frac[j]
                            if new_s < 1:
                                new_d = 1 - new_s
                                if new_d < cur_d:
                                    bset.remove(i)
                                    bset.add(j)
                                    bsum = new_s
                                    cur_d = new_d
                                    found = True
                                    break
                    if found:
                        break
                if not found:
                    break
            
            if bsum < 1:
                d = 1 - bsum
                if best_deficit is None or d < best_deficit:
                    best_deficit = d
                    best_indices = list(bset)
                    best_sum_float = float(bsum)

    if best_indices is None:
        return [0]
    
    return best_indices
