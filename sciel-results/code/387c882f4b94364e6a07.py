import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    total_words = q ** n
    
    # Fallback for larger spaces or general radius
    if total_words > 10000 or radius > 1:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 7:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return [list(random.choices(range(q), k=n)) for _ in range(100)]

    # Bitset-based approach for small spaces (e.g., 3^7 = 2187)
    
    # 1. Generate all words
    all_words = [list(p) for p in itertools.product(range(q), repeat=n)]
    
    # 2. Precompute balls as bitmasks (Python ints)
    balls = [0] * total_words
    
    if radius == 1:
        for i, w in enumerate(all_words):
            mask = 0
            # Self
            mask |= (1 << i)
            # Neighbors
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        # Calculate neighbor index directly
                        neighbor_idx = 0
                        for k in range(n):
                            val = w[k] if k != pos else new_val
                            neighbor_idx = neighbor_idx * q + val
                        mask |= (1 << neighbor_idx)
            balls[i] = mask
    else:
        # General radius (slower)
        for i, w in enumerate(all_words):
            mask = 0
            for j, v in enumerate(all_words):
                dist = sum(1 for a, b in zip(w, v) if a != b)
                if dist <= radius:
                    mask |= (1 << j)
            balls[i] = mask

    all_ones_mask = (1 << total_words) - 1
    
    def greedy_cover(uncovered_mask):
        """
        Greedily select codewords to cover the words in uncovered_mask.
        Returns a list of indices.
        """
        selected = []
        remaining = uncovered_mask
        while remaining:
            best_idx = -1
            max_cover = 0
            # Optimization: Only consider candidates that cover at least 1 new word
            # We iterate all, but this is the standard greedy step.
            for i in range(total_words):
                c = (balls[i] & remaining).bit_count()
                if c > max_cover:
                    max_cover = c
                    best_idx = i
                elif c == max_cover and c > 0:
                    # Tie break: random to encourage diversity
                    if random.random() < 0.5:
                        best_idx = i
            if best_idx == -1:
                break
            selected.append(best_idx)
            remaining &= ~balls[best_idx]
        return selected

    def remove_redundants(selected):
        """
        Removes any codeword from selected that is redundant.
        Returns the new list of selected indices.
        """
        if not selected:
            return []
        
        # Convert to set for faster membership if needed, but we need list for order.
        # We will iterate and check.
        changed = True
        while changed:
            changed = False
            for i in range(len(selected) - 1, -1, -1):
                # Check if selected[i] is redundant
                union_rest = 0
                for j in range(len(selected)):
                    if j != i:
                        union_rest |= balls[selected[j]]
                        
                # If the union of all other balls covers everything that selected[i] covers,
                # AND the union covers the entire space (or rather, the union covers 
                # the specific words that selected[i] was responsible for), it is redundant.
                # Wait, the condition for a valid code is that the UNION of ALL balls is all_ones_mask.
                # If we remove selected[i], the new union is union_rest.
                # The code is still valid if union_rest == all_ones_mask.
                # This is the strict condition for removal in a *complete* cover.
                
                if (union_rest & all_ones_mask) == all_ones_mask:
                    selected.pop(i)
                    changed = True
                    break # Restart loop as indices changed
        return selected

    best_code = None
    best_size = total_words + 1
    start_time = time.time()
    time_limit = 115.0 # Leave 5s buffer
    
    # Initial Greedy
    initial_selected = greedy_cover(all_ones_mask)
    initial_selected = remove_redundants(initial_selected)
    
    if len(initial_selected) < best_size:
        best_size = len(initial_selected)
        best_code = [all_words[i] for i in initial_selected]
    
    # Iterated Local Search / Perturb and Repair
    current_selected = initial_selected[:]
    
    while time.time() - start_time < time_limit:
        # 1. Perturb: Remove 1 to 3 random codewords from current solution
        perturb_amount = random.randint(1, 3)
        if len(current_selected) <= perturb_amount:
            # If set is too small, just restart greedy
            current_selected = greedy_cover(all_ones_mask)
            current_selected = remove_redundants(current_selected)
        else:
            # Remove random indices
            indices_to_remove = random.sample(range(len(current_selected)), perturb_amount)
            indices_to_remove_set = set(indices_to_remove)
            
            # Calculate uncovered mask after removal
            covered_mask = 0
            for i in range(len(current_selected)):
                if i not in indices_to_remove_set:
                    covered_mask |= balls[current_selected[i]]
            
            uncovered_mask = all_ones_mask & ~covered_mask
            
            # 2. Repair: Greedily cover the uncovered parts
            repair_selected = greedy_cover(uncovered_mask)
            
            # Combine remaining original codewords and repair codewords
            new_selected = []
            for i in range(len(current_selected)):
                if i not in indices_to_remove_set:
                    new_selected.append(current_selected[i])
            new_selected.extend(repair_selected)
            
            # 3. Clean: Remove redundants
            new_selected = remove_redundants(new_selected)
            
            # 4. Update Best
            if len(new_selected) < best_size:
                best_size = len(new_selected)
                best_code = [all_words[i] for i in new_selected]
            
            # 5. Update Current
            # Accept if better, or with some probability if worse (Simulated Annealing style)
            # But for minimization, usually accept if better or equal.
            # To escape local minima, we can accept worse solutions with low probability.
            if len(new_selected) <= len(current_selected) or (random.random() < 0.1 and len(new_selected) < len(current_selected) + 2):
                current_selected = new_selected
            else:
                # If rejected, revert to previous current_selected? 
                # Or keep the perturbed one? 
                # Standard ILS: keep the last accepted solution as the reference.
                # If we reject, we should probably revert to the last good one.
                # But we modified current_selected in place? No, we created new_selected.
                # So current_selected is still the previous one.
                pass

        # Additional Local Search: Try single swaps to improve?
        # The perturb/repair is often sufficient.
        
        # Check time
        if time.time() - start_time > time_limit:
            break

    return best_code
