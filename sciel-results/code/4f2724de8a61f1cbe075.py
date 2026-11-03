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
            for i in range(total_words):
                c = (balls[i] & remaining).bit_count()
                if c > max_cover:
                    max_cover = c
                    best_idx = i
                elif c == max_cover and c > 0:
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
        
        changed = True
        while changed:
            changed = False
            for i in range(len(selected) - 1, -1, -1):
                union_rest = 0
                for j in range(len(selected)):
                    if j != i:
                        union_rest |= balls[selected[j]]
                
                if (union_rest & all_ones_mask) == all_ones_mask:
                    selected.pop(i)
                    changed = True
                    break
        return selected

    def local_search_deficit(selected):
        """
        Perform local search using deficit analysis.
        1. Try to remove redundant nodes.
        2. Try swaps: Replace c with w if w covers the deficit created by removing c.
        """
        current = selected[:]
        improved = True
        while improved:
            improved = False
            
            # 1. Try removing any single node that is redundant
            for i in range(len(current) - 1, -1, -1):
                c_idx = current[i]
                # Check if removing c_idx leaves any word uncovered
                # Uncovered = All & ~(Union of others)
                # Faster: Calculate Union of others
                union_rest = 0
                for j in range(len(current)):
                    if j != i:
                        union_rest |= balls[current[j]]
                
                if (union_rest & all_ones_mask) == all_ones_mask:
                    current.pop(i)
                    improved = True
                    break # Restart loop as list changed
            
            if not improved:
                # 2. Try swaps
                # For each c in current, calculate Deficit
                # Deficit = All & ~(Union of others)
                # We need w such that Ball(w) covers Deficit
                for i in range(len(current)):
                    c_idx = current[i]
                    union_rest = 0
                    for j in range(len(current)):
                        if j != i:
                            union_rest |= balls[current[j]]
                    
                    deficit = all_ones_mask & ~union_rest
                    
                    if deficit == 0:
                        # Should have been caught in step 1, but handle just in case
                        current.pop(i)
                        improved = True
                        break
                    
                    # Find w not in current such that balls[w] covers deficit
                    # Optimization: Only check w that are not in current
                    # And maybe prioritize w that are "close" or random?
                    # Just iterate all w not in current
                    found_swap = False
                    # To speed up, we can shuffle the candidates or just iterate
                    # We want ANY valid swap.
                    for w_idx in range(total_words):
                        if w_idx in current:
                            continue
                        if (balls[w_idx] & deficit) == deficit:
                            # Valid swap
                            current[i] = w_idx
                            improved = True
                            found_swap = True
                            break
                    
                    if found_swap:
                        break # Restart loop with new configuration
        
        return current

    best_code = None
    best_size = total_words + 1
    start_time = time.time()
    time_limit = 115.0
    
    # Initial Greedy
    initial_selected = greedy_cover(all_ones_mask)
    initial_selected = remove_redundants(initial_selected)
    
    if len(initial_selected) < best_size:
        best_size = len(initial_selected)
        best_code = [all_words[i] for i in initial_selected]
    
    current_selected = initial_selected[:]
    
    while time.time() - start_time < time_limit:
        # 1. Apply Local Search Deficit to current
        current_selected = local_search_deficit(current_selected)
        
        if len(current_selected) < best_size:
            best_size = len(current_selected)
            best_code = [all_words[i] for i in current_selected]
        
        # 2. Perturb if stuck or to explore
        # Check if we can still improve
        # Simple strategy: Always perturb a bit to escape local minima
        perturb_amount = random.randint(1, 3)
        if len(current_selected) > perturb_amount:
            indices_to_remove = random.sample(range(len(current_selected)), perturb_amount)
            indices_to_remove_set = set(indices_to_remove)
            
            # Calculate uncovered mask after removal
            covered_mask = 0
            for i in range(len(current_selected)):
                if i not in indices_to_remove_set:
                    covered_mask |= balls[current_selected[i]]
            
            uncovered_mask = all_ones_mask & ~covered_mask
            
            # 3. Repair: Greedily cover the uncovered parts
            repair_selected = greedy_cover(uncovered_mask)
            
            # Combine
            new_selected = []
            for i in range(len(current_selected)):
                if i not in indices_to_remove_set:
                    new_selected.append(current_selected[i])
            new_selected.extend(repair_selected)
            
            # 4. Clean and Local Search
            new_selected = remove_redundants(new_selected)
            new_selected = local_search_deficit(new_selected)
            
            # 5. Accept if better or with probability
            if len(new_selected) < best_size:
                best_size = len(new_selected)
                best_code = [all_words[i] for i in new_selected]
            
            # Accept into current if better or with SA probability
            if len(new_selected) <= len(current_selected) or (random.random() < 0.1 and len(new_selected) < len(current_selected) + 2):
                current_selected = new_selected
            # Else keep current_selected

        if time.time() - start_time > time_limit:
            break

    return best_code
