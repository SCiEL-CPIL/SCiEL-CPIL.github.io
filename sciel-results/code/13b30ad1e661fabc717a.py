"""Model-and-optimize approach for integer sum-product extremal sets."""
import time
import random
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    n = 20
    max_val = 10000
    best_set = None
    best_score = float('inf')

    def fast_eval(A):
        """Compute max(|A+A|, |A*A|) for a set A."""
        a = sorted(A)
        sums = set()
        prods = set()
        for i in range(len(a)):
            for j in range(i, len(a)):
                sums.add(a[i] + a[j])
                prods.add(a[i] * a[j])
        return max(len(sums), len(prods))

    def get_counts(A):
        """Calculate reference counts for sums and products."""
        sum_counts = defaultdict(int)
        prod_counts = defaultdict(int)
        for i in range(n):
            for j in range(i, n):
                s = A[i] + A[j]
                p = A[i] * A[j]
                sum_counts[s] += 1
                prod_counts[p] += 1
        return sum_counts, prod_counts

    def get_active_sizes(sum_counts, prod_counts):
        """Count how many sums/prods have count > 0."""
        sum_size = sum(1 for v in sum_counts.values() if v > 0)
        prod_size = sum(1 for v in prod_counts.values() if v > 0)
        return sum_size, prod_size

    def structured_search():
        """
        Generate diverse initial candidates using AP+GP unions.
        This provides better starting points than random initialization.
        """
        nonlocal best_set, best_score
        candidates = []

        # 1. Pure Geometric Progressions
        for ratio in range(2, 15):
            for start in range(1, 100):
                A = []
                v = start
                for _ in range(n):
                    if v > max_val:
                        break
                    A.append(v)
                    v *= ratio
                if len(A) == n and max(A) <= max_val:
                    candidates.append(A)

        # 2. Pure Arithmetic Progressions
        for start in range(1, 50):
            for d in range(1, 20):
                A = [start + i * d for i in range(n)]
                if max(A) <= max_val:
                    candidates.append(A)

        # 3. AP + GP Unions
        # Take k elements from AP and n-k from GP
        for k_ap in range(5, 16):
            k_gp = n - k_ap
            if k_gp < 2:
                continue
            
            # Sample some AP parameters
            for a_start in range(1, 30):
                for d in range(1, 15):
                    ap_terms = [a_start + i * d for i in range(k_ap)]
                    if ap_terms[-1] > max_val:
                        continue
                    
                    # Sample some GP parameters
                    for g_start in range(1, 20):
                        for ratio in range(2, 8):
                            gp_terms = []
                            v = g_start
                            for _ in range(k_gp):
                                if v > max_val:
                                    break
                                gp_terms.append(v)
                                v *= ratio
                            if len(gp_terms) < k_gp:
                                continue
                            
                            combined = sorted(set(ap_terms + gp_terms))
                            if len(combined) == n and max(combined) <= max_val:
                                candidates.append(combined)
            
            if time.time() > deadline - 20:
                break

        # 4. Smooth Numbers (2,3,5-smooth)
        smooths = [1]
        for p in [2, 3, 5]:
            new_smooths = []
            for s in smooths:
                v = s
                while v * p <= max_val:
                    v *= p
                    new_smooths.append(v)
            smooths.extend(new_smooths)
        smooths = sorted(set(smooths))
        if len(smooths) >= n:
            # Take first n, last n, and some middle ones
            candidates.append(smooths[:n])
            candidates.append(smooths[-n:])
            if len(smooths) > 40:
                candidates.append(smooths[10:30])

        # Evaluate candidates
        valid_starts = []
        for A in candidates:
            if len(set(A)) != n:
                continue
            if max(A) > max_val or min(A) < 1:
                continue
            A_sorted = sorted(A)
            sc = fast_eval(A_sorted)
            if sc < best_score:
                best_score = sc
                best_set = A_sorted[:]
            if sc < 250: # Keep a pool of good starts
                valid_starts.append((A_sorted, sc))
        
        return valid_starts

    # --- Phase 1: Structured Search for Good Starts ---
    starts = structured_search()
    
    if not starts and best_set is None:
        # Fallback to a simple set
        best_set = list(range(1, 21))
        best_score = fast_eval(best_set)
        starts = [(best_set, best_score)]

    # --- Phase 2: Local Search with Reference Counting ---
    # We will run local search from the best few starts
    # Sort starts by score
    starts.sort(key=lambda x: x[1])
    top_starts = starts[:5]
    
    if best_set not in [s[0] for s in top_starts]:
        top_starts.append((best_set, best_score))

    random.seed(42)

    for start_A, start_score in top_starts:
        if time.time() > deadline - 10:
            break
            
        current = start_A[:]
        current_score = start_score
        
        # Initialize counts
        sum_counts, prod_counts = get_counts(current)
        sum_size, prod_size = get_active_sizes(sum_counts, prod_counts)
        current_max = max(sum_size, prod_size)
        
        if current_max < best_score:
            best_score = current_max
            best_set = current[:]

        no_improve = 0
        max_no_improve = 2000
        
        while no_improve < max_no_improve and time.time() < deadline - 5:
            improved = False
            i = random.randint(0, n - 1)
            old_val = current[i]
            
            # Generate new value
            # Strategy: random, or near existing, or multiple/divisor
            r = random.random()
            if r < 0.4:
                new_val = random.randint(1, max_val)
            elif r < 0.6:
                # Near an existing element
                base_idx = random.randint(0, n - 1)
                delta = random.randint(-5, 5)
                new_val = current[base_idx] + delta
            elif r < 0.8:
                # Multiple of existing
                base_idx = random.randint(0, n - 1)
                factor = random.choice([2, 3, 4, 5])
                new_val = current[base_idx] * factor
            else:
                # Divisor of existing
                base_idx = random.randint(0, n - 1)
                val = current[base_idx]
                divs = [d for d in range(1, int(val**0.5)+1) if val % d == 0]
                if divs:
                    new_val = random.choice(divs)
                    if new_val * new_val == val:
                        new_val = random.choice(divs) # Avoid self if possible, though logic handles it
                else:
                    new_val = random.randint(1, max_val)
            
            if new_val < 1 or new_val > max_val:
                no_improve += 1
                continue
            
            # Check if new_val is already in current (excluding i)
            if new_val in current and new_val != old_val:
                no_improve += 1
                continue
            
            # Apply move: Remove old_val contributions, Add new_val contributions
            # Remove old_val
            for j in range(n):
                if j == i:
                    continue
                s = old_val + current[j]
                p = old_val * current[j]
                sum_counts[s] -= 1
                prod_counts[p] -= 1
            s_ii = old_val + old_val
            p_ii = old_val * old_val
            sum_counts[s_ii] -= 1
            prod_counts[p_ii] -= 1
            
            # Add new_val
            current[i] = new_val
            for j in range(n):
                if j == i:
                    continue
                s = new_val + current[j]
                p = new_val * current[j]
                sum_counts[s] += 1
                prod_counts[p] += 1
            s_ii = new_val + new_val
            p_ii = new_val * new_val
            sum_counts[s_ii] += 1
            prod_counts[p_ii] += 1
            
            # Calculate new sizes
            # To avoid O(N) iteration over all keys every time, we can track active count incrementally
            # But checking if a key became 0 or 1 is O(1) if we check the specific keys touched.
            
            # We need to know if any sum/prod count dropped to 0 (size decreases) or rose from 0 (size increases)
            # However, we only touched specific keys.
            
            # Let's just recount active sizes for the touched keys? No, we need global size.
            # Efficient way:
            # Keep track of sum_size and prod_size.
            # When decrementing a count:
            #   if old_count == 1: size -= 1
            # When incrementing a count:
            #   if old_count == 0: size += 1
            
            # We didn't do this in the loop above. Let's refactor the loop to track sizes.
            
            # --- Refactored Move Logic ---
            # Undo the simple add/subtract above and do it properly with size tracking
            
            # Restore counts to state before move
            for j in range(n):
                if j == i:
                    continue
                s = new_val + current[j]
                p = new_val * current[j]
                sum_counts[s] -= 1
                prod_counts[p] -= 1
            sum_counts[s_ii] -= 1
            prod_counts[p_ii] -= 1
            
            current[i] = old_val
            
            for j in range(n):
                if j == i:
                    continue
                s = old_val + current[j]
                p = old_val * current[j]
                sum_counts[s] += 1
                prod_counts[p] += 1
            sum_counts[s_ii] += 1
            prod_counts[p_ii] += 1
            
            # Now apply move with size tracking
            # Remove old_val
            for j in range(n):
                if j == i:
                    continue
                s = old_val + current[j]
                p = old_val * current[j]
                if sum_counts[s] == 1: sum_size -= 1
                sum_counts[s] -= 1
                if prod_counts[p] == 1: prod_size -= 1
                prod_counts[p] -= 1
            
            s_ii = old_val + old_val
            p_ii = old_val * old_val
            if sum_counts[s_ii] == 1: sum_size -= 1
            sum_counts[s_ii] -= 1
            if prod_counts[p_ii] == 1: prod_size -= 1
            prod_counts[p_ii] -= 1
            
            # Add new_val
            current[i] = new_val
            for j in range(n):
                if j == i:
                    continue
                s = new_val + current[j]
                p = new_val * current[j]
                if sum_counts[s] == 0: sum_size += 1
                sum_counts[s] += 1
                if prod_counts[p] == 0: prod_size += 1
                prod_counts[p] += 1
            
            s_ii = new_val + new_val
            p_ii = new_val * new_val
            if sum_counts[s_ii] == 0: sum_size += 1
            sum_counts[s_ii] += 1
            if prod_counts[p_ii] == 0: prod_size += 1
            prod_counts[p_ii] += 1
            
            new_max = max(sum_size, prod_size)
            
            if new_max < current_max:
                current_max = new_max
                no_improve = 0
                improved = True
                current.sort()
                if current_max < best_score:
                    best_score = current_max
                    best_set = current[:]
            else:
                # Revert
                # Remove new_val
                for j in range(n):
                    if j == i:
                        continue
                    s = new_val + current[j]
                    p = new_val * current[j]
                    if sum_counts[s] == 1: sum_size -= 1
                    sum_counts[s] -= 1
                    if prod_counts[p] == 1: prod_size -= 1
                    prod_counts[p] -= 1
                
                s_ii = new_val + new_val
                p_ii = new_val * new_val
                if sum_counts[s_ii] == 1: sum_size -= 1
                sum_counts[s_ii] -= 1
                if prod_counts[p_ii] == 1: prod_size -= 1
                prod_counts[p_ii] -= 1
                
                # Restore old_val
                current[i] = old_val
                for j in range(n):
                    if j == i:
                        continue
                    s = old_val + current[j]
                    p = old_val * current[j]
                    if sum_counts[s] == 0: sum_size += 1
                    sum_counts[s] += 1
                    if prod_counts[p] == 0: prod_size += 1
                    prod_counts[p] += 1
                
                s_ii = old_val + old_val
                p_ii = old_val * old_val
                if sum_counts[s_ii] == 0: sum_size += 1
                sum_counts[s_ii] += 1
                if prod_counts[p_ii] == 0: prod_size += 1
                prod_counts[p_ii] += 1
                
                no_improve += 1
                
                # If stuck for a while, try a bigger jump or reset to best
                if no_improve > 500:
                    # Restart from best_set
                    current = best_set[:]
                    current_max = best_score
                    sum_counts, prod_counts = get_counts(current)
                    sum_size, prod_size = get_active_sizes(sum_counts, prod_counts)
                    no_improve = 0
                    break # Break while loop to restart outer for loop for this start? 
                    # Actually, let's just break and let the outer loop handle it if we have multiple starts.
                    # Or just restart the local search from the current best.

    # Final check
    if best_set is None:
        best_set = list(range(1, 21))
        
    return best_set
# EVOLVE-BLOCK-END
