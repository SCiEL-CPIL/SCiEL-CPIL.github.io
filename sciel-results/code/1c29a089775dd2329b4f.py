import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    limit = 145.0  # Safety margin for 150s limit
    
    total_words = q ** n
    powers = [q ** i for i in range(n)]
    
    def get_digits(idx):
        digits = [0] * n
        for i in range(n):
            digits[i] = idx % q
            idx //= q
        return digits
    
    # Precompute neighbor masks as Python integers (bitmasks)
    neighbor_masks = [0] * total_words
    all_mask = (1 << total_words) - 1
    
    for idx in range(total_words):
        mask = 1 << idx
        for i in range(n):
            original = (idx // powers[i]) % q
            base = idx - original * powers[i]
            for new_val in range(q):
                if new_val != original:
                    neighbor_idx = base + new_val * powers[i]
                    mask |= (1 << neighbor_idx)
        neighbor_masks[idx] = mask
    
    def is_complete(solution):
        """Check if solution covers all words."""
        covered = 0
        for w in solution:
            covered |= neighbor_masks[w]
        return covered == all_mask
    
    def remove_redundant(solution):
        """Remove codewords that don't cover any unique word."""
        if not solution:
            return solution
        sol = list(solution)
        changed = True
        while changed and (time.time() - start_time < limit):
            changed = False
            for i in range(len(sol) - 1, -1, -1):
                if time.time() - start_time > limit:
                    break
                w = sol[i]
                # Check if w covers any word not covered by others
                others_covered = 0
                for j in range(len(sol)):
                    if j != i:
                        others_covered |= neighbor_masks[sol[j]]
                if (neighbor_masks[w] & ~others_covered) == 0:
                    sol.pop(i)
                    changed = True
                    break
        return sol
    
    def greedy_cover(uncovered_mask, solution):
        """Greedy set cover starting from given solution and uncovered mask."""
        sol = list(solution)
        sol_set = set(sol)
        unc = uncovered_mask
        while unc:
            if time.time() - start_time > limit:
                break
            best_word = -1
            max_covered = 0
            for w in range(total_words):
                if w in sol_set:
                    continue
                count = (neighbor_masks[w] & unc).bit_count()
                if count > max_covered:
                    max_covered = count
                    best_word = w
            if best_word == -1:
                break
            sol.append(best_word)
            sol_set.add(best_word)
            unc &= ~neighbor_masks[best_word]
        return sol
    
    def local_search_swaps(solution, max_iters=200):
        """Try to improve solution by swapping codewords."""
        if not solution:
            return solution
        sol = list(solution)
        sol_set = set(sol)
        best_sol = list(sol)
        best_len = len(sol)
        
        for iteration in range(max_iters):
            if time.time() - start_time > limit:
                break
            
            # Compute coverage counts
            cov_counts = [0] * total_words
            for w in sol:
                m = neighbor_masks[w]
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    cov_counts[idx] += 1
                    m ^= lsb
            
            # Find words covered by exactly one codeword (critical words)
            critical = [i for i in range(total_words) if cov_counts[i] == 1]
            
            # Try random swaps
            improved = False
            if sol:
                # Pick a random codeword to potentially remove
                remove_idx = random.randint(0, len(sol) - 1)
                w_remove = sol[remove_idx]
                
                # Words only covered by w_remove
                only_covered_by_remove = [i for i in range(total_words) 
                                          if cov_counts[i] == 1 and (neighbor_masks[w_remove] >> i) & 1]
                
                if not only_covered_by_remove:
                    # w_remove is redundant, just remove it
                    sol.pop(remove_idx)
                    sol_set.discard(w_remove)
                    improved = True
                    continue
                
                # Find candidates that cover at least one of the critical words
                candidates = []
                for c in range(total_words):
                    if c in sol_set:
                        continue
                    # Check if c covers any critical word
                    for cw in only_covered_by_remove:
                        if (neighbor_masks[c] >> cw) & 1:
                            candidates.append(c)
                            break
                
                if not candidates:
                    continue
                
                # Try a random candidate
                c_add = random.choice(candidates)
                
                # Check if the swap is valid: after removing w_remove and adding c_add,
                # all words must still be covered
                new_sol = sol.copy()
                new_sol[remove_idx] = c_add
                
                # Quick check: is the new solution complete?
                covered = 0
                for w in new_sol:
                    covered |= neighbor_masks[w]
                if covered == all_mask:
                    sol = new_sol
                    sol_set = set(sol)
                    improved = True
                    if len(sol) < best_len:
                        best_sol = list(sol)
                        best_len = len(sol)
            
            if not improved and iteration > 50:
                # No improvement in a while, try a different strategy
                # Add a random codeword and then remove redundant ones
                if len(sol) < total_words:
                    random_word = random.randint(0, total_words - 1)
                    if random_word not in sol_set:
                        sol.append(random_word)
                        sol_set.add(random_word)
                        sol = remove_redundant(sol)
                        sol_set = set(sol)
        
        return best_sol if len(best_sol) <= len(sol) else sol
    
    def algebraic_initial_solution():
        """
        Construct an initial solution using the product structure of ternary space.
        For n=8, split into two halves of length 4.
        Use a covering code of length 4, radius 1 (or similar) for each half.
        """
        if n != 8 or q != 3:
            return None
        
        # Split into two halves of length 4
        half = n // 2
        half_words = q ** half  # 81 for q=3, half=4
        
        # Generate a covering code for length 4, radius 1 over ternary
        # We'll use a simple greedy approach for the half
        def greedy_half_cover(half_n, half_q, half_radius):
            hw = half_q ** half_n
            h_powers = [half_q ** i for i in range(half_n)]
            h_neighbor_masks = [0] * hw
            h_all_mask = (1 << hw) - 1
            
            for idx in range(hw):
                mask = 1 << idx
                for i in range(half_n):
                    original = (idx // h_powers[i]) % half_q
                    base = idx - original * h_powers[i]
                    for new_val in range(half_q):
                        if new_val != original:
                            neighbor_idx = base + new_val * h_powers[i]
                            mask |= (1 << neighbor_idx)
                h_neighbor_masks[idx] = mask
            
            # Greedy cover
            uncovered = h_all_mask
            code = []
            while uncovered:
                if time.time() - start_time > limit:
                    return None
                best_word = -1
                max_covered = 0
                for w in range(hw):
                    count = (h_neighbor_masks[w] & uncovered).bit_count()
                    if count > max_covered:
                        max_covered = count
                        best_word = w
                if best_word == -1:
                    break
                code.append(best_word)
                uncovered &= ~h_neighbor_masks[best_word]
            
            return code
        
        # Get covering code for each half
        code1 = greedy_half_cover(half, q, radius)
        code2 = greedy_half_cover(half, q, radius)
        
        if code1 is None or code2 is None:
            return None
        
        # Combine: for each pair (c1, c2) in code1 x code2, create a full word
        # c1 is the first half, c2 is the second half
        solution = []
        for c1 in code1:
            for c2 in code2:
                # Convert c1 and c2 to their digit representations
                c1_digits = [0] * half
                c2_digits = [0] * half
                temp = c1
                for i in range(half):
                    c1_digits[i] = temp % q
                    temp //= q
                temp = c2
                for i in range(half):
                    c2_digits[i] = temp % q
                    temp //= q
                
                # Combine into full word index
                full_idx = 0
                for i in range(half):
                    full_idx += c1_digits[i] * powers[i]
                for i in range(half):
                    full_idx += c2_digits[i] * powers[half + i]
                
                solution.append(full_idx)
        
        return solution
    
    # Phase 1: Try algebraic initial solution first, fall back to greedy
    solution = algebraic_initial_solution()
    
    if solution is None:
        # Fallback to greedy
        initial_solution = []
        uncovered = all_mask
        while uncovered:
            if time.time() - start_time > limit:
                break
            best_word = -1
            max_covered = 0
            for w in range(total_words):
                count = (neighbor_masks[w] & uncovered).bit_count()
                if count > max_covered:
                    max_covered = count
                    best_word = w
            if best_word == -1:
                break
            initial_solution.append(best_word)
            uncovered &= ~neighbor_masks[best_word]
        solution = initial_solution
    
    # Phase 2: Remove redundant codewords
    solution = remove_redundant(solution)
    
    # Phase 3: Local search with swaps
    solution = local_search_swaps(solution, max_iters=500)
    
    # Phase 4: Try a few more rounds of removal + swap
    for _ in range(3):
        if time.time() - start_time > limit:
            break
        solution = remove_redundant(solution)
        solution = local_search_swaps(solution, max_iters=200)
    
    # Final check
    if not is_complete(solution):
        # Fallback: complete the solution with greedy
        covered = 0
        for w in solution:
            covered |= neighbor_masks[w]
        uncovered = all_mask & ~covered
        solution = greedy_cover(uncovered, solution)
    
    return [get_digits(idx) for idx in solution]
