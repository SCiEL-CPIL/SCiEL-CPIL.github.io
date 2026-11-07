import math

# EVOLVE-BLOCK-START

def construct():
    MAX_VAL = 10210
    MIN_VAL = 211
    TARGET = 210
    
    # 1. Sieve primes up to MAX_VAL
    is_prime = [True] * (MAX_VAL + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, MAX_VAL + 1, i):
                is_prime[j] = False
                
    primes = [i for i, val in enumerate(is_prime) if val]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # 2. Function to get square-free kernel bitmask
    def get_kernel_mask(n):
        mask = 0
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                if count % 2 == 1:
                    mask |= (1 << prime_to_idx[p])
        if temp > 1:
            mask |= (1 << prime_to_idx[temp])
        return mask

    # Precompute kernels for all candidates
    candidates = list(range(MIN_VAL, MAX_VAL + 1))
    kernels = [get_kernel_mask(s) for s in candidates]
    target_mask = get_kernel_mask(TARGET)
    
    # 3. Incremental Gaussian Elimination
    # basis: list of masks
    # recipes: list of bitmasks of indices into 'candidates'
    basis = []
    recipes = []
    
    # We iterate through candidates, adding them one by one
    for i, mask in enumerate(kernels):
        # Try to reduce mask using current basis
        current_mask = mask
        current_recipe = 1 << i  # This candidate itself
        
        # Reduce
        for b_idx, b_mask in enumerate(basis):
            if current_mask & b_mask:
                # Find the highest set bit in b_mask to use for pivot?
                # Standard basis: pivot is the highest set bit.
                # We need to ensure basis is in row-echelon form.
                pass
        
        # Let's implement a simpler reduction:
        # Keep basis sorted by highest bit (pivot).
        # For each new mask, find if its highest bit matches a pivot.
        
        # Better approach:
        # Maintain a dictionary pivot_bit -> (basis_mask, recipe)
        # Or just a list and scan. Since dimension is small, list scan is fine.
        
        # Let's stick to a list but ensure we reduce properly.
        # We will maintain basis in a way that allows fast reduction.
        # Actually, let's just use a dictionary `basis_dict` mapping pivot_bit to (mask, recipe)
        
    # Restart with dictionary approach for clarity and speed
    basis_dict = {} # pivot_bit -> (mask, recipe)
    
    for i, mask in enumerate(kernels):
        current_mask = mask
        current_recipe = 1 << i
        
        # Reduce current_mask
        while current_mask:
            # Find highest set bit
            pivot = current_mask.bit_length() - 1
            if pivot in basis_dict:
                b_mask, b_recipe = basis_dict[pivot]
                current_mask ^= b_mask
                current_recipe ^= b_recipe
            else:
                # Add to basis
                basis_dict[pivot] = (current_mask, current_recipe)
                break
        
        # Check if target is in span
        # We can try to reduce target_mask using basis_dict
        test_mask = target_mask
        test_recipe = 0
        
        # To check if target is in span, we reduce it.
        # If it becomes 0, it is in the span.
        # The recipe accumulated during reduction is the combination.
        # Note: We must be careful. The reduction process finds a linear combination
        # of basis vectors that equals the target, IF the target is in the span.
        
        # However, doing this full reduction every step is O(N^2).
        # Given N <= 10000, and basis size is small (< 100), this is fine.
        
        # Optimization: Only check if we added a new basis vector?
        # No, the target might become reachable even if we didn't add a new vector?
        # No, if the span didn't change, the target status didn't change.
        # So we only need to check if we added a new basis vector.
        
        # Wait, if we didn't add a new vector, the span is the same.
        # So we only check after adding.
        
        # But we need to know the recipe.
        # Let's just check every step for simplicity, it's fast enough.
        
        # Actually, let's optimize:
        # We only check if the target was NOT already found.
        
        # Let's just do the check.
        # We need a helper to reduce a mask and return recipe if it reduces to 0.
        
        def reduce_mask(m, rec):
            while m:
                pivot = m.bit_length() - 1
                if pivot in basis_dict:
                    b_mask, b_recipe = basis_dict[pivot]
                    m ^= b_mask
                    rec ^= b_recipe
                else:
                    return None # Not in span
            return rec # Reduced to 0, so in span, return recipe
        
        # Check target
        # We only need to check if we haven't found a solution yet.
        # Let's store the best solution (min max value).
        # Since we iterate M upwards, the first time we find a solution is the optimal M.
        
        # However, the recipe depends on the current basis.
        # The basis changes as we add more numbers.
        # If the target is in the span, we found a solution.
        # The max value in this solution is the max index i used in the recipe.
        # Since we are iterating i, the current i is the max possible index in the recipe.
        # So the first time we find the target in the span, the max value is the current candidate.
        
        # Wait, is it possible that the target is in the span, but the recipe uses
        # indices that are not all <= i? No, we only added indices <= i.
        
        # So, if reduce_mask(target_mask, 0) is not None, we have a solution.
        
        # But we must ensure we don't re-check if we already found one.
        # Since we return immediately, we are fine.
        
        # One catch: The reduction `reduce_mask` uses the CURRENT basis_dict.
        # The basis_dict contains vectors formed from candidates 0..i.
        # So the recipe will only use indices 0..i.
        
        # Let's run the check.
        
        # Optimization: Only run check if we added a new basis vector?
        # If we didn't add a new vector, the span is unchanged.
        # If the target was not in the span before, it's not in the span now.
        # So we only check if we added a new vector.
        
        # But how do we know if we added a new vector?
        # In the loop above, if we broke out of the `while` loop without returning None
        # from the "not in basis" case, we added it.
        
        # Let's restructure to track this.
        
    # Let's rewrite the loop properly.
    
    basis_dict = {}
    
    for i, mask in enumerate(kernels):
        current_mask = mask
        current_recipe = 1 << i
        added = False
        
        while current_mask:
            pivot = current_mask.bit_length() - 1
            if pivot in basis_dict:
                b_mask, b_recipe = basis_dict[pivot]
                current_mask ^= b_mask
                current_recipe ^= b_recipe
            else:
                basis_dict[pivot] = (current_mask, current_recipe)
                added = True
                break
        
        if added:
            # Check if target is in span
            test_mask = target_mask
            test_recipe = 0
            in_span = True
            while test_mask:
                pivot = test_mask.bit_length() - 1
                if pivot in basis_dict:
                    b_mask, b_recipe = basis_dict[pivot]
                    test_mask ^= b_mask
                    test_recipe ^= b_recipe
                else:
                    in_span = False
                    break
            
            if in_span:
                # We found a combination.
                # test_recipe is a bitmask of indices.
                # Extract indices.
                selected_indices = []
                r = test_recipe
                while r:
                    idx = (r & -r).bit_length() - 1
                    selected_indices.append(idx)
                    r &= r - 1
                
                # Get the actual numbers
                selected_numbers = [candidates[idx] for idx in selected_indices]
                
                # Verify
                # Product of selected numbers * 210 must be a square.
                # We can verify this quickly.
                prod_mask = target_mask
                for idx in selected_indices:
                    prod_mask ^= kernels[idx]
                if prod_mask != 0:
                    # This shouldn't happen if logic is correct
                    continue
                
                # Return the list
                return selected_numbers

    # If no solution found (should not happen)
    return [211] # Fallback

# EVOLVE-BLOCK-END
