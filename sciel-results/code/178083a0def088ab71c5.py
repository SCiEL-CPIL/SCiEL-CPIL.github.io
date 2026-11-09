"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s kill
    
    # We are looking for a polynomial P(x) with exactly 12 non-zero terms.
    # The current best score is 1.1500, which corresponds to 13 non-zero terms in the square (13/15? No, seed is 1.0, current 1.15 implies improvement).
    # Score = Seed / Value. If Seed is 15 (max possible non-zero in square of 12 terms is 23, but maybe seed is normalized differently?).
    # Actually, the prompt says "Minimize nonzero terms in the EXACT integer square".
    # Let's assume the metric is (Base / Count). If Base=15, Count=13 -> 1.15.
    # We want to find Count < 13. Ideally 12, 11, or lower.
    
    # Strategy: Search for the support of the square.
    # Let P(x) = sum a_i x^i.
    # We fix the support of P to be {0, 1, ..., 11} for simplicity in the first pass, 
    # but we can also permute/shift. However, shifting doesn't change the number of non-zeros.
    # Permuting indices changes the structure of cancellations.
    
    # The most powerful approach:
    # 1. Guess a target number of non-zero terms in the square, K.
    # 2. Iterate over all possible subsets S of {0..22} of size K.
    # 3. For each S, check if there exists a P with 12 non-zero terms such that
    #    P(x)^2 has support exactly S.
    # 4. This is a system of quadratic equations. We can solve it via backtracking.
    
    # Optimization:
    # Instead of iterating all subsets (which is huge), we can use the structure.
    # The square of a polynomial with 12 terms has at most 23 terms.
    # To get cancellations, we need $a_i a_j + a_k a_l = 0$ for some pairs.
    # This implies $a_i a_j = - a_k a_l$.
    
    # Let's try a targeted search for specific "sparse square" structures.
    # Known small examples:
    # (1+x+x^2+x^3+x^4+x^5+x^6+x^7+x^8+x^9+x^10+x^11)^2 has 23 terms.
    # 
    # Let's use a backtracking solver on the coefficients.
    # We fix the support of P to be 0..11 first.
    # We iterate over possible supports of the square, starting from small sizes.
    
    def count_nonzero_square(coeffs):
        square_coeffs = [0] * 23
        for i in range(12):
            ai = coeffs[i]
            if ai == 0:
                continue
            for j in range(i, 12):
                aj = coeffs[j]
                if aj == 0:
                    continue
                k = i + j
                prod = ai * aj
                if i == j:
                    square_coeffs[k] += prod
                else:
                    square_coeffs[k] += 2 * prod
        return sum(1 for c in square_coeffs if c != 0)

    best_count = 23
    best_poly = None
    
    # We will try to find a polynomial with a square having <= 12 non-zero terms.
    # If we find 12, score might be 1.25 (if base 15).
    # If we find 11, score 1.36.
    
    # Approach:
    # 1. Generate random supports for the square (subsets of 0..22).
    # 2. For each support, solve for coefficients.
    #    To make solving easier, we can fix some coefficients to 1 or -1 and solve for others.
    #    Or use a heuristic search that respects the target support.
    
    # Alternative faster approach:
    # Use the fact that if P(x) has a sparse square, it often relates to cyclotomic polynomials or similar.
    # But let's stick to a robust search.
    
    # Heuristic:
    # Start with a random polynomial.
    # Compute its square.
    # Identify "collisions" (terms that cancel or are close to cancelling).
    # Adjust coefficients to force exact cancellation for a chosen set of degrees.
    
    # Let's implement a "Target Support" search.
    # We pick a target set T of degrees to be NON-ZERO in the square.
    # We want all other degrees to be 0.
    # We search for coefficients a_0...a_11 in [-64, 64] \ {0} such that this holds.
    
    # To speed this up, we can use a greedy backtracking.
    # Order the coefficients a_0...a_11.
    # When assigning a_i, we can check if previous constraints are violated.
    # Constraint: For all k not in T, sum_{i+j=k} a_i a_j = 0.
    
    # Let's try a few random target supports T of size K, for K=12, 11, 10...
    # For each T, run a limited backtracking search.
    
    import itertools
    
    # Precompute pairs (i,j) for each k
    pairs_for_k = [[] for _ in range(23)]
    for i in range(12):
        for j in range(i, 12):
            k = i + j
            pairs_for_k[k].append((i, j))
            
    def solve_for_support(T, time_limit=2.0):
        """
        Try to find coefficients a_0..a_11 in [-64,64]\{0}
        such that P(x)^2 has non-zero terms ONLY at indices in T.
        """
        start_t = time.time()
        T_set = set(T)
        # We must have 0 in T (a0^2) and 22 in T (a11^2)
        if 0 not in T_set or 22 not in T_set:
            return None
            
        # Backtracking
        # We assign a_0 ... a_11
        # To reduce symmetry, fix a_0 = 1 (we can scale later, but integers must be kept).
        # Actually, scaling by -1 is trivial. Scaling by other integers might break integrality if we don't start with integers.
        # Let's just search integers directly.
        
        # Heuristic ordering: Assign a_0, a_1, ...
        # But a_0 is constrained by a_0^2 != 0 (always true if a0!=0).
        # a_1 is constrained by 2 a_0 a_1 = 0 if 1 not in T.
        
        a = [0] * 12
        
        def backtrack(idx):
            nonlocal a
            if time.time() - start_t > time_limit:
                return False
            
            if idx == 12:
                # Verify full solution
                # Check all k
                for k in range(23):
                    val = 0
                    for (i, j) in pairs_for_k[k]:
                        prod = a[i] * a[j]
                        if i == j:
                            val += prod
                        else:
                            val += 2 * prod
                    if k in T_set:
                        if val == 0:
                            return False # Must be non-zero
                    else:
                        if val != 0:
                            return False # Must be zero
                return True
                
            # Pruning: Check constraints that involve only a_0...a_idx
            # A constraint for degree k is determined by pairs (i,j) with i+j=k.
            # If max(i,j) <= idx, the constraint is fully determined.
            # If min(i,j) > idx, it's not started.
            # If mixed, it's partial.
            
            # We can check "fully determined" constraints.
            for k in range(idx + 1): # Degrees 0 to idx
                # All pairs (i,j) with i+j=k have i,j <= k <= idx, so they are assigned.
                val = 0
                for (i, j) in pairs_for_k[k]:
                    prod = a[i] * a[j]
                    if i == j:
                        val += prod
                    else:
                        val += 2 * prod
                if k in T_set:
                    if val == 0:
                        return False
                else:
                    if val != 0:
                        return False
                        
            # Also check constraints for k > idx that might be forced to 0 by partial sums?
            # This is complex. Let's just assign and check fully determined ones.
            
            # Try values for a[idx]
            # Range: [-64, 64] \ {0}
            # To speed up, we can try small values first, or random.
            # Let's use a randomized order for the first few, then heuristic.
            
            # Simple heuristic: try 1, -1, 2, -2, 3, -3 ...
            # Or random.
            # Given the tight constraints, random might be better to escape local traps?
            # But backtracking is systematic.
            
            # Let's try a small set of candidate values to keep it fast.
            # If we restrict to [-4, 4], we might miss solutions.
            # But small integers are more likely to cancel.
            
            candidates = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5]
            # Shuffle for variety
            random.shuffle(candidates)
            
            for val in candidates:
                a[idx] = val
                if backtrack(idx + 1):
                    return True
            a[idx] = 0
            return False
            
        a[0] = 1 # Fix scale
        if backtrack(1):
            return a[:]
        a[0] = -1
        if backtrack(1):
            return a[:]
        return None

    # Try different target sizes K
    # We know K=13 is possible (current best).
    # We want K=12, 11, etc.
    
    # We will sample random subsets T of size K.
    # To be efficient, we focus on K=12 first.
    
    # Note: The support T must include 0 and 22.
    # Also, if i is in support of P, 2i is in support of P^2 (unless cancelled).
    # But P has full support 0..11, so 0, 2, 4...22 are "naturally" present.
    # Cancellations must remove them.
    
    # Let's iterate over K from 12 down to 10.
    for K in range(12, 9, -1):
        if time.time() > deadline:
            break
            
        attempts = 0
        max_attempts = 200 # Adjust based on time
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            # Generate random T containing 0 and 22, size K
            # Remaining K-2 from 1..21
            candidates = list(range(1, 22))
            random.shuffle(candidates)
            T = [0, 22] + candidates[:K-2]
            random.shuffle(T)
            
            # Solve
            sol = solve_for_support(T, time_limit=0.1) # Very short time per attempt
            if sol:
                # Verify
                c = count_nonzero_square(sol)
                if c < best_count:
                    best_count = c
                    best_poly = sol[:]
                    if best_count <= 10:
                        break
            # If we found a good one, we can break
            if best_count <= K:
                break
        
        if best_count <= 10:
            break

    # Fallback if the specific support search failed or didn't improve much
    # Use the original simulated annealing but with a better objective?
    # Or just return the best found so far.
    
    if best_poly is None or best_count > 13:
        # If we didn't find anything better than 13, use a quick random search to ensure we at least match or beat 1.15 (which is 13 terms? 15/13 = 1.153)
        # Actually, if seed is 1.0 and current is 1.15, and score = seed/value, then value = 1/1.15? No.
        # "search score is seed/value".
        # If seed=1.0, value=1.0 -> score 1.0.
        # If score=1.15, value = 1/1.15 = 0.869? No, value is "raw value (lower is better)".
        # So Value = 13? Seed = 15? 15/13 = 1.153.
        # So we are looking for 12 terms. 15/12 = 1.25.
        
        # Let's do a quick random search to find a 13-term square if we didn't find 12.
        for _ in range(1000):
            if time.time() > deadline:
                break
            state = [random.choice([-1, 1]) * random.randint(1, 5) for _ in range(12)]
            c = count_nonzero_square(state)
            if c < best_count:
                best_count = c
                best_poly = state[:]

    if best_poly is None:
        return [1] * 12

    return best_poly

# EVOLVE-BLOCK-END
