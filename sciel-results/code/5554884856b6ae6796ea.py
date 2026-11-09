import time
import math
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 210, 'maximum_window': 10210, 'maximum_size': 100}

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit to be safe
    
    n = P['n']
    max_val = P['maximum_window']
    max_size = P['maximum_size']
    
    # Primes up to max_val
    primes = []
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, max_val + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
                
    # Map primes to indices for vector representation
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    dim = len(primes)
    
    # Function to get square-free vector as a set of prime indices
    def get_sqfree_vec(x):
        vec = set()
        temp = x
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                if count % 2 == 1:
                    vec.add(prime_to_idx[p])
        if temp > 1:
            vec.add(prime_to_idx[temp])
        return frozenset(vec)
    
    target_vec = get_sqfree_vec(n)
    
    # We iterate M from n+1 to max_val
    # We maintain a basis of the vectors seen so far
    # basis: list of (vector, set_of_original_indices)
    # We also need to be able to reduce target_vec using the basis
    
    # To make it efficient, we can just re-run Gaussian elimination for each M?
    # No, incremental is better.
    
    # However, finding the *specific* subset requires backtracking the reduction steps.
    # Let's store the basis vectors and the original vectors they represent.
    
    # A simpler approach for small M:
    # For each M, build the matrix of vectors for 211..M.
    # Perform Gaussian elimination to find a basis.
    # Try to represent target_vec.
    
    # Since max_val is 10210, and we start from 211, the range is ~10000.
    # The dimension is ~1200.
    # Gaussian elimination on 10000 vectors of dim 1200 is fast.
    
    # Let's try iterating M and checking if target is in span.
    
    # Optimization: 
    # If we just want *a* solution, we can use the following:
    # Maintain a list of vectors V.
    # For each new vector v, try to add it to a basis.
    # If target is in span, we need to find the combination.
    
    # Let's implement a standard Gaussian elimination that tracks the linear combination.
    # basis_vecs: list of frozensets
    # basis_combos: list of sets of original indices (1-based offset from 211)
    
    # Actually, let's just do it for each M from scratch? 
    # No, that's O(M^2 * dim). M=10000, dim=1200. 10^8 * 1200 is too slow.
    
    # Incremental is necessary.
    
    # Let's maintain:
    # basis: list of (vec, combo_set)
    # where combo_set is the set of original integer values (or indices) that XOR to this vec.
    
    # When adding a new vector v_new with index i_new:
    # 1. Try to reduce v_new using existing basis.
    # 2. If it becomes 0, it's dependent. We can update some basis vectors to have smaller support? 
    #    Or just ignore? 
    #    If we ignore, we might miss a solution that requires this vector?
    #    No, if it's dependent, it's in the span of existing ones. So the span doesn't change.
    #    So if target was not in span before, it's not in span now.
    #    If target WAS in span before, we already found a solution.
    #    So we only need to check if target is in span when we ADD a new independent vector?
    #    Wait. If target is in span of {v1..vk}, it is also in span of {v1..vk, vk+1}.
    #    So once target is in span, we have a solution.
    #    We want the MINIMUM M.
    #    So we iterate M, add v(M) to the structure.
    #    If v(M) is independent, add to basis.
    #    Check if target is in span of current basis.
    #    If yes, find the combination and return.
    
    # How to check if target is in span and find combination?
    # We can maintain the basis in row-echelon form.
    # Then reduce target.
    
    # Let's implement this.
    
    basis = [] # list of (vec, combo_set)
    
    def reduce_vec(vec, combo):
        """
        Reduces vec using basis.
        Returns (reduced_vec, reduced_combo)
        """
        for i, (b_vec, b_combo) in enumerate(basis):
            if b_vec and len(b_vec) > 0:
                # Find the pivot (min element in b_vec for consistency? or just any?)
                # Let's use the smallest prime index as pivot for determinism
                pivot = min(b_vec)
                if pivot in vec:
                    vec = vec.symmetric_difference(b_vec)
                    combo = combo.symmetric_difference(b_combo)
        return vec, combo
    
    def add_to_basis(vec, combo):
        """
        Adds vec to basis if independent.
        Returns True if added, False if dependent.
        """
        # Reduce vec by existing basis
        reduced_vec, reduced_combo = reduce_vec(vec, combo)
        
        if not reduced_vec:
            return False
        
        # Find pivot for new vector
        pivot = min(reduced_vec)
        
        # Eliminate this pivot from other basis vectors to maintain reduced row echelon?
        # Or just insert and reorder?
        # For simplicity, let's just append and then maybe clean up?
        # Actually, for checking if target is in span, we just need a basis.
        # We don't strictly need RREF, just a basis.
        # But reduce_vec above assumes basis vectors have distinct pivots?
        # My reduce_vec above just checks if pivot is in vec.
        # If basis is not in echelon form, this might not reduce to 0 if it should.
        
        # Let's implement proper Gaussian Elimination.
        # We will maintain basis in row-echelon form.
        
        # 1. Reduce new vec by existing basis
        r_vec, r_combo = reduce_vec(vec, combo)
        if not r_vec:
            return False
            
        # 2. Find pivot
        pivot = min(r_vec)
        
        # 3. Eliminate this pivot from all existing basis vectors
        new_basis = []
        for i, (b_vec, b_combo) in enumerate(basis):
            if b_vec and pivot in b_vec:
                # Eliminate pivot from b_vec
                b_vec = b_vec.symmetric_difference(r_vec)
                b_combo = b_combo.symmetric_difference(r_combo)
            new_basis.append((b_vec, b_combo))
            
        # 4. Add new vector
        new_basis.append((r_vec, r_combo))
        
        # Update global basis
        basis[:] = new_basis
        return True

    # We need to check if target is in span.
    # We can do this by reducing target_vec using the current basis.
    # If it reduces to empty set, it's in the span.
    # The combo set will be the solution.
    
    for M in range(n + 1, max_val + 1):
        if time.time() > deadline:
            # Fallback: return a valid solution if found earlier? 
            # We are iterating from small M, so we haven't found one yet.
            # This shouldn't happen if the answer is small.
            break
            
        v_M = get_sqfree_vec(M)
        # The "combo" for v_M is just the set containing M
        combo_M = {M}
        
        # Try to add to basis
        # We need to know if it was added to check if span expanded.
        # Actually, even if not added, the span is the same.
        # So we only need to check if target is in span when span potentially expands.
        # But wait, if target was not in span before, and we add a dependent vector, it's still not in span.
        # So we only need to check if target is in span when we successfully add an independent vector.
        
        added = add_to_basis(v_M, combo_M)
        
        if added:
            # Check if target is in span
            t_vec = set(target_vec)
            t_combo = set()
            
            # Reduce t_vec using basis
            # We need a function that reduces a vector and returns the combo
            # My reduce_vec above takes combo and returns reduced combo.
            # But it modifies the local vec.
            
            r_t_vec, r_t_combo = reduce_vec(t_vec, t_combo)
            
            if not r_t_vec:
                # Found a solution
                # r_t_combo contains the integers
                sol = list(r_t_combo)
                if len(sol) <= max_size:
                    # Ensure types are int
                    return [int(x) for x in sol]
                else:
                    # Solution too large, try next M?
                    # The problem says 1..100 distinct integers.
                    # If the only solution for this M is too big, maybe a larger M has a smaller one?
                    # Unlikely, but possible.
                    # For now, continue.
                    pass
                    
    # If no solution found within deadline or range
    # Return a dummy valid solution? 
    # The task says "Any broken rule makes the whole result invalid".
    # We must return a valid list.
    # If we didn't find one, we are in trouble.
    # But 210 is small, so a solution should exist quickly.
    
    # Fallback: Just return [4*210] which is 840. 
    # 840 * 210 = 176400. sqrt(176400) = 420. 
    # 840 > 210. 840 <= 10210.
    # Distinct integers: [840]. Size 1.
    # Max - 210 = 630.
    # This is a valid solution, though not optimal.
    return [840]
# EVOLVE-BLOCK-END
