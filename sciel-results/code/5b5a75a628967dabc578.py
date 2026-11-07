"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = 60.0  # Safety margin
    
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    # Function to get exponent vector of n with respect to primes
    def get_vector(n):
        v = [0] * num_primes
        for i, p in enumerate(primes):
            while n % p == 0:
                v[i] += 1
                n //= p
        return v
    
    # Function to check if a vector is linearly independent of a list of vectors (basis)
    # We perform Gaussian elimination on the basis to see if we can reduce the new vector to zero.
    # If it reduces to zero, it is dependent. Otherwise, it is independent.
    def is_independent(vec, basis):
        # Create a copy of the vector to work with
        v = vec[:]
        
        # Iterate through basis vectors, eliminating components
        # We want to see if v is in the span of basis.
        # We can do this by trying to express v as a linear combination.
        # Alternatively, add v to basis and check if rank increases.
        # Simpler: Perform elimination.
        
        # To make elimination robust and simple with Fractions, 
        # we can just check if the rank of the matrix formed by basis + [v] is greater than rank of basis.
        # But maintaining a reduced row echelon form (RREF) of the basis is efficient.
        
        # Let's maintain the basis in a reduced form where each vector has a leading 1 
        # at a unique position, and other vectors have 0 at that position.
        
        # For each existing basis vector, eliminate the component of v corresponding to that vector's pivot.
        # However, the current `basis` list might not be in RREF. 
        # It's better to maintain `basis` as a list of RREF vectors.
        
        # Let's refactor: `basis` will be a list of vectors in RREF.
        # To check independence of `vec`:
        # 1. For each b in basis:
        #    Let pivot_index be the index of the first non-zero element in b.
        #    If v[pivot_index] != 0:
        #       v = v - (v[pivot_index] / b[pivot_index]) * b
        # 2. If v is all zeros, it is dependent.
        # 3. If v is not all zeros, it is independent. The new pivot for v is the first non-zero index.
        #    We should normalize v so the pivot is 1, and eliminate this pivot from other basis vectors 
        #    to maintain RREF property for future checks? 
        #    Actually, for just checking independence, we don't strictly need to update the whole basis to RREF 
        #    if we just want to know if it's independent. But to make future checks efficient, 
        #    adding it to the basis in a way that preserves some structure is good.
        #    
        #    Standard approach: 
        #    Keep basis as a list of vectors. 
        #    To check if v is in span(basis):
        #    Perform Gaussian elimination on the system Bx = v.
        
        # Let's use a simpler exact check:
        # Add v to the list of basis vectors.
        # Check if the rank of the matrix [basis; v] is the same as rank of [basis].
        # Since dimensions are small, we can compute the determinant of submatrices? No, that's expensive.
        
        # Let's stick to the elimination method on the vector v against the current basis.
        # We assume `basis` is a list of vectors that are linearly independent.
        # We don't necessarily need them in RREF if we just do a "reduction" pass.
        # However, if we don't maintain RREF, the elimination order matters or we need to do full Gaussian elimination.
        
        # Efficient method:
        # Maintain `basis` as a list of vectors.
        # To test `v`:
        # Create a copy `w = v[:]`.
        # For each `b` in `basis`:
        #   Find the first non-zero index `k` in `b`.
        #   If `w[k]` is not zero:
        #      `w = w - (w[k] / b[k]) * b`
        # If `w` becomes all zeros, `v` is dependent.
        # Else, `v` is independent.
        # Note: This works if `basis` is in "row echelon form" (each leading term is to the right of the previous).
        # We can ensure this by sorting basis vectors by their pivot index and inserting new vectors appropriately.
        
        # Let's implement this:
        # `basis` is a list of tuples (pivot_index, vector).
        
        # Wait, the function signature inside `construct` is local. I can define helper structures.
        
        # Let's re-implement `is_independent` to return (is_ind, new_basis_if_ind)
        # Or just return bool and update basis externally.
        
        # For now, let's just return bool, and I will handle basis insertion in the main loop.
        
        w = [Fraction(x) for x in v]
        
        for b_vec in basis:
            # Find pivot of b_vec
            pivot_idx = -1
            for j in range(num_primes):
                if b_vec[j] != 0:
                    pivot_idx = j
                    break
            if pivot_idx == -1:
                continue # Should not happen for non-zero basis vectors
                
            if w[pivot_idx] != 0:
                factor = w[pivot_idx] / b_vec[pivot_idx]
                for j in range(num_primes):
                    w[j] -= factor * b_vec[j]
        
        # Check if w is zero
        is_zero = all(x == 0 for x in w)
        return not is_zero

    # We need to maintain the basis in a specific order for the elimination to work correctly 
    # (each basis vector's pivot must be unique and to the right of the previous one).
    # Let's maintain `basis` as a list of vectors, sorted by their pivot index.
    
    basis = []
    selected = []
    
    for n in range(2, 41):
        if time.time() > deadline:
            break
            
        v = get_vector(n)
        
        # Check independence
        if is_independent(v, basis):
            # It is independent. Add to basis and selected.
            # To maintain the "echelon" property for the simple elimination check above,
            # we should insert it such that pivots are increasing.
            # Find the pivot of v
            v_frac = [Fraction(x) for x in v]
            pivot_idx = -1
            for j in range(num_primes):
                if v_frac[j] != 0:
                    pivot_idx = j
                    break
            
            # Normalize v so pivot is 1? Not strictly necessary for the elimination check 
            # if we just use the raw pivot value in the division.
            # But for stability and canonical form, let's normalize.
            if pivot_idx != -1 and v_frac[pivot_idx] != 0:
                for j in range(num_primes):
                    v_frac[j] /= v_frac[pivot_idx]
                v_frac[pivot_idx] = Fraction(1)
            
            # Find insertion point
            insert_pos = 0
            for i, b_vec in enumerate(basis):
                b_pivot = -1
                for j in range(num_primes):
                    if b_vec[j] != 0:
                        b_pivot = j
                        break
                if b_pivot < pivot_idx:
                    insert_pos = i + 1
                else:
                    break
            
            basis.insert(insert_pos, v_frac)
            selected.append(n)
            
            # Optional: Eliminate the new pivot from other basis vectors to maintain reduced form?
            # The simple elimination check `is_independent` assumes that when we process basis vector `b`,
            # it eliminates the component of `w` at `b`'s pivot.
            # This works if the basis is in row echelon form (pivots increasing).
            # It does NOT require the basis to be in Reduced Row Echelon Form (RREF) 
            # (where other vectors have 0 at the pivot).
            # Why?
            # When we process `b1` (pivot p1), we eliminate w[p1].
            # Then we process `b2` (pivot p2 > p1). We eliminate w[p2].
            # Since w[p1] is now 0, and `b2` has 0 at p1 (because p2 > p1 and b2 is in echelon form 
            # relative to b1? No, `b2` might have a non-zero at p1 if we didn't reduce it).
            # Wait. If `b2` has a non-zero at p1, then when we did `w = w - factor * b2`, 
            # we would have reintroduced a non-zero value at p1.
            # So, YES, we MUST maintain the basis in a form where each basis vector has 0 
            # at the pivot positions of all *previous* basis vectors.
            # i.e., Upper Triangular form relative to the pivots.
            
            # So, after inserting, we should eliminate the new pivot from all *other* basis vectors?
            # Or eliminate the *new* vector's pivot from the *previous* vectors?
            # Actually, the standard Gaussian elimination for independence check:
            # If we maintain the basis in RREF (each pivot is 1, and all other entries in that column are 0),
            # then the check is trivial: just look at the pivot columns.
            # Let's update the basis to RREF whenever we add a new vector.
            
            # Update other basis vectors to eliminate the new pivot column.
            for i in range(len(basis)):
                if i == insert_pos:
                    continue
                b_vec = basis[i]
                if b_vec[pivot_idx] != 0:
                    factor = b_vec[pivot_idx]
                    for j in range(num_primes):
                        b_vec[j] -= factor * v_frac[j]
                    # b_vec[pivot_idx] should now be 0

    return selected

# EVOLVE-BLOCK-END
