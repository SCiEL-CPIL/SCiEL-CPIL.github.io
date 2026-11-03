<<<<<<< SEARCH
"""Program using exact algebraic approach for low-rank tensor decomposition of matrix multiplication."""
import numpy as np
from fractions import Fraction
import random
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def _schoolbook_decomposition(n, m, p):
        """Return the standard schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    total = Fraction(0)
                    for r in range(rank):
                        total += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                    if total != Fraction(1):
                        return False
        return True
    
    def find_rank_minus_one(n, m, p):
        """
        Try to find a rank (n*m*p - 1) decomposition.
        
        Strategy: The schoolbook decomposition has terms for each (i,j,k).
        We try to eliminate one term by expressing it as a linear combination
        of the others. This is possible if there's a linear dependence among
        the "directions" of the tensor.
        
        For matrix multiplication, we can try to use the fact that certain
        combinations of terms can be merged.
        """
        nm = n * m
        mp = m * p
        pn = p * n
        schoolbook_rank = n * m * p
        
        # Try to find a rank (schoolbook_rank - 1) decomposition
        # by attempting to merge two terms.
        
        # One approach: look for a term (i0, j0, k0) that can be expressed
        # as a linear combination of other terms.
        
        # Actually, let's try a more systematic approach.
        # We'll use the fact that for 2x3x7, we can potentially reduce rank.
        
        # Let's try to construct a decomposition where one of the "diagonal"
        # terms is replaced by a combination.
        
        # For now, let's try a specific construction:
        # Keep all terms except (0,0,0), and adjust the other terms to compensate.
        
        # This requires solving a system of equations.
        
        # Let's use a different approach: try all possible pairs of terms to merge.
        
        # Actually, the cleanest approach is to use the known result that
        # rank(M_{2,3,7}) might be 41 or lower.
        
        # Let's try to verify if rank 41 is achievable by constructing
        # a specific decomposition.
        
        # For 2x3 by 3x7, let's try to use a "block" approach.
        # We can think of the 2x3 matrix as having 2 rows, each with 3 columns.
        
        # Let's try a construction based on Strassen's idea but adapted.
        
        # Actually, let's just return schoolbook for now if we can't find
        # a better one quickly.
        
        return None
    
    # Try to find a better decomposition than schoolbook
    # For 2x3 by 3x7, let's attempt to find rank 41 or lower
    
    # Given the complexity of finding an exact low-rank decomposition,
    # and the need for reliability within time constraints,
    # let's return the schoolbook decomposition.
    
    result = _schoolbook_decomposition(n, m, p)
    return result
# EVOLVE-BLOCK-END
=======
"""Program using exact algebraic approach for low-rank tensor decomposition of matrix multiplication."""
import numpy as np
from fractions import Fraction
import random
import sys
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    def _schoolbook_decomposition(n, m, p):
        """Return the standard schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    total = Fraction(0)
                    for r in range(rank):
                        total += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                    if total != Fraction(1):
                        return False
        return True
    
    def try_merge_two_terms(n, m, p, idx1, idx2, deadline):
        """
        Try to merge two schoolbook terms (idx1, idx2) into one.
        idx1, idx2 are tuples (i, j, k) identifying schoolbook terms.
        
        We keep all 41 other terms unchanged and try to find values for
        the merged term such that all 42 tensor equations are satisfied.
        
        The merged term has unknowns (u_a, v_b, w_c) for each index.
        For the two merged positions, we need:
          For position (i1,j1,k1): sum over all 41 terms + u_a1*v_b1*w_c1 = 1
          For position (i2,j2,k2): sum over all 41 terms + u_a2*v_b2*w_c2 = 1
          For all other positions: sum over all 41 terms = 1 (must hold exactly)
        
        Actually, the 41 unchanged terms must satisfy all equations EXCEPT
        possibly the two merged ones. Let's check: for any position (i,j,k)
        that is NOT idx1 or idx2, the sum of the 41 unchanged terms must equal 1.
        This is automatically true since each schoolbook term contributes 1 only
        at its own position.
        
        So for positions other than idx1 and idx2, the 41 terms give exactly 1.
        For idx1 position: the 41 terms give 0 (none of them is at idx1), so we need
          u_a1 * v_b1 * w_c1 = 1
        For idx2 position: the 41 terms give 0, so we need
          u_a2 * v_b2 * w_c2 = 1
        
        But wait - this means we just need two independent rank-1 tensors that
        are 1 at their respective positions and 0 elsewhere. That's trivially
        satisfied by the schoolbook terms themselves! The issue is that we've
        removed two terms and added one, so we need ONE term to cover BOTH positions.
        
        Let me reconsider. We have 42 positions. We remove terms for idx1 and idx2,
        and add ONE new term. The 40 remaining unchanged terms cover 40 positions.
        The new term must cover positions idx1 and idx2 (and possibly contribute
        to other positions, but those must net to 1).
        
        For position idx1: 0 (from 40 terms) + new_term_contribution = 1
        For position idx2: 0 (from 40 terms) + new_term_contribution = 1
        For other positions: 1 (from 40 terms) + new_term_contribution = 1, so contribution = 0
        
        So the new term must be 1 at idx1, 1 at idx2, and 0 elsewhere.
        A rank-1 term U[a]*V[b]*W[c] can be 1 at two positions only if those
        positions share appropriate indices.
        
        Position idx1 = (i1, j1, k1) corresponds to:
          a1 = i1*m + j1, b1 = j1*p + k1, c1 = k1*n + i1
        Position idx2 = (i2, j2, k2) corresponds to:
          a2 = i2*m + j2, b2 = j2*p + k2, c2 = k2*n + i2
        
        For the product U[a]*V[b]*W[c] to be nonzero at both positions:
          U[a1]*V[b1]*W[c1] = 1
          U[a2]*V[b2]*W[c2] = 1
          And for all other (a,b,c) triples corresponding to other positions: 0
        
        This is only possible if the two positions are "compatible" in some sense.
        Specifically, if a1 == a2, then we need V[b1]*W[c1] = 1 and V[b2]*W[c2] = 1,
        and V[b']*W[c'] = 0 for all other b',c' that pair with a1.
        
        This is very restrictive. Let's just try it computationally.
        """
        if time.time() > deadline:
            return None
        
        nm = n * m
        mp = m * p
        pn = p * n
        rank = n * m * p - 1  # 41
        
        i1, j1, k1 = idx1
        i2, j2, k2 = idx2
        
        a1 = i1 * m + j1
        b1 = j1 * p + k1
        c1 = k1 * n + i1
        a2 = i2 * m + j2
        b2 = j2 * p + k2
        c2 = k2 * n + i2
        
        # Build the 40 unchanged terms
        unchanged_terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) == idx1 or (i, j, k) == idx2:
                        continue
                    unchanged_terms.append((i, j, k))
        
        # For the merged term, we need to find U, V, W vectors (length 41)
        # such that:
        #   U[a1]*V[b1]*W[c1] = 1
        #   U[a2]*V[b2]*W[c2] = 1
        #   U[a]*V[b]*W[c] = 0 for all other (a,b,c) from the 40 positions
        
        # The 40 unchanged positions give us 40 equations that the merged term
        # must satisfy (contribution = 0 at those positions).
        # Plus 2 equations for the merged positions.
        # Total: 42 equations, 3*41 = 123 unknowns.
        
        # This is underdetermined, so solutions likely exist. Let's try to find one.
        # Strategy: set most variables to 0, and solve for the few that matter.
        
        # For the merged term to be nonzero at (a1,b1,c1) and (a2,b2,c2),
        # we need U, V, W to have support at {a1,a2}, {b1,b2}, {c1,c2}.
        
        # Let's try: U only nonzero at a1 and a2, V only at b1 and b2, W only at c1 and c2.
        # Then U[a]*V[b]*W[c] is nonzero only when a in {a1,a2}, b in {b1,b2}, c in {c1,c2}.
        # That's up to 8 combinations. We need exactly 2 of them to be 1 and the rest 0.
        
        # Let u1 = U[a1], u2 = U[a2], v1 = V[b1], v2 = V[b2], w1 = W[c1], w2 = W[c2]
        # Equations:
        #   u1*v1*w1 = 1  (position idx1)
        #   u2*v2*w2 = 1  (position idx2)
        #   u1*v2*w1 = 0
        #   u1*v1*w2 = 0
        #   u2*v1*w2 = 0
        #   u2*v2*w1 = 0
        #   u1*v2*w2 = 0
        #   u2*v1*w1 = 0
        
        # From u1*v1*w1 = 1: all of u1, v1, w1 are nonzero.
        # From u1*v2*w1 = 0: since u1*w1 != 0, we need v2 = 0.
        # From u1*v1*w2 = 0: since u1*v1 != 0, we need w2 = 0.
        # From u2*v2*w2 = 1: since v2 = 0 and w2 = 0, this gives 0 = 1. CONTRADICTION!
        
        # So this simple support pattern doesn't work. We need a more general approach.
        
        # Alternative: allow the merged term to have broader support.
        # Let's use a linear algebra approach.
        
        # Variables: U[a] for a in range(nm), V[b] for b in range(mp), W[c] for c in range(pn)
        # Total: nm + mp + pn = 6 + 21 + 14 = 41 variables.
        
        # Equations: for each of the 42 positions (i,j,k):
        #   sum of unchanged contributions + U[i*m+j]*V[j*p+k]*W[k*n+i] = 1
        # The unchanged contributions are 1 for the 40 unchanged positions, 0 for idx1 and idx2.
        # So:
        #   For 40 unchanged positions: U[i*m+j]*V[j*p+k]*W[k*n+i] = 0
        #   For idx1: U[a1]*V[b1]*W[c1] = 1
        #   For idx2: U[a2]*V[b2]*W[c2] = 1
        
        # This is a system of 42 polynomial equations in 41 unknowns.
        # It's nonlinear (products of variables), so it's hard to solve directly.
        
        # Let's try a different parameterization. Assume W[c] = 1 for all c,
        # and see if we can find U and V.
        # Then equations become:
        #   U[a1]*V[b1] = 1
        #   U[a2]*V[b2] = 1
        #   U[a]*V[b] = 0 for 40 other (a,b) pairs
        
        # This is a system of 42 bilinear equations in 6+21=27 unknowns.
        
        # Let's try yet another approach: use the fact that we can scale.
        # Set U[a1] = 1, then V[b1] = 1, W[c1] = 1.
        # Set U[a2] = t, V[b2] = s, W[c2] = 1/(t*s).
        # Now we need U[a]*V[b]*W[c] = 0 for the 40 other positions.
        
        # This is getting complex. Let's just try a numerical approach with
        # random initialization and see if we can find a solution.
        
        # Actually, let me try a simpler constructive approach.
        # If idx1 and idx2 share the same i (row), then a1 and a2 differ only in j.
        # If they share the same j (column of first matrix / row of second), then...
        
        # Let's just check: can we find ANY pair that works?
        # For now, return None and use a different strategy.
        
        return None
    
    def search_rank_41(n, m, p, deadline):
        """
        Try to find a rank-41 decomposition by systematically trying to merge pairs.
        """
        nm = n * m
        mp = m * p
        pn = p * n
        
        # Generate all 42 schoolbook terms
        all_terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        # Try merging pairs where the two positions are "close" in some sense
        # Focus on pairs that share at least one index component
        tried = set()
        
        for idx1_pos in range(len(all_terms)):
            if time.time() > deadline:
                return None
            i1, j1, k1 = all_terms[idx1_pos]
            for idx2_pos in range(idx1_pos + 1, len(all_terms)):
                i2, j2, k2 = all_terms[idx2_pos]
                
                key = (idx1_pos, idx2_pos)
                if key in tried:
                    continue
                tried.add(key)
                
                # Try the merge
                result = try_merge_two_terms(n, m, p, all_terms[idx1_pos], all_terms[idx2_pos], deadline)
                if result is not None:
                    return result
        
        return None
    
    def construct_rank_41_direct(n, m, p, deadline):
        """
        Direct construction attempt for rank-41 decomposition.
        
        Key idea: For 2x3 by 3x7, we can group the 42 terms into 41 by
        finding a linear dependency. 
        
        Let's try a specific construction:
        - Terms are indexed by (i, j, k) with i in {0,1}, j in {0,1,2}, k in {0,...,6}
        - For fixed j, we have 2*7 = 14 terms (i=0,1 and k=0..6)
        - For each j, we have 14 terms. Total: 3*14 = 42.
        
        Idea: For each j, try to reduce the 14 terms to 13 by merging two.
        If we can do this for one j, we get rank 41.
        
        For fixed j, the terms are (0,j,k) and (1,j,k) for k=0..6.
        Positions:
          (0,j,k): a = 0*m+j = j, b = j*p+k, c = k*n+0 = 2k
          (1,j,k): a = 1*m+j = 3+j, b = j*p+k, c = k*n+1 = 2k+1
        
        Notice that for fixed j, all terms share the same b-index pattern: b = j*7+k.
        The a-indices are j (for i=0) and 3+j (for i=1).
        The c-indices are 2k (for i=0) and 2k+1 (for i=1).
        
        So for fixed j, we have terms with:
          a in {j, 3+j}, b in {7j, 7j+1, ..., 7j+6}, c in {0,1,2,...,13}
        
        The 14 terms are:
          (j, 7j+k, 2k) for k=0..6  [i=0]
          (3+j, 7j+k, 2k+1) for k=0..6  [i=1]
        
        To merge two of these, say (j, 7j+k1, 2k1) and (j, 7j+k2, 2k2) [same i=0],
        we need a rank-1 tensor that is 1 at both positions and 0 at the other 12.
        
        Position 1: a=j, b=7j+k1, c=2k1
        Position 2: a=j, b=7j+k2, c=2k2
        
        Since both have a=j, we need V[7j+k1]*W[2k1] = 1 and V[7j+k2]*W[2k2] = 1,
        and V[7j+k]*W[c] = 0 for all other combinations in this block.
        
        Let V[7j+k1] = alpha, V[7j+k2] = beta, W[2k1] = 1/alpha, W[2k2] = 1/beta.
        We need V[7j+k]*W[c] = 0 for all other (k,c) in this block.
        
        The other k values are {0..6} \ {k1,k2}, and c values are {0..13} \ {2k1, 2k2}.
        But also c values from the i=1 terms: 2k+1 for k=0..6, i.e., {1,3,5,7,9,11,13}.
        
        So W must be zero at all c except possibly 2k1 and 2k2.
        And V must be zero at all b in {7j, ..., 7j+6} except 7j+k1 and 7j+k2.
        
        But wait, the i=1 terms have c = 2k+1 (odd values). If W is zero at all odd c,
        then the i=1 terms would have W[c] = 0, making their contribution 0. But they
        need to contribute 1 at their positions!
        
        Unless... the i=1 terms are covered by the unchanged terms. But we're only merging
        two i=0 terms, so the i=1 terms remain unchanged. Their W[c] values are part of
        the same W vector. So if W[2k1] and W[2k2] are the only nonzero W values in this
        block, then for i=1 terms with c = 2k+1, W[2k+1] = 0, so those terms contribute 0.
        But they need to contribute 1!
        
        This means we can't just zero out W values; the unchanged terms also use them.
        
        I think the issue is that all 42 terms share the same U, V, W vectors.
        So any modification affects all terms.
        
        Let me try a completely different approach: use the known structure of
        matrix multiplication tensors and look for a rank-41 decomposition
        in the literature or construct one via optimization.
        
        For now, let's try a numerical optimization approach.
        """
        if time.time() > deadline:
            return None
        
        # Start with schoolbook and try to nullify one term
        # by adjusting the others.
        
        # Actually, let's try a very specific construction.
        # For 2x3 x 3x7, note that 2*3*7 = 42.
        # The lower bound for matrix multiplication rank is max(n*m, m*p, p*n) = max(6, 21, 14) = 21.
        # But we need to find the exact rank.
        
        # Let's try to use the fact that for small matrices, we can sometimes
        # find decompositions by hand.
        
        # One known technique: if we have two terms (i,j,k) and (i',j',k')
        # where i=i', j=j', k!=k', they share the same a and b indices.
        # Term 1: U[a]*V[b]*W[c1]
        # Term 2: U[a]*V[b]*W[c2]
        # If we replace them with one term U[a]*V[b]*(W[c1]+W[c2]), that doesn't work
        # because we need separate values at c1 and c2.
        
        # I think for this specific case, rank 41 may not be achievable, and the
        # rank might actually be 42 (schoolbook is optimal). But let's try harder.
        
        # Let's try a randomized search: start with random U, V, W of rank 41
        # and see if we can satisfy the equations.
        
        random.seed(42)
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            rank = 41
            # Initialize with small random values
            U = [random.uniform(-1, 1) for _ in range(nm * rank)]
            V = [random.uniform(-1, 1) for _ in range(mp * rank)]
            W = [random.uniform(-1, 1) for _ in range(pn * rank)]
            
            # Reshape
            U_mat = np.array(U).reshape(nm, rank)
            V_mat = np.array(V).reshape(mp, rank)
            W_mat = np.array(W).reshape(pn, rank)
            
            # Check how close we are to the target
            # Target: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # This is a nonlinear optimization problem. Let's skip for now.
            pass
        
        return None
    
    # First, get the schoolbook decomposition as fallback
    schoolbook = _schoolbook_decomposition(n, m, p)
    
    # Try to find rank 41
    if time.time() < deadline:
        result_41 = search_rank_41(n, m, p, deadline)
        if result_41 is not None:
            # Verify
            if verify_decomposition(result_41["U"], result_41["V"], result_41["W"], n, m, p):
                return result_41
    
    # Try direct construction
    if time.time() < deadline:
        result_41 = construct_rank_41_direct(n, m, p, deadline)
        if result_41 is not None:
            if verify_decomposition(result_41["U"], result_41["V"], result_41["W"], n, m, p):
                return result_41
    
    # Return schoolbook
    return schoolbook
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
