import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 2, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # n=2, m=2, p=7
    # a in 0..3, b in 0..13, c in 0..13
    T = np.zeros((n*m, m*p, p*n), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = Fraction(1, 1)

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p

    def verify(U, V, W, tol=1e-9):
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        T_float = T.astype(float)
        return np.allclose(T_rec, T_float, atol=tol)

    def is_half_integer(arr, tol=1e-9):
        doubled = 2 * arr
        rounded = np.round(doubled)
        return np.allclose(doubled, rounded, atol=tol)

    # Strategy: Search for a decomposition of the form:
    # T = sum_{r=1}^{R} lambda_r * (u_r \otimes v_r \otimes w_r)
    # where u_r, v_r, w_r are chosen from a set of "canonical" patterns.
    
    # The tensor T is a sum of 28 terms: e_{i,j} \otimes e_{j,k} \otimes e_{k,i}
    # We can group these by the "multiplication pattern".
    # For a 2x2x1 multiplication, the patterns are defined by which output row (c)
    # depends on which input rows (a) and columns (b).
    # Actually, let's look at the structure of the terms.
    # Term (i,j,k): a=i*2+j, b=j*7+k, c=k*2+i.
    
    # Let's define a set of "basis" rank-1 tensors that are likely to appear.
    # A rank-1 tensor is defined by vectors u (4), v (14), w (14).
    # Since the tensor is sparse, the vectors u, v, w should be sparse.
    
    # Observation: The 28 terms can be grouped by the pair (i, k).
    # For a fixed (i, k), the term varies with j in {0, 1}.
    # Term 1 (j=0): a=i*2, b=0*7+k=k, c=k*2+i
    # Term 2 (j=1): a=i*2+1, b=1*7+k=7+k, c=k*2+i
    # Notice that for fixed (i, k), both terms share the same c index!
    # c = 2k + i.
    # The a indices are 2i and 2i+1.
    # The b indices are k and 7+k.
    
    # So for each pair (i, k) where i in {0,1} and k in {0..6}, we have a sub-tensor:
    # S_{i,k}[a,b,c] is non-zero only at c=2k+i.
    # S_{i,k}[2i, k, 2k+i] = 1
    # S_{i,k}[2i+1, 7+k, 2k+i] = 1
    
    # This sub-tensor has rank 2 (it's a sum of two orthogonal rank-1 terms).
    # Can we represent it with rank 1? No, because the support in (a,b) is (2i, k) and (2i+1, 7+k), which are distinct.
    # So each (i, k) block requires rank at least 2.
    # Total blocks: 2 * 7 = 14.
    # If independent, total rank = 14 * 2 = 28.
    
    # Can we share structure between different (i, k) blocks?
    # The c indices are distinct for each (i, k) pair?
    # c = 2k + i.
    # i=0: c = 0, 2, 4, 6, 8, 10, 12
    # i=1: c = 1, 3, 5, 7, 9, 11, 13
    # All 14 c indices are distinct!
    # Since the c indices are disjoint, the tensor is a direct sum of 14 sub-tensors S_{i,k}.
    # The rank of a direct sum is the sum of the ranks.
    # Rank(T) = sum_{i,k} Rank(S_{i,k}).
    # Each S_{i,k} has support {(2i, k), (2i+1, 7+k)} in the (a,b) plane and a single c.
    # This is a 2x2x1 tensor (restricted to the relevant a, b, c).
    # Specifically, it looks like:
    # [1 0; 0 1] at c.
    # This is the identity matrix. The rank of the identity matrix tensor (diag(1,1)) is 2.
    # Proof: If rank 1, u \otimes v \otimes w. w has only one non-zero (at c).
    # Then u \otimes v must equal the 2x2 identity.
    # u \otimes v has rank 1 as a matrix. Identity has rank 2. Contradiction.
    # So Rank(S_{i,k}) = 2.
    # Total Rank = 14 * 2 = 28.
    
    # Wait, is this logic sound?
    # T = sum_{i,k} S_{i,k}.
    # Since the c-supports are disjoint, any rank-1 term in a decomposition of T
    # can be assigned to exactly one S_{i,k} (or be zero).
    # Why? If a rank-1 term u \otimes v \otimes w has w non-zero at c_1 and c_2 (c_1 != c_2),
    # then it contributes to S_{i_1, k_1} and S_{i_2, k_2}.
    # But S_{i_1, k_1} and S_{i_2, k_2} have disjoint c-supports.
    # If w has support {c_1, c_2}, the term contributes to T at c_1 and c_2.
    # For the sum to equal T, the contribution at c_1 must match S_{i_1, k_1} and at c_2 match S_{i_2, k_2}.
    # This implies the term is a sum of two terms? No, it's a single rank-1 term.
    # If w = e_{c_1} + e_{c_2}, then u \otimes v \otimes w = (u \otimes v \otimes e_{c_1}) + (u \otimes v \otimes e_{c_2}).
    # This splits the rank-1 term into two "effective" rank-1 terms?
    # No, in the tensor decomposition T = sum_r u_r \otimes v_r \otimes w_r,
    # if w_r has multiple non-zeros, it couples the (a,b) slices.
    # However, since the target T has disjoint c-slices, we can argue that
    # we can always transform a decomposition into one where each w_r has support size 1.
    # Proof: Let T = sum_r u_r \otimes v_r \otimes w_r.
    # For each c, let R_c be the set of r such that w_r[c] != 0.
    # T[:,:,c] = sum_{r in R_c} (u_r \otimes v_r) w_r[c].
    # Since the c-slices of T are independent (disjoint support in c), the matrix
    # T[:,:,c] depends only on the terms with w_r[c] != 0.
    # We can view the decomposition as a block diagonal matrix in the "r" dimension
    # if we permute the terms? No.
    # But we can replace each term with w_r having support size > 1 by multiple terms.
    # Specifically, if w_r has support {c_1, ..., c_m}, we can write
    # u_r \otimes v_r \otimes w_r = sum_{j=1}^m (u_r \otimes v_r \otimes w_r^{(j)})
    # where w_r^{(j)} is w_r with only the c_j component.
    # This increases the rank? No, it replaces 1 term with m terms.
    # This would increase the rank. So this direction is wrong.
    # We want to MINIMIZE rank.
    
    # Let's re-evaluate.
    # Is it possible to have a rank-1 term that covers parts of two different S_{i,k} blocks?
    # Suppose a term covers part of S_{0,0} (c=0) and part of S_{0,1} (c=2).
    # Then w has non-zeros at 0 and 2.
    # The contribution to c=0 is u \otimes v * w[0].
    # The contribution to c=2 is u \otimes v * w[2].
    # The target at c=0 is S_{0,0}[:,:,0] = [[1,0],[0,1]] (in the relevant a,b subspace).
    # The target at c=2 is S_{0,1}[:,:,2] = [[1,0],[0,1]] (in the relevant a,b subspace).
    # Note that the a,b subspaces for S_{0,0} and S_{0,1} are:
    # S_{0,0}: a in {0,1}, b in {0, 7}.
    # S_{0,1}: a in {0,1}, b in {1, 8}.
    # These a,b subspaces are disjoint in b!
    # b=0,7 vs b=1,8.
    # So u \otimes v must have support in (a in {0,1}, b in {0,1,7,8}).
    # And u \otimes v * w[0] must be non-zero only at (0,0) and (1,7).
    # And u \otimes v * w[2] must be non-zero only at (0,1) and (1,8).
    # This implies u \otimes v has non-zeros at (0,0), (1,7), (0,1), (1,8).
    # A rank-1 matrix u \otimes v has the property that if M_{0,0} and M_{1,1} are non-zero,
    # then M_{0,1} and M_{1,0} must be non-zero (specifically M_{0,1}M_{1,0} = M_{0,0}M_{1,1}).
    # Here, let's map the indices.
    # Let the matrix M be 2x4 (a=0,1; b=0,1,7,8).
    # M[0,0] != 0, M[1,7] != 0.
    # M[0,1] != 0, M[1,8] != 0.
    # If M is rank 1, M = u v^T.
    # M[0,0] = u0 v0
    # M[1,7] = u1 v7
    # M[0,1] = u0 v1
    # M[1,8] = u1 v8
    # This is consistent with rank 1.
    # So it IS possible to have a rank-1 term that covers parts of two blocks.
    # In fact, this is the key to reducing the rank below 28.
    
    # Let's formulate the search.
    # We have 14 blocks S_{i,k}.
    # Each block requires "covering" two entries: (2i, k) and (2i+1, 7+k) at c=2k+i.
    # Let's call these "atoms".
    # Atom (i, k, 0) -> a=2i, b=k, c=2k+i.
    # Atom (i, k, 1) -> a=2i+1, b=7+k, c=2k+i.
    
    # A rank-1 term is defined by u, v, w.
    # It covers a set of atoms A.
    # For each atom (a,b,c) in A, we need u[a]v[b]w[c] = 1.
    # Also, for any (a',b',c') NOT in A, we need u[a']v[b']w[c'] = 0
    # (or rather, the sum of all terms must cancel out any spurious contributions).
    # But if we assume a "sparse" decomposition where each term only contributes to the atoms it covers,
    # then we need u, v, w to be zero everywhere else.
    
    # Let's try to find a set of rank-1 terms that covers all 28 atoms.
    # Each term covers a subset of atoms.
    # The subsets must be "compatible" (rank-1 constraint).
    
    # Let's generate all possible rank-1 terms that cover at least 2 atoms.
    # An atom is (a,b,c).
    # Possible atoms:
    # i=0:
    # k=0: (0,0,0), (1,7,0)
    # k=1: (0,1,2), (1,8,2)
    # ...
    # k=6: (0,6,12), (1,13,12)
    # i=1:
    # k=0: (2,0,1), (3,7,1)
    # k=1: (2,1,3), (3,8,3)
    # ...
    # k=6: (2,6,13), (3,13,13)
    
    # Let's list them:
    atoms = []
    for i in range(2):
        for k in range(7):
            a0 = 2*i
            b0 = k
            c0 = 2*k + i
            atoms.append((a0, b0, c0))
            
            a1 = 2*i + 1
            b1 = 7 + k
            c1 = 2*k + i
            atoms.append((a1, b1, c1))
            
    # Total 28 atoms.
    
    # A rank-1 term is a set of atoms such that there exist u, v, w with u[a]v[b]w[c]=1 for all (a,b,c) in the set.
    # This implies:
    # For any two atoms (a,b,c) and (a',b',c') in the set:
    # u[a]v[b]w[c] = 1
    # u[a']v[b']w[c'] = 1
    # This doesn't impose a simple constraint on the atoms alone, but it implies
    # that the matrix of (a,b) pairs must be rank 1 if we ignore c, etc.
    # Actually, if we fix c, the (a,b) pairs must form a rank-1 pattern.
    # If we fix a, the (b,c) pairs must form a rank-1 pattern.
    
    # Let's enumerate all subsets of atoms of size 2, 3, 4 that are rank-1 compatible.
    # Size 1: All 28 atoms are compatible.
    # Size 2: Pairs (atom1, atom2).
    #   Let atom1=(a1,b1,c1), atom2=(a2,b2,c2).
    #   We need u[a1]v[b1]w[c1]=1 and u[a2]v[b2]w[c2]=1.
    #   This is always possible unless there's a contradiction.
    #   Contradiction arises if a1=a2, b1=b2, c1!=c2. Then u[a1]v[b1]w[c1]=1 and u[a1]v[b1]w[c2]=1 => w[c1]=w[c2].
    #   If w[c1]=w[c2], it's fine.
    #   What if a1=a2, b1!=b2, c1=c2? Then u[a1]v[b1]w[c1]=1 and u[a1]v[b2]w[c1]=1 => v[b1]=v[b2].
    #   This is fine.
    #   So ANY pair of atoms is rank-1 compatible?
    #   Wait. If we have a set of atoms, we need to assign values to u, v, w.
    #   For a pair, we can always set u, v, w to satisfy the two equations.
    #   So any pair is a valid "term" in the sense that it can be part of a decomposition.
    #   BUT, we must ensure that the term does NOT contribute to other atoms.
    #   If we use a term to cover atom1 and atom2, it might also cover atom3.
    #   If it covers atom3, we must account for it.
    
    # Let's define the "support" of a term defined by a set of atoms S.
    # The term is u \otimes v \otimes w.
    # It contributes to (a,b,c) if u[a]v[b]w[c] != 0.
    # We want the support to be exactly S (or a subset of S, if we allow cancellation).
    # If we assume no cancellation, the support must be exactly S.
    # This means u, v, w must be zero at all indices not appearing in S.
    # Let A_S = {a | (a,b,c) in S}, B_S = {b | (a,b,c) in S}, C_S = {c | (a,b,c) in S}.
    # u is non-zero only on A_S.
    # v is non-zero only on B_S.
    # w is non-zero only on C_S.
    # And for all (a,b,c) in A_S x B_S x C_S, we must have u[a]v[b]w[c] = 1 if (a,b,c) in S, and 0 otherwise.
    # This is a strong constraint.
    # It implies that the "tensor" defined by S on the grid A_S x B_S x C_S must be rank 1.
    # I.e., the 3D array T_S[a,b,c] = 1 if (a,b,c) in S, 0 else, must be rank 1.
    # A 3D array is rank 1 if it can be written as u \otimes v \otimes w.
    # This is equivalent to: all 2x2 minors of all matricizations are 0.
    
    # Let's find all subsets S of atoms that form a rank-1 tensor.
    # Since the atoms are sparse, we can check this.
    
    def is_rank1_subset(S):
        # S is a set of (a,b,c)
        if not S:
            return True
        A = set(a for a,b,c in S)
        B = set(b for a,b,c in S)
        C = set(c for a,b,c in S)
        # Build the tensor on A x B x C
        # If any dimension is empty, it's rank 0 (or 1 if empty set?)
        # Let's just check the condition: T[a,b,c] = u[a]v[b]w[c]
        # This implies T[a,b,c] * T[a',b',c'] = T[a,b',c'] * T[a',b,c]
        # for all a,a' in A, b,b' in B, c,c' in C.
        # Let's check a few pairs.
        # Actually, simpler: 
        # If |A|=1, |B|=1, |C|=1: Rank 1.
        # If |A|=1, |B|=2, |C|=1: T[0,0,0]=1, T[0,1,0]=1. u[0]=1, v[0]=1, v[1]=1, w[0]=1. Rank 1.
        # If |A|=2, |B|=2, |C|=1: T[0,0,0]=1, T[1,1,0]=1, T[0,1,0]=0, T[1,0,0]=0.
        #   u[0]v[0]=1, u[1]v[1]=1, u[0]v[1]=0, u[1]v[0]=0.
        #   u[0]!=0, v[0]!=0 => u[0]v[1]=0 => v[1]=0.
        #   u[1]!=0, v[1]!=0 => u[1]v[0]=0 => v[0]=0.
        #   Contradiction. So this is NOT rank 1.
        
        # General check:
        # Pick a reference atom (a0, b0, c0) in S.
        # For any other atom (a,b,c) in S, we must have consistency.
        # Let's just iterate over all triples (a,a',b,b',c,c')? Too slow.
        # Let's use the property: T is rank 1 iff all 2x2 minors of the unfolding are 0.
        # Unfold 1: A x (B*C).
        # Unfold 2: B x (A*C).
        # Unfold 3: C x (A*B).
        
        # Let's just try to solve for u, v, w.
        # Set u[a0]=1.
        # Then v[b] = T[a0,b,c0] / w[c0]? No.
        # T[a,b,c] = u[a]v[b]w[c].
        # T[a0,b0,c0] = 1.
        # Let u[a0]=1.
        # Then v[b0]w[c0] = 1.
        # Let v[b0]=1. Then w[c0]=1.
        # Then for any (a,b,c) in S:
        # u[a] = T[a,b0,c0] (if b0,c0 in S? No, b0,c0 are from the reference atom).
        # This only works if the reference atom's b and c are "central".
        
        # Simpler: 
        # A tensor is rank 1 iff it is the outer product of three vectors.
        # This is equivalent to: 
        # For all a,a',b,b',c,c': T[a,b,c]T[a',b',c'] = T[a,b',c']T[a',b,c]
        # Let's check this condition.
        atoms_list = list(S)
        for i in range(len(atoms_list)):
            a1, b1, c1 = atoms_list[i]
            for j in range(len(atoms_list)):
                a2, b2, c2 = atoms_list[j]
                # Check T[a1,b1,c1] T[a2,b2,c2] = T[a1,b2,c2] T[a2,b1,c1]
                # LHS = 1 * 1 = 1
                # RHS:
                # T[a1,b2,c2] is 1 if (a1,b2,c2) in S, else 0.
                # T[a2,b1,c1] is 1 if (a2,b1,c1) in S, else 0.
                # So we need (a1,b2,c2) in S AND (a2,b1,c1) in S.
                
                if (a1, b2, c2) not in S or (a2, b1, c1) not in S:
                    return False
        return True

    # This check is O(|S|^2).
    # Let's find all rank-1 subsets of atoms.
    # We only care about subsets that are "maximal" or at least cover multiple atoms.
    # Let's find all pairs of atoms that form a rank-1 set.
    # Then triples, etc.
    
    valid_terms = []
    # Size 1
    for atom in atoms:
        valid_terms.append((atom,))
        
    # Size 2
    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)):
            S = {atoms[i], atoms[j]}
            if is_rank1_subset(S):
                valid_terms.append((atoms[i], atoms[j]))
                
    # Size 3
    # This might be too many. 28 choose 3 = 3276.
    # Let's just do size 2 for now and see if we can cover all atoms with size 2 terms.
    # If we use size 2 terms, we need 14 terms.
    # Can we do better?
    # If we find a size 3 term, we save 1 term.
    
    # Let's search for size 3 and 4 terms.
    # To speed up, only check triples that contain a valid pair.
    
    # Actually, let's just solve the set cover problem with the valid terms found.
    # We want to cover all 28 atoms with the minimum number of valid terms.
    
    # Let's generate more terms.
    # Size 3
    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)):
            for k in range(j+1, len(atoms)):
                S = {atoms[i], atoms[j], atoms[k]}
                if is_rank1_subset(S):
                    valid_terms.append((atoms[i], atoms[j], atoms[k]))
                    
    # Size 4
    # 28 choose 4 = 20475. Might be slow.
    # Let's skip size 4 for now.
    
    # Now, solve Set Cover.
    # Atoms to cover: 0..27
    # Terms: list of sets of atom indices.
    
    atom_to_idx = {atom: idx for idx, atom in enumerate(atoms)}
    term_sets = []
    for term in valid_terms:
        idxs = [atom_to_idx[atom] for atom in term]
        term_sets.append(frozenset(idxs))
        
    # Remove duplicate sets
    unique_term_sets = list(set(term_sets))
    
    # Greedy Set Cover
    covered = [False] * 28
    chosen_terms = []
    
    while not all(covered):
        if time.time() > deadline:
            break
        best_term = None
        best_count = 0
        for ts in unique_term_sets:
            count = sum(1 for idx in ts if not covered[idx])
            if count > best_count:
                best_count = count
                best_term = ts
            if count == 0:
                continue
        if best_term is None:
            break
        chosen_terms.append(best_term)
        for idx in best_term:
            covered[idx] = True
            
    # We have a set of terms (sets of atoms).
    # Now we need to construct U, V, W.
    # Each term corresponds to a rank-1 tensor.
    # We need to find u, v, w for each term such that u[a]v[b]w[c]=1 for all (a,b,c) in the term.
    # And 0 elsewhere (in the support of the term).
    
    R = len(chosen_terms)
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    for r, term_set in enumerate(chosen_terms):
        # Reconstruct the atoms
        term_atoms = [atoms[idx] for idx in term_set]
        # Find u, v, w
        # We can fix one value to break scaling.
        # Let's pick the first atom (a0, b0, c0).
        a0, b0, c0 = term_atoms[0]
        U[a0, r] = 1.0
        V[b0, r] = 1.0
        W[c0, r] = 1.0
        
        # Propagate to other atoms
        for a, b, c in term_atoms[1:]:
            # u[a]v[b]w[c] = 1
            # We know some of u, v, w?
            # This is a system of equations.
            # u[a]v[b]w[c] = 1
            # If we know u[a], v[b], then w[c] = 1/(u[a]v[b]).
            # If we know u[a], w[c], then v[b] = 1/(u[a]w[c]).
            # If we know v[b], w[c], then u[a] = 1/(v[b]w[c]).
            
            # Let's just solve it.
            # Since it's a rank-1 tensor, the values are consistent.
            # We can determine the values by traversing the "graph" of atoms.
            # Nodes are indices of a, b, c.
            # Edges are atoms.
            
            # Simpler: 
            # For each a in the term, if U[a, r] is 0, find an atom (a, b', c') in the term where V[b', r] and W[c', r] are non-zero.
            # Then U[a, r] = 1 / (V[b', r] * W[c', r]).
            
            # Iterate until all are set.
            changed = True
            while changed:
                changed = False
                for a, b, c in term_atoms:
                    if U[a, r] == 0 and V[b, r] != 0 and W[c, r] != 0:
                        U[a, r] = 1.0 / (V[b, r] * W[c, r])
                        changed = True
                    if V[b, r] == 0 and U[a, r] != 0 and W[c, r] != 0:
                        V[b, r] = 1.0 / (U[a, r] * W[c, r])
                        changed = True
                    if W[c, r] == 0 and U[a, r] != 0 and V[b, r] != 0:
                        W[c, r] = 1.0 / (U[a, r] * V[b, r])
                        changed = True
                        
    # Verify
    if verify(U, V, W):
        if R < best_rank:
            best_rank = R
            best_U, best_V, best_W = U, V, W
            
    # Return best found
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
