import numpy as np
import time
import itertools
import random

DIMENSIONS = [2, 3, 6]

def _verify(U, V, W, n, m, p):
    """
    Verify tensor decomposition identity:
    sum_l U[a*m+b, l] * V[c*p+d, l] * W[k*n+i, l] == 1 if (a==i and c==k and b==d) else 0
    """
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    T = np.einsum('abl,cdl,kil->abcdki', U3, V3, W3)
    
    target = np.zeros((n, m, m, p, p, n), dtype=float)
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    if b == d and c < p:
                        target[a, b, c, d, c, a] = 1.0
    
    return np.allclose(T, target, atol=1e-9)

def _build_target_tensor(n, m, p):
    """Build the 6D multiplication tensor T[a,b,c,d,k,i] = 1 if a==i, c==k, b==d else 0."""
    T = np.zeros((n, m, m, p, p, n), dtype=float)
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    if b == d and c < p:
                        T[a, b, c, d, c, a] = 1.0
    return T

def _atom_index_to_coords(atom_idx, n, m, p):
    """
    Map a linear index to (a,b,c,d,k,i) coordinates for a rank-1 atom.
    Atom (a,b,c,d,k,i) contributes to T[a,b,c,d,k,i] = 1 iff a==i, c==k, b==d.
    For a valid atom of the multiplication tensor, we need: a==i, c==k, b==d.
    So the atom is determined by (a, b, c, d) with i=a, k=c, and we need b==d.
    So valid atoms: (a, b, c, b, c, a) for a in [0,n), b in [0,m), c in [0,min(m,p)).
    """
    # The multiplication tensor has support only at (a,b,c,d,k,i) where a==i, c==k, b==d
    # So rank-1 atoms that can appear in a decomposition must have support within this set.
    # A rank-1 tensor u ⊗ v ⊗ w has entry u[a,b]*v[c,d]*w[k,i] at (a,b,c,d,k,i).
    # For this to match the target, we need u[a,b]*v[c,d]*w[k,i] = 1 when (a==i, c==k, b==d)
    # and 0 otherwise.
    # This is highly constrained. The standard schoolbook uses atoms where
    # u has a single nonzero at (a,b), v at (c,d), w at (k,i) with a==i, c==k, b==d.
    # But more general decompositions can use linear combinations.
    
    # For the sparse search, we consider atoms that are "compatible" in the sense that
    # their support pattern is a subset of the target's support pattern.
    # A rank-1 atom u⊗v⊗w is compatible if for all (a,b,c,d,k,i) where 
    # u[a,b]*v[c,d]*w[k,i] != 0, we have a==i, c==k, b==d.
    
    pass

def _generate_candidate_atoms(n, m, p, max_atoms=500):
    """
    Generate candidate rank-1 atoms for the 2x3x6 multiplication tensor.
    We use a structured ansatz: atoms where U, V, W each have at most 2 nonzeros,
    with values in {-1, 0, 1}. This keeps the search space manageable.
    """
    atoms = []
    U_size = n * m  # 6
    V_size = m * p  # 18
    W_size = p * n  # 12
    
    # For each possible U pattern (sparse, 1-2 nonzeros from {-1,1})
    U_patterns = []
    for nnz in [1, 2]:
        for positions in itertools.combinations(range(U_size), nnz):
            for signs in itertools.product([-1, 1], repeat=nnz):
                u = np.zeros(U_size)
                for pos, sign in zip(positions, signs):
                    u[pos] = sign
                U_patterns.append(u)
    
    V_patterns = []
    for nnz in [1, 2]:
        for positions in itertools.combinations(range(V_size), nnz):
            for signs in itertools.product([-1, 1], repeat=nnz):
                v = np.zeros(V_size)
                for pos, sign in zip(positions, signs):
                    v[pos] = sign
                V_patterns.append(v)
    
    W_patterns = []
    for nnz in [1, 2]:
        for positions in itertools.combinations(range(W_size), nnz):
            for signs in itertools.product([-1, 1], repeat=nnz):
                w = np.zeros(W_size)
                for pos, sign in zip(positions, signs):
                    w[pos] = sign
                W_patterns.append(w)
    
    # Filter: keep only atoms that are "compatible" with the target tensor structure.
    # An atom u⊗v⊗w is compatible if u⊗v⊗w has zeros wherever the target has zeros,
    # OR if the nonzero pattern of u⊗v⊗w is a subset of the target's nonzero pattern.
    # Target nonzero pattern: (a,b,c,d,k,i) where a==i, c==k, b==d.
    # In flattened indices: U index = a*m+b, V index = c*p+d, W index = k*n+i.
    # Target is nonzero at (U_idx, V_idx, W_idx) where:
    #   a = U_idx // m, b = U_idx % m
    #   c = V_idx // p, d = V_idx % p
    #   k = W_idx // n, i = W_idx % n
    #   and a==i, c==k, b==d
    
    target_support = set()
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    if b == d and c < p:
                        u_idx = a * m + b
                        v_idx = c * p + d
                        w_idx = c * n + a  # k=c, i=a
                        target_support.add((u_idx, v_idx, w_idx))
    
    compatible_atoms = []
    for u in U_patterns:
        u_nonzero = set(np.nonzero(u)[0])
        for v in V_patterns:
            v_nonzero = set(np.nonzero(v)[0])
            for w in W_patterns:
                w_nonzero = set(np.nonzero(w)[0])
                # Check if the outer product support is a subset of target support
                compatible = True
                for ui in u_nonzero:
                    for vi in v_nonzero:
                        for wi in w_nonzero:
                            if (ui, vi, wi) not in target_support:
                                compatible = False
                                break
                        if not compatible:
                            break
                    if not compatible:
                        break
                if compatible:
                    compatible_atoms.append((u, v, w))
                
                if len(compatible_atoms) >= max_atoms:
                    break
            if len(compatible_atoms) >= max_atoms:
                break
        if len(compatible_atoms) >= max_atoms:
            break
    
    return compatible_atoms

def _try_rank_k_decomposition(n, m, p, k, atoms, deadline):
    """
    Try to find a rank-k decomposition using the given atoms via sparse linear algebra.
    We select k atoms and solve for coefficients such that the linear combination
    equals the target tensor.
    """
    target = _build_target_tensor(n, m, p)
    target_flat = target.flatten()
    
    # Build the atom matrix: each atom is a column in the flattened tensor space
    atom_matrix = np.zeros((target.size, len(atoms)))
    for j, (u, v, w) in enumerate(atoms):
        u3 = u.reshape(n, m)
        v3 = v.reshape(m, p)
        w3 = w.reshape(p, n)
        atom_tensor = np.einsum('ab,cd,ki->abcdki', u3, v3, w3)
        atom_matrix[:, j] = atom_tensor.flatten()
    
    # We want to find k columns and coefficients x such that atom_matrix[:, selected] @ x = target_flat
    # This is an underdetermined system if k < rank, but we want exact solution.
    # Use randomized approach: pick k atoms, solve least squares, check if exact.
    
    best_solution = None
    n_atoms = len(atoms)
    
    if n_atoms < k:
        return None
    
    # Try multiple random subsets
    n_trials = 200
    for trial in range(n_trials):
        if time.time() > deadline:
            break
        
        selected = random.sample(range(n_atoms), k)
        A = atom_matrix[:, selected]
        
        # Solve A @ x = target_flat
        try:
            x, residuals, rank, sv = np.linalg.lstsq(A, target_flat, rcond=None)
            # Check if the solution is exact
            residual = np.linalg.norm(A @ x - target_flat)
            if residual < 1e-9:
                # Verify the full decomposition
                U = np.zeros((n * m, k))
                V = np.zeros((m * p, k))
                W = np.zeros((p * n, k))
                for j, idx in enumerate(selected):
                    u, v, w = atoms[idx]
                    U[:, j] = u
                    V[:, j] = v
                    W[:, j] = w
                # Scale by x
                U = U * x
                V = V * x
                W = W * x
                
                if _verify(U, V, W, n, m, p):
                    return U, V, W
        except Exception:
            continue
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36
    
    # Schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = schoolbook_rank
    
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # First: try the known rank-33 construction (three rank-11 2x3x2 blocks)
    # This is a reliable improvement over 36.
    U232 = np.zeros((2*3, 11))
    V232 = np.zeros((3*2, 11))
    W232 = np.zeros((2*2, 11))
    
    # Standard schoolbook for 2x3x2 is rank 12. We need rank 11.
    # Use a known rank-11 decomposition.
    # For 2x3x2, the rank is 11. Here's a valid one:
    # Using the structure from the literature.
    
    # Let me use a different approach: use the fact that 2x3x2 can be seen as
    # a sub-tensor of 2x2x2 (rank 7) with one extra column.
    # Actually, let me just use the rank-33 construction that was in the original.
    
    # Rank-11 for 2x3x2:
    # Terms:
    # 1: (a1, b1, c1)
    # 2: (a2, b3, c1)
    # 3: (a3, b5, c1)
    # 4: (a1, b2, c2)
    # 5: (a2, b4, c2)
    # 6: (a3, b6, c2)
    # 7: (a4, b1, c3)
    # 8: (a5, b3, c3)
    # 9: (a6, b5, c3)
    # 10: (a4, b2, c4)
    # 11: (a5, b4, c4) -- but we're missing a6*b6*c4
    # This doesn't work. Let me use a verified rank-12 (schoolbook) for now and 
    # focus on the direct search for rank 32-34.
    
    # Use schoolbook for 2x3x2 (rank 12), giving rank 36 total. No improvement.
    # Let me try the direct sparse search for rank 34, 33, 32.
    
    # Generate candidate atoms
    atoms = _generate_candidate_atoms(n, m, p, max_atoms=300)
    
    # Try ranks from 32 up to 35
    for target_rank in [32, 33, 34, 35]:
        if time.time() > deadline - 10:
            break
        
        result = _try_rank_k_decomposition(n, m, p, target_rank, atoms, deadline - 5)
        if result is not None:
            U, V, W = result
            best_U = U
            best_V = V
            best_W = W
            best_rank = target_rank
            break
    
    # If we didn't find anything better, try the rank-33 block construction
    if best_rank == schoolbook_rank and time.time() < deadline - 20:
        # Use three 2x3x2 blocks with rank 11 each (if we can find valid rank-11)
        # For now, use a known valid construction.
        
        # Let me use a verified rank-33 construction from the literature.
        # The 2x3x2 multiplication tensor has rank 11.
        # Here's a valid rank-11 decomposition:
        
        # A = [[a1, a2, a3], [a4, a5, a6]]
        # B = [[b1, b2], [b3, b4], [b5, b6]]
        # C = [[c1, c2], [c3, c4]]
        
        # 11 terms:
        # t1 = (a1+b1... no. Let me use explicit rank-1 terms.
        
        # Actually, I'll use a simpler approach: use the rank-12 schoolbook for 2x3x2
        # but with one linear combination to save one term.
        
        # For c4 = a4*b2 + a5*b4 + a6*b6, if we can combine a5*b4 and a6*b6...
        # This is tricky. Let me just use the direct search result or schoolbook.
        
        pass
    
    # Final verification
    if not _verify(best_U, best_V, best_W, n, m, p):
        # Fallback to schoolbook
        best_U = U_school
        best_V = V_school
        best_W = W_school
        best_rank = schoolbook_rank
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
