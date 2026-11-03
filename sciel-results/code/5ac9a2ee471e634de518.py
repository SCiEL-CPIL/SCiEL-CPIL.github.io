"""Program using structured search for low-rank tensor decomposition of 2x3 by 3x7 matrix multiplication."""
from fractions import Fraction
import random
import sys
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    def schoolbook_decomposition():
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

    def verify_decomposition(U, V, W):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        rank = len(U[0])
        valid_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    valid_set.add((a, b, c))
        
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    total = Fraction(0)
                    for r in range(rank):
                        ua = U[a][r]
                        vb = V[b][r]
                        wc = W[c][r]
                        if ua == 0 or vb == 0 or wc == 0:
                            continue
                        total += ua * vb * wc
                    expected = 1 if (a, b, c) in valid_set else 0
                    if total != expected:
                        return False
        return True

    def try_rank_41():
        """
        Try to find a rank-41 decomposition.
        
        Key insight: For rank n*m*p - 1, we need to find coefficients such that
        one schoolbook term can be expressed as a linear combination of the others
        with a rank-1 correction.
        
        Approach: Parameterize U and V with the schoolbook structure plus one
        additional degree of freedom, then solve for W.
        """
        random.seed(12345)
        
        # The schoolbook decomposition has 42 terms indexed by (i,j,k).
        # For rank 41, we need to find u, v, w such that:
        # sum_{(i,j,k) != special} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i} 
        #   + u ⊗ v ⊗ w = T
        # where T is the multiplication tensor.
        
        # This means: for the special (i0,j0,k0), we need:
        # u[i0*m+j0] * v[j0*p+k0] * w[k0*n+i0] = 1
        # And for all other (i,j,k), the contribution from u⊗v⊗w must be 0
        # (since the schoolbook terms already handle them).
        
        # Wait, that's not quite right. The sum of all 42 schoolbook terms equals T.
        # If we remove one term (say for (i0,j0,k0)) and add u⊗v⊗w, we need:
        # T - e_{i0*m+j0} ⊗ e_{j0*p+k0} ⊗ e_{k0*n+i0} + u⊗v⊗w = T
        # So: u⊗v⊗w = e_{i0*m+j0} ⊗ e_{j0*p+k0} ⊗ e_{k0*n+i0}
        # This means u, v, w must be standard basis vectors, which gives rank 42, not 41.
        
        # The correct approach: we need 41 rank-1 tensors whose sum equals T.
        # Let's try: 40 schoolbook terms + 1 general rank-1 term.
        # The 40 schoolbook terms handle 40 of the 42 positions.
        # The general term must handle the remaining 2 positions.
        
        # So we need: u[a]*v[b]*w[c] = 1 for 2 positions, and 0 for the other 40.
        
        # Let's enumerate which 2 positions are "special" and check feasibility.
        
        triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        num_triples = len(triples)  # 42
        
        # Precompute positions
        positions = []
        for i, j, k in triples:
            positions.append((i*m+j, j*p+k, k*n+i))
        
        # For each pair of positions, check if a rank-1 tensor can be 1 at both
        # and 0 at all other 40 positions.
        
        for idx1 in range(num_triples):
            for idx2 in range(idx1 + 1, num_triples):
                if time.time() > deadline:
                    return None
                
                a1, b1, c1 = positions[idx1]
                a2, b2, c2 = positions[idx2]
                
                # Collect all other positions
                other_positions = []
                for idx in range(num_triples):
                    if idx != idx1 and idx != idx2:
                        other_positions.append(positions[idx])
                
                # We need u[a]*v[b]*w[c] = 0 for all (a,b,c) in other_positions
                # and u[a1]*v[b1]*w[c1] = 1, u[a2]*v[b2]*w[c2] = 1.
                
                # Necessary condition: if a1 == a2, then we need v[b1]*w[c1] = v[b2]*w[c2]
                # If b1 == b2, then u[a1]*w[c1] = u[a2]*w[c2]
                # If c1 == c2, then u[a1]*v[b1] = u[a2]*v[b2]
                
                # Let's check if there's a simple pattern.
                # If a1 == a2 and b1 == b2: then c1 must equal c2 (otherwise impossible)
                # If a1 == a2: try u[a1] = 1, then v[b1]*w[c1] = 1 and v[b2]*w[c2] = 1
                #   For all other positions with a = a1, need v[b]*w[c] = 0
                #   For all other positions with b = b1, need v[b1]*w[c] = 0 => w[c] = 0
                #   For all other positions with b = b2, need v[b2]*w[c] = 0 => w[c] = 0
                
                # This is getting complex. Let's try a direct linear algebra approach.
                # Fix U and V with a specific structure, solve for W.
                
                # Strategy: Set U and V to be mostly schoolbook, with a few free parameters.
                # Specifically, for the two special positions, allow non-standard values.
                
                # Let's try: U and V are schoolbook for all 42 terms, but we merge
                # two columns by making them linearly dependent.
                
                # Actually, let's just try random sparse U and V and solve for W.
                # This is what the current program does. Let's improve the randomization.
                
                # Try multiple random initializations
                for attempt in range(20):
                    if time.time() > deadline:
                        return None
                    
                    U = [[Fraction(0)] * 41 for _ in range(n*m)]
                    V = [[Fraction(0)] * 41 for _ in range(m*p)]
                    
                    # Initialize with random sparse values
                    for a in range(n*m):
                        for r in range(41):
                            if random.random() < 0.25:
                                U[a][r] = Fraction(random.choice([-1, 1, 2, -2]))
                    
                    for b in range(m*p):
                        for r in range(41):
                            if random.random() < 0.25:
                                V[b][r] = Fraction(random.choice([-1, 1, 2, -2]))
                    
                    # Solve for W
                    W = [[Fraction(0)] * 41 for _ in range(p*n)]
                    feasible = True
                    
                    for c in range(p*n):
                        # Build linear system for W[c,:]
                        rows = []
                        rhs = []
                        for a in range(n*m):
                            for b in range(m*p):
                                row = [U[a][r] * V[b][r] for r in range(41)]
                                if any(x != 0 for x in row):
                                    rows.append(row)
                                    # T[a,b,c] = 1 iff (a,b,c) is a valid position
                                    expected = 0
                                    for i, j, k in triples:
                                        if i*m+j == a and j*p+k == b and k*n+i == c:
                                            expected = 1
                                            break
                                    rhs.append(Fraction(expected))
                        
                        if not rows:
                            continue
                        
                        # Gaussian elimination
                        aug = [rows[i][:] + [rhs[i]] for i in range(len(rows))]
                        num_rows = len(aug)
                        num_cols = 41
                        
                        rank = 0
                        for col in range(num_cols):
                            pivot = -1
                            for row in range(rank, num_rows):
                                if aug[row][col] != 0:
                                    pivot = row
                                    break
                            if pivot == -1:
                                continue
                            
                            aug[rank], aug[pivot] = aug[pivot], aug[rank]
                            pivot_val = aug[rank][col]
                            
                            for row in range(rank + 1, num_rows):
                                if aug[row][col] != 0:
                                    factor = aug[row][col] / pivot_val
                                    for j in range(col, num_cols + 1):
                                        aug[row][j] -= factor * aug[rank][j]
                            
                            rank += 1
                            if rank == num_cols:
                                break
                        
                        # Check consistency
                        for row in range(rank, num_rows):
                            if aug[row][num_cols] != 0:
                                feasible = False
                                break
                        
                        if not feasible:
                            break
                        
                        # Back-substitution
                        w_c = [Fraction(0)] * num_cols
                        for row in range(rank - 1, -1, -1):
                            lead_col = -1
                            for col in range(num_cols):
                                if aug[row][col] != 0:
                                    lead_col = col
                                    break
                            if lead_col == -1:
                                continue
                            w_c[lead_col] = aug[row][num_cols] / aug[row][lead_col]
                        
                        W[c] = w_c
                    
                    if feasible:
                        if verify_decomposition(U, V, W):
                            return {"U": U, "V": V, "W": W}
        
        return None

    # Try rank 41
    result_41 = try_rank_41()
    if result_41 is not None:
        if verify_decomposition(result_41["U"], result_41["V"], result_41["W"]):
            return result_41

    # Fallback to schoolbook
    return schoolbook_decomposition()
# EVOLVE-BLOCK-END
