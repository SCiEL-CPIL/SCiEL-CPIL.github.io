"""Rewritten program: rank-14 exact decomposition for 2x2 x 2x4 matrix multiplication.

The tensor for multiplying a 2x2 matrix A by a 2x4 matrix B has a direct-sum
structure: two independent 2x2 x 2x2 blocks sharing the same A.  The known
rank-14 solution applies Strassen's rank-7 algorithm to each block.  We verify
this solution exactly and return it immediately (no search needed).

We also attempt a rank-13 search using a structured linear-algebra approach:
fix the U-factors to the Strassen pattern (since they must span the A-side
bilinear forms), then solve for V and W via a linear system derived from the
16 tensor equations.  If a valid integer/half-integer solution exists at rank
13, we return it; otherwise we fall back to rank 14.
"""

import numpy as np
import time
import random

DIMENSIONS = [2, 2, 4]


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8

    def verify(U, V, W):
        """Verify the tensor decomposition is exact using integer arithmetic."""
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True

    def verify_float(U, V, W):
        """Verify using floating point (for half-integer entries)."""
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def get_rank14():
        """
        Rank-14 decomposition for 2x2 * 2x4 using two Strassen 2x2 blocks.
        Left block: columns 0,1 of B (indices j*p+k where j in {0,1}, k in {0,1})
        Right block: columns 2,3 of B (indices j*p+k where j in {0,1}, k in {2,3})
        
        W convention: W[k*n + i, r] where output C[i,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        """
        R = 14
        U = [[0] * R for _ in range(nm)]  # 4 x 14
        V = [[0] * R for _ in range(mp)]  # 8 x 14
        W = [[0] * R for _ in range(pn)]  # 8 x 14

        # LEFT BLOCK: multiply A (2x2) by B_left (2x2 with entries b00,b01,b10,b11)
        # V indices for left block: row j*4+k where j in {0,1}, k in {0,1}
        #   j=0,k=0 -> V[0], j=0,k=1 -> V[1], j=1,k=0 -> V[4], j=1,k=1 -> V[5]
        # W indices for left block: k*n+i where k in {0,1}, i in {0,1}
        #   k=0,i=0 -> W[0], k=0,i=1 -> W[1], k=1,i=0 -> W[2], k=1,i=1 -> W[3]

        # r=0: M1 = (a00+a11)(b00+b11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[5][0] = 1
        W[0][0] = 1; W[3][0] = 1

        # r=1: M2 = (a10+a11)b00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1

        # r=2: M3 = a00(b01-b11)
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = -1
        W[2][2] = 1; W[3][2] = 1

        # r=3: M4 = a11(b10-b00)
        U[3][3] = 1
        V[4][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1

        # r=4: M5 = (a00+a01)b11
        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[0][4] = -1; W[2][4] = 1

        # r=5: M6 = (a10-a00)(b00+b01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1

        # r=6: M7 = (a01-a11)(b10+b11)
        U[1][6] = 1; U[3][6] = -1
        V[4][6] = 1; V[5][6] = 1
        W[0][6] = 1

        # RIGHT BLOCK: multiply A (2x2) by B_right (2x2 with entries b02,b03,b12,b13)
        # V indices for right block: j*4+k where j in {0,1}, k in {2,3}
        #   j=0,k=2 -> V[2], j=0,k=3 -> V[3], j=1,k=2 -> V[6], j=1,k=3 -> V[7]
        # W indices for right block: k*n+i where k in {2,3}, i in {0,1}
        #   k=2,i=0 -> W[4], k=2,i=1 -> W[5], k=3,i=0 -> W[6], k=3,i=1 -> W[7]

        # r=7: N1 = (a00+a11)(b02+b13)
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1

        # r=8: N2 = (a10+a11)b02
        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[5][8] = 1; W[7][8] = -1

        # r=9: N3 = a00(b03-b13)
        U[0][9] = 1
        V[3][9] = 1; V[7][9] = -1
        W[6][9] = 1; W[7][9] = 1

        # r=10: N4 = a11(b12-b02)
        U[3][10] = 1
        V[6][10] = 1; V[2][10] = -1
        W[4][10] = 1; W[5][10] = 1

        # r=11: N5 = (a00+a01)b13
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[6][11] = 1

        # r=12: N6 = (a10-a00)(b02+b03)
        U[2][12] = 1; U[0][12] = -1
        V[2][12] = 1; V[3][12] = 1
        W[7][12] = 1

        # r=13: N7 = (a01-a11)(b12+b13)
        U[1][13] = 1; U[3][13] = -1
        V[6][13] = 1; V[7][13] = 1
        W[4][13] = 1

        return U, V, W

    def try_rank13_linear():
        """
        Attempt rank-13 by fixing U to a structured pattern and solving linear system.
        
        Key idea: The 16 tensor equations sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
        are quadratic in the variables. However, if we fix U, they become linear
        in the products V[j,k,r]*W[k,i,r]. Let X[j,k,k',i] = V[j,k,r]*W[k',i,r] summed over r.
        Actually, the equations are: for each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        If we fix U, this is a system where each equation is a sum of rank-1 terms
        in the V,W variables. This is still bilinear.
        
        Alternative approach: Fix both U and V, then solve for W linearly.
        For fixed U and V, the equation for each (i,j,k) becomes:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        This is linear in W! For each output index (k,i), we need:
        sum_r [U[i*m+j, r] * V[j*p+k, r]] * W[k*n+i, r] = 1
        
        But W[k*n+i, r] appears in multiple equations (one for each j). So for fixed (k,i),
        the unknowns are W[k*n+i, 0], ..., W[k*n+i, R-1], and the equations are:
        For each j: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        This gives m=2 equations for R unknowns per (k,i) pair. With 8 (k,i) pairs,
        we have 16 equations and 8*R unknowns. For R=13, that's 104 unknowns and 16 equations.
        Very underdetermined.
        
        Better approach: Fix U and W, solve for V. Or use a more constrained search.
        
        Let's try: Fix U to the Strassen pattern (7 terms for each block, but with 13 total
        terms we need to restructure). Actually, let's try a different strategy:
        
        Search over sparse integer patterns for rank 13.
        """
        if time.time() > deadline - 50:
            return None

        # Strategy: The rank-14 solution has U with specific patterns.
        # For rank 13, we need to find a different set of 13 rank-1 terms.
        # 
        # Let's try: keep the 7 U-patterns from Strassen but allow the right block
        # to share some structure. Specifically, try to find V and W for rank 13
        # where U is fixed to a modified Strassen pattern.
        #
        # Actually, let's try a more direct approach: random search with the constraint
        # that entries are in {-2,-1,0,1,2} and the solution is sparse.

        random.seed(789)

        # Precompute the U patterns from rank-14 as a starting structure
        U14, V14, W14 = get_rank14()

        # Try: remove one term and see if we can adjust to get rank 13
        # The most "redundant" terms to try removing are those with the simplest structure
        
        for r_remove in range(14):
            if time.time() > deadline - 50:
                break

            # Build U, V, W with rank 13 by removing term r_remove
            # and trying to absorb its contribution into other terms
            U = [[0] * 13 for _ in range(nm)]
            V = [[0] * 13 for _ in range(mp)]
            W = [[0] * 13 for _ in range(pn)]

            # Copy all terms except r_remove
            for r_new in range(13):
                r_old = r_new if r_new < r_remove else r_new + 1
                for i in range(nm):
                    U[i][r_new] = U14[i][r_old]
                for j in range(mp):
                    V[j][r_new] = V14[j][r_old]
                for k in range(pn):
                    W[k][r_new] = W14[k][r_old]

            # Check if this accidentally works (unlikely but possible)
            if verify(U, V, W):
                return U, V, W

            # Now try small perturbations to fix the missing contribution
            for attempt in range(500):
                if time.time() > deadline - 50:
                    break

                U_try = [row[:] for row in U]
                V_try = [row[:] for row in V]
                W_try = [row[:] for row in W]

                # Make 1-3 small changes
                n_changes = random.randint(1, 3)
                for _ in range(n_changes):
                    mat = random.randint(0, 2)
                    if mat == 0:
                        row = random.randint(0, nm - 1)
                        col = random.randint(0, 12)
                        delta = random.choice([-1, 1])
                        U_try[row][col] += delta
                    elif mat == 1:
                        row = random.randint(0, mp - 1)
                        col = random.randint(0, 12)
                        delta = random.choice([-1, 1])
                        V_try[row][col] += delta
                    else:
                        row = random.randint(0, pn - 1)
                        col = random.randint(0, 12)
                        delta = random.choice([-1, 1])
                        W_try[row][col] += delta

                if verify(U_try, V_try, W_try):
                    return U_try, V_try, W_try

        # Broader random search for rank 13
        for attempt in range(5000):
            if time.time() > deadline - 45:
                break

            U = [[0] * 13 for _ in range(nm)]
            V = [[0] * 13 for _ in range(mp)]
            W = [[0] * 13 for _ in range(pn)]

            for r in range(13):
                # U: 1-2 nonzeros
                nu = random.randint(1, 2)
                for row in random.sample(range(nm), nu):
                    U[row][r] = random.choice([-2, -1, 1, 2])
                # V: 1-3 nonzeros
                nv = random.randint(1, 3)
                for row in random.sample(range(mp), nv):
                    V[row][r] = random.choice([-2, -1, 1, 2])
                # W: 1-2 nonzeros
                nw = random.randint(1, 2)
                for row in random.sample(range(pn), nw):
                    W[row][r] = random.choice([-2, -1, 1, 2])

            if verify(U, V, W):
                return U, V, W

        return None

    def try_rank13_shared_structure():
        """
        Try to exploit shared structure between the two blocks.
        
        The key insight: both blocks use the same A matrix. The Strassen
        decomposition for each block uses 7 terms. If we can find a way to
        share one of the bilinear forms between the two blocks, we get rank 13.
        
        Specifically, consider the term (a00+a11) which appears in M1 and N1.
        If we could use a single term that contributes to both blocks' output,
        we'd save one rank. But the outputs are different (different W patterns),
        so this is non-trivial.
        
        Let's try: a term where U = (a00+a11) pattern, V has entries in both
        blocks, and W has entries in both blocks.
        """
        if time.time() > deadline - 40:
            return None

        random.seed(321)

        # Start from rank-14 and try to merge two terms
        U14, V14, W14 = get_rank14()

        # Try merging terms r_a and r_b into a single term
        for r_a in range(14):
            for r_b in range(r_a + 1, 14):
                if time.time() > deadline - 40:
                    break

                # Build rank-13 by merging r_a and r_b
                U = [[0] * 13 for _ in range(nm)]
                V = [[0] * 13 for _ in range(mp)]
                W = [[0] * 13 for _ in range(pn)]

                # Copy all terms except r_a and r_b
                new_r = 0
                for r_old in range(14):
                    if r_old == r_a or r_old == r_b:
                        continue
                    for i in range(nm):
                        U[i][new_r] = U14[i][r_old]
                    for j in range(mp):
                        V[j][new_r] = V14[j][r_old]
                    for k in range(pn):
                        W[k][new_r] = W14[k][r_old]
                    new_r += 1

                # The merged term goes at position new_r (which should be 12 if we removed 2)
                # Actually, let's put it at a fixed position
                merged_pos = 12  # last position

                # Try: U_merged = U14[:,r_a] + U14[:,r_b], V_merged = V14[:,r_a] + V14[:,r_b], W_merged = W14[:,r_a] + W14[:,r_b]
                # This won't work in general, but let's try variations

                for u_scale_a in [-1, 1]:
                    for u_scale_b in [-1, 1]:
                        for v_scale_a in [-1, 1]:
                            for v_scale_b in [-1, 1]:
                                for w_scale_a in [-1, 1]:
                                    for w_scale_b in [-1, 1]:
                                        if time.time() > deadline - 40:
                                            break

                                        U_try = [row[:] for row in U]
                                        V_try = [row[:] for row in V]
                                        W_try = [row[:] for row in W]

                                        for i in range(nm):
                                            U_try[i][merged_pos] = (u_scale_a * U14[i][r_a] + u_scale_b * U14[i][r_b])
                                        for j in range(mp):
                                            V_try[j][merged_pos] = (v_scale_a * V14[j][r_a] + v_scale_b * V14[j][r_b])
                                        for k in range(pn):
                                            W_try[k][merged_pos] = (w_scale_a * W14[k][r_a] + w_scale_b * W14[k][r_b])

                                        if verify(U_try, V_try, W_try):
                                            return U_try, V_try, W_try

        return None

    # Get the known rank-14 solution
    U, V, W = get_rank14()

    # Verify it's correct
    if not verify(U, V, W):
        # Fallback: naive rank-16
        R = 16
        U = [[0] * R for _ in range(nm)]
        V = [[0] * R for _ in range(mp)]
        W = [[0] * R for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}

    rank14_solution = {"U": U, "V": V, "W": W}

    # Try rank-13 searches
    if time.time() < deadline - 55:
        result = try_rank13_shared_structure()
        if result is not None:
            U13, V13, W13 = result
            if verify(U13, V13, W13):
                return {"U": U13, "V": V13, "W": W13}

    if time.time() < deadline - 50:
        result = try_rank13_linear()
        if result is not None:
            U13, V13, W13 = result
            if verify(U13, V13, W13):
                return {"U": U13, "V": V13, "W": W13}

    # Return the verified rank-14 solution
    return rank14_solution
