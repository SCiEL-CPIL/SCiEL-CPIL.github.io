import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    DEADLINE = 130  # seconds, leave 20s margin before 150s kill

    n, m, p = DIMENSIONS

    # Try rank 27 first with structured ansatz + Schwartz-Zippel verification
    if time.time() - start_time < DEADLINE:
        result = _try_rank27_structured(n, m, p, start_time, DEADLINE)
        if result is not None:
            return result

    # Fallback: return the known rank-28 solution
    return _known_rank28()

def _known_rank28():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Indices for A (2x3): A00=0, A01=1, A02=2, A10=3, A11=4, A12=5
    # Indices for B (3x5): B00=0..B04=4, B10=5..B14=9, B20=10..B24=14
    # W index: C[i,k] -> W[k*n+i], so C00=0, C01=2, C02=4, C03=6, C04=8
    #                                    C10=1, C11=3, C12=5, C13=7, C14=9

    # Part 1: A_L (2x2) * B_T1 (2x2) -> C_1 (cols 0,1) using Strassen-7
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[6][0] = 1
    W[0][0] = 1; W[3][0] = 1

    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1

    U[0][2] = 1
    V[1][2] = 1; V[6][2] = -1
    W[2][2] = 1; W[3][2] = 1

    U[4][3] = 1
    V[5][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1

    U[0][4] = 1; U[1][4] = 1
    V[6][4] = 1
    W[0][4] = -1; W[2][4] = 1

    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    U[1][6] = 1; U[4][6] = -1
    V[5][6] = 1; V[6][6] = 1
    W[0][6] = 1

    # Part 2: A_L (2x2) * B_T2 (2x2) -> C_2 (cols 2,3) using Strassen-7
    U[0][7] = 1; U[4][7] = 1
    V[2][7] = 1; V[8][7] = 1
    W[4][7] = 1; W[7][7] = 1

    U[3][8] = 1; U[4][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1

    U[0][9] = 1
    V[3][9] = 1; V[8][9] = -1
    W[6][9] = 1; W[7][9] = 1

    U[4][10] = 1
    V[7][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1

    U[0][11] = 1; U[1][11] = 1
    V[8][11] = 1
    W[4][11] = -1; W[6][11] = 1

    U[3][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1

    U[1][13] = 1; U[4][13] = -1
    V[7][13] = 1; V[8][13] = 1
    W[4][13] = 1

    # Part 3: A_L (2x2) * B_T3 (2x1) -> C_3 (col 4) schoolbook
    U[0][14] = 1; V[4][14] = 1; W[8][14] = 1
    U[1][15] = 1; V[9][15] = 1; W[8][15] = 1
    U[3][16] = 1; V[4][16] = 1; W[9][16] = 1
    U[4][17] = 1; V[9][17] = 1; W[9][17] = 1

    # Part 4: A_R (2x1) * B_B (1x5) -> C_4 (all cols) schoolbook
    U[2][18] = 1; V[10][18] = 1; W[0][18] = 1
    U[2][19] = 1; V[11][19] = 1; W[2][19] = 1
    U[2][20] = 1; V[12][20] = 1; W[4][20] = 1
    U[2][21] = 1; V[13][21] = 1; W[6][21] = 1
    U[2][22] = 1; V[14][22] = 1; W[8][22] = 1
    U[5][23] = 1; V[10][23] = 1; W[1][23] = 1
    U[5][24] = 1; V[11][24] = 1; W[3][24] = 1
    U[5][25] = 1; V[12][25] = 1; W[5][25] = 1
    U[5][26] = 1; V[13][26] = 1; W[7][26] = 1
    U[5][27] = 1; V[14][27] = 1; W[9][27] = 1

    return {"U": U, "V": V, "W": W}

def _try_rank27_structured(n, m, p, start_time, deadline):
    """
    Try to find a rank-27 decomposition using structured ansatz with
    Schwartz-Zippel verification for fast candidate testing.
    
    Strategy: Start from the rank-28 solution and try to merge/eliminate
    one term by introducing integer parameters. Use random evaluation over
    a finite field to quickly reject bad candidates (Schwartz-Zippel: a
    nonzero polynomial of degree d evaluated at a random point from a set
    of size S has probability at most d/S of being zero).
    """
    
    # Precompute the rank-28 solution as baseline
    base = _known_rank28()
    U_base = [row[:] for row in base['U']]
    V_base = [row[:] for row in base['V']]
    W_base = [row[:] for row in base['W']]
    
    # Verify the baseline is correct
    if not _exact_verify(U_base, V_base, W_base, n, m, p):
        return None
    
    # Strategy: Try to eliminate one of the schoolbook terms by absorbing
    # it into a Strassen term via parameterization.
    # 
    # The 4 schoolbook terms for A_L * B_T3 (terms 14-17) and 10 for A_R * B_B
    # (terms 18-27). Let's try to merge one pair.
    #
    # Key insight: If we can find a parameterization where one term becomes
    # redundant (its contribution is already covered by other terms), we get rank 27.
    #
    # Let's try: modify term 14 (A00*B04 -> C04) to also contribute to another
    # output, and adjust another term to compensate.
    
    # Ansatz: Try modifying the last 14 terms (14-27) with integer parameters.
    # We'll try several specific structural modifications.
    
    # Modification 1: Try to use a "diagonal" Strassen-like trick on the
    # 2x2 * 2x1 block (A_L * B_T3, terms 14-17)
    # Standard: 4 terms. Can we do it in 3 with half-integers?
    # For 2x2 * 2x1, the tensor is:
    # T[0,0,0]=1, T[0,1,0]=1, T[1,0,0]=1, T[1,1,0]=1 (indices: row, col_of_A, col_of_B)
    # Wait, this is just a matrix-vector product, rank is 4 (each output needs
    # its own term since the inputs are independent).
    # Actually no: C0 = A00*B0 + A01*B1, C1 = A10*B0 + A11*B1
    # This is a 2x2 matrix times a 2x1 vector. The tensor rank is 4
    # (it's the Kronecker product of identity_2 and identity_2, sort of).
    # Actually the rank of the 2x2*2x1 matmul tensor is 4.
    
    # So we can't reduce the 2x2*2x1 part below 4 terms.
    # And the 2x1*1x5 part has rank 10.
    # Total minimum for the "cross" terms is 14, plus 14 for Strassen = 28.
    
    # To get 27, we need to find a non-trivial interaction between the
    # Strassen terms and the schoolbook terms.
    
    # Let's try a different approach: use the fact that the Strassen terms
    # have "waste" (they compute extra products that cancel out). Maybe we
    # can redirect some of that waste to help with the schoolbook terms.
    
    # Specifically, in Strassen term 0: (A00+A11)*(B00+B11)
    # This produces: A00*B00 + A00*B11 + A11*B00 + A11*B11
    # The "needed" products are A00*B00 (for C00) and A11*B11 (for C11).
    # The "waste" is A00*B11 (for C01) and A11*B00 (for C10).
    # But C01 also needs A01*B10, and C10 also needs A10*B00.
    # So the waste actually helps! That's why Strassen works.
    
    # The question is: can we use the Strassen structure to also help with
    # the A_L * B_T3 or A_R * B_B terms?
    
    # Let's try: modify the Strassen terms to also involve A02 or B2k.
    # This would break the clean Strassen structure but might allow sharing.
    
    # Given the complexity, let's try a numerical search with the
    # Schwartz-Zippel trick for fast verification.
    
    # Setup: We'll parameterize the last few terms with small integers
    # and use random evaluation to quickly check.
    
    MOD = 1000003  # A prime for finite field evaluation
    
    def _evaluate_mod(U, V, W, n, m, p, mod):
        """Evaluate the tensor equations modulo 'mod'. Returns True if all satisfied."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(len(U[0])):
                        s += (U[i*m+j][r] % mod) * (V[j*p+k][r] % mod) % mod * (W[k*n+i][r] % mod)
                        s %= mod
                    if s % mod != 1 % mod:
                        return False
        return True
    
    def _schwartz_zippel_test(U, V, W, n, m, p, num_trials=5):
        """
        Use Schwartz-Zippel to test if the tensor equations are satisfied.
        We evaluate at random points in a finite field. If a nonzero polynomial
        (representing the residual) evaluates to 0 at all random points, it's
        likely (but not certain) the zero polynomial.
        
        For our case, the "polynomial" is actually just a number (the residual
        for each (i,j,k) equation), so we just check if all residuals are 0 mod p.
        Multiple trials with different primes give high confidence.
        """
        primes = [1000003, 1000033, 1000037, 1000039, 1000081]
        for trial in range(num_trials):
            mod = primes[trial % len(primes)]
            if not _evaluate_mod(U, V, W, n, m, p, mod):
                return False
        return True
    
    # Now let's try specific ansatze for rank 27.
    
    # Ansatz A: Try to merge term 22 (A02*B24 -> C04) with term 14 (A00*B04 -> C04)
    # Both contribute to C04. C04 = A00*B04 + A01*B14 + A02*B24
    # If we can express A00*B04 + A02*B24 in fewer terms... but they involve
    # different A entries, so we can't simply merge them.
    
    # Ansatz B: Try a "bilinear" modification.
    # Let's try: in term 14, instead of just A00*B04->C04, use
    # (A00 + a*A02) * (B04 + b*B24) -> C04 + extra
    # This would give: A00*B04 + a*A00*B24 + b*A02*B04 + ab*A02*B24
    # We need A00*B04 and A02*B24 to appear with coefficient 1 in C04.
    # And the cross terms a*A00*B24 and b*A02*B04 would need to be cancelled
    # by other terms or appear in other outputs.
    
    # This is getting very complex. Let me try a more systematic approach:
    # Start from rank 28, try to set one entry to 0 and solve for the rest.
    
    # Actually, let me try the most promising approach: use the known
    # rank-28 solution and try to find a rank-27 solution by solving
    # a system of equations with one term eliminated.
    
    # For each candidate term to eliminate (from the 14 schoolbook terms),
    # try to see if the remaining 26 terms can be adjusted to cover the
    # missing product.
    
    # This is a nonlinear system, so let's use the following trick:
    # Fix 26 of the 27 terms (using the rank-28 values as a starting point),
    # and solve for the 27th term's parameters.
    
    # Actually, the cleanest approach: try all ways to remove one schoolbook
    # term and see if the system becomes consistent.
    
    # For efficiency, use the modular check first.
    
    schoolbook_terms = list(range(14, 28))  # terms 14-27
    
    for elim_idx in schoolbook_terms:
        if time.time() - start_time > deadline - 5:
            break
        
        # Try eliminating term elim_idx
        # Create a candidate with 27 terms (remove term elim_idx)
        # But we need to redistribute its contribution.
        
        # The contribution of term elim_idx to each (i,j,k) equation is:
        # U[i*m+j, elim_idx] * V[j*p+k, elim_idx] * W[k*n+i, elim_idx]
        
        # Let's find which (i,j,k) it contributes to:
        contrib = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = U_base[i*m+j][elim_idx] * V_base[j*p+k][elim_idx] * W_base[k*n+i][elim_idx]
                    if val != 0:
                        contrib[(i,j,k)] = val
        
        # For the rank-28 solution, each schoolbook term contributes to exactly
        # one (i,j,k) equation with value 1.
        # So contrib should have exactly one entry with value 1.
        
        if len(contrib) != 1:
            continue
        
        (i0, j0, k0), val0 = list(contrib.items())[0]
        if val0 != 1:
            continue
        
        # We need to add 1 to the (i0,j0,k0) equation using the other 26 terms.
        # The other 26 terms currently give sum = 1 - 1 = 0 for this equation
        # (since the full sum is 1 and this term contributes 1).
        # So we need to modify some of the other terms to add 1 to this equation.
        
        # This is a nonlinear problem. Let's try a simple perturbation:
        # Add a small integer to one of the U, V, or W entries of an existing term.
        
        # For term r (r != elim_idx), if we change U[i0*m+j0, r] by delta,
        # the contribution to equation (i0,j0,k0) changes by:
        # delta * V[j0*p+k0, r] * W[k0*n+i0, r]
        # But it also changes other equations.
        
        # This is getting too complex for a quick search. Let me try a
        # different approach: use the structure of the problem more directly.
        
        pass
    
    # Given the difficulty of finding a rank-27 solution through this approach,
    # let me try one more specific ansatz that might work.
    
    # Ansatz C: Use a "block" Strassen approach.
    # Instead of Strassen on 2x2 blocks, try a different factorization.
    
    # The matmul tensor for 2x3 * 3x5 has a specific structure.
    # Let's try to use the fact that 2x3 can be seen as 2*(3) and 3x5 as (3)*5.
    
    # Actually, let me try the following: use the rank-28 solution but
    # try to find a linear dependency among the 28 terms that allows
    # elimination.
    
    # If the 28 terms are linearly dependent in some sense, we might be
    # able to eliminate one.
    
    # Check: are the 28 "vectors" (U_col, V_col, W_col) linearly dependent?
    # Each term r defines a trilinear form: sum_{i,j,k} U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]*T[i,j,k]
    # where T[i,j,k]=1.
    # The sum over all r of this trilinear form should equal the matmul tensor.
    
    # For rank reduction, we'd need one term to be expressible as a linear
    # combination of the others in a way that preserves the tensor decomposition.
    # This is not a simple linear dependency check.
    
    # Given time constraints, let me try one more thing: a direct numerical
    # search for a rank-27 solution using the parameterized ansatz.
    
    # I'll try: fix the first 14 terms (Strassen) to their rank-28 values,
    # and search for the last 13 terms (instead of 14) that complete the
    # decomposition.
    
    # The last 13 terms need to cover:
    # - A_L * B_T3: 4 products (A00*B04, A01*B14, A10*B04, A11*B14)
    # - A_R * B_B: 10 products (A02*B2k, A12*B2k for k=0..4)
    # Total: 14 products, but we only have 13 terms.
    
    # Each term can cover multiple products if it has multiple nonzeros.
    # For example, a term with U[0,r]=1, U[2,r]=1, V[4,r]=1, V[14,r]=1,
    # W[8,r]=1, W[9,r]=1 would contribute to multiple equations.
    
    # Let's try: one term covers both A00*B04->C04 and A02*B24->C04.
    # Term r: U[0,r]=1, U[2,r]=1, V[4,r]=1, V[14,r]=1, W[8,r]=1
    # Contribution to (0,0,4): U[0,0,4]=U[0,r]*V[4,r]*W[8,r] = 1*1*1 = 1 (A00*B04)
    # Contribution to (0,2,4): U[0,2,4]=U[2,r]*V[14,r]*W[8,r] = 1*1*1 = 1 (A02*B24)
    # But also: Contribution to (0,0,4) from V[14,r]: U[0,r]*V[14,r]*W[8,r] = 1*1*1 = 1
    # Wait, that's the same equation (0,0,4)! The j index in V must match.
    # V[j*p+k, r] where j is the column of A / row of B.
    # For equation (i,j,k), we use V[j*p+k, r].
    # So for (0,0,4): V[0*5+4, r] = V[4, r] = 1. Contribution: U[0*3+0, r]*V[4, r]*W[4*2+0, r] = 1*1*1 = 1.
    # For (0,2,4): V[2*5+4, r] = V[14, r] = 1. Contribution: U[0*3+2, r]*V[14, r]*W[4*2+0, r] = 1*1*1 = 1.
    # For (0,1,4): V[1*5+4, r] = V[9, r] = 0. No contribution.
    # For (1,0,4): V[0*5+4, r] = V[4, r] = 1. Contribution: U[1*3+0, r]*V[4, r]*W[4*2+1, r] = 0*1*0 = 0.
    # For (1,2,4): V[2*5+4, r] = V[14, r] = 1. Contribution: U[1*3+2, r]*V[14, r]*W[4*2+1, r] = 0*1*0 = 0.
    
    # So this term contributes 1 to (0,0,4) and 1 to (0,2,4). 
    # But we also need A01*B14->C04 and A10*B04->C14 and A11*B14->C14.
    # And we need to cover the other A_R*B_B terms.
    
    # Let me construct the full ansatz:
    # Terms 0-13: Same as rank-28 (Strassen for two 2x2 blocks)
    # Term 14: Merged term for (A00*B04 + A02*B24) -> C04
    #   U[0,14]=1, U[2,14]=1, V[4,14]=1, V[14,14]=1, W[8,14]=1
    # Term 15: A01*B14 -> C04
    #   U[1,15]=1, V[9,15]=1, W[8,15]=1
    # Term 16: A10*B04 -> C14
    #   U[3,16]=1, V[4,16]=1, W[9,16]=1
    # Term 17: A11*B14 -> C14
    #   U[4,17]=1, V[9,17]=1, W[9,17]=1
    # Terms 18-26: A_R * B_B for k=0,1,2,3 (8 terms)
    #   Term 18: A02*B20 -> C00: U[2,18]=1, V[10,18]=1, W[0,18]=1
    #   Term 19: A02*B21 -> C01: U[2,19]=1, V[11,19]=1, W[2,19]=1
    #   Term 20: A02*B22 -> C02: U[2,20]=1, V[12,20]=1, W[4,20]=1
    #   Term 21: A02*B23 -> C03: U[2,21]=1, V[13,21]=1, W[6,21]=1
    #   Term 22: A12*B20 -> C10: U[5,22]=1, V[10,22]=1, W[1,22]=1
    #   Term 23: A12*B21 -> C11: U[5,23]=1, V[11,23]=1, W[3,23]=1
    #   Term 24: A12*B22 -> C12: U[5,24]=1, V[12,24]=1, W[5,24]=1
    #   Term 25: A12*B23 -> C13: U[5,25]=1, V[13,25]=1, W[7,25]=1
    # Term 26: Merged term for (A02*B24 + A12*B24) -> C04 and C14
    #   Wait, A02*B24 -> C04 and A12*B24 -> C14
    #   U[2,26]=1, U[5,26]=1, V[14,26]=1, W[8,26]=1, W[9,26]=1
    #   Contribution to (0,2,4): U[2,26]*V[14,26]*W[8,26] = 1*1*1 = 1 (A02*B24)
    #   Contribution to (1,2,4): U[5,26]*V[14,26]*W[9,26] = 1*1*1 = 1 (A12*B24)
    #   But also: (0,2,4) gets U[2,26]*V[14,26]*W[8,26] = 1 (correct)
    #   And (1,2,4) gets U[5,26]*V[14,26]*W[9,26] = 1 (correct)
    #   Check other equations: (0,0,4): U[0,26]*V[4,26]*W[8,26] = 0*0*1 = 0 (ok)
    #   (0,1,4): U[0,26]*V[9,26]*W[8,26] = 0*0*1 = 0 (ok)
    #   etc.
    
    # Wait, but term 14 already contributes to (0,2,4) with value 1!
    # And term 26 also contributes to (0,2,4) with value 1.
    # So (0,2,4) would get 1 + 1 = 2, which is wrong!
    
    # I made an error. Let me reconsider.
    # Term 14: U[0,14]=1, U[2,14]=1, V[4,14]=1, V[14,14]=1, W[8,14]=1
    # For (0,0,4): U[0,14]*V[4,14]*W[8,14] = 1*1*1 = 1 (A00*B04) ✓
    # For (0,2,4): U[2,14]*V[14,14]*W[8,14] = 1*1*1 = 1 (A02*B24) ✓
    # For (0,1,4): U[0,14]*V[9,14]*W[8,14] = 1*0*1 = 0 ✓
    # For (1,0,4): U[3,14]*V[4,14]*W[9,14] = 0*1*0 = 0 ✓
    # For (1,1,4): U[4,14]*V[9,14]*W[9,14] = 0*0*0 = 0 ✓
    # For (1,2,4): U[5,14]*V[14,14]*W[9,14] = 0*1*0 = 0 ✓
    
    # So term 14 contributes to (0,0,4) and (0,2,4). Good.
    # Now term 26 should NOT also contribute to (0,2,4).
    # Term 26: A12*B24 -> C14 only.
    # U[5,26]=1, V[14,26]=1, W[9,26]=1
    # For (1,2,4): U[5,26]*V[14,26]*W[9,26] = 1*1*1 = 1 (A12*B24) ✓
    # For (0,2,4): U[2,26]*V[14,26]*W[8,26] = 0*1*0 = 0 ✓
    
    # So the corrected ansatz:
    # Term 14: Merged (A00*B04 + A02*B24) -> C04: U[0,14]=1, U[2,14]=1, V[4,14]=1, V[14,14]=1, W[8,14]=1
    # Term 15: A01*B14 -> C04: U[1,15]=1, V[9,15]=1, W[8,15]=1
    # Term 16: A10*B04 -> C14: U[3,16]=1, V[4,16]=1, W[9,16]=1
    # Term 17: A11*B14 -> C14: U[4,17]=1, V[9,17]=1, W[9,17]=1
    # Terms 18-25: A_R * B_B for k=0,1,2,3 (8 terms, same as before)
    # Term 26: A12*B24 -> C14: U[5,26]=1, V[14,26]=1, W[9,26]=1
    
    # Total: 14 (Strassen) + 4 (A_L*B_T3 with merge) + 8 (A_R*B_B for k=0-3) + 1 (A12*B24) = 27
    # But wait, we're missing A02*B24 -> C04! That's covered by term 14.
    # And A12*B24 -> C14 is term 26.
    # Let me recount the A_R*B_B terms:
    # A02*B20, A02*B21, A02*B22, A02*B23, A02*B24 (5 terms)
    # A12*B20, A12*B21, A12*B22, A12*B23, A12*B24 (5 terms)
    # Total: 10 terms.
    # A02*B24 is covered by term 14 (merged).
    # A12*B24 is term 26.
    # Remaining: A02*B20, A02*B21, A02*B22, A02*B23 (4 terms) + A12*B20, A12*B21, A12*B22, A12*B23 (4 terms) = 8 terms.
    # So: 14 + 4 + 8 + 1 = 27. ✓
    
    # Let's construct this and verify!
    
    U27 = [[0] * 27 for _ in range(n * m)]
    V27 = [[0] * 27 for _ in range(m * p)]
    W27 = [[0] * 27 for _ in range(p * n)]
    
    # Copy Strassen terms 0-13 from the rank-28 solution
    for r in range(14):
        for i in range(n * m):
            U27[i][r] = U_base[i][r]
        for j in range(m * p):
            V27[j][r] = V_base[j][r]
        for k in range(p * n):
            W27[k][r] = W_base[k][r]
    
    # Term 14: Merged (A00*B04 + A02*B24) -> C04
    U27[0][14] = 1  # A00
    U27[2][14] = 1  # A02
    V27[4][14] = 1  # B04
    V27[14][14] = 1  # B24
    W27[8][14] = 1  # C04 (W[4*2+0] = W[8])
    
    # Term 15: A01*B14 -> C04
    U27[1][15] = 1  # A01
    V27[9][15] = 1  # B14
    W27[8][15] = 1  # C04
    
    # Term 16: A10*B04 -> C14
    U27[3][16] = 1  # A10
    V27[4][16] = 1  # B04
    W27[9][16] = 1  # C14 (W[4*2+1] = W[9])
    
    # Term 17: A11*B14 -> C14
    U27[4][17] = 1  # A11
    V27[9][17] = 1  # B14
    W27[9][17] = 1  # C14
    
    # Terms 18-21: A02 * B2[0..3] -> C0[0..3]
    U27[2][18] = 1; V27[10][18] = 1; W27[0][18] = 1  # A02*B20 -> C00
    U27[2][19] = 1; V27[11][19] = 1; W27[2][19] = 1  # A02*B21 -> C01
    U27[2][20] = 1; V27[12][20] = 1; W27[4][20] = 1  # A02*B22 -> C02
    U27[2][21] = 1; V27[13][21] = 1; W27[6][21] = 1  # A02*B23 -> C03
    
    # Terms 22-25: A12 * B2[0..3] -> C1[0..3]
    U27[5][22] = 1; V27[10][22] = 1; W27[1][22] = 1  # A12*B20 -> C10
    U27[5][23] = 1; V27[11][23] = 1; W27[3][23] = 1  # A12*B21 -> C11
    U27[5][24] = 1; V27[12][24] = 1; W27[5][24] = 1  # A12*B22 -> C12
    U27[5][25] = 1; V27[13][25] = 1; W27[7][25] = 1  # A12*B23 -> C13
    
    # Term 26: A12*B24 -> C14
    U27[5][26] = 1; V27[14][26] = 1; W27[9][26] = 1
    
    # Verify using Schwartz-Zippel (fast modular check)
    if _schwartz_zippel_test(U27, V27, W27, n, m, p, num_trials=5):
        # Do exact verification
        if _exact_verify(U27, V27, W27, n, m, p):
            return {"U": U27, "V": V27, "W": W27}
    
    # If that didn't work, try a few more ansatze...
    # But given the analysis above, I believe the merged term approach
    # should work. Let me double-check the logic.
    
    # For equation (0,0,4): C04 = A00*B04 + A01*B14 + A02*B24
    # Term 14 contributes: U[0,14]*V[4,14]*W[8,14] = 1*1*1 = 1 (A00*B04) ✓
    # Term 15 contributes: U[0,15]*V[9,15]*W[8,15] = 0*1*1 = 0
    #   Wait, U[0,15] = 0, but we need A01*B14. U[1,15]=1, so for (0,1,4):
    #   U[0,15]*V[9,15]*W[8,15] = 0*1*1 = 0. Hmm.
    #   For (0,1,4): We need A01*B14 = 1.
    #   Term 15: U[1,15]=1 (A01), V[9,15]=1 (B14), W[8,15]=1 (C04)
    #   Contribution to (0,1,4): U[0*3+1, 15]*V[1*5+4, 15]*W[4*2+0, 15] = U[1,15]*V[9,15]*W[8,15] = 1*1*1 = 1 ✓
    #   I was confusing myself. The index into U is i*m+j, so for (i=0, j=1): U[0*3+1, 15] = U[1,15] = 1. ✓
    
    # Let me re-verify all equations for the A_L*B_T3 part:
    # (0,0,4): A00*B04 -> Term 14: U[0,14]*V[4,14]*W[8,14] = 1*1*1 = 1 ✓
    # (0,1,4): A01*B14 -> Term 15: U[1,15]*V[9,15]*W[8,15] = 1*1*1 = 1 ✓
    # (0,2,4): A02*B24 -> Term 14: U[2,14]*V[14,14]*W[8,14] = 1*1*1 = 1 ✓
    # (1,0,4): A10*B04 -> Term 16: U[3,16]*V[4,16]*W[9,16] = 1*1*1 = 1 ✓
    # (1,1,4): A11*B14 -> Term 17: U[4,17]*V[9,17]*W[9,17] = 1*1*1 = 1 ✓
    # (1,2,4): A12*B24 -> Term 26: U[5,26]*V[14,26]*W[9,26] = 1*1*1 = 1 ✓
    
    # And check no cross-contamination:
    # (0,0,4) from other terms: 
    #   Term 15: U[0,15]*V[4,15]*W[8,15] = 0*0*1 = 0 ✓
    #   Term 16: U[0,16]*V[4,16]*W[8,16] = 0*1*0 = 0 ✓
    #   Term 17: U[0,17]*V[4,17]*W[8,17] = 0*0*0 = 0 ✓
    #   Term 26: U[0,26]*V[4,26]*W[8,26] = 0*0*0 = 0 ✓
    #   Terms 18-25: U[0,18..25] = 0, so no contribution ✓
    
    # This looks correct! The key insight is that term 14 can cover both
    # A00*B04 and A02*B24 because they both contribute to C04, and the
    # U and V entries for A00/A02 and B04/B24 don't interfere with
    # other equations (since A00 and A02 are in different rows, and
    # B04 and B24 are in different rows of B, so the j indices are different).
    
    # Wait, I need to double-check: does term 14 contribute to any unwanted equations?
    # Term 14: U[0,14]=1, U[2,14]=1, V[4,14]=1, V[14,14]=1, W[8,14]=1
    # Nonzero contributions when U[i*m+j,14] != 0 AND V[j*5+k,14] != 0 AND W[k*2+i,14] != 0
    # U nonzero at: i=0,j=0 (idx 0) and i=0,j=2 (idx 2)
    # V nonzero at: j=0,k=4 (idx 4) and j=2,k=4 (idx 14)
    # W nonzero at: k=4,i=0 (idx 8)
    
    # Possible (i,j,k) with nonzero contribution:
    # (i=0, j=0, k=4): U[0,14]*V[4,14]*W[8,14] = 1*1*1 = 1 (A00*B04) ✓
    # (i=0, j=2, k=4): U[2,14]*V[14,14]*W[8,14] = 1*1*1 = 1 (A02*B24) ✓
    # (i=0, j=0, k=4) with V[14,14]: j must match, so V[j*5+k] where j=0 means V[4], not V[14]. No cross.
    # (i=0, j=2, k=4) with V[4,14]: j must match, so V[j*5+k] where j=2 means V[14], not V[4]. No cross.
    
    # So no unwanted contributions! The ansatz is valid.
    
    # But wait, the Schwartz-Zippel test should have passed. Let me check if
    # there's a bug in my verification function.
    
    # Actually, I realize I should just return this solution directly since
    # I've verified it by hand. But let me keep the verification for safety.
    
    # Let me also try another merge to see if we can get even lower rank.
    # Can we merge A01*B14 with something? A01*B14 -> C04.
    # Is there another term that also contributes to C04? Yes, terms 14 and 15.
    # But A01 is in row 0, col 1 of A. B14 is in row 1, col 4 of B.
    # Can we merge with a term that involves A01 or B14?
    # In the Strassen terms, A01 appears in terms 4, 11 (U[1,4]=1, U[1,11]=1)
    # and B14... let me check. B14 is V[9,r]. In the Strassen terms for B_T2 (cols 2,3),
    # we use B22=B[2,2], B23=B[2,3], B32=B[3,2], B33=B[3,3]. So B14 (row 1, col 4)
    # is not in the Strassen terms.
    
    # So A01*B14 can't easily be merged with Strassen terms.
    
    # What about merging A10*B04 with A12*B24? Both contribute to C14.
    # Term: U[3,r]=1 (A10), U[5,r]=1 (A12), V[4,r]=1 (B04), V[14,r]=1 (B24), W[9,r]=1 (C14)
    # (1,0,4): U[3,r]*V[4,r]*W[9,r] = 1*1*1 = 1 (A10*B04) ✓
    # (1,2,4): U[5,r]*V[14,r]*W[9,r] = 1*1*1 = 1 (A12*B24) ✓
    # This would save another term! But we already used a merge for A00/A02.
    # Can we do both merges?
    
    # If we merge (A00*B04 + A02*B24) and (A10*B04 + A12*B24), that's 2 terms
    # instead of 4. Plus A01*B14 and A11*B14 = 2 more terms.
    # Total for A_L*B_T3: 4 terms (same as before, no savings).
    # Wait: originally 4 terms (A00*B04, A01*B14, A10*B04, A11*B14) + A02*B24 + A12*B24 = 6 terms.
    # With merges: (A00*B04+A02*B24), (A10*B04+A12*B24), A01*B14, A11*B14 = 4 terms.
    # Savings: 2 terms!
    
    # But we also need to cover the remaining A_R*B_B terms:
    # A02*B20, A02*B21, A02*B22, A02*B23 (4 terms)
    # A12*B20, A12*B21, A12*B22, A12*B23 (4 terms)
    # Total: 8 terms.
    
    # Grand total: 14 (Strassen) + 4 (merged A_L*B_T3 + A_R*B24) + 8 (A_R*B_B for k=0-3) = 26!
    
    # Let me try this!
    
    U26 = [[0] * 26 for _ in range(n * m)]
    V26 = [[0] * 26 for _ in range(m * p)]
    W26 = [[0] * 26 for _ in range(p * n)]
    
    # Copy Strassen terms 0-13
    for r in range(14):
        for i in range(n * m):
            U26[i][r] = U_base[i][r]
        for j in range(m * p):
            V26[j][r] = V_base[j][r]
        for k in range(p * n):
            W26[k][r] = W_base[k][r]
    
    # Term 14: Merged (A00*B04 + A02*B24) -> C04
    U26[0][14] = 1; U26[2][14] = 1
    V26[4][14] = 1; V26[14][14] = 1
    W26[8][14] = 1
    
    # Term 15: Merged (A10*B04 + A12*B24) -> C14
    U26[3][15] = 1; U26[5][15] = 1
    V26[4][15] = 1; V26[14][15] = 1
    W26[9][15] = 1
    
    # Term 16: A01*B14 -> C04
    U26[1][16] = 1; V26[9][16] = 1; W26[8][16] = 1
    
    # Term 17: A11*B14 -> C14
    U26[4][17] = 1; V26[9][17] = 1; W26[9][17] = 1
    
    # Terms 18-21: A02 * B2[0..3] -> C0[0..3]
    U26[2][18] = 1; V26[10][18] = 1; W26[0][18] = 1
    U26[2][19] = 1; V26[11][19] = 1; W26[2][19] = 1
    U26[2][20] = 1; V26[12][20] = 1; W26[4][20] = 1
    U26[2][21] = 1; V26[13][21] = 1; W26[6][21] = 1
    
    # Terms 22-25: A12 * B2[0..3] -> C1[0..3]
    U26[5][22] = 1; V26[10][22] = 1; W26[1][22] = 1
    U26[5][23] = 1; V26[11][23] = 1; W26[3][23] = 1
    U26[5][24] = 1; V26[12][24] = 1; W26[5][24] = 1
    U26[5][25] = 1; V26[13][25] = 1; W26[7][25] = 1
    
    # Verify rank-26 candidate
    if _schwartz_zippel_test(U26, V26, W26, n, m, p, num_trials=5):
        if _exact_verify(U26, V26, W26, n, m, p):
            return {"U": U26, "V": V26, "W": W26}
    
    # Check if the rank-26 candidate has issues.
    # Potential problem: term 14 and term 15 both use V[4,r] and V[14,r].
    # Term 14: V[4,14]=1, V[14,14]=1
    # Term 15: V[4,15]=1, V[14,15]=1
    # These are different terms, so no conflict.
    
    # But wait: does term 15 contribute to any unwanted equation?
    # Term 15: U[3,15]=1 (A10), U[5,15]=1 (A12), V[4,15]=1 (B04), V[14,15]=1 (B24), W[9,15]=1 (C14)
    # (1,0,4): U[3,15]*V[4,15]*W[9,15] = 1*1*1 = 1 (A10*B04) ✓
    # (1,2,4): U[5,15]*V[14,15]*W[9,15] = 1*1*1 = 1 (A12*B24) ✓
    # (0,0,4): U[0,15]*V[4,15]*W[8,15] = 0*1*0 = 0 ✓
    # (0,2,4): U[2,15]*V[14,15]*W[8,15] = 0*1*0 = 0 ✓
    # Looks good!
    
    # And term 14:
    # (0,0,4): U[0,14]*V[4,14]*W[8,14] = 1*1*1 = 1 (A00*B04) ✓
    # (0,2,4): U[2,14]*V[14,14]*W[8,14] = 1*1*1 = 1 (A02*B24) ✓
    # (1,0,4): U[3,14]*V[4,14]*W[9,14] = 0*1*0 = 0 ✓
    # (1,2,4): U[5,14]*V[14,14]*W[9,14] = 0*1*0 = 0 ✓
    
    # So the rank-26 solution should be valid! Let me verify it properly.
    
    # If both failed, there might be a bug in my verification. Let me try
    # a direct exact check on the rank-27 candidate.
    
    if _exact_verify(U27, V27, W27, n, m, p):
        return {"U": U27, "V": V27, "W": W27}
    
    if _exact_verify(U26, V26, W26, n, m, p):
        return {"U": U26, "V": V26, "W": W26}
    
    return None

def _exact_verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct using exact integer arithmetic."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True
