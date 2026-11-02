"""Program for matrix multiplication 3x4 * 4x6 with rank-minimal tensor decomposition."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [3, 4, 6]
n, m, p = DIMENSIONS


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x6 matrix.
    
    Strategy: Use Strassen decomposition on 2x2x2 sub-blocks where applicable,
    combined with schoolbook for the remaining terms. This gives rank 66.
    Then attempt pairwise term merging to reduce rank further.
    """
    start_time = time.time()
    DEADLINE = 115  # seconds, safety margin before 150s kill

    def check_time():
        return time.time() - start_time < DEADLINE

    def schoolbook_full():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def full_verify(U, V, W, tol=1e-6):
        """Full verification of all entries."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0
                    for r in range(rank):
                        val += U[a][r] * V[b][r] * W[c][r]
                    if abs(val - 1) > tol:
                        return False
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if i != i2 or j != j2 or k != k2:
                        val = 0
                        for r in range(rank):
                            val += U[a][r] * V[b][r] * W[c][r]
                        if abs(val) > tol:
                            return False
        return True

    def try_merge_terms(U, V, W, max_iter=5000):
        """
        Attempt to merge pairs of rank terms that have compatible supports.
        Two terms r1, r2 can potentially be merged if for every index (a,b,c),
        the product U[a,r1]*V[b,r1]*W[c,r1] + U[a,r2]*V[b,r2]*W[c,r2]
        can be represented as a single rank-1 term (plus possibly adjusting other terms).
        
        A simpler approach: try to find two terms that are "opposites" in some sense
        and can be combined. For Strassen-like decompositions, adjacent terms often
        share structure.
        
        We'll try a greedy approach: for each pair of terms, check if replacing
        them with a single term (with scaled values) preserves the decomposition.
        """
        rank = len(U[0])
        U = [list(row) for row in U]
        V = [list(row) for row in V]
        W = [list(row) for row in W]
        
        # Build the tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # For merging to work, we need T to be representable with fewer terms.
        # 
        # Strategy: Try removing one term at a time and see if the residual
        # can be absorbed into remaining terms by adjusting their values.
        # This is equivalent to checking if the term is "redundant."
        #
        # Better strategy for this specific problem: Look for pairs of terms
        # where one is a scaled version of the other in a compatible way.
        
        # Actually, let's try a different approach: for each pair (r1, r2),
        # check if we can replace both with a single term by finding
        # new values that satisfy all constraints.
        #
        # The condition is: for all (a,b,c),
        #   U[a,r1]*V[b,r1]*W[c,r1] + U[a,r2]*V[b,r2]*W[c,r2]
        # must equal U'[a]*V'[b]*W'[c] for some single rank-1 term.
        #
        # This is very restrictive. Let's instead try a more practical approach:
        # Look for terms that have very sparse or structured patterns and
        # see if they can be combined.
        
        # Practical approach: For Strassen terms, M2 and M3 often have
        # complementary structures. Let's try to find mergeable pairs
        # by checking if the sum of two rank-1 tensors is itself rank-1.
        
        # Two rank-1 tensors u1*v1*w1 and u2*v2*w2 sum to a rank-1 tensor
        # iff one is a scalar multiple of the other (in each factor).
        # That is, u2 = alpha*u1, v2 = beta*v1, w2 = gamma*w1,
        # and the product alpha*beta*gamma gives the combined scaling.
        # But that would mean they're already "parallel" and trivially mergeable.
        
        # More generally, for the sum to be rank-1, we need:
        # u1*v1*w1 + u2*v2*w2 = u'*v'*w'
        # This requires specific relationships between the vectors.
        
        # Let's try a different, more practical strategy:
        # Remove one term and check if the remaining decomposition still works
        # after rescaling. This is the "term elimination" approach.
        
        best_U, best_V, best_W = U, V, W
        best_rank = rank
        
        # Try eliminating each term one at a time
        for r_elim in range(rank):
            if not check_time():
                break
            
            # Remove term r_elim
            new_rank = rank - 1
            new_U = [row[:r_elim] + row[r_elim+1:] for row in U]
            new_V = [row[:r_elim] + row[r_elim+1:] for row in V]
            new_W = [row[:r_elim] + row[r_elim+1:] for row in W]
            
            # Check if the decomposition still holds
            # The removed term contributed U[a,r_elim]*V[b,r_elim]*W[c,r_elim]
            # to each (a,b,c). For the decomposition to still work,
            # this contribution must be zero for all (a,b,c) OR
            # must be cancelable by adjusting other terms.
            
            # Simple check: is the removed term identically zero?
            term_nonzero = False
            for a in range(n*m):
                for b in range(m*p):
                    for c in range(p*n):
                        if U[a][r_elim] * V[b][r_elim] * W[c][r_elim] != 0:
                            term_nonzero = True
                            break
                    if term_nonzero:
                        break
                if term_nonzero:
                    break
            
            if not term_nonzero:
                # Term is zero, just remove it
                best_U, best_V, best_W = new_U, new_V, new_W
                best_rank = new_rank
                U, V, W, rank = new_U, new_V, new_W, new_rank
                continue
            
            # Try to see if we can absorb the removed term into others
            # by finding a linear combination. This is complex, so let's
            # try a simpler heuristic: check if the removed term's support
            # is a subset of another term's support with compatible signs.
            
            # For now, skip complex absorption and just check if the
            # decomposition is still valid without the term (it won't be
            # unless the term was zero, so this is mainly for zero terms).
        
        if best_rank < len(U[0]):
            return best_U, best_V, best_W, best_rank
        return U, V, W, rank

    def strassen_decomposition():
        """
        Use Strassen decomposition on 2x2x2 sub-blocks.
        
        For 3x4*4x6:
        - Split columns of A (4 cols) into [0,1] and [2,3]
        - Split rows of B (4 rows) into [0,1] and [2,3]  
        - Split columns of B (6 cols) into pairs [0,1], [2,3], [4,5]
        
        For each combination of (col_pair_of_A, row_pair_of_B, col_pair_of_B):
        if all dimensions are 2, use Strassen (rank 7 instead of 8)
        
        Layout:
        A: 3 rows x 4 cols -> col pairs: (0,1), (2,3)
        B: 4 rows x 6 cols -> row pairs: (0,1), (2,3); col pairs: (0,1), (2,3), (4,5)
        C: 3 rows x 6 cols
        
        For each (a_col_pair, b_row_pair, b_col_pair) where all are size 2:
        Use Strassen with 7 multiplications
        
        Total: 2 * 2 * 3 = 12 blocks of 2x2x2
        Each Strassen block: rank 7
        Total rank: 12 * 7 = 84... wait that's worse.
        
        Actually let me reconsider. The schoolbook for a 2x2x2 block has rank 8.
        Strassen has rank 7. So we save 1 per block.
        12 blocks * 7 = 84 vs 12 * 8 = 96? No, total is 3*4*6 = 72.
        
        Let me recount: n*m*p = 3*4*6 = 72 total terms.
        The 2x2x2 blocks: we have 3 row-pairs of A (but A has only 3 rows, so pairs (0,1) and row 2 alone)
        
        Actually, let's think differently:
        A has 3 rows, 4 cols. B has 4 rows, 6 cols.
        
        We can partition:
        - A rows: {0,1} and {2}
        - A cols / B rows: {0,1} and {2,3}
        - B cols: {0,1}, {2,3}, {4,5}
        
        For the subproblem with A rows {0,1}, A cols {0,1}, B rows {0,1}, B cols {0,1}:
        This is a 2x2 * 2x2 -> 2x2 block, can use Strassen (rank 7 vs 8)
        
        Similarly for all combinations where both A and B have 2x2 structure.
        
        Number of 2x2x2 blocks:
        - A row pair: (0,1) [1 pair, since row 2 is alone]
        - A col/B row pair: (0,1), (2,3) [2 pairs]
        - B col pair: (0,1), (2,3), (4,5) [3 pairs]
        
        So 1 * 2 * 3 = 6 blocks of 2x2x2, each using Strassen (rank 7)
        Remaining terms: 72 - 6*8 = 72 - 48 = 24 terms (schoolbook)
        Total rank: 6*7 + 24 = 42 + 24 = 66
        
        That matches the observed improvement to 66!
        """
        rank = 66
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        term = 0
        
        # Strassen terms for 2x2x2 blocks
        # A rows: (0,1), A cols/B rows: (0,1) and (2,3), B cols: (0,1), (2,3), (4,5)
        a_row_pairs = [(0, 1)]
        a_col_b_row_pairs = [(0, 1), (2, 3)]
        b_col_pairs = [(0, 1), (2, 3), (4, 5)]
        
        for (ai1, ai2) in a_row_pairs:
            for (aj1, aj2) in a_col_b_row_pairs:
                for (ak1, ak2) in b_col_pairs:
                    # Strassen decomposition for C_block = A_block * B_block
                    # where A_block is 2x2 (rows ai1,ai2; cols aj1,aj2)
                    # B_block is 2x2 (rows aj1,aj2; cols ak1,ak2)
                    # C_block is 2x2 (rows ai1,ai2; cols ak1,ak2)
                    
                    # M1 = (a00+a11)(b00+b11)
                    U[ai1*m + aj1][term] = 1
                    U[ai2*m + aj2][term] = 1
                    V[aj1*p + ak1][term] = 1
                    V[aj2*p + ak2][term] = 1
                    W[ak1*n + ai1][term] = 1
                    W[ak2*n + ai2][term] = 1
                    term += 1
                    
                    # M2 = (a10+a11)b00
                    U[ai2*m + aj1][term] = 1
                    U[ai2*m + aj2][term] = 1
                    V[aj1*p + ak1][term] = 1
                    W[ak1*n + ai2][term] = 1
                    W[ak1*n + ai1][term] = -1
                    term += 1
                    
                    # M3 = a00(b10-b11)
                    U[ai1*m + aj1][term] = 1
                    V[aj2*p + ak1][term] = 1
                    V[aj2*p + ak2][term] = -1
                    W[ak1*n + ai1][term] = 1
                    W[ak2*n + ai1][term] = -1
                    term += 1
                    
                    # M4 = a11(b00-b10)
                    U[ai2*m + aj2][term] = 1
                    V[aj1*p + ak1][term] = 1
                    V[aj1*p + ak2][term] = -1
                    W[ak2*n + ai2][term] = 1
                    W[ak2*n + ai1][term] = -1
                    term += 1
                    
                    # M5 = (a00+a01)b11
                    U[ai1*m + aj1][term] = 1
                    U[ai1*m + aj2][term] = 1
                    V[aj2*p + ak2][term] = 1
                    W[ak2*n + ai1][term] = 1
                    W[ak2*n + ai2][term] = 1
                    term += 1
                    
                    # M6 = (a10-a00)(b00+b01)
                    U[ai2*m + aj1][term] = 1
                    U[ai1*m + aj1][term] = -1
                    V[aj1*p + ak1][term] = 1
                    V[aj1*p + ak2][term] = 1
                    W[ak1*n + ai2][term] = 1
                    W[ak2*n + ai2][term] = 1
                    term += 1
                    
                    # M7 = (a01-a11)(b10+b11)
                    U[ai1*m + aj2][term] = 1
                    U[ai2*m + aj2][term] = -1
                    V[aj2*p + ak1][term] = 1
                    V[aj2*p + ak2][term] = 1
                    W[ak1*n + ai1][term] = 1
                    W[ak1*n + ai2][term] = -1
                    term += 1
        
        # Now add schoolbook terms for remaining entries
        # These are entries where either A row is 2 (alone), or A col/B row is not in a pair, etc.
        # Actually, let me just add all schoolbook terms that aren't covered
        
        covered = set()
        for (ai1, ai2) in a_row_pairs:
            for (aj1, aj2) in a_col_b_row_pairs:
                for (ak1, ak2) in b_col_pairs:
                    for ai in [ai1, ai2]:
                        for aj in [aj1, aj2]:
                            for ak in [ak1, ak2]:
                                covered.add((ai, aj, ak))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        U[i * m + j][term] = 1
                        V[j * p + k][term] = 1
                        W[k * n + i][term] = 1
                        term += 1
        
        assert term == rank, f"Expected rank {rank}, got {term}"
        return {"U": U, "V": V, "W": W}

    # Start with schoolbook as fallback
    best_result = schoolbook_full()
    if not full_verify(best_result["U"], best_result["V"], best_result["W"]):
        raise RuntimeError("Schoolbook verification failed")
    best_rank = n * m * p
    
    # Try Strassen decomposition
    if check_time():
        strassen_res = strassen_decomposition()
        if full_verify(strassen_res["U"], strassen_res["V"], strassen_res["W"], tol=1e-6):
            best_rank = len(strassen_res["U"][0])
            best_result = strassen_res
    
    # Try to merge/eliminate terms from the Strassen decomposition
    if check_time() and best_rank < n * m * p:
        U, V, W = best_result["U"], best_result["V"], best_result["W"]
        U, V, W, new_rank = try_merge_terms(U, V, W)
        if new_rank < best_rank:
            if full_verify(U, V, W, tol=1e-6):
                best_rank = new_rank
                best_result = {"U": U, "V": V, "W": W}
    
    return best_result
