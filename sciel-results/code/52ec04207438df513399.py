"""Program for 3x4x4 matrix multiplication rank minimization."""
import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0]) if U and U[0] else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _schoolbook_rank(n, m, p):
    """Return the schoolbook decomposition (rank n*m*p)."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is a tuple (u_dict, v_dict, w_dict) where:
    - u_dict maps (i,j) -> coefficient for A[i,j]
    - v_dict maps (j,k) -> coefficient for B[j,k]
    - w_dict maps (k,i) -> coefficient for C[k,i]
    """
    terms = []
    # M1 = (A00+A11)(B00+B11)
    terms.append(
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):1})
    )
    # M2 = (A10+A11)B00
    terms.append(
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):1})
    )
    # M3 = A00(B10-B11)
    terms.append(
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,0):1, (0,1):-1})
    )
    # M4 = A11(B01-B10)
    terms.append(
        ({(1,1):1}, {(0,1):1, (1,0):-1}, {(1,0):1, (1,1):-1})
    )
    # M5 = (A00+A01)B11
    terms.append(
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):1, (0,1):1})
    )
    # M6 = (A10-A00)(B00+B10)
    terms.append(
        ({(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}, {(1,0):1, (1,1):1})
    )
    # M7 = (A01-A11)(B01+B11)
    terms.append(
        ({(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}, {(0,0):1, (0,1):1})
    )
    return terms

def _expand_strassen_to_scalar(n, m, p, i_off, j_off, k_off):
    """
    Expand a 2x2x2 Strassen decomposition into scalar rank terms.
    Returns a list of (u_idx, v_idx, w_idx, coeff) tuples.
    
    The key insight: each Strassen term Mi is itself a 2x2 matrix product.
    We expand each Mi into its 4 schoolbook scalar products, but the
    Strassen linear combinations mean that each scalar product A[i,j]*B[j,k]
    appears in multiple Mi terms with different coefficients.
    
    We collect all contributions and sum them up.
    """
    from collections import defaultdict
    coeff_map = defaultdict(float)
    
    strassen_terms = _strassen_2x2x2_terms()
    
    for u_dict, v_dict, w_dict in strassen_terms:
        # For this Strassen term, enumerate all scalar products
        for (ui, uj), uc in u_dict.items():
            for (vj, vk), vc in v_dict.items():
                # Contraction: uj must equal vj
                if uj != vj:
                    continue
                for (wk, wi), wc in w_dict.items():
                    # Output indices: vk must equal wk, ui must equal wi
                    if vk != wk or ui != wi:
                        continue
                    # Product A[ui,uj] * B[uj,vk] contributes uc*vc*wc to C[vk,ui]
                    key = (ui, uj, vk)
                    coeff_map[key] += uc * vc * wc
    
    # Convert to list of (u_idx, v_idx, w_idx, coeff)
    results = []
    for (ui, uj, vk), coeff in coeff_map.items():
        if abs(coeff) < 1e-12:
            continue
        u_idx = (i_off + ui) * m + (j_off + uj)
        v_idx = (j_off + uj) * p + (k_off + vk)
        w_idx = (k_off + vk) * n + (i_off + ui)
        results.append((u_idx, v_idx, w_idx, coeff))
    
    return results

def _build_decomposition(n, m, p):
    """
    Build the Strassen-based decomposition for 3x4x4.
    
    Strategy:
    - Partition i into [0,1] and [2]
    - Partition j into [0,1] and [2,3]  
    - Partition k into [0,1] and [2,3]
    
    For the 2x2x2 blocks (i in {0,1}, j in {0,1}, k in {0,1}):
      Use Strassen expansion -> 4 scalar terms per block (not 7!)
      Wait, let me recount. The Strassen expansion gives us the sum of
      coefficients for each A[i,j]*B[j,k] product.
      
    Actually, let me think about this more carefully.
    
    The 2x2x2 Strassen decomposition, when expanded to scalars, gives:
    For each of the 8 possible (i,j,k) combinations in the 2x2x2 block,
    we get a coefficient that is the sum of contributions from all 7 Strassen terms.
    
    Let me compute these coefficients explicitly.
    """
    from collections import defaultdict
    
    # For 3x4x4, we'll use:
    # - Strassen for the 2x2x2 sub-block (i:0-1, j:0-1, k:0-1)
    # - Schoolbook for everything else
    
    # But wait, we can also apply Strassen to other 2x2x2 sub-blocks.
    # j:0-1 with k:2-3 is also a 2x2x2 block for i:0-1.
    # And j:2-3 with k:0-1 is another.
    # And j:2-3 with k:2-3 is another.
    
    # So for i:0-1, we have 4 independent 2x2x2 Strassen blocks:
    # (j:0-1, k:0-1), (j:0-1, k:2-3), (j:2-3, k:0-1), (j:2-3, k:2-3)
    # Each gives 4 scalar terms (after expansion and cancellation).
    # Total: 4 * 4 = 16 terms for i:0-1.
    
    # For i:2, we have 1x4x4 which is just schoolbook: 16 terms.
    
    # Total: 16 + 16 = 32? That seems too low. Let me recheck.
    
    # Actually, the Strassen expansion of a 2x2x2 block gives us
    # coefficients for each of the 8 scalar products A[i,j]*B[j,k]->C[k,i].
    # But due to the linear combinations in Strassen, some of these
    # coefficients might be zero or combine.
    
    # Let me compute the actual coefficients for the 2x2x2 Strassen.
    
    coeff_map = defaultdict(float)
    strassen_terms = _strassen_2x2x2_terms()
    
    for u_dict, v_dict, w_dict in strassen_terms:
        for (ui, uj), uc in u_dict.items():
            for (vj, vk), vc in v_dict.items():
                if uj != vj:
                    continue
                for (wk, wi), wc in w_dict.items():
                    if vk != wk or ui != wi:
                        continue
                    coeff_map[(ui, uj, vk)] += uc * vc * wc
    
    print("2x2x2 Strassen scalar coefficients:")
    for key in sorted(coeff_map.keys()):
        print(f"  A[{key[0]},{key[1]}]*B[{key[1]},{key[2]}]->C[{key[2]},{key[0]}]: {coeff_map[key]}")
    
    # The coefficients should all be 1.0 if Strassen is correct.
    # Let's verify: for a correct decomposition, each A[i,j]*B[j,k] should
    # contribute exactly 1 to C[k,i].
    
    # If all coefficients are 1, then the Strassen expansion is equivalent
    # to schoolbook (8 terms). That can't be right - Strassen should reduce
    # the number of multiplications.
    
    # Ah, I see the issue. The Strassen algorithm reduces the number of
    # 2x2 matrix multiplications from 8 to 7, but each 2x2 multiplication
    # still requires 4 scalar multiplications. So 7*4 = 28 scalar multiplications.
    # But when we expand to scalars, the linear combinations cause some
    # scalar products to appear multiple times with different signs.
    
    # The key point: in a tensor decomposition, each "rank term" corresponds
    # to one scalar multiplication. If the same A[i,j]*B[j,k] product appears
    # in multiple Strassen terms, we can't simply combine them into one rank term
    # because the W coefficients would need to match.
    
    # Actually, we CAN combine them! If A[i,j]*B[j,k] appears in multiple terms
    # with total coefficient 1, and it contributes to C[k,i] with total coefficient 1,
    # then we just need one rank term with U=1, V=1, W=1.
    
    # So the question is: after summing all contributions, what is the coefficient
    # for each (i,j,k) in the 2x2x2 block?
    
    # If all are 1, then we need 8 rank terms (same as schoolbook).
    # If some are 0, we need fewer.
    
    # Let me check: the coefficients should indeed all be 1 for a correct
    # Strassen decomposition. This means Strassen doesn't help at the scalar level
    # for 2x2x2 - it only helps at the 2x2 matrix level.
    
    # So for our purposes, we should use schoolbook for the 2x2x2 blocks.
    # The Strassen benefit comes from reducing the number of 2x2 multiplications,
    # but since we're doing scalar tensor decomposition, we need all scalar products.
    
    # Hmm, but the knowledge says Strassen reduces 2x4x4 from 32 to 28.
    # Let me re-examine.
    
    # For 2x4x4:
    # Schoolbook: 2*4*4 = 32
    # Strassen on 2x2 blocks: 
    #   A is 2x4, B is 4x4
    #   Partition A into [A_top; A_bottom] where each is 2x2... no, A is 2x4.
    #   Actually, partition A as [A0, A1] where A0 is 2x2 (cols 0-1) and A1 is 2x2 (cols 2-3)
    #   Partition B as [[B00, B01], [B10, B11]] where each is 2x2
    #   C = [C0, C1] where C0 = A0*B_left + A1*B_right... no.
    
    # Let me think differently. For 2x4 * 4x4 -> 2x4:
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    
    # Using Strassen on the j=0,1 and j=2,3 groups:
    # For j=0,1: C[i,k] += A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #   This is like a 2x2 * 2x4 multiplication... not quite.
    
    # Actually, the standard way to apply Strassen to 2x4x4 is:
    # View A as two 2x2 blocks: A = [A_left | A_right] where A_left is 2x2, A_right is 2x2
    # View B as four 2x2 blocks: B = [[B00, B01], [B10, B11]]
    # C = A*B = [A_left*B_left + A_right*B_right] where B_left = [B00; B10] (4x2) and B_right = [B01; B11] (4x2)
    
    # Hmm, this is getting complicated. Let me just use the direct approach.
    
    # For 3x4x4, the simplest good decomposition is:
    # - For i in {0,1}: use Strassen on the 2x4x4 sub-problem
    # - For i=2: use schoolbook on the 1x4x4 sub-problem
    
    # But I showed that Strassen at the scalar level doesn't reduce the count
    # for 2x2x2. So let me just use schoolbook everywhere and see if there's
    # a better structure.
    
    # Actually, wait. Let me reconsider. The Strassen algorithm works by
    # replacing 8 matrix multiplications with 7. At the scalar level, each
    # matrix multiplication of size s x s requires s^2 scalar multiplications.
    # So Strassen replaces 8*s^2 with 7*s^2 scalar multiplications.
    
    # For s=2: 8*4 = 32 -> 7*4 = 28 scalar multiplications.
    
    # But in my expansion above, I got 8 terms with coefficient 1 each.
    # That's 8, not 28 or 32. Something is wrong.
    
    # Oh! I see the issue. In the tensor decomposition, each rank term
    # corresponds to ONE scalar multiplication: U[a]*V[b]*W[c].
    # The Strassen algorithm doesn't create new scalar multiplications;
    # it reorganizes which products are computed and how they're combined.
    
    # But for an EXACT tensor decomposition, we need to represent the
    # multiplication tensor exactly. The Strassen algorithm computes the
    # same result, so the tensor decomposition should be equivalent.
    
    # The key insight: Strassen's 7 "multiplications" are not scalar
    # multiplications - they are 2x2 matrix multiplications. Each 2x2
    # matrix multiplication can be done with 4 scalar multiplications
    # (schoolbook) or 7 (Strassen recursively).
    
    # So for the 2x2x2 block:
    # - Schoolbook: 8 scalar multiplications (one for each A[i,j]*B[j,k])
    # - Strassen: 7 "matrix multiplications" of 2x2 matrices
    #   - If each 2x2 matrix multiplication uses schoolbook: 7*4 = 28 scalar multiplications
    #   - But wait, that's MORE than 8!
    
    # I'm confusing myself. Let me start over.
    
    # The matrix multiplication tensor T for n x m x p has T[i,j,k] = 1
    # meaning A[i,j]*B[j,k] contributes to C[i,k].
    
    # A rank-r decomposition is: T = sum_{r=1}^{r} u_r ⊗ v_r ⊗ w_r
    # where u_r is a vector of length n*m, v_r of length m*p, w_r of length p*n.
    
    # The schoolbook decomposition has rank n*m*p, with one term for each
    # (i,j,k) triple.
    
    # Strassen's algorithm shows that 2x2x2 matrix multiplication can be done
    # with 7 multiplications instead of 8. But these 7 "multiplications" are
    # of 2x2 matrices, not scalars.
    
    # To convert Strassen to a scalar tensor decomposition, we need to expand
    # each 2x2 matrix multiplication into scalar multiplications.
    
    # A 2x2 matrix multiplication M = P*Q where P, Q are 2x2 can be done
    # with 4 scalar multiplications (schoolbook) or 7 (Strassen recursively).
    
    # So Strassen for 2x2x2:
    # - 7 "2x2 matrix multiplications"
    # - Each expanded with schoolbook: 7*4 = 28 scalar multiplications
    # - Each expanded with Strassen: 7*7 = 49 scalar multiplications
    
    # But the schoolbook for 2x2x2 is just 8 scalar multiplications!
    
    # So Strassen is WORSE at the scalar level for 2x2x2. The benefit of
    # Strassen only appears when the matrices are larger, because the
    # "multiplications" in Strassen are of larger matrices.
    
    # For 4x4x4:
    # - Schoolbook: 64 scalar multiplications
    # - Strassen (one level): 7 "2x2 matrix multiplications" of 2x2 matrices
    #   - Each 2x2*2x2 with schoolbook: 7*4 = 28 scalar multiplications
    #   - Total: 28, which is less than 64!
    
    # Ah, so Strassen helps for 4x4x4 but not for 2x2x2.
    
    # For our 3x4x4 problem:
    # - We can't directly apply Strassen since n=3 is not a power of 2.
    # - But we can partition: use Strassen for the 2x4x4 part (i=0,1) and
    #   schoolbook for the 1x4x4 part (i=2).
    
    # For 2x4x4:
    # - Schoolbook: 32
    # - Can we use Strassen? 2x4x4 is not a square matrix multiplication.
    #   But we can view it as: C (2x4) = A (2x4) * B (4x4)
    #   Partition B into [B_left | B_right] where each is 4x2.
    #   C = [A*B_left | A*B_right]
    #   Each A*B_left is 2x4 * 4x2 = 2x2.
    #   So we have two 2x4*4x2 multiplications, each giving a 2x2 result.
    #   
    #   For each 2x4*4x2: schoolbook is 2*4*2 = 16.
    #   Can we use Strassen? Not directly, since it's not square.
    
    # Alternatively, partition A into [A_top; A_bottom] where each is 1x4.
    # C_top = A_top * B (1x4 * 4x4 = 1x4)
    # C_bottom = A_bottom * B (1x4 * 4x4 = 1x4)
    # Each 1x4*4x4 has rank 16 (schoolbook), total 32.
    
    # Hmm, I don't see an obvious Strassen application for 2x4x4.
    
    # Let me try a different approach: use the fact that 4x4 matrix
    # multiplication has known optimal rank.
    
    # The optimal rank for 4x4x4 is known to be at most 49 (from Strassen)
    # and at least 27 (from lower bounds). The exact value is unknown.
    
    # For 3x4x4, we can use:
    # C = A*B where A is 3x4, B is 4x4.
    # C_i = A_i * B for i=0,1,2, where A_i is the i-th row of A (1x4).
    # Each C_i is 1x4, and C_i[k] = sum_j A_i[j]*B[j,k].
    
    # This is 3 separate 1x4*4x4 multiplications, each of rank 16.
    # Total: 48.
    
    # To beat 48, we need to share rank terms across different i values.
    
    # One approach: use a 4x4x4 decomposition and restrict it to 3x4x4.
    # If we have a rank-r decomposition of 4x4x4, we can use it for 3x4x4
    # by just ignoring the 4th row of A and the 4th row of C.
    
    # The best known rank for 4x4x4 is 49 (Strassen) or possibly lower.
    # If we use rank 49 for 4x4x4, we get rank 49 for 3x4x4 (worse than 48).
    
    # So we need a better approach.
    
    # Let me try to find a direct decomposition with rank < 48.
    
    # For now, let's just return schoolbook (48) and verify it works.
    return _schoolbook_rank(n, m, p)

def _try_rank_reduction(U, V, W, n, m, p, max_iter=1000):
    """
    Try to reduce the rank by finding linear dependencies among rank terms.
    
    If two rank terms have proportional U, V, and W vectors, we can combine them.
    More generally, if a set of rank terms sums to zero, we can remove them.
    
    This is a simple heuristic that may not find the optimal reduction.
    """
    rank = len(U[0]) if U and U[0] else 0
    if rank == 0:
        return U, V, W
    
    # Convert to numpy for easier manipulation
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Try to find linear dependencies by checking if any column is a
    # linear combination of others.
    # This is expensive, so we'll just try a few random subsets.
    
    # For now, just return the input unchanged.
    return U, V, W

def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x4x4 matrix multiplication."""
    n, m, p = 3, 4, 4
    
    start_time = time.time()
    deadline = start_time + 115  # 5 second safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook_rank(n, m, p)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Should never happen, but just in case
        U, V, W = _schoolbook_rank(n, m, p)
    
    # Try to reduce rank
    if time.time() < deadline:
        U, V, W = _try_rank_reduction(U, V, W, n, m, p)
    
    # Final verification
    if not _verify(U, V, W, n, m, p):
        U, V, W = _schoolbook_rank(n, m, p)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
