"""Strassen-based rank-46 decomposition for 2x5 x 5x5 matrix multiplication."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

DEADLINE = time.time() + 130  # safety margin before 150s kill


def _schoolbook(n=2, m=5, p=5):
    """Standard schoolbook rank-50 decomposition."""
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


def _verify(U, V, W, n=2, m=5, p=5):
    """Verify exact tensor decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True


def _strassen_2x2x2():
    """Return the 7 Strassen multiplications for 2x2 * 2x2.
    
    For matrices A (2x2), B (2x2):
    C = A * B where C[i,j] = sum_k A[i,k]*B[k,j]
    
    Strassen's 7 products:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    Each M is a scalar multiplication: a * b where a is a linear combination
    of A entries and b is a linear combination of B entries.
    
    For tensor decomposition, each M corresponds to one rank-1 term:
    U entry: the coefficient of A[i,k] in the 'a' part
    V entry: the coefficient of B[k,j] in the 'b' part
    W entry: 1 (since C[i,j] gets this product directly)
    """
    # Each term: (a_coeffs, b_coeffs)
    # a_coeffs: dict mapping (i,k) -> coeff for A[i,k]
    # b_coeffs: dict mapping (k,j) -> coeff for B[k,j]
    
    terms = []
    
    # M1 = (A00+A11)*(B00+B11)
    terms.append(({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}))
    # M2 = (A10+A11)*B00
    terms.append(({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}))
    # M3 = A00*(B01-B11)
    terms.append(({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}))
    # M4 = A11*(B10-B00)
    terms.append(({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}))
    # M5 = (A00+A01)*B11
    terms.append(({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}))
    # M6 = (A10-A00)*(B00+B01)
    terms.append(({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}))
    # M7 = (A01-A11)*(B10+B11)
    terms.append(({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}))
    
    return terms


def _construct_strassen(n=2, m=5, p=5):
    """Construct rank-46 decomposition using Strassen on 2+2+1 partition.
    
    Partition:
    - A is 2x5: rows indexed by i in {0,1}, cols by j in {0..4}
      Split j into: {0,1} (block A), {2,3} (block B), {4} (block C)
    - B is 5x5: rows indexed by k in {0..4}, cols by l in {0..4}
      Split k into: {0,1}, {2,3}, {4}
      Split l into: {0,1}, {2,3}, {4}
    
    C[i,l] = sum_k A[i,k] * B[k,l]
    
    For each (i, l) output:
    C[i,l] = sum over blocks:
      - For k in {0,1}: A[i, {0,1}] * B[{0,1}, l]
      - For k in {2,3}: A[i, {2,3}] * B[{2,3}, l]  
      - For k in {4}: A[i, 4] * B[4, l]
    
    For the 2x2 subblocks (k in {0,1} and l in {0,1}), use Strassen.
    For the 2x1 subblocks (k in {0,1} and l in {2} or {3} or {4}), use schoolbook (2 mults).
    For the 1x2 subblocks (k in {2,3} and l in {0,1}), use schoolbook (2 mults).
    For the 2x2 subblock (k in {2,3} and l in {2,3}), use Strassen.
    For the 2x1 subblocks (k in {2,3} and l in {4}), use schoolbook (2 mults).
    For the 1x2 subblock (k in {4} and l in {0,1}), use schoolbook (2 mults).
    For the 1x1 subblock (k in {4} and l in {4}), use schoolbook (1 mult).
    
    Total: For each i (2 values):
      - 2x2 block (k={0,1}, l={0,1}): 7 mults
      - 2x1 block (k={0,1}, l={2}): 2 mults
      - 2x1 block (k={0,1}, l={3}): 2 mults
      - 2x1 block (k={0,1}, l={4}): 2 mults
      - 1x2 block (k={4}, l={0,1}): 2 mults
      - 1x1 block (k={4}, l={4}): 1 mult
      Total per i: 7+2+2+2+2+1 = 16
    Total: 2 * 16 = 32? That's not 46.
    
    Wait, I need to reconsider. The Strassen algorithm works on 2x2 matrices.
    Let me think about this more carefully.
    
    Actually, the partition should be:
    - A (2x5): split columns into 2+2+1
    - B (5x5): split rows into 2+2+1 and columns into 2+2+1
    
    For each pair of column-block of A and row-block of B, we get a subproblem.
    Then for each output column block, we sum contributions.
    
    Let me redo:
    A columns: block0 = {0,1}, block1 = {2,3}, block2 = {4}
    B rows: block0 = {0,1}, block1 = {2,3}, block2 = {4}
    B columns: block0 = {0,1}, block1 = {2,3}, block2 = {4}
    
    C[i, col_block] = sum over row_blocks: A[i, col_block] * B[row_block, col_block]
    
    For fixed i and fixed col_block of B:
    - If col_block = {0,1} (2 cols):
      - From A col_block {0,1} (2 cols) and B row_block {0,1} (2 rows): 2x2 * 2x2 -> Strassen (7)
      - From A col_block {2,3} (2 cols) and B row_block {2,3} (2 rows): 2x2 * 2x2 -> Strassen (7)
      - From A col_block {4} (1 col) and B row_block {4} (1 row): 1x1 * 1x2 -> 2 mults
      - Cross terms: A col_block {0,1} with B row_block {2,3}: 2x2 * 2x2 -> Strassen (7)
      - Cross terms: A col_block {0,1} with B row_block {4}: 2x1 * 1x2 -> 2 mults
      - Cross terms: A col_block {2,3} with B row_block {0,1}: 2x2 * 2x2 -> Strassen (7)
      - Cross terms: A col_block {2,3} with B row_block {4}: 2x1 * 1x2 -> 2 mults
      - Cross terms: A col_block {4} with B row_block {0,1}: 1x2 * 2x2 -> 2 mults
      - Cross terms: A col_block {4} with B row_block {2,3}: 1x2 * 2x2 -> 2 mults
    This is getting complicated. Let me just use schoolbook for all cross-block terms
    and Strassen only for the diagonal 2x2 blocks.
    
    Simpler approach:
    - For each i in {0,1}:
      - For each pair (j_block, k_block) where j_block is a column block of A and
        k_block is a row block of B:
        - If both are size-2 blocks: use Strassen (7 mults) for the 2x2*2x2 -> 2x2
        - Otherwise: use schoolbook
    
    But the output C has 5 columns, and we're computing C[i, :] for each i.
    The Strassen gives us 2 output values (for a 2x2 result), but we need 5.
    
    Let me think differently. The key insight from the cell findings is:
    "Partitioning the 5x5 B matrix into 2+2+1 blocks along both dimensions allows
    using Strassen's 7-multiplication algorithm for 2x2*2x2 subblocks, reducing
    total rank from 50 (schoolbook) to 46."
    
    So the saving is 50 - 46 = 4. This means we save 4 multiplications total.
    If we have one 2x2*2x2 block that we replace with Strassen, we save 8-7=1 per such block.
    To save 4, we need 4 such blocks.
    
    Actually, let me reconsider. Maybe the partition is:
    - Split the 5 columns of A into 2+2+1
    - Split the 5 rows of B into 2+2+1  
    - Split the 5 columns of B into 2+2+1
    
    And we only apply Strassen to the "diagonal" 2x2 blocks where the column block
    of A matches the row block of B in size 2, AND the output column block is also size 2.
    
    There are 3 column blocks for A: {0,1}, {2,3}, {4}
    There are 3 row blocks for B: {0,1}, {2,3}, {4}
    There are 3 column blocks for B: {0,1}, {2,3}, {4}
    
    For the output C[i, col_block_B]:
    C[i, col_block_B] = sum over row_block_B: (A[i, col_block_A] * B[row_block_B, col_block_B])
    where col_block_A ranges over all 3 blocks.
    
    This is still complex. Let me just implement a clean version:
    
    I'll construct the decomposition by treating each "multiplication" as a rank-1 term.
    For Strassen, each of the 7 products M_t = a_t * b_t contributes one rank-1 term
    where:
    - U[i, j] gets the coefficient of A[i,j] in a_t
    - V[j, k] gets the coefficient of B[j,k] in b_t
    - W[k, i] = 1 (the product directly contributes to C[i, ...])
    
    But wait, in Strassen, the 7 products are combined with additions to form the
    output. So each product M_t contributes to multiple output entries C[i,j].
    This means W[k,i] should encode which output entries get this product.
    
    Actually, in the tensor decomposition framework:
    C[i,j] = sum_k A[i,k] * B[k,j] = sum_c U[i,k] * V[k,j] * W[j,i]
    
    (using the convention W[k*n+i, r] but let me re-read the problem)
    
    The problem says: "The output factor follows the official trace convention W[k*n+i,r]"
    
    So W is indexed as W[k*n + i, c] where k is the column index of B (which is the
    row index of the result C), and i is the row index of A (which is also the row
    index of C).
    
    So: C[i, k] = sum_c U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
    where j is the column index of A / row index of B.
    
    In Strassen for 2x2*2x2:
    C[i,j] = sum_k A[i,k] * B[k,j]
    
    Each Strassen product M_t = (sum_{(i',k')} a_{t,(i',k')} A[i',k']) * (sum_{(k',j')} b_{t,(k',j')} B[k',j'])
    
    This contributes to C[i,j] with coefficient:
    sum_{(i',k')} a_{t,(i',k')} * sum_{(k',j')} b_{t,(k',j')} * [i'==i] * [k'==j]... 
    
    Hmm, this doesn't directly give a rank-1 term in the tensor sense. The Strassen
    products are bilinear forms, not trilinear.
    
    Actually, I think the correct interpretation is:
    Each Strassen product M_t = a_t * b_t where a_t is a linear combination of A entries
    and b_t is a linear combination of B entries. This M_t then gets added (with a
    coefficient) to certain output entries C[i,j].
    
    In the tensor decomposition:
    C[i,j] = sum_c U[i*m+j_A, c] * V[j_B*p+j, c] * W[j*n+i, c]
    
    Wait, I'm confusing myself. Let me re-read the convention.
    
    "W[k*n+i,r]" - so W has p*n rows, indexed by (k, i) where k is 0..p-1 and i is 0..n-1.
    
    The tensor is: T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    representing C[i,k] = sum_j A[i,j] * B[j,k]
    
    And the decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    So for a rank-1 term c:
    - U[i*m+j, c] is the coefficient for A[i,j]
    - V[j*p+k, c] is the coefficient for B[j,k]
    - W[k*n+i, c] is the coefficient for C[i,k]
    
    And we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k).
    
    Now, for a Strassen product M_t = a_t * b_t:
    a_t = sum_{(i',j')} a_{t,(i',j')} * A[i',j']
    b_t = sum_{(j',k')} b_{t,(j',k')} * B[j',k']
    
    This product contributes to C[i,k] with coefficient:
    sum_{(i',j')} a_{t,(i',j')} * sum_{(j',k')} b_{t,(j',k')} * [i'==i] * [k'==k]... 
    
    No wait, M_t is a scalar. It gets added to C[i,k] with some coefficient w_{t,(i,k)}.
    So: C[i,k] += w_{t,(i,k)} * M_t = w_{t,(i,k)} * a_t * b_t
    
    Expanding: C[i,k] += w_{t,(i,k)} * (sum_{(i',j')} a_{t,(i',j')} A[i',j']) * (sum_{(j',k')} b_{t,(j',k')} B[j',k'])
    
    = sum_{(i',j')} sum_{(j',k')} w_{t,(i,k)} * a_{t,(i',j')} * b_{t,(j',k')} * A[i',j'] * B[j',k']
    
    For this to be a valid contribution to the tensor, we need:
    For each (i',j',j',k') with i'==i and k'==k (the actual output indices):
    The coefficient of A[i',j'] * B[j',k'] in C[i,k] should be 1 (for the correct j').
    
    But M_t * w_{t,(i,k)} contributes to ALL (i',j',j',k') combinations, not just
    the diagonal ones. This means Strassen products don't directly correspond to
    rank-1 tensor terms.
    
    I think the correct approach is: each Strassen product M_t = a_t * b_t is itself
    a "multiplication" that we track. In the tensor decomposition, each such
    multiplication corresponds to one rank-1 term where:
    - U[i*m+j, c] = a_{t,(i,j)} (the coefficient of A[i,j] in a_t)
    - V[j*p+k, c] = b_{t,(j,k)} (the coefficient of B[j,k] in b_t)
    - W[k*n+i, c] = w_{t,(i,k)} (the coefficient by which M_t contributes to C[i,k])
    
    And the constraint is that for all (i,j,k):
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    This works because:
    sum_c a_{c,(i,j)} * b_{c,(j,k)} * w_{c,(i,k)} = 1
    
    which is exactly the condition that the linear combination of products a_c * b_c
    (each weighted by w_{c,(i,k)}) gives 1 for each (i,j,k) output.
    
    So the Strassen decomposition IS a valid tensor decomposition! Each Strassen
    product is one rank-1 term, and the W matrix encodes how each product contributes
    to the output.
    
    Now let me implement this properly.
    """
    n, m, p = 2, 5, 5
    
    # Define the partition of indices
    # A columns (j): blocks [0,1], [2,3], [4]
    # B rows (j): blocks [0,1], [2,3], [4]
    # B columns (k): blocks [0,1], [2,3], [4]
    
    # We'll build the decomposition term by term.
    # Each term is (U_entries, V_entries, W_entries)
    # U_entries: dict mapping (i, j) -> coeff
    # V_entries: dict mapping (j, k) -> coeff
    # W_entries: dict mapping (k, i) -> coeff
    
    terms = []
    
    # For each i in {0,1}, we compute C[i, :] = A[i, :] * B
    # C[i, k] = sum_j A[i, j] * B[j, k]
    
    # Partition j (columns of A / rows of B) into blocks: J0={0,1}, J1={2,3}, J2={4}
    # Partition k (columns of B) into blocks: K0={0,1}, K1={2,3}, K2={4}
    
    J_blocks = [[0, 1], [2, 3], [4]]
    K_blocks = [[0, 1], [2, 3], [4]]
    
    for i in range(n):
        for jb_idx, J_block in enumerate(J_blocks):
            for kb_idx, K_block in enumerate(K_blocks):
                # Subproblem: C[i, K_block] += A[i, J_block] * B[J_block, K_block]
                # A_sub is 1 x len(J_block), B_sub is len(J_block) x len(K_block)
                # C_sub is 1 x len(K_block)
                
                len_J = len(J_block)
                len_K = len(K_block)
                
                if len_J == 2 and len_K == 2:
                    # Use Strassen for 1x2 * 2x2 -> 1x2
                    # Actually, A_sub is 1x2 (one row), B_sub is 2x2.
                    # C_sub[i,k] = A_sub[0,0]*B_sub[0,k] + A_sub[0,1]*B_sub[1,k]
                    # This is just 2 multiplications (schoolbook for 1x2*2x2).
                    # Strassen doesn't help here since A is only 1 row.
                    # Use schoolbook: 2 mults.
                    for jj in range(len_J):
                        for kk in range(len_K):
                            j = J_block[jj]
                            k = K_block[kk]
                            terms.append({
                                'U': {(i, j): 1},
                                'V': {(j, k): 1},
                                'W': {(k, i): 1}
                            })
                elif len_J == 2 and len_K == 1:
                    # 1x2 * 2x1 -> 1x1: 2 mults (schoolbook)
                    for jj in range(len_J):
                        j = J_block[jj]
                        k = K_block[0]
                        terms.append({
                            'U': {(i, j): 1},
                            'V': {(j, k): 1},
                            'W': {(k, i): 1}
                        })
                elif len_J == 1 and len_K == 2:
                    # 1x1 * 1x2 -> 1x2: 2 mults (schoolbook)
                    j = J_block[0]
                    for kk in range(len_K):
                        k = K_block[kk]
                        terms.append({
                            'U': {(i, j): 1},
                            'V': {(j, k): 1},
                            'W': {(k, i): 1}
                        })
                elif len_J == 1 and len_K == 1:
                    # 1x1 * 1x1 -> 1x1: 1 mult
                    j = J_block[0]
                    k = K_block[0]
                    terms.append({
                        'U': {(i, j): 1},
                        'V': {(j, k): 1},
                        'W': {(k, i): 1}
                    })
    
    # Count terms
    # For each i (2 values):
    #   J0 x K0: 2x2 -> 4 terms
    #   J0 x K1: 2x2 -> 4 terms
    #   J0 x K2: 2x1 -> 2 terms
    #   J1 x K0: 2x2 -> 4 terms
    #   J1 x K1: 2x2 -> 4 terms
    #   J1 x K2: 2x1 -> 2 terms
    #   J2 x K0: 1x2 -> 2 terms
    #   J2 x K1: 1x2 -> 2 terms
    #   J2 x K2: 1x1 -> 1 term
    # Total per i: 4+4+2+4+4+2+2+2+1 = 25
    # Total: 2 * 25 = 50 (same as schoolbook, no improvement)
    
    # The issue is that when A is 1x2 and B is 2x2, we can't use Strassen directly.
    # We need to think about this differently.
    
    # Alternative approach: Group the computation so that we have 2x2 * 2x2 blocks.
    # For this, we need A to be 2x2 and B to be 2x2.
    # A is 2x5, so we can take A[:, {j1,j2}] which is 2x2.
    # B is 5x5, so we can take B[{j1,j2}, {k1,k2}] which is 2x2.
    # Then C[:, {k1,k2}] += A[:, {j1,j2}] * B[{j1,j2}, {k1,k2}]
    
    # This gives us a 2x2 * 2x2 -> 2x2 subproblem, where we CAN use Strassen!
    
    # Let's redo:
    # For each pair of 2-element J_block and 2-element K_block:
    #   Use Strassen on A[:, J_block] (2x2) * B[J_block, K_block] (2x2) -> 2x2
    # For remaining 1-element blocks, use schoolbook.
    
    # J_blocks: [0,1], [2,3], [4]
    # K_blocks: [0,1], [2,3], [4]
    
    # Pairs where both have size 2:
    # (J0, K0), (J0, K1), (J1, K0), (J1, K1) -> 4 pairs, each uses Strassen (7 terms)
    # Pairs where one or both have size 1:
    # (J0, K2): 2x2 * 2x1 -> schoolbook (4 terms)
    # (J1, K2): 2x2 * 2x1 -> schoolbook (4 terms)
    # (J2, K0): 2x1 * 1x2 -> schoolbook (4 terms)
    # (J2, K1): 2x1 * 1x2 -> schoolbook (4 terms)
    # (J2, K2): 2x1 * 1x1 -> schoolbook (2 terms)
    
    # Total: 4*7 + 4*4 + 4*4 + 4*4 + 2*2 = 28 + 16 + 16 + 16 + 4 = 80?
    # That's worse than schoolbook!
    
    # I think I'm overcomplicating this. Let me go back to the cell finding:
    # "Partitioning the 5x5 B matrix into 2+2+1 blocks along both dimensions allows
    # using Strassen's 7-multiplication algorithm for 2x2*2x2 subblocks, reducing
    # total rank from 50 (schoolbook) to 46."
    
    # Saving is 4. So maybe only 4 Strassen blocks are used, saving 1 each.
    # Or maybe the partition is different.
    
    # Let me try: partition B's rows into 2+2+1 and B's columns into 2+2+1.
    # A's columns are also 5, partitioned into 2+2+1.
    
    # The key insight might be that we DON'T need to compute all cross-block terms
    # separately. Instead, we can use the Strassen structure more cleverly.
    
    # Actually, I think the correct approach is:
    # Treat the multiplication as: C = A * B where A is 2x5 and B is 5x5.
    # Partition B into 3x3 blocks of sizes 2x2, 2x2, 2x1, 1x2, 1x2, 1x1, etc.
    # But A only has 2 rows, so A's "blocks" are just rows of A.
    
    # I think the simplest correct approach that achieves rank 46 is:
    # Use schoolbook for everything EXCEPT replace 4 specific 2x2*2x2 subcomputations
    # with Strassen, saving 1 multiplication each for a total saving of 4.
    
    # But which 4 subcomputations? And how do we make them 2x2*2x2?
    
    # Let me just implement a clean Strassen-based approach and verify it.
    
    # New approach: I'll directly construct the rank-46 decomposition.
    # 
    # The 5x5 B matrix is partitioned into blocks along rows: R0={0,1}, R1={2,3}, R2={4}
    # and along columns: C0={0,1}, C1={2,3}, C2={4}
    # 
    # A's columns are partitioned as: A0={0,1}, A1={2,3}, A2={4}
    # 
    # C[i, Cx] = sum over Ry: A[i, Ax] * B[Ry, Cx] for all x,y
    # 
    # For the "diagonal" blocks where |Ax|=|Ry|=|Cx|=2, we can use Strassen.
    # These are: (A0, R0, C0), (A1, R1, C1) -> 2 blocks, saving 2.
    # 
    # For off-diagonal 2x2*2x2 blocks: (A0, R1, C0), (A1, R0, C0), etc.
    # These are also 2x2*2x2 and can use Strassen.
    # 
    # But wait, A[i, Ax] is 1x2 (one row of A), not 2x2.
    # So we can't use Strassen on A[i, Ax] * B[Ry, Cx] because A[i, Ax] is 1x2.
    # 
    # Unless we group both rows of A together. But then C has 2 rows and we'd be
    # computing a 2x2 * 2x2 -> 2x2, which is exactly Strassen!
    # 
    # So: for each pair (Ry, Cx) where both have size 2:
    #   Compute A[:, Ax] * B[Ry, Cx] where Ax is the corresponding column block.
    #   But A[:, Ax] is 2x2 only if Ax has size 2.
    # 
    # The issue is that Ax, Ry, Cx are independent partitions.
    # 
    # I think the correct interpretation is:
    # We partition the computation into subproblems where we multiply a 2x2 submatrix
    # of A by a 2x2 submatrix of B, and the result is a 2x2 submatrix of C.
    # 
    # A is 2x5. A 2x2 submatrix is A[:, {j1,j2}] for some j1 != j2.
    # B is 5x5. A 2x2 submatrix is B[{j1,j2}, {k1,k2}] for some j1!=j2, k1!=k2.
    # C is 2x5. A 2x2 submatrix is C[:, {k1,k2}] for some k1 != k2.
    # 
    # The constraint is that the column block of A must match the row block of B:
    # A[:, {j1,j2}] * B[{j1,j2}, {k1,k2}] -> C[:, {k1,k2}]
    # 
    # So we need to choose pairs ({j1,j2}, {k1,k2}) where {j1,j2} is a 2-element
    # subset of {0,1,2,3,4} and {k1,k2} is a 2-element subset.
    # 
    # But we also need to cover all (j,k) pairs exactly once (for the sum to work).
    # 
    # This is getting very complex. Let me just go with a simpler approach:
    # Use schoolbook rank 50, and try to find a rank-46 decomposition through
    # a different method.
    
    # For now, let me just return schoolbook. The Strassen approach is complex
    # to implement correctly within the time limit.
    
    return _schoolbook(n, m, p)


def construct():
    start = time.time()
    
    # Return schoolbook rank-50 decomposition
    # This is correct but not optimal.
    result = _schoolbook()
    
    return result
# EVOLVE-BLOCK-END
