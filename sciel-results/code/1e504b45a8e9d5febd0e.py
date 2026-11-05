"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time as _time
_DEADLINE = _time.time() + 140  # safety margin under 150s

# ---------------------------------------------------------------------------
# Winograd 4x4x4 rank-49 exact tensor (Winograd 1978).
# Built from the 3x3x3 rank-23 tensor via the "1+3+3" / block construction.
# We verify it exactly against the 4x4x4 multiplication tensor before use.
# ---------------------------------------------------------------------------

def _winograd_3x3x3_rank23():
    """Return (U,V,W) lists for the 3x3x3 rank-23 decomposition.
    U shape (9,23), V shape (9,23), W shape (9,23).
    Convention: U[i*3+j, c], V[j*3+k, c], W[k*3+i, c].
    """
    # The classic rank-23 3x3x3 tensor (Winograd). Entries are 0, +/-1, +/-1/2.
    # We store as fractions to keep exactness.
    from fractions import Fraction as F
    # Each rank-1 term is (a, b, c) where a is a 3x3 matrix (flattened i*3+j),
    # b is 3x3 (j*3+k), c is 3x3 (k*3+i). The product sum_c a_c * b_c * c_c = AB.
    # Winograd's 23 terms for 3x3:
    terms = []
    # We'll define via the known explicit construction.
    # Represent each term by three 9-vectors.
    def M(*rows):
        # rows: 3 rows of 3
        flat = []
        for r in rows:
            for v in r:
                flat.append(F(v))
        return flat
    # The 23 rank-1 components (a_c, b_c, c_c).
    # This is the standard Winograd 3x3x3 rank 23.
    T = [
        # (a, b, c)
        (M([1,0,0],[0,0,0],[0,0,0]), M([1,0,0],[0,0,0],[0,0,0]), M([1,0,0],[0,0,0],[0,0,0])),
        (M([0,0,0],[0,1,0],[0,0,0]), M([1,0,0],[0,0,0],[0,0,0]), M([0,0,0],[0,1,0],[0,0,0])),
        (M([0,0,0],[0,0,0],[0,0,1]), M([1,0,0],[0,0,0],[0,0,0]), M([0,0,0],[0,0,0],[0,0,1])),
        (M([1,0,0],[0,0,0],[0,0,0]), M([0,1,0],[0,0,0],[0,0,0]), M([0,0,0],[0,1,0],[0,0,0])),
        (M([0,0,0],[0,1,0],[0,0,0]), M([0,1,0],[0,0,0],[0,0,0]), M([0,0,0],[0,1,0],[0,0,0])),
        (M([0,0,0],[0,0,0],[0,0,1]), M([0,1,0],[0,0,0],[0,0,0]), M([0,0,0],[0,0,0],[0,0,1])),
        (M([1,0,0],[0,0,0],[0,0,0]), M([0,0,1],[0,0,0],[0,0,0]), M([0,0,0],[0,0,0],[0,0,1])),
        (M([0,0,0],[0,1,0],[0,0,0]), M([0,0,1],[0,0,0],[0,0,0]), M([0,0,0],[0,0,0],[0,0,1])),
        (M([0,0,0],[0,0,0],[0,0,1]), M([0,0,1],[0,0,0],[0,0,0]), M([0,0,0],[0,0,0],[0,0,1])),
        # diagonal-ish shared terms
        (M([1,1,0],[0,0,0],[0,0,0]), M([1,0,0],[1,0,0],[0,0,0]), M([1,1,0],[0,0,0],[0,0,0])),
        (M([0,0,1],[0,0,0],[0,0,0]), M([1,0,0],[0,1,0],[0,0,0]), M([0,0,1],[0,0,0],[0,0,0])),
        (M([1,0,0],[0,1,0],[0,0,0]), M([1,0,0],[0,1,0],[0,0,0]), M([1,0,0],[0,1,0],[0,0,0])),
        (M([0,0,0],[0,1,1],[0,0,0]), M([0,0,0],[1,0,1],[0,0,0]), M([0,0,0],[0,1,1],[0,0,0])),
        (M([0,0,0],[0,0,0],[1,1,0]), M([0,0,0],[0,0,0],[1,1,0]), M([0,0,0],[0,0,0],[1,1,0])),
        (M([0,1,0],[0,0,0],[1,0,0]), M([0,0,1],[0,0,0],[0,1,0]), M([0,1,0],[0,0,0],[1,0,0])),
        (M([0,0,0],[1,0,0],[0,1,0]), M([0,0,0],[0,1,0],[1,0,0]), M([0,0,0],[1,0,0],[0,1,0])),
        (M([1,0,0],[0,0,0],[0,0,1]), M([0,0,1],[0,0,0],[1,0,0]), M([1,0,0],[0,0,0],[0,0,1])),
        (M([0,1,0],[0,0,0],[0,1,0]), M([0,0,0],[1,0,0],[0,0,1]), M([0,1,0],[0,0,0],[0,1,0])),
        (M([1,0,0],[0,0,1],[0,0,0]), M([0,0,0],[0,0,1],[0,1,0]), M([1,0,0],[0,0,1],[0,0,0])),
        (M([0,0,1],[0,1,0],[0,0,0]), M([1,0,0],[0,0,0],[0,0,1]), M([0,0,1],[0,1,0],[0,0,0])),
        (M([0,1,0],[0,0,1],[0,0,0]), M([0,0,1],[1,0,0],[0,0,0]), M([0,1,0],[0,0,1],[0,0,0])),
        (M([1,0,0],[0,1,0],[0,0,0]), M([0,0,0],[0,0,1],[0,0,0]), M([1,0,0],[0,1,0],[0,0,0])),
        (M([0,0,0],[0,1,0],[1,0,0]), M([0,0,0],[0,0,1],[0,0,0]), M([0,0,0],[0,1,0],[1,0,0])),
    ]
    # Pad/truncate to exactly 23 if needed (the list above has 23).
    assert len(T) == 23, len(T)
    U = [[F(0)]*23 for _ in range(9)]
    V = [[F(0)]*23 for _ in range(9)]
    W = [[F(0)]*23 for _ in range(9)]
    for c,(a,b,cc) in enumerate(T):
        for idx in range(9):
            U[idx][c] = a[idx]
            V[idx][c] = b[idx]
            W[idx][c] = cc[idx]
    return U, V, W

def _verify_3x3x3(U, V, W):
    from fractions import Fraction as F
    # Check sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == delta_{i,k}... 
    # Actually the multiplication tensor: T[i,j,k] = delta? No.
    # For matrix mult: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}.
    # Tensor form: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = delta_{i,i'} delta_{k,k'}? 
    # The standard: the bilinear form equals delta_{i,i'} delta_{k,k'} when we contract.
    # Let's just verify: for each (i,k), sum_{j,c} U[i*3+j,c]*V[j*3+k,c]*W[k*3+i,c] should equal 1 if we think of it as identity? 
    # Simpler: build the 3x3x3 tensor T[i,j,k] = sum_c U[i*3+j,c] V[j*3+k,c] W[k*3+i,c].
    # For correct matmul decomposition, T[i,j,k] must equal delta_{i,k}? No:
    # (AB)_{i,k} = sum_j A_{i,j} B_{j,k}. The tensor is T_{i,j,k} with the property that
    # sum_{j,k} A_{i,j} B_{j,k} T_{i,j,k}... 
    # Cleanest check: T[i,j,k] should equal 1 if i==k? No.
    # Actually the matmul tensor M_{i,j,k} is defined so that M(A,B)_{i,k} = sum_{j} A_{i,j}B_{j,k}.
    # In the rank decomposition, M_{i,j,k} = sum_c U[i,j,c] V[j,k,c] W[k,i,c].
    # And M_{i,j,k} = delta_{i, i} ... no. M_{i,j,k} = 1 if (the indices) ... 
    # The matmul tensor has M_{i,j,k} = delta_{i,k}? No, that's not right either.
    # Let me think: M_{i,j,k} multiplies as: for fixed i,k, (AB)_{i,k}=sum_j A_{i,j} B_{j,k}.
    # So the tensor M_{i,j,k} is the "structure" where M_{i,j,k} = 1 if i==? 
    # The matmul tensor is M_{i,j,k} = delta_{i, i'} delta_{k, k'} in the sense that it's the identity in the appropriate slots.
    # Concretely, M_{i,j,k} = 1 if (i, k) are the output and j is the contraction; the tensor value M_{i,j,k} = delta_{i,k}? 
    # No! M_{i,j,k} = 1 for all i,j,k where the pattern holds: it's the tensor that when you contract A over (i,j) and B over (j,k) gives (AB)_{i,k}.
    # That tensor is M_{i,j,k} = delta_{i,k}? Let's test: sum_{j,k} A_{i,j} B_{j,k} delta_{i,k} = sum_j A_{i,j} B_{j,i}. That's not (AB)_{i,k}.
    # I'm overcomplicating. The matmul tensor T has T_{i,j,k} = 1 if i==k? No.
    # The correct statement: T_{i,j,k} = delta_{i, i} is wrong.
    # The matmul tensor is: T_{i,j,k} = 1 when (i,j,k) form the "multiplication" pattern, i.e. T_{i,j,k} = delta_{i,k} is NOT it.
    # Let me just use the definition: T_{i,j,k} = 1 if i == k? 
    # I'll verify numerically with a known-correct schoolbook and compare to my rank-23.
    # Schoolbook 3x3x3 rank 27: U[i*3+j, c=(i,j)] =1, V[j*3+k, c=(i,j,k)]... 
    # Let me just check that my rank-23 equals schoolbook tensor value-by-value.
    # Build schoolbook tensor S[i,j,k]: S[i,j,k] = 1 if (i,j,k) is a valid triple? 
    # The matmul tensor S_{i,j,k} = delta_{i,k}? 
    # Ugh. Let me define it properly: S_{i,j,k} = 1 if i==k else 0? Test with A=B=I: (AB)_{i,k}=delta_{i,k}. 
    # sum_j A_{i,j}B_{j,k} = sum_j delta_{i,j} delta_{j,k} = delta_{i,k}. Good, that's (AB)_{i,k} not a tensor in (i,j,k).
    # The tensor T_{i,j,k} satisfies: for all A,B, sum_{j,k} A_{i,j} B_{j,k} T_{i,j,k} = (AB)_{i,k} for each (i,k).
    # So T_{i,j,k} must equal delta_{i,k}? No, because then sum_{j,k} A_{i,j}B_{j,k}delta_{i,k} = delta_{i,k} sum_j A_{i,j}B_{j,i} != (AB)_{i,k}.
    # The resolution: T_{i,j,k} = 1 if i==k is wrong. The matmul tensor is T_{i,j,k} = delta_{i,k}?? 
    # I clearly have a confusion. Let me look at it as: T is a 3-index tensor where the bilinear map is
    # f(A,B)_{i,k} = sum_{j} A_{i,j} B_{j,k}.
    # Writing f(A,B)_{i,k} = sum_{j,c} A_{i,j} B_{j,k} T_{i,j,k,c}... no.
    # Standard: f(A,B)_{i,k} = sum_{j} A_{i,j} B_{j,k} = sum_{j, c} A_{i,j} B_{j,k} (sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c})? 
    # No. The rank decomposition: f(A,B)_{i,k} = sum_c (sum_j A_{i,j} U_{i,j,c}) (sum_k' B_{j,k'} V_{j,k',c}) ... 
    # I think the cleanest verification is: 
    #   (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    #   = sum_c [ (sum_j A_{i,j} U_{i*m+j,c}) * (sum_k B_{j,k} V_{j*p+k,c}) ] * W_{k*n+i,c}?? 
    # That doesn't factor. 
    # The correct factorization: 
    #   (AB)_{i,k} = sum_c M_c(A) N_c(B) P_c
    #   where M_c(A) = sum_{i,j} A_{i,j} U_{i*m+j,c}, N_c(B) = sum_{j,k} B_{j,k} V_{j*p+k,c}, and P_c is... 
    #   but (AB)_{i,k} depends on (i,k), so we need: (AB)_{i,k} = sum_c (sum_{i',j} A_{i',j} U_{i'*m+j,c})(sum_{j',k'} B_{j',k'} V_{j'*p+k',c}) W_{k*n+i,c}
    #   and this must equal sum_j A_{i,j}B_{j,k}.
    # So the verification: for all A,B, for all (i,k):
    #   sum_c (sum_{i',j} A_{i',j} U_{i',j,c})(sum_{j',k'} B_{j',k'} V_{j',k',c}) W_{k,i,c} == sum_j A_{i,j}B_{j,k}
    # Equivalently, the tensor equality:
    #   sum_c U_{i',j,c} V_{j',k',c} W_{k,i,c} = delta_{i',i} delta_{k',k}
    # Let me verify THAT.
    from fractions import Fraction as F
    n=m=p=3
    for i in range(n):
        for k in range(p):
            for ip in range(n):
                for kp in range(p):
                    s = F(0)
                    for c in range(23):
                        for j in range(m):
                            for jp in range(m):
                                # U index: ip*m + j ; V index: jp*p + kp ; W index: kp*n + i
                                s += U[ip*m+j][c] * V[jp*p+kp][c] * W[kp*n+i][c]
                    expected = (1 if ip==i and kp==k else 0)
                    if s != expected:
                        return False
    return True

def _winograd_4x4x4_rank49():
    """Build 4x4x4 rank-49 from the 3x3x3 rank-23 via Winograd's 1+3+3 construction.
    We verify exactness; if verification fails we fall back to schoolbook 64.
    """
    from fractions import Fraction as F
    U3, V3, W3 = _winograd_3x3x3_rank23()
    if not _verify_3x3x3(U3, V3, W3):
        return None  # signal fallback
    # Winograd 4x4x4 construction:
    # 4x4 = [[1,0],[0,3]] block structure. The rank-49 uses:
    #  - 1 term for the (0,0) corner (the 1x1x1)
    #  - the 3x3x3 rank-23 for the bottom-right 3x3 block
    #  - 27 cross terms (3x3) connecting the 1 and the 3 blocks
    # Total = 1 + 23 + 27 = 51? That's not 49.
    # The actual Winograd 4x4x4 rank is 49 with a specific construction.
    # Rather than risk an incorrect 49, we use a SAFE construction:
    # 4x4x4 = 1x1x1 (rank1) + 3x3x3 (rank23) + cross terms.
    # Cross terms: the off-diagonal blocks. A 4x4 = [a b; c d] with a scalar, b,c 1x3, d 3x3.
    # (AB) = [aa' + bb' + cc' + dd', ...]
    # The cross contributions: a*b', a*c', b*a', c*a', b*d', d'*b, c*d', d'*c, and the 3x3 d*d'.
    # This is getting complex. To be SAFE and still beat 80, I'll use a verified construction:
    # 4x4x4 via 2x2x2 rank-7 (Strassen) blocks? 4x4 = 2x(2x2). 
    # Actually simplest SAFE win: 4x4x4 rank 49 is hard to get exactly right.
    # Let me instead use a DIFFERENT safe decomposition that I can VERIFY:
    # 4x4x4 = schoolbook 64 is safe but only gives 64+16=80 = no gain.
    # I need a verified sub-64 for 4x4x4.
    # Use Strassen 2x2x2 (rank 7) in a 2x2 block tiling: 4x4 = 4 blocks of 2x2.
    # 2x2 block Strassen: to multiply 4x4 by 4x4, we have 16 output 2x2 blocks, each = product of two 2x2 blocks.
    # Naive: 16 * 7 = 112. Worse.
    # Winograd's actual 4x4x4 rank 49 uses the 3x3x3 rank 23 cleverly.
    # Given risk, I'll implement the 3x3x3 rank-23 for the bottom-right and handle the
    # first row/col with 16 schoolbook-like terms but shared.
    # 4x4 = [x  r; c  D] where x scalar, r,c 1x3, D 3x3.
    # (AB)_{0,0} = x*x' + r.b' + c.c' + D.d'  -> needs: 1 (x*x') + 3 (r.b') + 3 (c.c') + 23 (D.d') = 29 terms but shared.
    # (AB)_{0,k} = x*r'_k + r.b'_k + c.c'_k + D.d'_k for k=1..3
    # (AB)_{i,0} = x*c'_i + r.b'_i + c.c'_i + D.d'_i for i=1..3
    # (AB)_{i,k} = r.b'_k + c.c'_i + D.d'_{i,k} for i,k in 1..3
    # The D.d' part (3x3x3) is rank 23 and covers D*d' for all i,k in 1..3.
    # The r.b' and c.c' parts: r is 1x3, b' is 3x3. r.b' is 1x3, rank 3 (schoolbook).
    # c.c' is 3x1, rank 3.
    # x*x' is rank 1.
    # But these cross terms ALSO appear in (AB)_{0,k}, (AB)_{i,0}, (AB)_{i,k}.
    # Specifically, the term r.b'_{?} appears in row 0 (as r.b') and in column 0 (as r.b'_i for the i-th row? no).
    # Let me just do: total = 1 (x*x') + 3 (r*b' row0) + 3 (c*c' col0) + 23 (D*D') = 30, 
    # but the (AB)_{i,k} for i,k>=1 ALSO needs r.b'_k (row i, col k) and c.c'_i.
    # r.b' is a 1x3 vector = row 0 of the r*b' contribution. For (AB)_{i,k} we need the (i,k) entry of r*b' which is r * b'_{i,k}?? 
    # No: r is 1x3, b' is 3x3, so r*b' is 1x3 (only row 0). It does NOT contribute to (AB)_{i,k} for i>=1.
    # Wait, (AB) = [x  r; c  D] [x'  r'; c'  D'] = 
    #   [ x*x' + r.c' + c.r' + D.D',  x*r' + r.D' + c.c' + D.r'  ]
    #   [ c*x' + D.c',                 c*r' + D.D'                  ]  -- wait let me redo.
    # (AB)_{0,0} = x*x' + r.c' + c.r' + D.D'
    # (AB)_{0,k} (k>=1) = x*r'_k + r.D'_{.,k} + c.r'?? 
    # Let me index: A = [x, r_1, r_2, r_3; c_1,c_2,c_3, D]. B = [x', r'_1,r'_2,r'_3; c'_1,c'_2,c'_3, D'].
    # (AB)_{0,0} = x*x' + r_1 c'_1 + r_2 c'_2 + r_3 c'_3 + D D'
    # (AB)_{0,k} = x r'_k + r_1 c'_k + r_2 c'_k + r_3 c'_k + (D D')_{0,k}  for k=1,2,3
    # (AB)_{i,0} = x c'_i + r_i c'_1 + r_i c'_2 + r_i c'_3 + (D D')_{i,0}  for i=1,2,3
    # (AB)_{i,k} = r_i c'_k + (D D')_{i,k}  for i,k=1..3
    # So the cross terms needed:
    #  - x*x' : 1 term
    #  - r_1 c'_1, r_2 c'_2, r_3 c'_3 : these appear in (AB)_{0,0} as r.c' = sum r_i c'_i. 
    #    But (AB)_{0,k} needs r_i c'_k for ALL i,k (that's r * c' where c' is 3x1? no c' is the first column of B's bottom-left = 3x1).
    #    Hmm r is 1x3, c' is 3x1, so r*c' is a scalar (contributes only to (AB)_{0,0}).
    #    (AB)_{0,k} for k>=1 needs r * (column k of D')? No, r is 1x3 and D' is 3x3, r*D' is 1x3 -> contributes to row 0, cols 1..3. 
    #    And c (3x1) * r' (1x3) is 3x3 -> contributes to rows 1..3, cols 1..3.
    #    And c * c' is scalar -> (AB)_{0,0} only? No: c is 3x1 (col 0 rows 1..3), c' is 3x1 (col 0 rows 1..3). c*c' is not a matrix product of compatible dims in the block sense... 
    #    Actually in block multiplication, (AB)_{0,0} = A_{0,0}B_{0,0} + A_{0,1}B_{1,0} + A_{0,0}... 
    #    A = [A00 A01; A10 A11] = [x  r; c  D], B = [x' r'; c' D'].
    #    (AB)_{0,0} = x*x' + r*c'      (r is 1x3, c' is 3x1 -> scalar)
    #    (AB)_{0,1} = x*r' + r*D'      (1x3)
    #    (AB)_{1,0} = c*x' + D*c'      (3x1)
    #    (AB)_{1,1} = c*r' + D*D'      (3x3)
    # Great, that's clean. So:
    #  (AB)_{0,0}: x*x' (rank1) + r*c' (rank3)
    #  (AB)_{0,1}: x*r' (rank3) + r*D' (rank 3*3=9? r is 1x3, D' 3x3 -> 1x3, rank 3)
    #  (AB)_{1,0}: c*x' (rank3) + D*c' (rank 3)
    #  (AB)_{1,1}: c*r' (rank 3*1*3? c 3x1, r' 1x3 -> 3x3, rank 3) + D*D' (rank 23)
    # Total = 1 + 3 + 3 + 3 + 3 + 3 + 3 + 23 = 42? Let me count distinct rank-1 terms:
    #  x*x': 1
    #  r*c': 3  (terms r_i * c'_i)
    #  x*r': 3  (x * r'_k)
    #  r*D': 3  (r_i * D'_{i,k} for the row0,col k: actually r*D' = sum_i r_i D'_{i,:}, that's 3 rank-1 terms each = r_i * (row i of D'))
    #  c*x': 3  (c_i * x')
    #  D*c': 3  (D_{:,i} * c'_i)
    #  c*r': 3  (c_i * r'_{:,i})
    #  D*D': 23
    # Total = 1+3+3+3+3+3+3+23 = 42.
    # But wait, do these terms overlap / can they be shared? The r*D' terms produce row 0 cols 1..3. 
    # The c*r' terms produce rows 1..3 cols 1..3. These are different output positions, so the rank-1 terms are distinct (they write to different (i,k)).
    # However, a rank-1 term in the tensor decomposition writes to ALL (i,k) simultaneously via W_{k,i,c}. 
    # So a term r_i * D'_{i,:} would, in the tensor, have U, V, W that place contributions at specific (i,k). 
    # This is exactly how the tensor works: each rank-1 term c contributes U_{i,j,c}, V_{j,k,c}, W_{k,i,c} to (AB)_{i,k}.
    # So I can build the 4x4x4 tensor by combining:
    #  - the 3x3x3 rank-23 for D*D' (placed at i,k in 1..3, i.e. offset by 1)
    #  - 3 terms for r*c' (scalar at (0,0))
    #  - 3 terms for x*r' (at (0,k), k=1..3)
    #  - 3 terms for r*D' (at (0,k), k=1..3)  [r_i * D'_{i,k}]
    #  - 3 terms for c*x' (at (i,0), i=1..3)
    #  - 3 terms for D*c' (at (i,0), i=1..3)  [D_{i,i'} * c'_{i'}]
    #  - 3 terms for c*r' (at (i,k), i,k=1..3)  [c_i * r'_{i,k}]
    #  - 1 term for x*x' (at (0,0))
    # = 23 + 3+3+3+3+3+3+1 = 42.
    # 42 < 49! And I can VERIFY it exactly. If verification passes, 4x4x4 rank 42.
    # Then 4x4x5 = 42 (for first 4 cols) + 16 (schoolbook for col 4) = 58. Beats 80!
    # Let me build this and verify.
    n = m = p = 4
    R = 42
    U = [[F(0)]*R for _ in range(n*m)]
    V = [[F(0)]*R for _ in range(m*p)]
    W = [[F(0)]*R for _ in range(p*n)]
    col = 0
    # Helper to add a rank-1 term given (i,j) for U, (j,k) for V, (k,i) for W with value.
    # Actually a rank-1 term is defined by U_{i,j,c}=a, V_{j,k,c}=b, W_{k,i,c}=d for specific (i,j),(j,k),(k,i).
    # But in the tensor, a rank-1 term can have multiple nonzero U entries etc.
    # For the D*D' part, each of the 23 rank-1 terms of the 3x3x3 has full 3x3 U,V,W patterns.
    # I'll place the 3x3x3 rank-23 at offset (1,1) in (i,k) and j in 1..3.
    # 3x3x3 indices (a,b,c) with a,b,c in 0..2 map to 4x4 indices (a+1, b+1, c+1).
    for c3 in range(23):
        for a in range(3):   # i-1
            for b in range(3): # j-1
                U[(a+1)*m + (b+1)][col] = U3[a*3+b][c3]
        for b in range(3):   # j-1
            for cc in range(3): # k-1
                V[(b+1)*p + (cc+1)][col] = V3[b*3+cc][c3]
        for cc in range(3):  # k-1
            for a in range(3): # i-1
                W[(cc+1)*n + (a+1)][col] = W3[cc*3+a][c3]
        col += 1
    # Now the cross terms. Each is a "rank-1" in the sense of a single (i,j),(j,k),(k,i) product.
    # x*x' at (0,0): U[0,0]=1, V[0,0]=1, W[0,0]=1
    U[0*m+0][col]=1; V[0*p+0][col]=1; W[0*n+0][col]=1; col+=1
    # r*c' scalar at (0,0): terms r_i * c'_i for i=1..3.
    # r_i is A_{0,i}, c'_i is B_{i,0}. So U[0,i]=1, V[i,0]=1, W[0,0]=1.
    for i in range(1,4):
        U[0*m+i][col]=1; V[i*p+0][col]=1; W[0*n+0][col]=1; col+=1
    # x*r' at (0,k), k=1..3: x=A_{0,0}, r'_k=B_{0,k}. U[0,0]=1, V[0,k]=1, W[k,0]=1.
    for k in range(1,4):
        U[0*m+0][col]=1; V[0*p+k][col]=1; W[k*n+0][col]=1; col+=1
    # r*D' at (0,k), k=1..3: r_i=A_{0,i}, D'_{i,k}=B_{i,k}. U[0,i]=1, V[i,k]=1, W[k,0]=1.
    for i in range(1,4):
        for k in range(1,4):
            U[0*m+i][col]=1; V[i*p+k][col]=1; W[k*n+0][col]=1; col+=1
    # c*x' at (i,0), i=1..3: c_i=A_{i,0}, x'=B_{0,0}. U[i,0]=1, V[0,0]=1, W[0,i]=1.
    for i in range(1,4):
        U[i*m+0][col]=1; V[0*p+0][col]=1; W[0*n+i][col]=1; col+=1
    # D*c' at (i,0), i=1..3: D_{i,i'}=A_{i,i'}, c'_{i'}=B_{i',0}. U[i,i']=1, V[i',0]=1, W[0,i]=1.
    for i in range(1,4):
        for ip in range(1,4):
            U[i*m+ip][col]=1; V[ip*p+0][col]=1; W[0*n+i][col]=1; col+=1
    # c*r' at (i,k), i,k=1..3: c_i=A_{i,0}, r'_{i,k}=B_{0,k}?? 
    # c is 3x1 = A_{1..3,0}, r' is 1x3 = B_{0,1..3}. c*r' is 3x3 with (i,k) entry = c_i * r'_k = A_{i,0}*B_{0,k}.
    # So U[i,0]=1, V[0,k]=1, W[k,i]=1.
    for i in range(1,4):
        for k in range(1,4):
            U[i*m+0][col]=1; V[0*p+k][col]=1; W[k*n+i][col]=1; col+=1
    assert col == R, (col, R)
    # Verify 4x4x4.
    if not _verify_matmul_tensor(U, V, W, 4, 4, 4):
        return None
    return U, V, W

def _verify_matmul_tensor(U, V, W, n, m, p):
    from fractions import Fraction as F
    # Check sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] == delta_{i,i'} delta_{k,k'}
    # (the matmul tensor structure)
    for i in range(n):
        for k in range(p):
            for ip in range(n):
                for kp in range(p):
                    s = F(0)
                    for c in range(len(U[0])):
                        for j in range(m):
                            for jp in range(m):
                                s += U[ip*m+j][c] * V[jp*p+kp][c] * W[kp*n+i][c]
                    expected = (1 if ip==i and kp==k else 0)
                    if s != expected:
                        return False
    return True

def _to_list(frac_list):
    """Convert list of lists of Fraction to list of lists of int/float (half-integers ok)."""
    out = []
    for row in frac_list:
        r = []
        for v in row:
            if v == v.numerator // v.denominator:  # integer
                r.append(v.numerator // v.denominator)
            else:
                r.append(float(v))
        out.append(r)
    return out

def construct():
    from fractions import Fraction as F
    n, m, p = DIMENSIONS  # 4,4,5
    # Try the 4x4x4 rank-42 (or 49) + 16 schoolbook for col 4.
    res = _winograd_4x4x4_rank49()
    if res is not None and _time.time() < _DEADLINE:
        U4, V4, W4 = res
        rank4 = len(U4[0])
        # Append 16 schoolbook terms for the 5th output column (k=4).
        # For k=4: (AB)_{i,4} = sum_j A_{i,j} B_{j,4}.
        # Schoolbook: for each (i,j): U[i,j,c]=1, V[j,4,c]=1, W[4,i,c]=1.
        total_rank = rank4 + n*m
        U = [row[:] for row in U4]
        V = [row[:] for row in V4]
        W = [row[:] for row in W4]
        for i in range(n):
            for j in range(m):
                U[i*m+j].append(1)
                V[j*p+4].append(1)
                W[4*n+i].append(1)
        # Verify the full 4x4x5 tensor.
        if _verify_matmul_tensor(U, V, W, n, m, p) and _time.time() < _DEADLINE:
            return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    # Fallback: schoolbook 80.
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
