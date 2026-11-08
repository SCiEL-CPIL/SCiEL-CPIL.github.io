"""Tensor decomposition for matrix multiplication 3x3 * 3x8.

We build a structural decomposition:
- Split the inner index j into j in {0,1} and j=2.
- For j in {0,1}: treat A[0:2, 0:2] as a 2x2 block and use Strassen (rank 7)
  for each pair of output columns. For the remaining row i=2, use 2 direct terms per k-pair.
- For j=2: use 3 direct terms per k-pair (one for each i).

Per k-pair (2 columns):
  Strassen: 7 terms
  Row 2, j in {0,1}: 2 terms per column * 2 columns = 4 terms
  j=2: 3 terms per column * 2 columns = 6 terms
  Total per k-pair: 7 + 4 + 6 = 17

4 k-pairs: 4 * 17 = 68 terms.

That's worse than the current 48. Let me reconsider.

Actually the current program gets 48. Let me recount the current:
Per k-pair:
  Strassen block: 7
  A[2,0]*B[0,k] and A[2,1]*B[1,k] for k and k+1: 4 terms
  A[i,2]*B[2,k] for i=0,1,2 and k=k0,k0+1: 6 terms
  Total: 7+4+6 = 17 per pair, 4*17 = 68? But the code says 48.

Wait, let me re-read the code. The code adds:
- strassen_block: 7 terms
- 4 direct terms for row 2, j in {0,1}
- 6 direct terms for j=2
That's 17 per k-pair, 4 k-pairs = 68.

But the score is 1.0588 which corresponds to rank 48 (since 72/48 = 1.5, but 72/68 ≈ 1.0588).
So the current rank IS 68, and 72/68 ≈ 1.0588. Yes that matches.

So current rank is 68. Can we do better?

Let me think about a better decomposition.

Alternative: Use the fact that 3x3 * 3x8 can be seen as 3x3 * 3x3 + 3x3 * 3x3 + 3x3 * 3x2.
Each 3x3*3x3 has rank at most 23 (Landsberg's bound is 23, and it's known that rank of 3x3 is 23).
But we can't easily combine them.

Actually, let me try a different split. What if I use Strassen on a 2x2 sub-block but overlap more cleverly?

Or better: what about using the rank-7 Strassen for the top-left 2x2 of A with the full B?
No, Strassen is for 2x2*2x2.

Let me try: split A into rows {0,1} and row 2.
For rows {0,1} of A (2x3) times B (3x8):
  This is a 2x3 * 3x8 multiplication.
  Split B into columns... no, split the inner index.
  
  For j in {0,1}: A[0:2, 0:2] * B[0:2, :] -> 2x2 * 2x8
    We can use Strassen per 2x2 block of B: 4 blocks * 7 = 28
  For j=2: A[0:2, 2] * B[2, :] -> 2x1 * 1x8 = 16 terms
  
  Total for rows {0,1}: 28 + 16 = 44

For row 2 of A (1x3) times B (3x8):
  For j in {0,1}: A[2, 0:2] * B[0:2, :] -> 1x2 * 2x8 = 16 terms
  For j=2: A[2, 2] * B[2, :] -> 1x1 * 1x8 = 8 terms
  Total for row 2: 16 + 8 = 24

Grand total: 44 + 24 = 68. Same as before.

Hmm. Let me think if there's a way to reduce the j=2 part.

What if I use a different Strassen-like decomposition for 2x2 * 2x8?
The rank of 2x2 * 2x8 is at most 2*2*8 = 32 naively, but with Strassen per 2x2 block it's 4*7=28.
Can we do better? The tensor [2,2,8] has rank at most 2*2*8=32, and using Strassen per pair gives 28.
Actually, the minimal rank of [2,2,8] is known to be... let me think. It's at most 2*rank([2,2,2]) = 2*7=14? No, that's not right.

Actually [2,2,8] can be decomposed as a sum of 4 [2,2,2] tensors, each of rank 7, giving 28. But maybe we can share structure.

Alternatively, think of [2,2,8] as [2,2,2] ⊗ [1,1,4] in some sense... no.

Let me just try to be smarter about the j=2 terms. Instead of 16 + 8 = 24 for the j=2 contributions, can we combine them?

For rows {0,1}, j=2: A[0,2]*B[2,k] and A[1,2]*B[2,k] for k=0..7. That's 16 terms.
For row 2, j=2: A[2,2]*B[2,k] for k=0..7. That's 8 terms.
Total j=2: 24 terms.

What if I group the j=2 terms with the Strassen terms? Probably not straightforward.

Let me try yet another approach. What about using the full 3x3 structure?

The rank of [3,3,3] is 23. So 3x3*3x3 has rank 23.
3x3*3x8 = 3x3*3x3 + 3x3*3x3 + 3x3*3x2.
If each 3x3*3x3 has rank 23, and 3x3*3x2 has rank at most 3*3*2=18 (or less), then total ≤ 23+23+18 = 64.
But we need to verify this is achievable with explicit factorizations.

Actually, I recall that the rank of [3,3,3] is indeed 23, but constructing the explicit decomposition is non-trivial. Let me see if I can find a known rank-23 decomposition.

Alternatively, let me try a simpler improvement: can I reduce the 68-rank decomposition?

Looking at the j=2 part: 24 terms. What if I use a rank-5 algorithm for 2x1 * 1x8? No, 2x1*1x8 is just 16 rank-1 terms, no savings.

What about combining the row-2, j∈{0,1} terms (16 terms) with the j=2 terms (24 terms)? They all involve B[*, k] for the same k. But the U parts are different.

I think 68 might be hard to beat with this structural approach. Let me try a completely different split.

What if I split the output columns into 4+4 instead of 2+2+2+2?
For 3x3 * 3x4:
  Using Strassen per 2x2 block of B (2 blocks): 2*7 = 14
  Row 2, j∈{0,1}: 2*4 = 8
  j=2: 3*4 = 12
  Total: 14+8+12 = 34 per 4-column block.
  2 blocks: 68. Same.

OK let me just verify the current 68-rank decomposition is correct and return it. The score is 72/68 ≈ 1.0588.

Actually wait - I should double-check my Strassen implementation is correct. Let me trace through it.

Strassen for 2x2*2x2:
m1 = (a+d)(e+h)
m2 = (c+d)e
m3 = a(f-h)
m4 = d(g-e)
m5 = (a+b)h
m6 = (c-a)(e+f)
m7 = (b-d)(g+h)

C00 = m1 + m4 - m5 + m7
C01 = m3 + m5
C10 = m2 + m4
C11 = m1 - m2 + m3 + m6

Let me verify: 
C00 = (a+d)(e+h) + d(g-e) - (a+b)h + (b-d)(g+h)
= ae+ah+de+dh + dg-de - ah-bh + bg+bh-dg-dh
= ae+ah+de+dh+dg-de-ah-bh+bg+bh-dg-dh
= ae + (ah-ah) + (de-de) + (dh-dh) + (dg-dg) + (-bh+bh) + bg
= ae + bg ✓

C01 = a(f-h) + (a+b)h = af-ah+ah+bh = af+bh ✓

C10 = (c+d)e + d(g-e) = ce+de+dg-de = ce+dg ✓

C11 = (a+d)(e+h) - (c+d)e + a(f-h) + (c-a)(e+f)
= ae+ah+de+dh - ce-de + af-ah + ce+cf-ae-af
= (ae-ae) + (ah-ah) + (de-de) + (-ce+ce) + (-af+af) + dh + cf
= dh + cf ✓

Great, Strassen is correct.

So the current decomposition should be valid with rank 68. Let me just clean up the code and make sure it runs correctly.
"""

import time
import numpy as np

def construct():
    n, m, p = 3, 3, 8
    start_time = time.time()
    DEADLINE = 145.0

    U_terms = []
    V_terms = []
    W_terms = []

    def add_term(u_list, v_list, w_list):
        U_terms.append(u_list)
        V_terms.append(v_list)
        W_terms.append(w_list)

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(i, k):
        return k * n + i

    def strassen_block(i_off, k_off):
        a = (u_idx(i_off, 0), 1)
        b = (u_idx(i_off, 1), 1)
        c = (u_idx(i_off + 1, 0), 1)
        d = (u_idx(i_off + 1, 1), 1)
        
        e = (v_idx(0, k_off), 1)
        f = (v_idx(0, k_off + 1), 1)
        g = (v_idx(1, k_off), 1)
        h = (v_idx(1, k_off + 1), 1)
        
        C00 = (w_idx(i_off, k_off), 1)
        C01 = (w_idx(i_off, k_off + 1), 1)
        C10 = (w_idx(i_off + 1, k_off), 1)
        C11 = (w_idx(i_off + 1, k_off + 1), 1)
        
        add_term([a, d], [e, h], [C00, C11])
        add_term([c, d], [e], [C10, (w_idx(i_off + 1, k_off + 1), -1)])
        add_term([a], [f, (v_idx(1, k_off + 1), -1)], [C01, C11])
        add_term([d], [g, (v_idx(0, k_off), -1)], [C00, C10])
        add_term([a, b], [h], [(w_idx(i_off, k_off), -1), C01])
        add_term([c, (u_idx(i_off, 0), -1)], [e, f], [C11])
        add_term([b, (u_idx(i_off + 1, 1), -1)], [g, h], [C00])

    for k0 in range(0, p, 2):
        strassen_block(0, k0)
        
        add_term([(u_idx(2, 0), 1)], [(v_idx(0, k0), 1)], [(w_idx(2, k0), 1)])
        add_term([(u_idx(2, 1), 1)], [(v_idx(1, k0), 1)], [(w_idx(2, k0), 1)])
        add_term([(u_idx(2, 0), 1)], [(v_idx(0, k0 + 1), 1)], [(w_idx(2, k0 + 1), 1)])
        add_term([(u_idx(2, 1), 1)], [(v_idx(1, k0 + 1), 1)], [(w_idx(2, k0 + 1), 1)])
        
        for i in range(n):
            add_term([(u_idx(i, 2), 1)], [(v_idx(2, k0), 1)], [(w_idx(i, k0), 1)])
        for i in range(n):
            add_term([(u_idx(i, 2), 1)], [(v_idx(2, k0 + 1), 1)], [(w_idx(i, k0 + 1), 1)])
    
    rank = len(W_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for idx in range(rank):
        for (flat_idx, coeff) in U_terms[idx]:
            U[flat_idx][idx] = coeff
        for (flat_idx, coeff) in V_terms[idx]:
            V[flat_idx][idx] = coeff
        for (flat_idx, coeff) in W_terms[idx]:
            W[flat_idx][idx] = coeff
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    def verify_decomp(Ua, Va, Wa):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_decomp(U_arr, V_arr, W_arr):
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
