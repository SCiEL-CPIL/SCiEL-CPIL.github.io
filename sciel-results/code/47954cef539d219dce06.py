<<<<<<< SEARCH
    u2 = [0]*12; u2[4] = 1; u2[5] = 1
    v2 = [0]*20; v2[0] = 1
    w2 = [0]*15; w2[1] = 1
    U.append(u2); V.append(v2); W.append(w2)
    
    # Term 3: M3 = A00(B10 - B11)
    # U: A00 (i=0,j=0) -> idx 0
    # V: B10 (j=1,k=0) -> idx 5, B11 (j=1,k=1) -> idx 6
    # W: C01 (k=1,i=0) -> idx 3
    u3 = [0]*12; u3[0] = 1
    v3 = [0]*20; v3[5] = 1; v3[6] = -1
    w3 = [0]*15; w3[3] = 1
    U.append(u3); V.append(v3); W.append(w3)
=======
    u2 = [0]*12; u2[4] = 1; u2[5] = 1
    v2 = [0]*20; v2[0] = 1
    w2 = [0]*15; w2[1] = 1; w2[4] = -1
    U.append(u2); V.append(v2); W.append(w2)
    
    # Term 3: M3 = A00(B10 - B11)
    # U: A00 (i=0,j=0) -> idx 0
    # V: B10 (j=1,k=0) -> idx 5, B11 (j=1,k=1) -> idx 6
    # W: C01 (k=1,i=0) -> idx 3, C11 (k=1,i=1) -> idx 4
    u3 = [0]*12; u3[0] = 1
    v3 = [0]*20; v3[5] = 1; v3[6] = -1
    w3 = [0]*15; w3[3] = 1; w3[4] = 1
    U.append(u3); V.append(v3); W.append(w3)
>>>>>>> REPLACE

Now let me verify the total rank: 7 (Strassen) + (60 - 8) = 7 + 52 = 59. Wait, that's not right.

The 2×2×2 subproblem has 8 terms (i,j,k each in {0,1}). Strassen uses 7 terms instead of 8, saving 1 term. The remaining terms are 60 - 8 = 52. So total rank = 7 + 52 = 59.

But the inspiration says rank 58. Let me reconsider. Maybe I can use Strassen for a different block, or maybe there's a way to save another term.

Actually, looking at the inspiration again: "2 Strassen blocks (2×2×2) covering i∈{0,1}, j". So maybe they're using two Strassen blocks? But we only have a 3×4×5 problem, so we can only fit one 2×2×2 block in the i,j,k dimensions.

Wait, maybe the second Strassen block is for a different subset. Let me think...

Actually, I think the rank 58 might come from a different decomposition strategy. For now, let me proceed with rank 59 and see if it's valid. If it is, the score would be 60/59 ≈ 1.017.

Let me finalize the code:

