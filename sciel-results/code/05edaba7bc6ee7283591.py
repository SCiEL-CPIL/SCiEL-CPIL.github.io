"""Matrix multiplication 3x5 by 5x7: Strassen on 2x2x2 blocks + schoolbook remainder."""
import time

DIMENSIONS = [3, 5, 7]

# Strassen 2x2x2 decomposition: 7 rank-1 terms
# A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]], C = A*B
# M1 = (a00+a11)*(b00+b11)
# M2 = (a10+a11)*b00
# M3 = a00*(b10-b11)
# M4 = a11*(b01-b00)
# M5 = (a00+a01)*b11
# M6 = (a10-a00)*(b00+b01)
# M7 = (a01-a11)*(b10+b11)
# C00 = M1 + M4 - M5 + M7
# C01 = M3 + M5
# C10 = M2 + M4
# C11 = M1 - M2 + M3 + M6

# For each term, define (U_entries, V_entries, W_entries) where each is a list of (offset, coeff)
# U is indexed by (i, j) -> i*2+j for the 2x2 block
# V is indexed by (j, k) -> j*2+k
# W is indexed by (k, i) -> k*2+i  (trace convention: W[k*n+i, r])

STRASSEN_TERMS = [
    # (U: list of (i,j,coeff), V: list of (j,k,coeff), W: list of (k,i,coeff))
    # M1 = (a00+a11)*(b00+b11)
    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
    # M2 = (a10+a11)*b00
    ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1)]),
    # M3 = a00*(b10-b11)
    ([(0,0,1)], [(1,0,1),(1,1,-1)], [(0,0,1)]),
    # M4 = a11*(b01-b00)
    ([(1,1,1)], [(0,1,1),(0,0,-1)], [(0,1,1)]),
    # M5 = (a00+a01)*b11
    ([(0,0,1),(0,1,1)], [(1,1,1)], [(1,0,1)]),
    # M6 = (a10-a00)*(b00+b01)
    ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
    # M7 = (a01-a11)*(b10+b11)
    ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(1,0,1)]),
]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # We'll use a list to track which (i,j,k) triples are covered by Strassen blocks
    # and which columns are used for each.
    
    col = 0
    # Track covered (i,j,k) to avoid double-counting in schoolbook
    covered = set()
    
    # Place Strassen 2x2x2 blocks at all possible positions
    # A is n x m = 3x5, B is m x p = 5x7, C is n x p = 3x7
    # A 2x2x2 block uses rows i,i+1 of A, cols j,j+1 of A (=rows j,j+1 of B), cols k,k+1 of B
    # So i in [0, n-2], j in [0, m-2], k in [0, p-2]
    # n=3, m=5, p=7 -> i in [0,1], j in [0,3], k in [0,5]
    # Total possible blocks: 2*4*6 = 48
    # But blocks overlap in A, B, C indices, so we can't use all simultaneously with simple Strassen.
    # We need to select a set of non-overlapping (or carefully handled) blocks.
    # 
    # Actually, the Strassen decomposition works on the tensor: for a 2x2x2 sub-tensor,
    # it replaces 8 schoolbook terms with 7 Strassen terms. The sub-tensor involves
    # A indices {i, i+1} x {j, j+1}, B indices {j, j+1} x {k, k+1}, C indices {i, i+1} x {k, k+1}.
    # 
    # For the decomposition to be valid, we need to partition the 105 (i,j,k) triples.
    # Each Strassen block covers 8 triples (2x2x2), replacing them with 7 rank-1 terms.
    # Each schoolbook term covers 1 triple with 1 rank-1 term.
    #
    # We need to find a packing of 2x2x2 blocks that covers as many triples as possible
    # without overlap.
    
    # Let's try a greedy approach: select non-overlapping 2x2x2 blocks.
    # Two blocks overlap if they share any (i,j,k) triple.
    # Block (i,j,k) covers triples {(i+a, j+b, k+c) for a in {0,1}, b in {0,1}, c in {0,1}}
    
    # Greedy: sort blocks by some criterion, pick if no overlap with already-picked blocks.
    blocks = []
    for i in range(n - 1):
        for j in range(m - 1):
            for k in range(p - 1):
                blocks.append((i, j, k))
    
    # Sort by (i, j, k) for determinism
    blocks.sort()
    
    selected_blocks = []
    used_triples = set()
    
    for (i, j, k) in blocks:
        triples = set()
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    triples.add((i+a, j+b, k+c))
        # Check for overlap
        overlap = triples & used_triples
        if not overlap:
            selected_blocks.append((i, j, k))
            used_triples |= triples
    
    # For each selected block, add 7 Strassen terms
    for (bi, bj, bk) in selected_blocks:
        for (u_entries, v_entries, w_entries) in STRASSEN_TERMS:
            # Add to U
            for (ui, uj, coeff) in u_entries:
                U[ui * m + uj][col] = coeff
            # Add to V
            for (vj, vk, coeff) in v_entries:
                V[vj * p + vk][col] = coeff
            # Add to W
            for (wk, wi, coeff) in w_entries:
                W[wk * n + wi][col] = coeff
            col += 1
    
    # For remaining (i,j,k) triples not covered, use schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in used_triples:
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
    
    # Trim to actual rank used
    rank_used = col
    U = [row[:rank_used] for row in U]
    V = [row[:rank_used] for row in V]
    W = [row[:rank_used] for row in W]
    
    return {"U": U, "V": V, "W": W}
