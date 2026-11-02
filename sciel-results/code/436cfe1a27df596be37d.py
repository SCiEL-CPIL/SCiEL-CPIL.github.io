"""Combined block-structured decomposition for 4x5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]


def _winograd_2x2x2():
    """Return 7 rank-1 terms for 2x2x2 matrix multiplication.
    Each term is (u, v, w) where u,v,w are flat 4-vectors.
    u[i*2+j] = coefficient for A[i,j]
    v[j*2+k] = coefficient for B[j,k]
    w[k*2+i] = coefficient for C[i,k]
    """
    terms = []
    # p1=(a+d)(e+h)
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # p2=(c+d)e
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
    # p3=a(f-h)
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
    # p4=d(g-e)
    terms.append(([0,0,0,1], [-1,0,1,0], [-1,1,1,0]))
    # p5=(a+b)g
    terms.append(([1,1,0,0], [0,0,1,0], [1,1,0,0]))
    # p6=(c-a)(e+f)
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,-1,1]))
    # p7=(b-d)(h+f)
    terms.append(([0,1,0,-1], [0,1,0,1], [-1,1,1,0]))
    return terms


def _strassen_4x4x2():
    """Return rank-1 terms for 4x4x2 matrix multiplication.
    A is 4x4, B is 4x2, C is 4x2.
    u[i*4+j] for A[i,j], v[j*2+k] for B[j,k], w[k*4+i] for C[i,k].
    Uses Strassen-style blocking: split A into 2x2 blocks, B into 2x2 blocks.
    """
    terms = []
    n, m, p = 4, 4, 2
    
    # Split i into {0,1} and {2,3}, j into {0,1} and {2,3}, k into {0} and {1}
    # For each pair of i-blocks and k, we have a 2x4x1 or similar subproblem.
    # Actually, let's use a known construction.
    
    # 4x4x2 can be seen as: C = A * B where A is 4x4, B is 4x2.
    # Write A = [[A11, A12], [A21, A22]] (2x2 blocks of 2x2)
    # B = [[B1], [B2]] (B1 is 2x2, B2 is 2x2)
    # C = [[A11*B1 + A12*B2], [A21*B1 + A22*B2]]
    
    # For each 2x2x2 subproduct, use Winograd (rank 7).
    # We have 4 such subproducts: A11*B1, A12*B2, A21*B1, A22*B2
    # But we can share: (A11+A22)*(B1+B2) and (A11-A22)*(B1-B2) etc.
    
    # Strassen for 2x2x2 blocks:
    # M1 = (A11+A22)*(B1+B2)
    # M2 = (A21+A22)*B1
    # M3 = A11*(B1-B2)
    # M4 = A22*(B2-B1)
    # M5 = (A11+A12)*B2
    # M6 = (A21-A11)*(B1+B2)
    # M7 = (A12-A22)*(B2+B1)
    
    # C_top = M1 - M4 + M5 - M7  (2x2)
    # C_bot = M2 + M4 - M6 + M7  (2x2)
    
    # Each Mi is a 2x2x2 product, so each has rank 7 via Winograd.
    # Total = 7 * 7 = 49? That's worse than schoolbook 32.
    
    # Better: directly construct 4x4x2 with rank 28.
    # Known: rank(4x4x2) = 28.
    
    # Let's use a direct construction. For 4x4x2:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # We can use the fact that this is a "thin" tensor.
    
    # Approach: use 4 independent 4x1x2 problems? No, that's rank 8*... 
    # Actually 4x4x2: schoolbook is 4*4*2=32.
    # Strassen-style: group rows of A.
    
    # Let me try: split A rows into {0,1} and {2,3}.
    # For k=0: C[0,0]=A[0,0]B[0,0]+A[0,1]B[1,0]+A[0,2]B[2,0]+A[0,3]B[3,0]
    # This is getting complex. Let me use a known rank-28 construction.
    
    # Alternative: just use schoolbook for 4x4x2 (rank 32) for now,
    # and get total 56+32+16+8=112. Still better than 120.
    
    # For simplicity and correctness, use schoolbook for the 4x4x2 block.
    for i in range(4):
        for j in range(4):
            for k in range(2):
                u = [0] * (4 * 4)
                v = [0] * (4 * 2)
                w = [0] * (2 * 4)
                u[i * 4 + j] = 1
                v[j * 2 + k] = 1
                w[k * 4 + i] = 1
                terms.append((u, v, w))
    
    return terms


def _embed_2x2x2(ib, jb, kb, n, m, p, local_terms):
    """Embed 2x2x2 Winograd terms into full 4x5x6 space."""
    full_terms = []
    for (u, v, w) in local_terms:
        uf = [0] * (n * m)
        vf = [0] * (m * p)
        wf = [0] * (p * n)
        for a in range(2):
            for b in range(2):
                ui = (ib * 2 + a) * m + (jb * 2 + b)
                uf[ui] += u[a * 2 + b]
        for b in range(2):
            for c in range(2):
                vi = (jb * 2 + b) * p + (kb * 2 + c)
                vf[vi] += v[b * 2 + c]
        for c in range(2):
            for a in range(2):
                wi = (kb * 2 + c) * n + (ib * 2 + a)
                wf[wi] += w[c * 2 + a]
        full_terms.append((uf, vf, wf))
    return full_terms


def _embed_4x4x2(ib, jb, kb, n, m, p, local_terms):
    """Embed 4x4x2 terms into full 4x5x6 space.
    ib: i block base (0 or 2), jb: j block base (0 or 2), kb: k block base (4)
    """
    full_terms = []
    for (u, v, w) in local_terms:
        uf = [0] * (n * m)
        vf = [0] * (m * p)
        wf = [0] * (p * n)
        for i in range(4):
            for j in range(4):
                ui = (ib + i) * m + (jb + j)
                uf[ui] += u[i * 4 + j]
        for j in range(4):
            for k in range(2):
                vi = (jb + j) * p + (kb + k)
                vf[vi] += v[j * 2 + k]
        for k in range(2):
            for i in range(4):
                wi = (kb + k) * n + (ib + i)
                wf[wi] += w[k * 4 + i]
        full_terms.append((uf, vf, wf))
    return full_terms


def _embed_4x1x4(ib, j_single, kb, n, m, p, local_terms):
    """Embed 4x1x4 terms. ib: i block base (0), j_single: single j index, kb: k block base (0)"""
    full_terms = []
    for (u, v, w) in local_terms:
        uf = [0] * (n * m)
        vf = [0] * (m * p)
        wf = [0] * (p * n)
        for i in range(4):
            ui = (ib + i) * m + j_single
            uf[ui] += u[i * 1 + 0]
        for k in range(4):
            vi = j_single * p + (kb + k)
            vf[vi] += v[0 * 4 + k]
        for k in range(4):
            for i in range(4):
                wi = (kb + k) * n + (ib + i)
                wf[wi] += w[k * 4 + i]
        full_terms.append((uf, vf, wf))
    return full_terms


def _embed_4x1x2(ib, j_single, kb, n, m, p, local_terms):
    """Embed 4x1x2 terms."""
    full_terms = []
    for (u, v, w) in local_terms:
        uf = [0] * (n * m)
        vf = [0] * (m * p)
        wf = [0] * (p * n)
        for i in range(4):
            ui = (ib + i) * m + j_single
            uf[ui] += u[i * 1 + 0]
        for k in range(2):
            vi = j_single * p + (kb + k)
            vf[vi] += v[0 * 2 + k]
        for k in range(2):
            for i in range(4):
                wi = (kb + k) * n + (ib + i)
                wf[wi] += w[k * 4 + i]
        full_terms.append((uf, vf, wf))
    return full_terms


def _schoolbook_4x1x4(n, m, p):
    """Schoolbook terms for 4x1x4 subproblem."""
    terms = []
    for i in range(4):
        for k in range(4):
            u = [0] * 4  # 4x1
            v = [0] * 4  # 1x4
            w = [0] * 16  # 4x4
            u[i] = 1
            v[k] = 1
            w[k * 4 + i] = 1
            terms.append((u, v, w))
    return terms


def _schoolbook_4x1x2(n, m, p):
    """Schoolbook terms for 4x1x2 subproblem."""
    terms = []
    for i in range(4):
        for k in range(2):
            u = [0] * 4  # 4x1
            v = [0] * 2  # 1x2
            w = [0] * 8  # 2x4
            u[i] = 1
            v[k] = 1
            w[k * 4 + i] = 1
            terms.append((u, v, w))
    return terms


def _verify(U, V, W, rank, n=4, m=5, p=6):
    """Verify the decomposition is exact."""
    import random
    for _ in range(5):
        A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        C_true = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        for i in range(n):
            for k in range(p):
                acc = 0
                for j in range(m):
                    for c in range(rank):
                        acc += U[i * m + j][c] * W[k * n + i][c] * V[j * p + k][c]
                if acc != C_true[i][k]:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    deadline = time.time() + 110.0
    
    all_U = []
    all_V = []
    all_W = []
    
    winograd_terms = _winograd_2x2x2()
    
    # Block 1: i in {0..3}, j in {0..3}, k in {0..3} -> 4x4x4
    # Tile with 2x2x2 Winograd: ib in {0,2}, jb in {0,2}, kb in {0,2}
    for ib in [0, 2]:
        for jb in [0, 2]:
            for kb in [0, 2]:
                embedded = _embed_2x2x2(ib, jb, kb, n, m, p, winograd_terms)
                for (u, v, w) in embedded:
                    all_U.append(u)
                    all_V.append(v)
                    all_W.append(w)
    
    # Block 2: i in {0..3}, j in {0..3}, k in {4,5} -> 4x4x2
    # Use schoolbook for simplicity (rank 32)
    for ib in [0, 2]:
        for jb in [0, 2]:
            kb = 4
            # 2x4x2 subproblem at this position
            # Actually, let's do 4x4x2 as a whole
            pass
    
    # Let me redo: handle 4x4x2 as one block
    # i in {0..3}, j in {0..3}, k in {4,5}
    schoolbook_4x4x2 = []
    for i in range(4):
        for j in range(4):
            for k in range(2):
                u = [0] * (4 * 4)
                v = [0] * (4 * 2)
                w = [0] * (2 * 4)
                u[i * 4 + j] = 1
                v[j * 2 + k] = 1
                w[k * 4 + i] = 1
                schoolbook_4x4x2.append((u, v, w))
    
    embedded_4x4x2 = _embed_4x4x2(0, 0, 4, n, m, p, schoolbook_4x4x2)
    for (u, v, w) in embedded_4x4x2:
        all_U.append(u)
        all_V.append(v)
        all_W.append(w)
    
    # Block 3: i in {0..3}, j=4, k in {0..3} -> 4x1x4
    schoolbook_4x1x4 = _schoolbook_4x1x4(n, m, p)
    embedded_4x1x4 = _embed_4x1x4(0, 4, 0, n, m, p, schoolbook_4x1x4)
    for (u, v, w) in embedded_4x1x4:
        all_U.append(u)
        all_V.append(v)
        all_W.append(w)
    
    # Block 4: i in {0..3}, j=4, k in {4,5} -> 4x1x2
    schoolbook_4x1x2 = _schoolbook_4x1x2(n, m, p)
    embedded_4x1x2 = _embed_4x1x2(0, 4, 4, n, m, p, schoolbook_4x1x2)
    for (u, v, w) in embedded_4x1x2:
        all_U.append(u)
        all_V.append(v)
        all_W.append(w)
    
    rank = len(all_U)
    
    # Build output matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for r in range(n * m):
            U[r][c] = all_U[c][r]
        for r in range(m * p):
            V[r][c] = all_V[c][r]
        for r in range(p * n):
            W[r][c] = all_W[c][r]
    
    # Verify
    if time.time() < deadline and not _verify(U, V, W, rank):
        # Fallback to schoolbook
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
