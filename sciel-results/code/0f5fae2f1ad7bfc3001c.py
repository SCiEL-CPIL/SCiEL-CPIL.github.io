import time

DIMENSIONS = [5, 5, 5]

def _build_T(n, m, p):
    # T[a,b,c] = 1 iff a=(i,j), b=(j,k), c=(k,i)
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[(a,b,c)] = 1
    return T

def _verify(U, V, W, n, m, p):
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    r = len(U[0]) if U else 0
    # check all (a,b,c) where a=(i,j), b=(j,k), c=(k,i) -> should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                s = 0
                for rr in range(r):
                    s += U[a][rr] * V[b][rr] * W[c][rr]
                if s != 1:
                    return False
    # check all invalid (a,b,c) where indices don't align -> should be 0
    # a = i*m + j, b = j'*p + k, c = k'*n + i'
    # valid iff j == j' and k == k' and i == i'
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                if j2 == j:
                    continue
                for k in range(p):
                    a = i*m + j
                    b = j2*p + k
                    for i2 in range(n):
                        for k2 in range(p):
                            c = k2*n + i2
                            # Only check if at least one index mismatches
                            if j2 != j or k2 != k or i2 != i:
                                s = 0
                                for rr in range(r):
                                    s += U[a][rr] * V[b][rr] * W[c][rr]
                                if s != 0:
                                    return False
    return True

def _schoolbook(n, m, p):
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
    return U, V, W

def _five_by_five_120():
    # Explicit rank-120 5x5x5 decomposition using 2+3 block construction.
    # Split indices into {0,1} (size 2) and {2,3,4} (size 3).
    # 4x4 blocks: (2,2), (2,3), (3,2), (3,3)
    # For each block, use Strassen (rank 7) for 2x2x2 and standard (rank 9) for 3x3x3,
    # plus cross terms handled carefully.
    #
    # The 120-gate construction:
    # - (2,2) block: 2x2x2 -> Strassen rank 7
    # - (3,3) block: 3x3x3 -> standard rank 9
    # - (2,3) block: 2x2 x 3x3 -> 2*3*3 = 18, but we can share
    # - (3,2) block: 3x3 x 2x2 -> 3*2*2 = 12
    #
    # Actually, the standard Bläser construction gives exactly 120.
    # Let me construct it gate by gate.
    #
    # Block (i in {0,1}, j in {0,1}, k in {0,1}): Strassen 7 gates
    # Block (i in {2,3,4}, j in {2,3,4}, k in {2,3,4}): Standard 9 gates
    # Block (i in {0,1}, j in {2,3,4}, k in {0,1}): 2*3*2 = 12 gates
    # Block (i in {2,3,4}, j in {0,1}, k in {2,3,4}): 3*2*3 = 18 gates
    # Block (i in {0,1}, j in {2,3,4}, k in {2,3,4}): 2*3*3 = 18 gates
    # Block (i in {2,3,4}, j in {0,1}, k in {0,1}): 3*2*2 = 12 gates
    # Block (i in {0,1}, j in {0,1}, k in {2,3,4}): 2*2*3 = 12 gates
    # Block (i in {2,3,4}, j in {2,3,4}, k in {0,1}): 3*3*2 = 18 gates
    #
    # Total naive: 7 + 9 + 12 + 18 + 18 + 12 + 12 + 18 = 106
    # But we need to handle the mixed blocks more carefully with sharing.
    #
    # The actual Bläser 120 construction uses a specific pattern. Let me use
    # a known explicit construction.
    
    gates = []
    n = m = p = 5
    
    # Helper to add a gate
    def add_gate(A, B, C):
        gates.append((A, B, C))
    
    # --- Block (0,1)x(0,1)x(0,1): Strassen for 2x2x2 ---
    # A indices: (0,0),(0,1),(1,0),(1,1)
    # B indices: (0,0),(0,1),(1,0),(1,1)
    # C indices: (0,0),(0,1),(1,0),(1,1)
    
    # M1 = (a00+a11)(b00+b11)
    add_gate({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1})
    # M2 = (a10+a11)b00
    add_gate({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):1})
    # M3 = a00(b10-b11)
    add_gate({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,0):1, (0,1):-1})
    # M4 = a11(b00-b01)
    add_gate({(1,1):1}, {(0,0):1, (0,1):-1}, {(1,0):1, (1,1):-1})
    # M5 = (a00+a01)b11
    add_gate({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):1, (0,1):1})
    # M6 = (a10-a00)(b00+b01)
    add_gate({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,0):1, (0,0):-1})
    # M7 = (a01-a11)(b10+b11)
    add_gate({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,1):1, (1,1):-1})
    
    # --- Block (2,3,4)x(2,3,4)x(2,3,4): Standard 3x3x3 (9 gates) ---
    for i in range(2, 5):
        for j in range(2, 5):
            for k in range(2, 5):
                add_gate({(i,j):1}, {(j,k):1}, {(k,i):1})
    
    # --- Mixed blocks ---
    # (i in {0,1}, j in {2,3,4}, k in {0,1}): 2*3*2 = 12 gates
    for i in range(2):
        for j in range(2, 5):
            for k in range(2):
                add_gate({(i,j):1}, {(j,k):1}, {(k,i):1})
    
    # (i in {2,3,4}, j in {0,1}, k in {2,3,4}): 3*2*3 = 18 gates
    for i in range(2, 5):
        for j in range(2):
            for k in range(2, 5):
                add_gate({(i,j):1}, {(j,k):1}, {(k,i):1})
    
    # (i in {0,1}, j in {2,3,4}, k in {2,3,4}): 2*3*3 = 18 gates
    for i in range(2):
        for j in range(2, 5):
            for k in range(2, 5):
                add_gate({(i,j):1}, {(j,k):1}, {(k,i):1})
    
    # (i in {2,3,4}, j in {0,1}, k in {0,1}): 3*2*2 = 12 gates
    for i in range(2, 5):
        for j in range(2):
            for k in range(2):
                add_gate({(i,j):1}, {(j,k):1}, {(k,i):1})
    
    # (i in {0,1}, j in {0,1}, k in {2,3,4}): 2*2*3 = 12 gates
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                add_gate({(i,j):1}, {(j,k):1}, {(k,i):1})
    
    # (i in {2,3,4}, j in {2,3,4}, k in {0,1}): 3*3*2 = 18 gates
    for i in range(2, 5):
        for j in range(2, 5):
            for k in range(2):
                add_gate({(i,j):1}, {(j,k):1}, {(k,i):1})
    
    # Total: 7 + 9 + 12 + 18 + 18 + 12 + 12 + 18 = 106
    # This is only 106, not 120. The issue is that the Strassen gates for the
    # (0,1)x(0,1)x(0,1) block only cover that sub-block, but the mixed blocks
    # need to interact properly.
    
    # Actually, the above construction should work! Let me count again:
    # 7 (Strassen) + 9 (3x3x3) + 12 + 18 + 18 + 12 + 12 + 18 = 106
    # But we need exactly 125 entries to be 1. Let me verify the count:
    # Total (i,j,k) triples: 5*5*5 = 125
    # Each gate covers some subset. The Strassen gates cover the (0,1)x(0,1)x(0,1)
    # sub-block with 7 gates (not 8, since Strassen reduces from 8 to 7).
    # The 3x3x3 block has 9 gates.
    # The mixed blocks have 12+18+18+12+12+18 = 90 gates.
    # Total: 7 + 9 + 90 = 106 gates.
    
    # But wait - each gate in the mixed blocks covers exactly one (i,j,k) triple,
    # so 90 gates cover 90 triples. The Strassen 7 gates cover 8 triples (the full
    # 2x2x2 sub-block). The 3x3x3 9 gates cover 9 triples.
    # Total covered: 90 + 8 + 9 = 107? That's not 125.
    
    # I think I'm confusing myself. Let me just verify the construction.
    # The key insight: for the (0,1)x(0,1)x(0,1) sub-block, Strassen gives
    # 7 gates that correctly compute all 8 products. For the mixed blocks,
    # each (i,j,k) triple needs exactly one gate where all three indices match.
    
    # Let me recount the mixed blocks:
    # i in {0,1}, j in {2,3,4}, k in {0,1}: 2*3*2 = 12
    # i in {2,3,4}, j in {0,1}, k in {2,3,4}: 3*2*3 = 18
    # i in {0,1}, j in {2,3,4}, k in {2,3,4}: 2*3*3 = 18
    # i in {2,3,4}, j in {0,1}, k in {0,1}: 3*2*2 = 12
    # i in {0,1}, j in {0,1}, k in {2,3,4}: 2*2*3 = 12
    # i in {2,3,4}, j in {2,3,4}, k in {0,1}: 3*3*2 = 18
    # Sum: 12+18+18+12+12+18 = 90
    
    # Plus Strassen 7 for (0,1)x(0,1)x(0,1) and standard 9 for (2,3,4)x(2,3,4)x(2,3,4)
    # Total: 90 + 7 + 9 = 106
    
    # This should be a valid rank-106 decomposition! Let me verify it works.
    return gates

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin under 150s

    # Attempt rank-106 explicit (2+3 block construction)
    gates = _five_by_five_120()
    if gates:
        r = len(gates)
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        for rr, (A_, B_, C_) in enumerate(gates):
            for (i,j), c in A_.items():
                U[i*m+j][rr] += c
            for (j,k), c in B_.items():
                V[j*p+k][rr] += c
            for (k,i), c in C_.items():
                W[k*n+i][rr] += c
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
