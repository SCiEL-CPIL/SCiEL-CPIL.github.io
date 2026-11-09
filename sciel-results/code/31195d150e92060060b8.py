"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 4, 4

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def strassen28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        u_coeffs = [
            [[1,0],[0,1]], [[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[0,1]],
            [[1,1],[0,0]], [[-1,0],[1,0]], [[0,1],[0,-1]],
        ]
        v_coeffs = [
            [[1,0],[0,1]], [[1,0],[0,0]], [[0,1],[0,-1]], [[-1,0],[1,0]],
            [[0,0],[0,1]], [[1,1],[0,0]], [[0,0],[1,1]],
        ]
        w_coeffs = [
            [[1,0],[0,1]], [[0,1],[0,-1]], [[0,0],[1,1]], [[1,1],[0,0]],
            [[-1,1],[0,0]], [[0,0],[0,1]], [[1,0],[0,0]],
        ]
        for s in range(4):
            a = s // 2
            b = s // 2
            c = s % 2
            for f in range(7):
                col = s * 7 + f
                for pi in range(2):
                    for pj in range(2):
                        val = u_coeffs[f][pi][pj]
                        if val != 0:
                            U[pi * m + (2*a + pj)][col] += val
                for pj in range(2):
                    for qk in range(2):
                        val = v_coeffs[f][pj][qk]
                        if val != 0:
                            V[(2*b + pj) * p + (2*c + qk)][col] += val
                for qk in range(2):
                    for pi in range(2):
                        val = w_coeffs[f][qk][pi]
                        if val != 0:
                            W[(2*c + qk) * n + pi][col] += val
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    # --- Attempt rank-7: merge the 4 Strassen instances into one rank-7 decomposition.
    # For each factor f, choose (u,v,w) 2x2 matrices so that the three cores
    # (over a,b for U; b,c for V; c,a for W) all equal a common 2x2x2 core.
    # Then U[(2a+pi)*4+(2b+pj),f]=u[pi][pj], V[(2b+pj)*4+(2c+qk),f]=v[pj][qk],
    # W[(2c+qk)*2+(2a+pi),f]=w[qk][pi].
    def mat_to_list(M):
        return [M[0][0], M[0][1], M[1][0], M[1][1]]

    def list_to_mat(x):
        return [[x[0], x[1]], [x[2], x[3]]]

    def try_rank7():
        if time.time() > deadline - 5:
            return None
        # Candidate coefficient sets (integers, kept small for speed).
        cand = [-2, -1, 0, 1, 2]
        # All 2x2 matrices over cand, as a flat list of 4-tuples.
        mats = list(itertools.product(cand, repeat=4))

        # Precompute the 10 core-index pairs (pi,pj) and (qk,pi) -> (a,b) / (c,a).
        # We'll build cores as dict keyed by (x,y,z).
        best = None
        # For each f, we need (u,v,w). We search per-f but must keep cores consistent
        # across f? No: cores can differ per f; the constraint is only within a single f
        # that the three cores (U-core, V-core, W-core) are equal.
        # So for each f independently find (u,v,w) with matching cores; combine.
        per_f = []
        for f in range(7):
            found = None
            for ux in mats:
                u = list_to_mat(ux)
                # Build U-core: key (a,b,pi,pj) -> u[pi][pj]
                # We want a core C[pi][pj][?]; define core C indexed by (pi,pj) is not enough
                # because U-core also depends on a,b. Let's define the shared core as a
                # 2x2x2 tensor T[a][b][?]... Actually the shared object across U,V,W must be
                # consistent. Let me define: U-core over (a,b) with inner (pi,pj) must equal
                # a tensor S_u[a][b][pi][pj]; V-core S_v[b][c][pj][qk]; W-core S_w[c][a][qk][pi].
                # For these to be "the same" core, we need a single 4-index tensor that all
                # three are views of. This is only possible if the contraction structure
                # allows it. Simpler: require that there exists a 2x2x2x2 tensor T such that
                # S_u[a,b,pi,pj]=T[a,b,pi,pj], S_v[b,c,pj,qk]=T[b,c,pj,qk] (renamed), etc.
                # This is an over-constrained matching; instead just require the three
                # 4D arrays (with their specific index roles) be consistent when contracted.
                # Given time, fall back: require u, v, w each equal a common 2x2 matrix M
                # (i.e., u=v=w=M). Then U-core[a,b,pi,pj]=M[pi][pj] independent of a,b,
                # V-core[b,c,pj,qk]=M[pj][qk], W-core[c,a,qk,pi]=M[qk][pi]. These are NOT the
                # same tensor (different index roles), so this is a heuristic, not exact.
                # Therefore rank-7 via simple matching is not guaranteed. We verify at the end.
                break
            # This heuristic search is unreliable; instead directly verify candidate rank-7
            # tensors built from a guessed Strassen-like assignment and rely on verify().
        # Given the complexity, we do a direct verification of a constructed rank-7 tensor
        # using the standard 2x2 Strassen pattern extended, then verify().
        # Build rank-7 with u=v=w = standard Strassen matrices (same triple), and test.
        u_coeffs = [
            [[1,0],[0,1]], [[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[0,1]],
            [[1,1],[0,0]], [[-1,0],[1,0]], [[0,1],[0,-1]],
        ]
        # For a merged single-instance, set the same (u,v,w) triple for all a,b,c.
        v_coeffs = [
            [[1,0],[0,1]], [[1,0],[0,0]], [[0,1],[0,-1]], [[-1,0],[1,0]],
            [[0,0],[0,1]], [[1,1],[0,0]], [[0,0],[1,1]],
        ]
        w_coeffs = [
            [[1,0],[0,1]], [[0,1],[0,-1]], [[0,0],[1,1]], [[1,1],[0,0]],
            [[-1,1],[0,0]], [[0,0],[0,1]], [[1,0],[0,0]],
        ]
        # Try several assignments of which (u,v,w) triple maps to the single instance.
        triples = [(u_coeffs, v_coeffs, w_coeffs)]
        for (uc, vc, wc) in triples:
            U = [[0] * 7 for _ in range(n * m)]
            V = [[0] * 7 for _ in range(m * p)]
            W = [[0] * 7 for _ in range(p * n)]
            for f in range(7):
                for a in range(2):
                    for b in range(2):
                        for c in range(2):
                            for pi in range(2):
                                for pj in range(2):
                                    U[(2*a+pi)*4 + (2*b+pj)][f] += uc[f][pi][pj]
                            for pj in range(2):
                                for qk in range(2):
                                    V[(2*b+pj)*4 + (2*c+qk)][f] += vc[f][pj][qk]
                            for qk in range(2):
                                for pi in range(2):
                                    W[(2*c+qk)*2 + (2*a+pi)][f] += wc[f][qk][pi]
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}
        return None

    # 1) Try rank-7.
    r7 = try_rank7()
    if r7 is not None:
        return r7

    # 2) Fall back to rank-28 Strassen (verified).
    r28 = strassen28()
    if verify(r28["U"], r28["V"], r28["W"]):
        return r28

    # 3) Final fallback: schoolbook.
    return schoolbook()
# EVOLVE-BLOCK-END
