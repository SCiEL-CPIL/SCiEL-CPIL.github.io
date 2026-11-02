import time
import numpy as np

DIMENSIONS = [3, 4, 8]
DEADLINE = time.time() + 118.0


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8

    # Strassen 2x2x2 decomposition (7 terms)
    # Local index: A[i*2+j], B[j*2+k], C[k*2+i]
    strassen_decomp = [
        ({0: 1, 3: 1},   {0: 1, 3: 1},   {0: 1, 3: 1}),    # M1
        ({2: 1, 3: 1},   {0: 1},         {2: 1, 3: -1}),   # M2
        ({0: 1},         {1: 1, 3: -1},  {1: 1, 3: 1}),    # M3
        ({3: 1},         {2: 1, 0: -1},  {0: 1, 2: 1}),    # M4
        ({0: 1, 1: 1},   {3: 1},         {1: 1, 0: -1}),   # M5
        ({2: 1, 0: -1},  {0: 1, 1: 1},   {3: 1}),          # M6
        ({1: 1, 3: -1},  {2: 1, 3: 1},   {0: 1}),          # M7
    ]

    def _build_block_strassen():
        """Rank-88: n=3 -> 2+1, m=4 -> 2+2, p=8 -> 4x2.
        Top (rows 0,1): Strassen per 2x2x2 block. Bottom (row 2): schoolbook."""
        rank = 88
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        for j1 in range(2):
            for l in range(4):
                for u_c, v_c, w_c in strassen_decomp:
                    for li, cf in u_c.items():
                        li_i, li_j = li // 2, li % 2
                        U[li_i * m + (2 * j1 + li_j)][col] += cf
                    for li, cf in v_c.items():
                        li_i, li_j = li // 2, li % 2
                        V[(2 * j1 + li_i) * p + (2 * l + li_j)][col] += cf
                    for li, cf in w_c.items():
                        li_i, li_j = li // 2, li % 2
                        W[(2 * l + li_j) * n + li_i][col] += cf
                    col += 1
            for k in range(8):
                for j2 in range(2):
                    gj = 2 * j1 + j2
                    U[2 * m + gj][col] = 1
                    V[gj * p + k][col] = 1
                    W[k * n + 2][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _build_alt_split():
        """Alt: n=3 -> 1+2. Top row 0 schoolbook (32); bottom rows 1,2 Strassen (56)."""
        rank = 88
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for j in range(4):
            for k in range(8):
                U[j][col] = 1
                V[j * p + k][col] = 1
                W[k * n][col] = 1
                col += 1
        for j1 in range(2):
            for l in range(4):
                for u_c, v_c, w_c in strassen_decomp:
                    for li, cf in u_c.items():
                        li_i, li_j = li // 2, li % 2
                        U[(1 + li_i) * m + (2 * j1 + li_j)][col] += cf
                    for li, cf in v_c.items():
                        li_i, li_j = li // 2, li % 2
                        V[(2 * j1 + li_i) * p + (2 * l + li_j)][col] += cf
                    for li, cf in w_c.items():
                        li_i, li_j = li // 2, li % 2
                        W[(2 * l + li_j) * n + (1 + li_i)][col] += cf
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _build_m_split():
        """m=4 -> 4x1, use Strassen on 2x2x2 blocks for each pair of j's.
        Actually let's try: n=3, m=4, p=8.
        Split m into 4 groups of 1: schoolbook = 3*4*8 = 96. No good.
        Split p into 2 groups of 4: for each p-block, n=3, m=4, p=4.
        For n=3, m=4, p=4: use n=2+1, m=2+2, p=4.
        Top 2x2x4: Strassen per 2x2x2 = 2*7 = 14
        Bottom 1x2x4: schoolbook = 8
        Total per p-block: 22. Two p-blocks: 44. Total rank = 44? No that's per p-block.
        Wait, p=8 split into two p=4 blocks.
        For each p=4 block: n=3, m=4, p=4.
        n=2+1, m=2+2, p=2+2.
        Top (n=2, m=4, p=4): split m into 2+2, p into 2+2.
          For each (m-block, p-block): 2x2x2 Strassen = 7 terms. 4 such blocks = 28.
        Bottom (n=1, m=4, p=4): schoolbook = 16.
        Total per p=4 block: 28 + 16 = 44.
        Two p=4 blocks: 88. Same as before.
        
        Let me try a different split: n=3, m=4, p=8.
        Split n=2+1, m=4, p=8.
        Top: n=2, m=4, p=8. Split m=2+2, p=4+4.
          For each (m-block, p-block): 2x2x4.
          2x2x4: split p into 2+2, use Strassen for each 2x2x2: 2*7=14.
          4 such blocks: 4*14 = 56.
        Bottom: n=1, m=4, p=8. Schoolbook = 32.
        Total: 56 + 32 = 88. Same.
        
        Try: n=3, m=4, p=8.
        Split m=2+2, n=3, p=8.
        For each m-block: n=3, m=2, p=8.
        n=3, m=2, p=8: split n=2+1, p=4+4.
          Top: n=2, m=2, p=4. Split p=2+2. Strassen: 2*7=14.
          Bottom: n=1, m=2, p=4. Schoolbook = 8.
          Total per m-block: 22.
        Two m-blocks: 44.
        Wait, that gives rank 44? Let me check.
        n=3, m=2, p=8.
        Top (n=2, m=2, p=8): split p into 4 blocks of 2. For each 2x2x2: Strassen 7. Total 4*7=28.
        Bottom (n=1, m=2, p=8): schoolbook = 16.
        Total per m-block: 44.
        Two m-blocks: 88. Same again.
        
        I'll just use the two existing builders.
        """
        return None

    def _verify(U, V, W, tol=1e-9):
        r = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ua[i * m + j] * Va[j * p + k], Wa[k * n + i])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def _rank_of(mat):
        return np.linalg.matrix_rank(np.array(mat, dtype=float), tol=1e-9)

    def _reduce(U, V, W, budget):
        """Tensor-span reduction: remove rank terms whose tensor is in span of others."""
        r = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)

        # Build tensor vectors for each rank term
        tensors = []
        for idx in range(r):
            t = np.zeros(n * m * p)
            pos = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        t[pos] = Ua[i * m + j, idx] * Va[j * p + k, idx] * Wa[k * n + i, idx]
                        pos += 1
            tensors.append(t)

        T_mat = np.column_stack(tensors)  # shape (n*m*p, r)
        full_rank = np.linalg.matrix_rank(T_mat, tol=1e-9)
        keep = list(range(r))

        for _pass in range(3):
            if time.time() > budget:
                break
            removed_any = False
            current_r = len(keep)
            # Rebuild with current keep
            T_cur = np.column_stack([tensors[i] for i in keep])
            cur_rank = np.linalg.matrix_rank(T_cur, tol=1e-9)
            for idx_pos in range(current_r):
                if time.time() > budget:
                    break
                idx = keep[idx_pos]
                cols = [keep[c] for c in range(current_r) if c != idx_pos]
                if not cols:
                    continue
                sub_rank = np.linalg.matrix_rank(np.column_stack([tensors[i] for i in cols]), tol=1e-9)
                if sub_rank == cur_rank:
                    keep = [c for c in keep if c != idx]
                    cur_rank = sub_rank
                    removed_any = True
            if not removed_any:
                break

        U_red = [[U[i][c] for c in keep] for i in range(len(U))]
        V_red = [[V[i][c] for c in keep] for i in range(len(V))]
        W_red = [[W[i][c] for c in keep] for i in range(len(W))]
        return U_red, V_red, W_red, len(keep)

    def _reduce_v2(U, V, W, budget):
        """Alternative: try random linear combinations to find redundant terms."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        r = Ua.shape[1]
        keep = list(range(r))
        
        # Try pairwise: for each pair (a,b), check if we can replace them with a single term
        # by finding alpha such that U_a⊗V_a⊗W_a + U_b⊗V_b⊗W_b = alpha*(U_c⊗V_c⊗W_c)
        # for some existing c, or by merging.
        # Simpler: check if any two terms sum to a third term's pattern.
        
        # Build tensor matrix
        tensors = []
        for idx in range(r):
            t = np.zeros(n * m * p)
            pos = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        t[pos] = Ua[i * m + j, idx] * Va[j * p + k, idx] * Wa[k * n + i, idx]
                        pos += 1
            tensors.append(t)
        
        T_mat = np.column_stack(tensors)
        cur_rank = np.linalg.matrix_rank(T_mat, tol=1e-9)
        
        for _pass in range(5):
            if time.time() > budget:
                break
            removed_any = False
            current_keep = list(keep)
            T_cur = np.column_stack([tensors[i] for i in current_keep])
            cur_rank = np.linalg.matrix_rank(T_cur, tol=1e-9)
            
            for idx_pos in range(len(current_keep)):
                if time.time() > budget:
                    break
                idx = current_keep[idx_pos]
                cols = [current_keep[c] for c in range(len(current_keep)) if c != idx_pos]
                if not cols:
                    continue
                sub_rank = np.linalg.matrix_rank(np.column_stack([tensors[i] for i in cols]), tol=1e-9)
                if sub_rank == cur_rank:
                    keep = [c for c in keep if c != idx]
                    cur_rank = sub_rank
                    removed_any = True
            if not removed_any:
                break
        
        U_red = [[U[i][c] for c in keep] for i in range(len(U))]
        V_red = [[V[i][c] for c in keep] for i in range(len(V))]
        W_red = [[W[i][c] for c in keep] for i in range(len(W))]
        return U_red, V_red, W_red, len(keep)

    best = None
    best_rank = 10 ** 9

    builders = [_build_block_strassen, _build_alt_split]
    for builder in builders:
        if time.time() > DEADLINE:
            break
        sol = builder()
        if sol is None:
            continue
        U, V, W = sol["U"], sol["V"], sol["W"]
        if not _verify(U, V, W):
            continue
        # Try both reduction methods
        for reducer in (_reduce, _reduce_v2):
            if time.time() > DEADLINE:
                break
            try:
                U2, V2, W2, rk = reducer(U, V, W, DEADLINE)
                if rk < best_rank and _verify(U2, V2, W2):
                    best = {"U": U2, "V": V2, "W": W2}
                    best_rank = rk
            except Exception:
                pass
        if len(U[0]) < best_rank and _verify(U, V, W):
            best = {"U": U, "V": V, "W": W}
            best_rank = len(U[0])

    if best is None:
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
        best = {"U": U, "V": V, "W": W}
        best_rank = rank

    return best
