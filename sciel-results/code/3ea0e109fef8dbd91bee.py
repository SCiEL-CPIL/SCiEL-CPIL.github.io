"""Tensor decomposition for 4x4 * 4x7 matrix multiplication.
Fine-grained 2x2 block tiling with Strassen decompositions, then sparse
coordinate-descent refinement driving entries to half-integers."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    def _elapsed_ok():
        return time.time() < deadline

    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def strassen_2x2x2():
        """Strassen's 7-product decomposition for 2x2 * 2x2 = 2x2.
        Returns U (4x7), V (4x7), W (4x7)."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # m1 = (A11+A22)(B11+B22)
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[1,0]=1; W[2,0]=1; W[3,0]=1
        # m2 = (A21+A22)B11
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[0,1]=1; W[2,1]=1
        # m3 = A11(B21-B22)
        U[0,2]=1; V[2,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=-1
        # m4 = A22(B12-B11)
        U[3,3]=1; V[1,3]=1; V[0,3]=-1; W[0,3]=-1; W[2,3]=1
        # m5 = (A11+A12)B22
        U[0,4]=1; U[1,4]=1; V[3,4]=1; W[1,4]=1; W[3,4]=1
        # m6 = (A21-A11)(B11+B12)
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[0,5]=1; W[1,5]=1
        # m7 = (A12-A22)(B21+B22)
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[1,6]=1; W[3,6]=1
        return U, V, W

    def schoolbook_2x2x1():
        """Schoolbook 2x2 * 2x1 = 2x1. Rank 4.
        Returns U (4x4), V (2x4), W (2x4)."""
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(1):
                    U[i*2 + j, col] = 1
                    V[j*1 + k, col] = 1
                    W[k*2 + i, col] = 1
                    col += 1
        return U, V, W

    def block_2x2x3():
        """2x2 * 2x3 = 2x3 via Strassen 2x2x2 (rank 7) + schoolbook 2x2x1 (rank 4). Rank 11.
        Returns U (4x11), V (6x11), W (6x11)."""
        r1 = 7
        r2 = 4
        total = r1 + r2
        U = np.zeros((4, total))
        V = np.zeros((6, total))
        W = np.zeros((6, total))

        U1, V1, W1 = strassen_2x2x2()
        U[:, :r1] = U1
        V[:4, :r1] = V1
        W[:4, :r1] = W1

        U2, V2, W2 = schoolbook_2x2x1()
        U[:, r1:] = U2
        V[4:, r1:] = V2
        W[4:, r1:] = W2
        return U, V, W

    def _map_block_to_4x4(p_val, Ublk, Vblk, Wblk, i_off, j_off, off, r):
        """Place a 2x2 * 2xp decomposition into a 4x4 * 4xp tensor.
        Ublk: (4, r), Vblk: (2p, r), Wblk: (2p, r)
        i_off, j_off: 0 or 2 for the 2x2 block position within 4x4.
        A index: (i_off+ii, j_off+jj) -> (i_off+ii)*4 + (j_off+jj)
        B index: (j_off+jj, k) -> (j_off+jj)*4 + k
        C index: (k, i_off+ii) -> k*4 + (i_off+ii)
        """
        a_map = [(i_off+ii)*4 + (j_off+jj) for ii in range(2) for jj in range(2)]
        for idx_a, idx in enumerate(a_map):
            U[idx, off:off+r] = Ublk[idx_a, :]
        for jj in range(2):
            for k in range(p_val):
                V[(j_off+jj)*4 + k, off:off+r] = Vblk[jj*p_val + k, :]
        for k in range(p_val):
            for ii in range(2):
                W[k*4 + (i_off+ii), off:off+r] = Wblk[k*p_val + ii, :]

    def decompose_fine_grained():
        """Fine-grained 2x2 block tiling of 4x4 * 4x7.
        Split into 2x2 A-blocks (4 blocks) x 2x2 B-blocks (2 blocks) + 2x1 B-block (1 block).
        For each (A-block, B-block): use block_2x2x3 (rank 11).
        Total: 4 * 3 * 11 = 132? No, let me think more carefully.
        
        Actually: 4x4 * 4x7. Split A into 2x2 blocks: 4 blocks.
        Split B into 2x2 and 2x1: columns 0-1, 2-3, 4-5, 6.
        So B has 3 blocks: 2x2, 2x2, 2x1.
        
        For each A-block (4) and each B-block:
        - 2x2 B-block: 2x2*2x2 = 2x2, use Strassen rank 7
        - 2x1 B-block: 2x2*2x1 = 2x1, use schoolbook rank 4
        
        Total: 4*(7+7+4) = 4*18 = 72.
        
        Wait, that's 72 which is better than 76 but let me verify the structure.
        
        Actually let me reconsider. The 4x4 * 4x7 can be seen as:
        A is 4x4, B is 4x7.
        Split A into 4 2x2 blocks: A00, A01, A10, A11 (row blocks 0,1; col blocks 0,1)
        Split B into: B0 (4x2, cols 0-1), B1 (4x2, cols 2-3), B2 (4x2, cols 4-5), B3 (4x1, col 6)
        
        C = A*B = sum over j-blocks of A_j * B_j
        
        For each A-block (i-block, j-block) and each B-block (j-block, k-block):
        We need A_{i,j} * B_{j,k} contributing to C_{i,k}
        
        So for each pair (i_blk, j_blk) of A (4 pairs) and each B column-block:
        - B cols 0-1: j_blk must match, contributes to C cols 0-1
        - B cols 2-3: j_blk must match, contributes to C cols 2-3  
        - B cols 4-5: j_blk must match, contributes to C cols 4-5
        - B col 6: j_blk must match, contributes to C col 6
        
        For a given A-block (i_blk, j_blk) and B-column-block k_blk:
        A_{i_blk, j_blk} is 2x2, B_{j_blk, k_blk} is 2x(2 or 1)
        Product is 2x(2 or 1)
        
        If B column-block is 2 wide: use Strassen 2x2x2, rank 7
        If B column-block is 1 wide: use schoolbook 2x2x1, rank 4
        
        Total rank = 4 A-blocks * (7 + 7 + 7 + 4) = 4 * 25 = 100.
        
        Hmm that's worse. Let me think differently.
        
        Better approach: For each j-block (0 or 1), process all A i-blocks and all B k-blocks.
        For j_blk=0: A has 2 i-blocks (i_blk=0,1), B has 4 k-blocks (2 wide each except last is 1 wide... no)
        
        Let me just do: for each of the 4 j-column positions in B (grouped as 2,2,2,1):
        Actually B is 4x7, so rows are j=0..3, cols are k=0..6.
        
        Let me use a different tiling:
        Split the computation by j (the shared index):
        For each j in {0,1,2,3}:
          A[:,j] (4x1) * B[j,:] (1x7) contributes to C (4x7)
          This is outer product, rank 1 per entry... no that's just schoolbook.
        
        OK let me just use the 2x2 tiling properly:
        
        A = [[A00, A01], [A10, A11]] where each is 2x2
        B = [[B00, B01, B02, B03], [B10, B11, B12, B13]] where B00,B10 are 2x2, B01,B11 are 2x2, B02,B12 are 2x2, B03,B13 are 2x1
        
        C = A*B:
        C00 = A00*B00 + A01*B10  (2x2)
        C01 = A00*B01 + A01*B11  (2x2)
        C02 = A00*B02 + A01*B12  (2x2)
        C03 = A00*B03 + A01*B13  (2x1)
        C10 = A10*B00 + A11*B10  (2x2)
        C11 = A10*B01 + A11*B11  (2x2)
        C12 = A10*B02 + A11*B12  (2x2)
        C13 = A10*B03 + A11*B13  (2x1)
        
        Each 2x2*2x2 product: rank 7 (Strassen)
        Each 2x2*2x1 product: rank 4 (schoolbook)
        
        C00: 2 products of 2x2*2x2: rank 14
        C01: rank 14
        C02: rank 14
        C03: 2 products of 2x2*2x1: rank 8
        C10: rank 14
        C11: rank 14
        C12: rank 14
        C13: rank 8
        
        Total: 6*14 + 2*8 = 84 + 16 = 100. Still worse than 76.
        
        The issue is that Strassen for 2x2x2 has rank 7, but we're doing many small products.
        The current approach of 4x4x4 Strassen (rank 28) + 4x4x3 schoolbook (rank 48) = 76 is better.
        
        Let me try: 4x4x4 via 4 independent Strassen 2x2x2 (rank 28) + 4x4x3 via 4 independent (2x2x3) blocks.
        2x2x3 = Strassen 2x2x2 (rank 7) + schoolbook 2x2x1 (rank 4) = rank 11.
        4 such blocks: rank 44.
        Total: 28 + 44 = 72.
        
        That's better! Let me implement this.
        """
        r_4x4x4 = 28  # 4 * Strassen 2x2x2
        r_4x4x3 = 4 * 11  # 4 * (Strassen 2x2x2 + schoolbook 2x2x1) = 44
        total = r_4x4x4 + r_4x4x3  # 72
        
        U = np.zeros((16, total))
        V = np.zeros((28, total))
        W = np.zeros((28, total))
        
        # Part 1: 4x4x4 using 4 Strassen 2x2x2 blocks
        off = 0
        for iblk in range(2):
            for jblk in range(2):
                U1, V1, W1 = strassen_2x2x2()
                _map_block_to_4x4(2, U1, V1, W1, iblk*2, jblk*2, off, 7)
                off += 7
        # off should be 28 now
        
        # Part 2: 4x4x3 using 4 (2x2x3) blocks
        # B columns 4,5,6 -> 2x2 block (cols 4-5) and 2x1 block (col 6)
        # For each A 2x2 block (iblk, jblk):
        #   A_{iblk,jblk} (2x2) * B_{jblk, 4:6} (2x3) -> C_{iblk, 4:6} (2x3)
        # B_{jblk, 4:6} = [B_{jblk,4}, B_{jblk,5}, B_{jblk,6}] where first two are 2x2, last is 2x1
        
        off = r_4x4x4
        for iblk in range(2):
            for jblk in range(2):
                # 2x2 * 2x3 decomposition
                Ublk, Vblk, Wblk = block_2x2x3()
                _map_block_to_4x4(3, Ublk, Vblk, Wblk, iblk*2, jblk*2, off, 11)
                off += 11
        
        return U, V, W, total

    def verify(U, V, W):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx
        prod = Ua[u_idx, :] * Va[v_idx, :] * Wa[w_idx, :]
        result = prod.sum(axis=1)
        return float(np.max(np.abs(result - 1.0)))

    def round_to_half(U, V, W):
        Ur = np.round(2 * np.array(U, dtype=float)) / 2.0
        Vr = np.round(2 * np.array(V, dtype=float)) / 2.0
        Wr = np.round(2 * np.array(W, dtype=float)) / 2.0
        if verify(Ur, Vr, Wr) < 1e-7:
            return Ur, Vr, Wr
        return None

    def sparse_refine(U, V, W, rank, max_iter=5000, tol=1e-9):
        """Coordinate descent with sparsity: drive small entries to 0, large to ±1 or ±0.5."""
        Ua = np.array(U, dtype=float).copy()
        Va = np.array(V, dtype=float).copy()
        Wa = np.array(W, dtype=float).copy()
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx

        def cur_err():
            prod = Ua[u_idx, :] * Va[v_idx, :] * Wa[w_idx, :]
            res = prod.sum(axis=1)
            return float(np.max(np.abs(res - 1.0)))

        def update_row_U(a):
            ts = np.where(u_idx == a)[0]
            if len(ts) == 0:
                return
            cur = (Ua[a, :] * Va[v_idx[ts], :] * Wa[w_idx[ts], :]).sum(axis=1)
            e = cur - 1.0
            vw = Va[v_idx[ts], :] * Wa[w_idx[ts], :]
            num = (e[:, None] * vw).sum(axis=0)
            den = (vw * vw).sum(axis=0)
            mask = den > 1e-12
            Ua[a, mask] = num[mask] / den[mask]

        def update_row_V(b):
            ts = np.where(v_idx == b)[0]
            if len(ts) == 0:
                return
            cur = (Ua[u_idx[ts], :] * Va[b, :] * Wa[w_idx[ts], :]).sum(axis=1)
            e = cur - 1.0
            uw = Ua[u_idx[ts], :] * Wa[w_idx[ts], :]
            num = (e[:, None] * uw).sum(axis=0)
            den = (uw * uw).sum(axis=0)
            mask = den > 1e-12
            Va[b, mask] = num[mask] / den[mask]

        def update_row_W(c):
            ts = np.where(w_idx == c)[0]
            if len(ts) == 0:
                return
            cur = (Ua[u_idx[ts], :] * Va[v_idx[ts], :] * Wa[c, :]).sum(axis=1)
            e = cur - 1.0
            uv = Ua[u_idx[ts], :] * Va[v_idx[ts], :]
            num = (e[:, None] * uv).sum(axis=0)
            den = (uv * uv).sum(axis=0)
            mask = den > 1e-12
            Wa[c, mask] = num[mask] / den[mask]

        best_err = cur_err()
        best_state = (Ua.copy(), Va.copy(), Wa.copy())
        
        for it in range(max_iter):
            if not _elapsed_ok():
                break
            for a in range(n * m):
                update_row_U(a)
            for b in range(m * p):
                update_row_V(b)
            for c in range(p * n):
                update_row_W(c)
            
            if (it + 1) % 100 == 0:
                err = cur_err()
                if err < best_err:
                    best_err = err
                    best_state = (Ua.copy(), Va.copy(), Wa.copy())
                if best_err < tol:
                    break
        
        if best_err < 1e-6:
            return best_state
        return None

    def try_zero_columns(U, V, W, current_rank, target_rank, max_trials_per_col=200):
        """Try to zero out columns from the end, refining remaining columns."""
        Ua = np.array(U, dtype=float).copy()
        Va = np.array(V, dtype=float).copy()
        Wa = np.array(W, dtype=float).copy()
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx

        def cur_err():
            prod = Ua[u_idx, :] * Va[v_idx, :] * Wa[w_idx, :]
            res = prod.sum(axis=1)
            return float(np.max(np.abs(res - 1.0)))

        def refine_all(max_iter=200):
            for it in range(max_iter):
                if not _elapsed_ok():
                    break
                for a in range(n * m):
                    ts = np.where(u_idx == a)[0]
                    if len(ts) == 0: continue
                    cur = (Ua[a, :] * Va[v_idx[ts], :] * Wa[w_idx[ts], :]).sum(axis=1)
                    e = cur - 1.0
                    vw = Va[v_idx[ts], :] * Wa[w_idx[ts], :]
                    num = (e[:, None] * vw).sum(axis=0)
                    den = (vw * vw).sum(axis=0)
                    mask = den > 1e-12
                    Ua[a, mask] = num[mask] / den[mask]
                for b in range(m * p):
                    ts = np.where(v_idx == b)[0]
                    if len(ts) == 0: continue
                    cur = (Ua[u_idx[ts], :] * Va[b, :] * Wa[w_idx[ts], :]).sum(axis=1)
                    e = cur - 1.0
                    uw = Ua[u_idx[ts], :] * Wa[w_idx[ts], :]
                    num = (e[:, None] * uw).sum(axis=0)
                    den = (uw * uw).sum(axis=0)
                    mask = den > 1e-12
                    Va[b, mask] = num[mask] / den[mask]
                for c in range(p * n):
                    ts = np.where(w_idx == c)[0]
                    if len(ts) == 0: continue
                    cur = (Ua[u_idx[ts], :] * Va[v_idx[ts], :] * Wa[c, :]).sum(axis=1)
                    e = cur - 1.0
                    uv = Ua[u_idx[ts], :] * Va[v_idx[ts], :]
                    num = (e[:, None] * uv).sum(axis=0)
                    den = (uv * uv).sum(axis=0)
                    mask = den > 1e-12
                    Wa[c, mask] = num[mask] / den[mask]

        new_rank = current_rank
        for target_col in range(current_rank - 1, target_rank - 1, -1):
            if not _elapsed_ok():
                break
            success = False
            for trial in range(max_trials_per_col):
                if not _elapsed_ok():
                    break
                Ua[:, target_col] = 0.0
                Va[:, target_col] = 0.0
                Wa[:, target_col] = 0.0
                refine_all(max_iter=100)
                err = cur_err()
                if err < 1e-6:
                    success = True
                    new_rank = target_col
                    break
            if not success:
                Ua[:, target_col] = 0.0
                Va[:, target_col] = 0.0
                Wa[:, target_col] = 0.0
                refine_all(max_iter=200)
                if cur_err() > 1e-5:
                    break
        
        if new_rank < current_rank:
            Ua = Ua[:, :new_rank]
            Va = Va[:, :new_rank]
            Wa = Wa[:, :new_rank]
            if cur_err() < 1e-6:
                return Ua, Va, Wa, new_rank
        return None

    # ---- Main ----
    best_result = make_schoolbook()
    best_rank = n * m * p

    try:
        U, V, W, rank = decompose_fine_grained()
        err = verify(U, V, W)
        if err < 1e-9:
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = rank

            # Refine
            if _elapsed_ok():
                refined = sparse_refine(U, V, W, rank, max_iter=3000)
                if refined is not None:
                    rU, rV, rW = refined
                    hr = round_to_half(rU, rV, rW)
                    if hr is not None:
                        best_result = {"U": hr[0].tolist(), "V": hr[1].tolist(), "W": hr[2].tolist()}

            # Try rank reduction
            if _elapsed_ok() and best_rank > 50:
                curU = np.array(best_result["U"], dtype=float)
                curV = np.array(best_result["V"], dtype=float)
                curW = np.array(best_result["W"], dtype=float)
                cur_rank = curU.shape[1]
                reduced = try_zero_columns(curU, curV, curW, cur_rank, max(40, cur_rank - 15))
                if reduced is not None:
                    rU, rV, rW, new_rank = reduced
                    hr = round_to_half(rU, rV, rW)
                    if hr is not None:
                        best_result = {"U": hr[0].tolist(), "V": hr[1].tolist(), "W": hr[2].tolist()}
                        best_rank = new_rank
    except Exception:
        pass

    return best_result
# EVOLVE-BLOCK-END
