"""Hybrid Strassen(4x4x4) + schoolbook(4x4x3) exact decomposition for 4x4x7."""
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 4, 4, 7

    def base2x2_terms(u, v, w):
        def a(x, y):
            d = dict(x)
            for k, val in y.items():
                d[k] = d.get(k, 0) + val
            return d
        def s(x, y):
            d = dict(x)
            for k, val in y.items():
                d[k] = d.get(k, 0) - val
            return d
        M1u = a({(0,0): u[(0,0)], (1,1): u[(1,1)]}, {})
        M1v = a({(0,0): v[(0,0)], (1,1): v[(1,1)]}, {})
        M2u = a({(1,0): u[(1,0)], (1,1): u[(1,1)]}, {})
        M2v = {(0,0): v[(0,0)]}
        M3u = {(0,0): u[(0,0)]}
        M3v = s({(0,1): v[(0,1)], (1,1): v[(1,1)]}, {})
        M4u = {(1,1): u[(1,1)]}
        M4v = s({(1,0): v[(1,0)], (0,0): v[(0,0)]}, {})
        M5u = a({(0,0): u[(0,0)], (0,1): u[(0,1)]}, {})
        M5w = {(1,1): w[(1,1)]}
        M6u = s({(1,0): u[(1,0)], (0,0): u[(0,0)]}, {})
        M6w = a({(0,0): w[(0,0)], (0,1): w[(0,1)]}, {})
        M7u = s({(1,1): u[(1,1)], (0,1): u[(0,1)]}, {})
        M7w = a({(1,0): w[(1,0)], (0,1): w[(0,1)]}, {})
        return [
            (M1u, M1v, {}),
            (M2u, M2v, {}),
            (M3u, M3v, {}),
            (M4u, M4v, {}),
            (M5u, {}, M5w),
            (M6u, {}, M6w),
            (M7u, {}, M7w),
        ]

    def blk(rows, cols):
        return {(r, c) for r in rows for c in cols}
    A = blk((0,1),(0,1)); B = blk((0,1),(2,3))
    C = blk((2,3),(0,1)); D = blk((2,3),(2,3))
    E = blk((0,1),(0,1)); F = blk((0,1),(2,3))
    G = blk((2,3),(0,1)); H = blk((2,3),(2,3))
    I = blk((0,1),(0,1)); J = blk((0,1),(2,3))
    K = blk((2,3),(0,1)); L = blk((2,3),(2,3))

    def il_of(block):
        keys = sorted(block)
        return { keys[r*2+c]: (r,c) for r in range(2) for c in range(2) }
    def lm_of(block):
        keys = sorted(block)
        return { (r,c): keys[r*2+c] for r in range(2) for c in range(2) }

    I_local = il_of(I)
    def remap_wg(wg_over_I, target):
        tlm = lm_of(target)
        out = {}
        for k, v in wg_over_I.items():
            local = I_local[k]
            out[tlm[local]] = v
        return out

    def expand(u_block, v_block, w_block):
        ui = il_of(u_block); vi = il_of(v_block); wi = il_of(w_block)
        u2 = {ui[k]: 1 for k in u_block}
        v2 = {vi[k]: 1 for k in v_block}
        w2 = {wi[k]: 1 for k in w_block}
        local_terms = base2x2_terms(u2, v2, w2)
        out = []
        for (ul, vl, wl) in local_terms:
            ug = {ui[k]: val for k, val in ul.items()} if ul else {}
            vg = {vi[k]: val for k, val in vl.items()} if vl else {}
            wg = {wi[k]: val for k, val in wl.items()} if wl else {}
            out.append((ug, vg, wg))
        return out

    def comb(*blocks, signs=None):
        d = {}
        for i, b in enumerate(blocks):
            s = signs[i] if signs else 1
            for k in b:
                d[k] = d.get(k, 0) + s
        return d

    all_terms = []
    def processM(ug, vg, targets):
        exp = expand(ug, vg, I)
        for (ug_t, vg_t, wg_t) in exp:
            for (target, sign) in targets:
                wg_target = remap_wg(wg_t, target)
                wg_signed = {k: sign * v for k, v in wg_target.items()}
                all_terms.append((ug_t, vg_t, wg_signed))

    processM(comb(A, D), comb(E, H), [(I, 1), (L, 1)])
    processM(comb(C, D), E, [(K, 1), (L, -1)])
    processM(A, comb(F, H, signs=(1, -1)), [(J, 1)])
    processM(D, comb(G, E, signs=(1, -1)), [(J, 1), (I, -1)])
    processM(comb(A, B), L, [(K, 1), (L, 1)])
    processM(comb(C, A, signs=(1, -1)), comb(I, J), [(I, 1), (L, 1)])
    processM(comb(D, B, signs=(1, -1)), comb(K, J), [(I, 1)])

    assert len(all_terms) == 49, len(all_terms)

    U1 = [[0]*49 for _ in range(16)]
    V1 = [[0]*49 for _ in range(16)]
    W1 = [[0]*49 for _ in range(16)]
    for c, (ug, vg, wg) in enumerate(all_terms):
        for k, v in ug.items():
            U1[k[0]*4 + k[1]][c] = v
        for k, v in vg.items():
            V1[k[0]*4 + k[1]][c] = v
        for k, v in wg.items():
            W1[k[0]*4 + k[1]][c] = v

    U2 = [[0]*48 for _ in range(16)]
    V2 = [[0]*48 for _ in range(12)]
    W2 = [[0]*48 for _ in range(12)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U2[i*4 + j][col] = 1
                V2[j*3 + k][col] = 1
                W2[k*4 + i][col] = 1
                col += 1

    U = [U1[r] + U2[r] for r in range(16)]
    V = [[0]*97 for _ in range(28)]
    for r in range(16):
        j = r // 4; k = r % 4
        Vrow = j*7 + k
        V[Vrow] = [V1[r][c] for c in range(49)] + [0]*48
    for r in range(12):
        j = r // 3; k = r % 3
        Vrow = j*7 + (k + 4)
        V[Vrow] = [0]*49 + [V2[r][c] for c in range(48)]
    W = [[0]*97 for _ in range(28)]
    for r in range(16):
        W[r] = [W1[r][c] for c in range(49)] + [0]*48
    for r in range(12):
        k = r // 4; i = r % 4
        Wrow = (k + 4)*4 + i
        W[Wrow] = [0]*49 + [W2[r][c] for c in range(48)]

    if time.time() > DEADLINE:
        # Should never happen; construction is fast. Return best so far.
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
