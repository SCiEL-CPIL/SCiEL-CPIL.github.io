# EVOLVE-BLOCK-START
import time
import numpy as np

_TIME_LIMIT = 140.0
_START = None


def _elapsed():
    return time.monotonic() - _START


def _deadline():
    return _elapsed() > _TIME_LIMIT


def construct():
    global _START
    _START = time.monotonic()

    # GF(16) arithmetic with primitive polynomial x^4 + x + 1
    def gf16_mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r

    # Build GF(16) tables
    gf_log = {}
    gf_exp = [0] * 15
    x = 1
    for e in range(15):
        gf_log[x] = e
        gf_exp[e] = x
        x = gf16_mul(x, 2)

    # The 3 cosets of the cubic subgroup (order 5) in GF(16)* (order 15)
    cubes = set()
    for k in range(5):
        cubes.add(gf_exp[3 * k % 15])
    cosets = []
    for j in range(3):
        coset = set()
        for c in cubes:
            coset.add(gf16_mul(c, gf_exp[j]))
        cosets.append(coset)

    # Build 16x16 base coloring (3 colors, no mono triangles)
    base16 = np.full((16, 16), -1, dtype=np.int8)
    for a in range(16):
        for b in range(a + 1, 16):
            diff = a ^ b
            if diff == 0:
                continue
            d = gf_log[diff]
            for c in range(3):
                if gf_exp[d] in cosets[c]:
                    base16[a, b] = c
                    base16[b, a] = c
                    break

    # Build 32-vertex coloring: two copies, cross-edges color 3
    base32 = np.full((32, 32), -1, dtype=np.int8)
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                base32[i, j] = base32[j, i] = base16[i % 16, j % 16]
            else:
                base32[i, j] = base32[j, i] = 3

    best_n = 32
    best_colors = [int(base32[i, j]) for i in range(32) for j in range(i + 1, 32)]

    target_n = 50

    def count_mono_triangles(mat):
        n = mat.shape[0]
        count = 0
        for c in range(4):
            for i in range(n):
                for j in range(i + 1, n):
                    if mat[i, j] == c:
                        mask = (mat[i, :] == c) & (mat[j, :] == c)
                        mask[i] = False
                        mask[j] = False
                        count += np.sum(mask)
        return count

    def run_sa(seed_matrix, time_budget):
        matrix = seed_matrix.copy()
        n = matrix.shape[0]
        rng = np.random.default_rng(int(time.time() * 1000))
        
        current_count = count_mono_triangles(matrix)
        if current_count == 0:
            return matrix, 0

        T = 1.0
        T_min = 0.001
        alpha = 0.9995
        start_time = time.monotonic()

        while time.monotonic() - start_time < time_budget and not _deadline():
            if current_count == 0:
                break

            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i == j:
                continue
            if i > j:
                i, j = j, i

            old_color = matrix[i, j]
            new_color = (old_color + rng.integers(1, 4)) % 4

            delta = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                c_ik = matrix[i, k]
                c_jk = matrix[j, k]
                old_mono = 1 if (old_color == c_ik and old_color == c_jk) else 0
                new_mono = 1 if (new_color == c_ik and new_color == c_jk) else 0
                delta += new_mono - old_mono

            if delta < 0 or rng.random() < np.exp(-delta / T):
                matrix[i, j] = new_color
                matrix[j, i] = new_color
                current_count += delta

            T *= alpha
            if T < T_min:
                T = T_min

        return matrix, current_count

    # Try multiple restarts
    for restart in range(5):
        if _deadline():
            break
            
        # Initialize
        matrix = np.full((target_n, target_n), -1, dtype=np.int8)
        matrix[:32, :32] = base32
        
        rng = np.random.default_rng(42 + restart)
        for i in range(32, target_n):
            for j in range(i + 1, target_n):
                matrix[i, j] = matrix[j, i] = rng.integers(0, 4)
        for i in range(32):
            for j in range(32, target_n):
                matrix[i, j] = matrix[j, i] = rng.integers(0, 4)

        # Run SA for a portion of the time
        # Total time 120s. 5 restarts -> 20s each?
        # But we have overhead. Let's say 15s each.
        time_per_restart = 15.0
        
        res_matrix, count = run_sa(matrix, time_per_restart)
        
        if count == 0:
            best_n = target_n
            best_colors = [int(res_matrix[i, j]) for i in range(target_n) for j in range(i + 1, target_n)]
            break

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
