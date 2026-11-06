import numpy as np
import time
from scipy import sparse

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Step 1: Build the 256-word [17,9,6] shortened quadratic residue code
    poly = 0x11D

    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= poly
        return res & 0xFF

    def gf_pow(a, n):
        res = 1
        base = a
        while n:
            if n & 1:
                res = gf_mul(res, base)
            base = gf_mul(base, base)
            n >>= 1
        return res

    beta = None
    for candidate in range(2, 256):
        if gf_pow(candidate, 17) == 1 and candidate != 1:
            beta = candidate
            break

    if beta is None:
        # Fallback: 32-word seed
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # Build generator polynomial
    poly_coeffs = [1]
    current_beta = beta
    for _ in range(8):
        new_coeffs = [0] * (len(poly_coeffs) + 1)
        for i in range(len(poly_coeffs)):
            new_coeffs[i+1] = poly_coeffs[i]
        for i in range(len(poly_coeffs)):
            new_coeffs[i] ^= gf_mul(poly_coeffs[i], current_beta)
        poly_coeffs = new_coeffs
        current_beta = gf_mul(current_beta, current_beta)

    g_bits = [0] * 9
    for i, c in enumerate(poly_coeffs):
        g_bits[i] = 1 if c == 1 else 0

    # Generate 256 codewords via extend-and-shorten
    code_list = []
    seen = set()
    for u in range(512):
        c = 0
        for k in range(9):
            if (u >> k) & 1:
                for j in range(9):
                    if g_bits[j]:
                        c ^= 1 << (j + k)
        parity = bin(c).count('1') % 2
        ec = c | (parity << 17)
        if (ec >> 17) & 1 == 0:
            w = ec & ((1 << 17) - 1)
            if w not in seen:
                seen.add(w)
                code_list.append(w)

    if len(code_list) < 256:
        return code_list

    # Step 2: Greedy expansion using scipy sparse for fast distance checks
    used = np.zeros(2**17, dtype=bool)
    for w in code_list:
        used[w] = True

    # Precompute sparse bit-matrix for each code word
    def make_sparse_bitmat(w, n=17):
        bits = [int((w >> i) & 1) for i in range(n)]
        return sparse.csr_matrix(bits, shape=(n, 1), dtype=np.int8)

    # For fast popcount of XOR, use numpy
    def fast_popcount(arr):
        """Vectorized popcount for uint32 array."""
        x = arr.copy()
        x = x - ((x >> 1) & 0x55555555)
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
        x = (x + (x >> 4)) & 0x0F0F0F0F
        return (x * 0x01010101) >> 24

    # Build numpy array of current code
    code_arr = np.array(code_list, dtype=np.uint32)

    def check_batch(candidates_batch):
        """Return indices of candidates with min distance >= 6 to all code words."""
        if len(candidates_batch) == 0:
            return []
        c_arr = np.array(candidates_batch, dtype=np.uint32)
        # Chunk to avoid huge matrices
        results = []
        chunk = 512
        for i in range(0, len(c_arr), chunk):
            sub = c_arr[i:i+chunk]
            xor_mat = sub[:, np.newaxis] ^ code_arr[np.newaxis, :]
            dist_mat = fast_popcount(xor_mat)
            min_d = np.min(dist_mat, axis=1)
            valid = np.where(min_d >= 6)[0]
            results.extend(i + valid.tolist())
        return results

    # All candidate words (not yet used)
    all_candidates = [w for w in range(2**17) if not used[w]]

    # Multi-pass greedy with different orderings
    orderings = [
        ('asc', sorted(all_candidates)),
        ('desc', sorted(all_candidates, reverse=True)),
        ('weight_asc', sorted(all_candidates, key=lambda w: w.bit_count())),
        ('weight_desc', sorted(all_candidates, key=lambda w: w.bit_count(), reverse=True)),
    ]

    for pass_name, ordering in orderings:
        if time.time() > deadline - 5.0:
            break
        batch_sz = 4096
        for start in range(0, len(ordering), batch_sz):
            if time.time() > deadline - 5.0:
                break
            batch = ordering[start:start+batch_sz]
            # Filter already used
            fresh = [w for w in batch if not used[w]]
            if not fresh:
                continue
            valid_idx = check_batch(fresh)
            if not valid_idx:
                continue
            # Greedily pick mutually compatible words from valid set
            picked = []
            for idx in valid_idx:
                w = fresh[idx]
                if used[w]:
                    continue
                ok = True
                for pw in picked:
                    if (w ^ pw).bit_count() < 6:
                        ok = False
                        break
                if ok:
                    picked.append(w)
            for w in picked:
                used[w] = True
                code_list.append(w)
                code_arr = np.append(code_arr, w)

    return code_list

if __name__ == "__main__":
    words = construct()
    print(f"Number of words: {len(words)}")
