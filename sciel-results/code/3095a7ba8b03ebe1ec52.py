import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    # Construct the [18, 9, 6] extended quadratic-residue code
    # via GF(2^8) polynomial arithmetic

    def gf28_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= 0x11D
        return res

    def gf28_pow(a, e):
        res = 1
        while e:
            if e & 1:
                res = gf28_mul(res, a)
            a = gf28_mul(a, a)
            e >>= 1
        return res

    # Find a primitive 17th root of unity in GF(2^8)
    alpha = None
    for candidate in range(2, 256):
        if gf28_pow(candidate, 17) == 1 and candidate != 1:
            alpha = candidate
            break

    if alpha is None:
        return _greedy_fallback(32, deadline)

    # Generator polynomial roots: quadratic non-residues mod 17
    non_residues = [3, 5, 6, 7, 10, 11, 12, 14]

    # Build g(x) = prod (x - alpha^i) over non-residues
    g_coeffs = [1]
    for i in non_residues:
        root = gf28_pow(alpha, i)
        new_g = [0] * (len(g_coeffs) + 1)
        for j in range(len(g_coeffs)):
            new_g[j] ^= gf28_mul(g_coeffs[j], root)
            new_g[j + 1] ^= g_coeffs[j]
        g_coeffs = new_g

    # Coefficients should be binary (0 or 1) due to cyclotomic closure
    bin_g = [1 if c == 1 else 0 for c in g_coeffs]

    # Build 9x17 generator matrix for [17, 9, 5] cyclic code
    G_17 = np.zeros((9, 17), dtype=np.uint8)
    base_vec = bin_g + [0] * (17 - len(bin_g))
    for i in range(9):
        for j in range(9):
            idx = (i + j) % 17
            G_17[i, idx] = base_vec[j]

    # Extend to length 18 with parity bit
    parity = G_17.sum(axis=1) % 2
    G_18 = np.zeros((9, 18), dtype=np.uint8)
    G_18[:, :17] = G_17
    G_18[:, 17] = parity

    # Generate all 512 codewords using NumPy
    # For each message 0..511, compute the XOR of selected rows
    messages = np.arange(512, dtype=np.uint16)
    bits = ((messages[:, None] >> np.arange(9)) & 1).astype(np.uint8)  # 512x9
    codeword_matrix = (bits @ G_18) % 2  # 512x18

    # Convert to list of integers
    codewords = []
    for i in range(512):
        val = 0
        for j in range(18):
            if codeword_matrix[i, j]:
                val |= (1 << j)
        codewords.append(val)

    # Verify using NumPy for speed
    if _verify_fast(codewords):
        return codewords

    # Fallback
    return _greedy_fallback(512, deadline)


def _verify_fast(words):
    """Fast verification using NumPy."""
    n = len(words)
    if n < 2:
        return True
    # Check duplicates
    if len(set(words)) != n:
        return False
    # Convert to numpy array
    arr = np.array(words, dtype=np.uint32)
    # Use broadcasting: arr[:, None] ^ arr[None, :] gives all pairwise XORs
    # But this is 512x512 = 262144 elements, manageable
    xor_matrix = arr[:, None] ^ arr[None, :]
    # Count bits: use lookup or loop
    # For 512x512, we can use a vectorized popcount
    # np doesn't have direct popcount for uint32, so use a trick
    # Split into bytes and use lookup
    # Actually, let's use a simpler approach: check upper triangle only
    # Use the fact that we just need min distance >= 6
    # Compute popcount for each element
    # Method: iterative bit counting
    popcount = np.zeros(xor_matrix.shape, dtype=np.uint8)
    x = xor_matrix.copy()
    for _ in range(8):
        popcount += (x & 1)
        x >>= 1
    # Check all off-diagonal entries
    np.fill_diagonal(popcount, 255)  # Mark diagonal as valid
    if popcount.min() < 6:
        return False
    return True


def _verify(words):
    """Check that all pairs have Hamming distance >= 6."""
    n = len(words)
    seen = set()
    for w in words:
        if w in seen:
            return False
        seen.add(w)
    for i in range(n):
        for j in range(i + 1, n):
            if (words[i] ^ words[j]).bit_count() < 6:
                return False
    return True


def _greedy_fallback(target_size, deadline=None):
    """Greedy random search to build a code with min distance 6."""
    if deadline is None:
        deadline = time.time() + 75.0

    codewords = [0]
    random.seed(42)

    while len(codewords) < target_size:
        if time.time() > deadline:
            break
        candidate = random.getrandbits(18)
        valid = True
        for w in codewords:
            if (candidate ^ w).bit_count() < 6:
                valid = False
                break
        if valid:
            codewords.append(candidate)

    return codewords

if __name__ == "__main__":
    words = construct()
    print(f"Generated {len(words)} codewords")
    valid = _verify_fast(words)
    print(f"Valid: {valid}")
