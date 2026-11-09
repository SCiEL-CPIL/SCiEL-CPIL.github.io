"""Combined program: known size-67 seed + exact branch-and-bound over the
small-number region (a structural change from randomized local search)."""
import time

# EVOLVE-BLOCK-START

# ---------- fast validity ----------
def _valid(S):
    """True iff no three distinct a<b<c in S have a+b, a+c, b+c all in S."""
    Sset = set(S)
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(i + 1, n):
            b = S[j]
            if (a + b) not in Sset:
                continue
            for k in range(j + 1, n):
                c = S[k]
                if (a + c) in Sset and (b + c) in Sset:
                    return False
    return True

# ---------- bit-parallel "forbidden pair" test ----------
# bit i of mask  <=>  integer (i+1) present.
# x is addable to mask iff for NO a in mask does the "forbidden partner" b
# (b = x + a - c for some c in mask) also lie in mask.  Equivalently:
#   forb = OR over a in mask of (mask << (x + a)) ;  x addable iff forb == 0.
def _forb(x, mask):
    f = 0
    m = mask
    while m:
        lb = m & (-m)
        a = lb.bit_length()          # a = (bit index)+1 = element value
        sh = x + a
        if sh <= 128:
            f |= (mask << sh)
        m ^= lb
    return f

def _cost(x, mask):
    """Number of (a,b) pairs in mask that would make {x,a,b} forbidden."""
    return _forb(x, mask).bit_count()

# ---------- branch and bound over small numbers 2..63 ----------
def _b_and_b(base_mask, best_mask, best_size, deadline):
    """Maximize |small| over small subset of {2..63} such that
    base_mask | small_mask is valid.  Returns (best_small_mask, best_size)."""
    small = list(range(2, 64))
    N = len(small)

    # order candidates by ascending current "cost" (least damaging first)
    order = sorted(range(N), key=lambda i: _cost(small[i], base_mask))
    vals = [small[i] for i in order]

    result = [0, 0]  # [best_small_mask, best_size]

    def rec(d, mask, size):
        if time.time() > deadline:
            return
        # prune: even taking everything left cannot beat current best
        if size + (N - d) <= result[1]:
            return
        if d == N:
            if size > result[1]:
                result[1] = size
                result[0] = mask
            return
        x = vals[d]
        f = _forb(x, mask)
        if f == 0:
            # x is addable -> explore adding it first (greedy bias)
            rec(d + 1, mask | (1 << (x - 1)), size + 1)
            rec(d + 1, mask, size)
        else:
            # x not addable -> must skip it
            rec(d + 1, mask, size)

    rec(0, 0, 0)
    return result[0], result[1]

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before the 90s kill

    # Known-valid seed of size 67: {1,62} U {64..128}.
    # - Triples inside {62,64..128}: pair sums are only 126,127,128 (or >128);
    #   no triple has all three sums present.
    # - Triples with 1: {1,b,c}, b,c>=64 => b+c>=129 (not in S).
    base = [1, 62] + list(range(64, 129))
    base_mask = 0
    for v in base:
        base_mask |= (1 << (v - 1))
    best_mask = base_mask
    best_size = len(base)

    # Exact branch-and-bound to maximize extra small numbers (2..63).
    small_mask, small_size = _b_and_b(base_mask, base_mask, best_size, deadline)
    if small_size > 0:
        cand_mask = base_mask | small_mask
        cand_list = [i + 1 for i in range(128) if (cand_mask >> i) & 1]
        if _valid(cand_list):
            best_mask = cand_mask
            best_size = len(cand_list)

    # Final safety: verify; fall back to the guaranteed seed if anything is off.
    best_list = [i + 1 for i in range(128) if (best_mask >> i) & 1]
    if not _valid(best_list):
        best_list = [1, 62] + list(range(64, 129))

    return [int(x) for x in sorted(best_list)]

# EVOLVE-BLOCK-END
