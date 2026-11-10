"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start = time.time()
    deadline = 80.0  # safety margin well before the 90 s kill

    n = 12
    P = 27720  # lcm(1..12); the pattern of divisibility is P-periodic in m

    # Precompute, for each position p (0-based, value = m+p), the set of indices
    # i (1..12) that divide that value.  This depends only on p mod P.
    # divset[p] = bitmask of i dividing (P + p)  [value m+p with m a multiple of P]
    divset = [0] * P
    for p in range(P):
        val = P + p
        mask = 0
        for i in range(1, n + 1):
            if val % i == 0:
                mask |= (1 << i)
        divset[p] = mask

    def can_assign(m, L):
        """Hall's condition via bitsets: can we pick distinct b_i in (m, m+L] with i|b_i?"""
        # Window positions: values m+1 .. m+L  ->  positions 1..L (mod P)
        # For each index i, the set of window positions where i divides the value.
        # Hall: for every subset S of indices, |union of their position sets| >= |S|.
        # We check via maximal subsets (the family is closed under union),
        # but simplest: iterate over all 2^n subsets of indices.
        # n=12 -> 4096 subsets; each union computed incrementally.
        # Build per-index position bitsets over the window.
        # Represent window positions as a Python int bitmask (bit j = position j, 0-based).
        idx_sets = [0] * (n + 1)  # 1-based index i -> bitmask of window positions
        # position j (1-based) corresponds to value m+j, global pos = (m + j) mod P
        # precompute per-index: we can build by scanning window once
        # For each window position j (1..L), value v = m+j, gpos = v mod P (0..P-1)
        # every i dividing v gets bit (j-1) set.
        for j in range(1, L + 1):
            gpos = (m + j) % P
            mask = divset[gpos]
            bit = 1 << (j - 1)
            # set bit for each i in mask
            mm = mask
            while mm:
                low = mm & (-mm)
                i = low.bit_length() - 1
                idx_sets[i] |= bit
                mm ^= low
        # Hall over all subsets of {1..n}
        # union of sets for subset S; need popcount(union) >= popcount(S)
        # Iterate subsets; compute union incrementally using DP over subsets.
        # union_bits[S] = union of idx_sets[i] for i in S
        # popcount of subset S = S.bit_count()
        # We need union for all 4096 subsets; compute via DP:
        # union_bits[0]=0; for S>0, let lb = S & -S, i = idx, union_bits[S] = union_bits[S^lb] | idx_sets[i]
        # That's 4096 ORs of big ints (up to 144 bits) - cheap.
        all_sub = 1 << n
        union_bits = [0] * all_sub
        for S in range(1, all_sub):
            lb = S & (-S)
            i = lb.bit_length() - 1  # index i (1..12)
            union_bits[S] = union_bits[S ^ lb] | idx_sets[i]
        for S in range(1, all_sub):
            if union_bits[S].bit_count() < S.bit_count():
                return False
        return True

    def f_of(m):
        """Least positive L with a valid assignment; cap at 144."""
        # binary search on L in [1, 144]
        lo, hi = 1, 144
        # find minimal L that works
        # monotone: if works at L, works at L+1? Adding positions only adds candidates,
        # so feasibility is monotone non-decreasing in L. Yes.
        ans = 144
        while lo <= hi:
            mid = (lo + hi) // 2
            if can_assign(m, mid):
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return ans

    # Compute the full f table over one period.
    # Use periodicity: f(m) depends on m mod P.
    table = [0] * P
    for r in range(P):
        if time.time() > deadline:
            # fallback: return a safe default
            return 0
        table[r] = f_of(r)

    maxL = max(table)
    # Find an m in [0, 1000000] attaining maxL.
    # m = r works if r <= 1000000 (it is, since P=27720).
    best_m = None
    for r in range(P):
        if table[r] == maxL:
            best_m = r
            break
    if best_m is None:
        best_m = 0

    return int(best_m)
# EVOLVE-BLOCK-END
