"""
Combined program: Bitmask-based direct search for minimal start s for 5 consecutive QRs mod 1009.

Strategy: Represent the set of non-zero quadratic residues as a single integer bitmask.
Bit v is set if v is a quadratic residue mod 1009. To check if s..s+4 are all QRs,
we test (mask >> s) & 0x1F == 0x1F. This is a single bitwise operation per candidate s.

We iterate s from 1 to 1004, returning the first valid s (which is minimal by construction).

Refinements:
- Precompute the roots list as a Python list indexed by residue value for O(1) access
  instead of dictionary lookups.
- Use a tight deadline guard well within the 90s wall-clock limit.
- The bitmask approach reduces the 5-consecutive-residue check from 5 lookups to
  a single bitwise AND and comparison.
"""
import time

P = 1009
M = 5


def construct():
    t_start = time.time()
    deadline = t_start + 60.0  # safety margin before 90s kill

    # Build mapping from residue to root
    # roots[r] = x such that x^2 % P == r, with 1 <= x <= P-1
    # Use a list indexed by residue for O(1) access
    roots = [0] * P
    for x in range(1, P):
        r = (x * x) % P
        if r != 0 and roots[r] == 0:
            roots[r] = x

    # Build bitmask: bit v is set if v is a non-zero quadratic residue
    mask = 0
    for r in range(1, P):
        if roots[r] != 0:
            mask |= (1 << r)

    # Search for minimal s where s, s+1, ..., s+4 are all quadratic residues
    # s must be in [1, P-M] = [1, 1004] to avoid wraparound
    target = (1 << M) - 1  # 0b11111 = 31

    for s in range(1, P - M + 1):
        # Check if bits s, s+1, s+2, s+3, s+4 are all set
        if (mask >> s) & target == target:
            # Construct result with roots (list access is O(1))
            result_roots = [roots[s + i] for i in range(M)]
            return {'start': s, 'roots': result_roots}

        # Time guard (should never trigger for p=1009, but per directive)
        if time.time() > deadline:
            break

    # Fallback: should not happen for p=1009, but return best effort
    for s in range(1, P - M + 1):
        if (mask >> s) & target == target:
            result_roots = [roots[s + i] for i in range(M)]
            return {'start': s, 'roots': result_roots}

    # Ultimate fallback (mathematically impossible for p=1009)
    return {'start': 1004, 'roots': [1, 1, 1, 1, 1]}
