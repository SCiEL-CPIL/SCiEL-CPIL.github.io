import time

def construct():
    # The problem space is tiny (p=1009, residues 1..1008).
    # An exhaustive search over all start positions with efficient
    # validity checks can find the true maximum interval length
    # well within the time budget.
    #
    # Key observations:
    # 1. Self-inverse residues (x where x^2 ≡ 1 mod p) cannot be in any valid interval.
    #    For p=1009 (prime), x^2 ≡ 1 mod p has solutions x=1 and x=p-1=1008.
    # 2. For any x in the interval, inv(x) must NOT be in the interval.
    # 3. We need to find the longest contiguous [start, start+len-1] ⊆ [1, 1008]
    #    such that for all x,y in the interval, x*y % 1009 != 1.
    #
    # Strategy: For each possible start, extend the interval as far as possible
    # while maintaining validity. Use a precomputed inverse array for O(1) lookup.

    p = 1009
    deadline = time.time() + 85.0

    # Precompute modular inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)

    # Self-inverse residues: x where x*x % p == 1
    # For prime p, these are 1 and p-1
    self_inv = set()
    for x in range(1, p):
        if x * x % p == 1:
            self_inv.add(x)
    # self_inv = {1, 1008}

    best_start = 2
    best_len = 1

    # For each start position, find the maximum valid interval length
    for s in range(1, p):
        if time.time() > deadline:
            break

        # If s is a self-inverse residue, no interval starting here is valid
        if s in self_inv:
            continue

        # Try to extend the interval from s
        # Maintain a set of elements in the current interval
        # For each new element e, check that inv[e] is not in the set
        # and that e is not self-inverse
        current_set = set()
        current_set.add(s)

        for e in range(s + 1, p):
            if time.time() > deadline:
                break

            # e cannot be self-inverse
            if e in self_inv:
                break  # Contiguous interval must stop

            # inv[e] must not be in current_set
            if inv[e] in current_set:
                break

            current_set.add(e)

        length = len(current_set)
        if length > best_len:
            best_len = length
            best_start = s

    return [best_start, best_len]
