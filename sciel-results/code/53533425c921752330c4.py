import time
import bisect
import random

def construct():
    start = time.time()
    deadline = start + 85.0
    n = 256

    def full_validate(S_list):
        n2 = len(S_list)
        for i in range(n2):
            a = S_list[i]
            for j in range(i+1, n2):
                b = S_list[j]
                for k in range(j, n2):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def can_add(a, S_sorted):
        """Check if a can be added to S_sorted (all elements > a).
        For every b, c in S_sorted with b < c (or b == c), we need (b+c) % a != 0.
        Since all elements in S_sorted are > a, b+c > 2a.
        The condition (b+c) % a == 0 means b+c is a multiple of a.
        We check: does S_sorted contain two elements (possibly same) whose sum is divisible by a?
        Using residues: for each x in S_sorted, compute r = x % a.
        We need no two residues r1, r2 (from possibly same element if 2*r1 % a == 0)
        such that (r1 + r2) % a == 0.
        """
        if a <= 1:
            return False
        # Build residue counts
        # For efficiency, use a set of residues and check
        residues = set()
        for x in S_sorted:
            r = x % a
            residues.add(r)
        # Now check: for each r in residues, is (a - r) % a also in residues?
        # If r == (a-r) % a, i.e., 2r % a == 0, then we need at least one element with that residue
        # (since b=c is allowed, one element suffices for the pair (x,x))
        # If r != (a-r) % a, we need both residues present.
        for r in residues:
            r2 = (-r) % a
            if r == r2:
                # 2r % a == 0, need at least one element with residue r (we have it)
                # But wait: b=c means we use the same element twice. So if there's one element
                # with residue r, then (x+x) % a == 2r % a == 0, which is a violation.
                return False
            else:
                if r2 in residues:
                    return False
        return True

    def greedy_add(S_list, S_set, order, deadline):
        """Try adding elements in given order."""
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for a in order:
                if a in S_set:
                    continue
                if can_add(a, S_list):
                    pos = bisect.bisect_left(S_list, a)
                    S_list.insert(pos, a)
                    S_set.add(a)
                    changed = True
        return S_list, S_set

    # Start with the known good base: [171, 256]
    best_S = set(range(171, n+1))
    best_size = len(best_S)

    # Try various starting thresholds
    for m_start in [171, 170, 169, 168, 167, 166, 165, 160, 155, 150, 145, 140, 135, 130, 128, 120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20]:
        if time.time() > deadline - 15:
            break

        # Check if [m_start, n] is valid
        base = list(range(m_start, n+1))
        valid = True
        for i in range(len(base)):
            a = base[i]
            for j in range(i+1, len(base)):
                b = base[j]
                for k in range(j, len(base)):
                    c = base[k]
                    if (b + c) % a == 0:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

        if not valid:
            continue

        S_list = base[:]
        S_set = set(S_list)

        if len(S_list) > best_size:
            best_S = set(S_list)
            best_size = len(S_list)

        # Add elements below m_start, trying from largest to smallest
        if time.time() < deadline - 10:
            S_list, S_set = greedy_add(S_list, S_set, list(range(m_start - 1, 0, -1)), deadline - 5)
            if len(S_list) > best_size:
                best_S = set(S_list)
                best_size = len(S_list)

        # Also try ascending order
        if time.time() < deadline - 8:
            S_list2 = base[:]
            S_set2 = set(S_list2)
            S_list2, S_set2 = greedy_add(S_list2, S_set2, list(range(1, m_start)), deadline - 5)
            if len(S_list2) > best_size:
                best_S = set(S_list2)
                best_size = len(S_list2)

    # Final polish: try to add any remaining elements to best_S
    if time.time() < deadline - 20:
        S_list = sorted(best_S)
        S_set = set(S_list)
        candidates = [a for a in range(n, 0, -1) if a not in S_set]
        S_list, S_set = greedy_add(S_list, S_set, candidates, deadline - 10)
        if len(S_list) > best_size:
            best_S = set(S_list)
            best_size = len(S_list)

        # Second pass: try ascending order
        if time.time() < deadline - 10:
            S_list = sorted(best_S)
            S_set = set(S_list)
            candidates = [a for a in range(1, n + 1) if a not in S_set]
            S_list, S_set = greedy_add(S_list, S_set, candidates, deadline - 5)
            if len(S_list) > best_size:
                best_S = set(S_list)
                best_size = len(S_list)

    # Local search: try removing one element and adding one or two others
    if time.time() < deadline - 15:
        S_list = sorted(best_S)
        S_set = set(S_list)
        for _ in range(50):  # limited number of local search iterations
            if time.time() > deadline - 10:
                break
            improved = False
            # Try removing each element and see if we can add more
            for i in range(len(S_list)):
                if time.time() > deadline - 10:
                    break
                removed = S_list[i]
                S_list2 = S_list[:i] + S_list[i+1:]
                S_set2 = set(S_list2)
                # Try adding all missing elements
                candidates = [a for a in range(n, 0, -1) if a not in S_set2]
                S_list2, S_set2 = greedy_add(S_list2, S_set2, candidates, deadline - 5)
                if len(S_list2) > len(S_list):
                    S_list = S_list2
                    S_set = S_set2
                    improved = True
                    if len(S_list) > best_size:
                        best_S = set(S_list)
                        best_size = len(S_list)
                    break
            if not improved:
                break

    # Final validation
    if time.time() < deadline:
        S_final = sorted(best_S)
        if len(S_final) > 5 and not full_validate(S_final):
            S_final = list(range(171, n+1))
        return S_final

    return sorted(best_S)
