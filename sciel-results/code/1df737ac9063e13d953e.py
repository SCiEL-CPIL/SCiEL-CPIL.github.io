"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _is_valid(S):
    """Check that no three distinct a,b,c in S have all three pair sums in S."""
    s = set(S)
    for i, a in enumerate(S):
        for j in range(i+1, len(S)):
            b = S[j]
            ab = a + b
            if ab not in s:
                continue
            for k in range(j+1, len(S)):
                c = S[k]
                if a + c in s and b + c in s:
                    return False
    return True

def construct():
    deadline = time.time() + 75.0  # safety margin before 90s kill
    best = list(range(17, 33)) + list(range(65, 129))  # seed: 16 + 64 = 80
    best_size = len(best)

    def try_add(S, x):
        """Try to add x to S. Returns new list or None if invalid."""
        s = set(S)
        if x in s:
            return None
        # Adding x creates violation iff exist distinct a,b in S with
        # a+b in S, a+x in S, b+x in S.
        # For each a in S with a+x in S: need b in S such that b+x in S and a+b in S.
        # b+x in S => b in S-x. a+b in S => b in S-a.
        # So b in S ∩ (S-x) ∩ (S-a), and b != a.
        S_minus_x = {y - x for y in s if 1 <= y - x <= 128}
        if not S_minus_x:
            return S + [x]
        # For each a where a+x is in S
        for a in s:
            if a + x in s:
                # Need b in S ∩ S_minus_x ∩ (S-a), b != a
                # S-a = {y-a for y in S if 1 <= y-a <= 128}
                S_minus_a = {y - a for y in s if 1 <= y - a <= 128}
                # Find intersection: iterate over smallest set
                inter = S_minus_x & S_minus_a
                if inter:
                    # Check if any element in inter is in S and != a
                    for b in inter:
                        if b in s and b != a:
                            return None
        return S + [x]

    def try_add_to_set(S_set, x):
        """Try to add x given S as a set. Returns set or None."""
        if x in S_set:
            return None
        S_minus_x = {y - x for y in S_set if 1 <= y - x <= 128}
        if not S_minus_x:
            return S_set | {x}
        for a in S_set:
            if a + x in S_set:
                S_minus_a = {y - a for y in S_set if 1 <= y - a <= 128}
                inter = S_minus_x & S_minus_a
                if inter:
                    for b in inter:
                        if b in S_set and b != a:
                            return None
        return S_set | {x}

    # Phase 1: Greedy add
    S = list(best)
    S_set = set(S)
    for x in range(1, 129):
        if x in S_set:
            continue
        if time.time() > deadline:
            break
        new_S = try_add(S, x)
        if new_S is not None:
            S = new_S
            S_set = set(S)

    if len(S) > best_size:
        best = S
        best_size = len(S)

    # Phase 2: Single-swap local search
    all_vals = list(range(1, 129))
    for iteration in range(5000):
        if time.time() > deadline:
            break
        S_set = set(best)
        n = len(best)
        # Try removing a random element and adding a random missing one
        idx = random.randint(0, n - 1)
        removed = best[idx]
        S2_set = S_set - {removed}
        # Try adding back removed or a new element
        candidates = [removed]
        for _ in range(5):
            candidates.append(random.choice(all_vals))
        improved = False
        for x in candidates:
            if x in S2_set:
                continue
            new_set = try_add_to_set(S2_set, x)
            if new_set is not None and len(new_set) > best_size:
                best = list(new_set)
                best_size = len(new_set)
                improved = True
                break
            elif new_set is not None and len(new_set) == n:
                # Same size swap - keep for diversity with some probability
                if random.random() < 0.3:
                    best = list(new_set)
                    improved = True
                    break

    # Phase 3: Multi-pair swap (remove 2, add 2-3)
    for iteration in range(2000):
        if time.time() > deadline:
            break
        S_set = set(best)
        n = len(best)
        if n < 4:
            break
        # Remove 2 random elements
        idx1 = random.randint(0, n - 1)
        idx2 = random.randint(0, n - 2)
        if idx2 >= idx1:
            idx2 += 1
        r1 = best[idx1]
        r2 = best[idx2]
        S2_set = S_set - {r1, r2}
        # Try to add back r1, r2 and new elements
        add_candidates = [r1, r2]
        for _ in range(4):
            add_candidates.append(random.choice(all_vals))
        # Try adding all candidates one by one
        new_set = set(S2_set)
        added = []
        for x in add_candidates:
            if x in new_set:
                continue
            res = try_add_to_set(new_set, x)
            if res is not None:
                new_set = res
                added.append(x)
        if len(new_set) > best_size:
            best = list(new_set)
            best_size = len(new_set)

    # Phase 4: Exhaustive final polish - try all single additions
    S_set = set(best)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, 129):
            if x in S_set:
                continue
            new_set = try_add_to_set(S_set, x)
            if new_set is not None:
                S_set = new_set
                changed = True
                if len(S_set) > best_size:
                    best = list(S_set)
                    best_size = len(best)

    # Phase 5: One more round of single-swap after polish
    for iteration in range(2000):
        if time.time() > deadline:
            break
        S_set = set(best)
        n = len(best)
        idx = random.randint(0, n - 1)
        removed = best[idx]
        S2_set = S_set - {removed}
        candidates = [removed] + [random.choice(all_vals) for _ in range(5)]
        for x in candidates:
            if x in S2_set:
                continue
            new_set = try_add_to_set(S2_set, x)
            if new_set is not None and len(new_set) > best_size:
                best = list(new_set)
                best_size = len(new_set)
                break

    # Final verification
    if not _is_valid(best):
        best = list(range(17, 33)) + list(range(65, 129))

    return [int(x) for x in best]
# EVOLVE-BLOCK-END
