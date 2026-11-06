"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    if limit >= 0:
        sieve[0] = False
    if limit >= 1:
        sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def construct():
    start_time = time.time()
    deadline = start_time + 270.0

    # Precompute primes up to N
    all_primes = _primes(N)
    primes_set = set(all_primes)

    # Phase 1: Top-down removal (current strategy)
    A = set(range(N + 1))
    primes = _primes(N)

    for i, p in enumerate(primes):
        if time.time() > deadline:
            break
        if p > len(A):
            break

        counts = [0] * p
        for x in A:
            counts[x % p] += 1

        min_count = min(counts)
        if min_count == 0:
            continue

        candidates = [r for r in range(p) if counts[r] == min_count]
        r_to_remove = candidates[0]

        # Lookahead: pick candidate that maximizes min count for next prime
        if len(candidates) > 1 and i + 1 < len(primes):
            p_next = primes[i + 1]
            new_size_est = len(A) - min_count
            if p_next <= new_size_est:
                best_r = candidates[0]
                best_score = -1
                for r in candidates:
                    next_counts = [0] * p_next
                    for x in A:
                        if x % p != r:
                            next_counts[x % p_next] += 1
                    score = min(next_counts)
                    if score > best_score:
                        best_score = score
                        best_r = r
                r_to_remove = best_r

        A = {x for x in A if x % p != r_to_remove}
        if not A:
            break

    # Phase 2: Greedy addition - try to add back elements
    # Build candidate list: prefer primes first (they're "sparse"), then composites
    # Sort by: (is_composite, value) so primes come first
    candidates = [x for x in range(N + 1) if x not in A]
    # Sort: primes first (0 if prime), then by value
    candidates.sort(key=lambda x: (0 if x in primes_set else 1, x))

    # For fast checking, maintain residue counts for each relevant prime
    # We only need primes p <= len(A) + 1 (since adding one element increases size by 1)
    
    def is_admissible_after_add(A_set, cand):
        """Check if adding cand to A_set keeps it admissible."""
        new_size = len(A_set) + 1
        # Only need to check primes p <= new_size
        for p in all_primes:
            if p > new_size:
                break
            # Count residues in A_set
            counts = [0] * p
            for x in A_set:
                counts[x % p] += 1
            # Check if cand's residue is already full
            r = cand % p
            if counts[r] == 0:
                # This residue was missing - adding cand fills it
                # But we need to check: was it already all-present before?
                # If counts[r] == 0, then residue r was missed, so adding is fine
                # UNLESS all other residues are also present (they are, since min was 0 only for r)
                # Actually if counts[r]==0, the set was already admissible for p (missing r)
                # Adding cand makes r present. Now all residues present -> NOT admissible!
                # Wait: admissible means SOME residue is missed. If we fill the last missing one,
                # then all residues are present -> NOT admissible.
                return False
            # If counts[r] > 0, residue r already present, adding doesn't change "missed" status
            # But we need to verify that at least one residue remains missed
            # Since counts[r] > 0 for cand's residue, and the set was admissible before,
            # there was some residue r' with counts[r'] == 0. That's still 0 now.
            # So it's still admissible.
            # Actually wait - we need to check if ANY residue is missing after adding.
            # Before adding: some residue was missing (set is admissible).
            # After adding cand: cand's residue count increases by 1.
            # If cand's residue was the ONLY missing one (count was 0), now it's 1 -> all present -> bad.
            # If cand's residue was already present (count > 0), some other residue is still 0 -> good.
            # So the check is: is counts[cand % p] == 0? If yes, bad. If no, good.
            # We already handled that above.
            pass
        return True

    # Faster: just check for each p <= new_size if adding cand fills the last gap
    def can_add(A_set, cand):
        new_size = len(A_set) + 1
        for p in all_primes:
            if p > new_size:
                break
            r = cand % p
            # Count how many elements in A_set have residue r mod p
            # If it's 0, then adding cand makes all residues present -> bad
            cnt = 0
            for x in A_set:
                if x % p == r:
                    cnt += 1
            if cnt == 0:
                return False
        return True

    # Optimize: precompute for each candidate, which primes would be violated
    # Use a different approach: maintain counts incrementally
    # For each prime p, maintain residue counts
    # But this is complex. Let's just do the greedy with a time budget.

    # Sort candidates: primes first, then by "rarity"
    # Actually, let's try adding in order and stop when time runs out
    added = 0
    for cand in candidates:
        if time.time() > deadline:
            break
        if cand in A:
            continue
        if can_add(A, cand):
            A.add(cand)
            added += 1

    return list(A)
# EVOLVE-BLOCK-END
