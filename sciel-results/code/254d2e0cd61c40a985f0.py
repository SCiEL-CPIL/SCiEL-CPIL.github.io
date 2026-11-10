import time
import random

N = 256
DEADLINE = 58.0  # Safety margin for 90s limit

def construct():
    start_time = time.time()

    def time_left():
        return DEADLINE - (time.time() - start_time)

    # Precompute 3-APs: for each element, list of masks of the OTHER two elements
    # so that (elem, other1, other2) forms a 3-AP.
    aps_by_elem = [[] for _ in range(N + 1)]
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            # For a: other two are b,c
            aps_by_elem[a].append((1 << (b-1)) | (1 << (c-1)))
            # For b: other two are a,c
            aps_by_elem[b].append((1 << (a-1)) | (1 << (c-1)))
            # For c: other two are a,b
            aps_by_elem[c].append((1 << (a-1)) | (1 << (b-1)))

    # Precompute 4-APs: for each element, list of masks of the OTHER three elements
    four_aps_by_v = [[] for _ in range(N + 1)]
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            four_aps_by_v[a].append((1 << (b-1)) | (1 << (c-1)) | (1 << (e-1)))
            four_aps_by_v[b].append((1 << (a-1)) | (1 << (c-1)) | (1 << (e-1)))
            four_aps_by_v[c].append((1 << (a-1)) | (1 << (b-1)) | (1 << (e-1)))
            four_aps_by_v[e].append((1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)))

    def get_set_bits(S):
        bits = []
        temp = S
        i = 0
        while temp:
            if temp & 1:
                bits.append(i + 1)
            temp >>= 1
            i += 1
        return bits

    def get_unset_bits(S):
        bits = []
        full_mask = (1 << N) - 1
        unset = full_mask ^ S
        temp = unset
        i = 0
        while temp:
            if temp & 1:
                bits.append(i + 1)
            temp >>= 1
            i += 1
        return bits

    def count_3aps(S):
        count = 0
        for x in range(1, N + 1):
            if not (S >> (x-1)) & 1:
                continue
            for mask in aps_by_elem[x]:
                if (S & mask) == mask:
                    count += 1
        return count

    # --- Initial constructions ---
    def greedy_construct():
        S = 0
        for x in range(1, N + 1):
            bit = 1 << (x - 1)
            valid = True
            for mask in four_aps_by_v[x]:
                if (S & mask) == mask:
                    valid = False
                    break
            if valid:
                S |= bit
        return S

    def quadratic_residue_set():
        p = 251
        residues = set()
        for x in range(1, p):
            residues.add((x * x) % p)
        S = 0
        for r in residues:
            if 1 <= r <= N:
                S |= (1 << (r - 1))
        return S

    def clean_4ap(S):
        """Greedy removal to make S 4-AP-free."""
        changed = True
        while changed and time_left() > 3.0:
            changed = False
            for x in range(1, N + 1):
                if not (S >> (x-1)) & 1:
                    continue
                bit = 1 << (x - 1)
                for mask in four_aps_by_v[x]:
                    if (S & mask) == mask:
                        S &= ~bit
                        changed = True
                        break
        return S

    # --- Local search: swap u (in) -> v (out) if it improves and stays 4-AP-free ---
    def local_search(S, max_rounds=200):
        current_score = count_3aps(S)
        best_S = S
        best_score = current_score
        no_improve = 0

        while time_left() > 5.0 and no_improve < max_rounds:
            elements = get_set_bits(S)
            non_elements = get_unset_bits(S)
            if not elements or not non_elements:
                break
            random.shuffle(elements)
            random.shuffle(non_elements)

            improved = False
            for u in elements:
                if time_left() < 5.0:
                    break
                u_bit = 1 << (u - 1)
                for v in non_elements:
                    if time_left() < 5.0:
                        break
                    v_bit = 1 << (v - 1)
                    # Validity: after swap, no 4-AP fully contained.
                    # A 4-AP becomes complete iff all other 3 are in S and u is NOT one of them.
                    valid = True
                    for mask in four_aps_by_v[v]:
                        if (S & mask) == mask:
                            if not (mask & u_bit):
                                valid = False
                                break
                    if not valid:
                        continue
                    # Score delta
                    gained = 0
                    for mask in aps_by_elem[v]:
                        if (S & mask) == mask:
                            if not (mask & u_bit):
                                gained += 1
                    lost = 0
                    for mask in aps_by_elem[u]:
                        if (S & mask) == mask:
                            lost += 1
                    delta = gained - lost
                    if delta > 0:
                        S = (S & ~u_bit) | v_bit
                        current_score += delta
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_S = S
                        break
                if improved:
                    break

            if not improved:
                no_improve += 1
                # Kick to escape local optimum
                if no_improve >= 2:
                    kick_done = False
                    for _ in range(40):
                        if time_left() < 5.0:
                            break
                        u = random.choice(elements)
                        v = random.choice(non_elements)
                        u_bit = 1 << (u - 1)
                        v_bit = 1 << (v - 1)
                        valid = True
                        for mask in four_aps_by_v[v]:
                            if (S & mask) == mask:
                                if not (mask & u_bit):
                                    valid = False
                                    break
                        if not valid:
                            continue
                        gained = 0
                        for mask in aps_by_elem[v]:
                            if (S & mask) == mask:
                                if not (mask & u_bit):
                                    gained += 1
                        lost = 0
                        for mask in aps_by_elem[u]:
                            if (S & mask) == mask:
                                lost += 1
                        delta = gained - lost
                        if delta >= -8:
                            S = (S & ~u_bit) | v_bit
                            current_score += delta
                            if current_score > best_score:
                                best_score = current_score
                                best_S = S
                            kick_done = True
                            break
                    if kick_done:
                        no_improve = 0
                    else:
                        no_improve = 0

        return best_S, best_score

    # --- Multi-start: combine greedy and quadratic-residue starts, plus a few random perturbations ---
    best_S = 0
    best_score = -1

    # Start 1: greedy
    S1 = greedy_construct()
    S1, sc1 = local_search(S1, max_rounds=80)
    if sc1 > best_score:
        best_score = sc1
        best_S = S1

    # Start 2: quadratic residues, cleaned
    S2 = clean_4ap(quadratic_residue_set())
    if time_left() > 5.0:
        S2, sc2 = local_search(S2, max_rounds=80)
        if sc2 > best_score:
            best_score = sc2
            best_S = S2

    # Start 3: greedy with random order
    if time_left() > 5.0:
        order = list(range(1, N + 1))
        random.shuffle(order)
        S3 = 0
        for x in order:
            bit = 1 << (x - 1)
            valid = True
            for mask in four_aps_by_v[x]:
                if (S3 & mask) == mask:
                    valid = False
                    break
            if valid:
                S3 |= bit
        S3, sc3 = local_search(S3, max_rounds=80)
        if sc3 > best_score:
            best_score = sc3
            best_S = S3

    # Final safeguard: ensure best_S is 4-AP-free (should already be)
    best_S = clean_4ap(best_S)

    return get_set_bits(best_S)
