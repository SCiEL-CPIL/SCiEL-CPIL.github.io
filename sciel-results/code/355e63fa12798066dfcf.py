import numpy as np
import time
import random

N = 16
NPOINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 295

    # Precompute squared distances and isosceles relationships
    # For each point p, compute iso_pairs[p] = set of (q, r) pairs such that (p,q,r) is isosceles
    # But we need: for maximality, p not in S can be added iff for all q,r in S, (p,q,r) is NOT isosceles
    # So p is "blocked" by S iff there exist q,r in S such that (p,q,r) is isosceles

    # Precompute: for each p, the list of (q, r) with q < r such that (p,q,r) is isosceles
    # Then p is blocked by S iff exists (q,r) in iso_pairs[p] with q in S and r in S

    # For efficiency, for each p, precompute a list of bitmasks: for each q, the set of r such that (p,q,r) is isosceles
    # Then p is blocked by S iff exists q in S such that iso_r[p][q] & S != 0

    iso_r = [[0] * NPOINTS for _ in range(NPOINTS)]
    for p in range(NPOINTS):
        px, py = p // N, p % N
        for q in range(NPOINTS):
            if q == p:
                continue
            qx, qy = q // N, q % N
            bits = 0
            for r in range(NPOINTS):
                if r == p or r == q:
                    continue
                rx, ry = r // N, r % N
                d1 = (px - qx)**2 + (py - qy)**2
                d2 = (px - rx)**2 + (py - ry)**2
                d3 = (qx - rx)**2 + (qy - ry)**2
                if d1 == d2 or d1 == d3 or d2 == d3:
                    bits |= (1 << r)
            iso_r[p][q] = bits

    # can_add(p, S_bits): p can be added to S iff for all q in S, iso_r[p][q] & S == 0
    def can_add(p, S_bits):
        if S_bits & (1 << p):
            return False
        s = S_bits
        while s:
            lowbit = s & (-s)
            q = lowbit.bit_length() - 1
            s ^= lowbit
            if iso_r[p][q] & S_bits:
                return False
        return True

    # is_blocked(p, S_bits): p is blocked by S (cannot be added)
    def is_blocked(p, S_bits):
        if S_bits & (1 << p):
            return True  # already in S, trivially "blocked"
        s = S_bits
        while s:
            lowbit = s & (-s)
            q = lowbit.bit_length() - 1
            s ^= lowbit
            if iso_r[p][q] & S_bits:
                return True
        return False

    # is_maximal: all points not in S are blocked
    def is_maximal(S_bits):
        complement = ((1 << NPOINTS) - 1) & ~S_bits
        s = complement
        while s:
            lowbit = s & (-s)
            p = lowbit.bit_length() - 1
            s ^= lowbit
            if not is_blocked(p, S_bits):
                return False
        return True

    # Greedy maximal construction
    def greedy_maximal(order):
        S_bits = 0
        for p in order:
            if can_add(p, S_bits):
                S_bits |= (1 << p)
        # Ensure maximality by adding points if needed (greedy should give maximal)
        return S_bits

    best_size = NPOINTS + 1
    best_set_bits = 0

    # Phase 1: Multiple greedy constructions with different orderings
    for trial in range(100):
        if time.time() > deadline - 30:
            break
        rng = random.Random(42 + trial)
        order = list(range(NPOINTS))
        rng.shuffle(order)
        S_bits = greedy_maximal(order)
        if is_maximal(S_bits):
            size = S_bits.bit_count()
            if size < best_size:
                best_size = size
                best_set_bits = S_bits

    # Phase 2: Local search - try to remove points while maintaining maximality
    if best_set_bits and time.time() < deadline - 20:
        current_bits = best_set_bits

        for iteration in range(200):
            if time.time() > deadline - 10:
                break

            improved = False
            # Try removing each point
            for p in range(NPOINTS):
                if current_bits & (1 << p):
                    new_bits = current_bits & ~(1 << p)
                    if is_maximal(new_bits):
                        current_bits = new_bits
                        improved = True
                        break

            if not improved:
                # Try 2-point removals
                found = False
                for p in range(NPOINTS):
                    if time.time() > deadline - 10:
                        break
                    if not (current_bits & (1 << p)):
                        continue
                    for q in range(p + 1, NPOINTS):
                        if not (current_bits & (1 << q)):
                            continue
                        new_bits = current_bits & ~(1 << p) & ~(1 << q)
                        if is_maximal(new_bits):
                            current_bits = new_bits
                            found = True
                            improved = True
                            break
                    if improved:
                        break

            if not improved:
                # Try swap: remove p, add q, then try to remove more
                found_swap = False
                for p in range(NPOINTS):
                    if time.time() > deadline - 10:
                        break
                    if not (current_bits & (1 << p)):
                        continue
                    temp = current_bits & ~(1 << p)
                    for q in range(NPOINTS):
                        if temp & (1 << q):
                            continue
                        if can_add(q, temp):
                            new_bits = temp | (1 << q)
                            # Try to remove a point from new_bits
                            for r in range(NPOINTS):
                                if new_bits & (1 << r):
                                    check = new_bits & ~(1 << r)
                                    if is_maximal(check):
                                        current_bits = check
                                        found_swap = True
                                        improved = True
                                        break
                            if improved:
                                break
                    if improved:
                        break

            if not improved:
                break

            size = current_bits.bit_count()
            if size < best_size:
                best_size = size
                best_set_bits = current_bits

    # Phase 3: Simulated annealing style - allow temporary worsening
    if time.time() < deadline - 25:
        current_bits = best_set_bits
        current_size = current_bits.bit_count()
        temp = 5.0

        for sa_iter in range(5000):
            if time.time() > deadline - 10:
                break
            if temp < 0.1:
                break

            # Random move: remove a random point from S, or add a random point not in S
            rng = random.Random(sa_iter + 7777)
            move_type = rng.random()

            if move_type < 0.7:
                # Try to remove a random point
                s_points = []
                s = current_bits
                while s:
                    lowbit = s & (-s)
                    p = lowbit.bit_length() - 1
                    s ^= lowbit
                    s_points.append(p)
                if not s_points:
                    continue
                p = rng.choice(s_points)
                new_bits = current_bits & ~(1 << p)
                if is_maximal(new_bits):
                    new_size = new_bits.bit_count()
                    if new_size < current_size:
                        current_bits = new_bits
                        current_size = new_size
                        if current_size < best_size:
                            best_size = current_size
                            best_set_bits = current_bits
                else:
                    # Accept with probability exp(-delta/temp) where delta = 1 (size increased by 0, but not maximal)
                    # Actually, if not maximal, it's invalid. So we can't accept.
                    pass
            else:
                # Try to swap: remove p, add q
                s_points = []
                s = current_bits
                while s:
                    lowbit = s & (-s)
                    p = lowbit.bit_length() - 1
                    s ^= lowbit
                    s_points.append(p)
                if not s_points:
                    continue
                p = rng.choice(s_points)
                temp_bits = current_bits & ~(1 << p)
                # Find a random q that can be added
                candidates = []
                for q in range(NPOINTS):
                    if not (temp_bits & (1 << q)) and can_add(q, temp_bits):
                        candidates.append(q)
                if candidates:
                    q = rng.choice(candidates)
                    new_bits = temp_bits | (1 << q)
                    new_size = new_bits.bit_count()
                    if new_size < current_size:
                        if is_maximal(new_bits):
                            current_bits = new_bits
                            current_size = new_size
                            if current_size < best_size:
                                best_size = current_size
                                best_set_bits = current_bits
                    elif new_size == current_size and rng.random() < 0.1:
                        # Same size, explore
                        if is_maximal(new_bits):
                            current_bits = new_bits

            temp *= 0.999

    # Convert best_set_bits to list of (x, y) tuples
    result = []
    for p in range(NPOINTS):
        if best_set_bits & (1 << p):
            result.append((p // N, p % N))

    if not result:
        # Fallback
        S_bits = 0
        for p in range(NPOINTS):
            if can_add(p, S_bits):
                S_bits |= (1 << p)
        for p in range(NPOINTS):
            if S_bits & (1 << p):
                result.append((p // N, p % N))

    return result
