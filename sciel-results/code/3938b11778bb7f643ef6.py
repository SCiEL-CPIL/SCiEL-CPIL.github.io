import math
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit

    def compute_score(u):
        u = sorted(set(u))
        if not u or 0 not in u:
            return -1.0
        max_u = max(u)
        m = 2 * max_u + 1
        if m <= 1:
            return 1.0
        sumset = set()
        diffset = set()
        for a in u:
            for b in u:
                sumset.add(a + b)
                diffset.add(a - b)
        r = len(diffset) / len(sumset)
        if r <= 0:
            return -1.0
        return 1 + math.log(r) / math.log(m)

    def is_golomb(u):
        u = sorted(u)
        diffs = set()
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                d = u[j] - u[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    best_set = [0, 1, 3]
    best_score = compute_score(best_set)

    def try_set(cand):
        nonlocal best_set, best_score
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if max(cand) == 0:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # Phase 1: Exhaustive search for small sets (size <= 6)
    max_u_limit_small = 20
    for max_u in range(1, max_u_limit_small + 1):
        if time.time() > deadline - 5:
            break
        elements = list(range(1, max_u + 1))
        for size in range(3, 7):
            if len(elements) < size:
                continue
            for comb in combinations(elements, size):
                cand = sorted([0] + list(comb))
                try_set(cand)

    # Phase 2: Structured sets with larger max_u
    for max_u in range(10, 200):
        if time.time() > deadline - 5:
            break

        # Two blocks
        for a in range(1, 10):
            for b in range(a + 1, max_u - a + 1):
                if time.time() > deadline - 5:
                    break
                block1 = list(range(0, a + 1))
                block2 = list(range(b, b + a + 1))
                cand = sorted(set(block1 + block2))
                if max(cand) <= max_u:
                    try_set(cand)

        # Geometric with small ratios
        for r in [2, 3, 4, 5]:
            geo = [0, 1]
            while geo[-1] * r <= max_u:
                geo.append(geo[-1] * r)
            try_set(geo)

        # Arithmetic progressions
        for d in range(1, 10):
            ap = []
            v = 0
            while v <= max_u:
                ap.append(v)
                v += d
            try_set(ap)

        # Mixed: {0, 1, ..., a} U {b}
        for a in range(1, 10):
            for b in range(a + 1, max_u + 1):
                cand = sorted(list(range(0, a + 1)) + [b])
                try_set(cand)

        # Mixed: {0, 1, ..., a} U {b, c}
        for a in range(1, 5):
            for b in range(a + 1, max_u + 1):
                for c in range(b + 1, max_u + 1):
                    cand = sorted(list(range(0, a + 1)) + [b, c])
                    try_set(cand)

    # Phase 3: Known Golomb rulers
    known_golomb = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 20, 24],
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 34],
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 34, 38, 41],
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 34, 38, 41, 46, 52],
    ]
    for g in known_golomb:
        if time.time() > deadline - 5:
            break
        try_set(g)

    # Phase 4: Extend Golomb rulers via local search
    def try_extend_golomb(base_u, max_n, time_limit=25):
        start = time.time()
        best_u = list(base_u)
        best_s = compute_score(best_u)

        for target_n in range(len(base_u) + 1, max_n + 1):
            if time.time() - start > time_limit or time.time() > deadline - 10:
                break
            improved = True
            while improved:
                improved = False
                if time.time() - start > time_limit or time.time() > deadline - 10:
                    break
                current_max = max(best_u)
                best_candidate = None
                best_candidate_score = best_s
                search_hi = min(current_max + target_n * target_n * 2, 200)
                for new_mark in range(current_max + 1, search_hi + 1):
                    candidate = sorted(best_u + [new_mark])
                    if not is_golomb(candidate):
                        continue
                    s = compute_score(candidate)
                    if s > best_candidate_score:
                        best_candidate_score = s
                        best_candidate = candidate
                    if time.time() - start > time_limit or time.time() > deadline - 10:
                        break
                if best_candidate is not None:
                    best_u = best_candidate
                    best_s = best_candidate_score
                    improved = True
        return best_u, best_s

    if time.time() < deadline - 30:
        ext_u, ext_s = try_extend_golomb([0, 1, 4, 10, 12, 17], 12, time_limit=25)
        if ext_u and len(ext_u) > 6:
            try_set(ext_u)

    # Phase 5: Greedy Golomb construction
    def greedy_golomb(n, time_limit=15):
        start = time.time()
        u = [0]
        for i in range(1, n):
            if time.time() - start > time_limit or time.time() > deadline - 10:
                break
            pos = 0
            found = False
            while not found:
                pos += 1
                candidate = sorted(u + [pos])
                if is_golomb(candidate):
                    u = candidate
                    found = True
                if pos > n * n * 30:
                    break
        return u

    if time.time() < deadline - 20:
        greedy_u = greedy_golomb(12, time_limit=15)
        if greedy_u and len(greedy_u) >= 6:
            try_set(greedy_u)

    # Phase 6: Broader structured candidates
    if time.time() < deadline - 20:
        # Geometric progressions
        for ratio in [2, 3]:
            geo = [0, 1]
            while geo[-1] * ratio <= 500:
                geo.append(geo[-1] * ratio)
            for n in range(3, len(geo) + 1):
                if time.time() > deadline - 10:
                    break
                try_set(geo[:n])

        # Fibonacci-like
        fib = [0, 1]
        while fib[-1] <= 500:
            fib.append(fib[-1] + fib[-2])
        for n in range(3, len(fib)):
            if time.time() > deadline - 10:
                break
            try_set(fib[:n])

        # Random sets with various sizes
        random.seed(42)
        for _ in range(200):
            if time.time() > deadline - 10:
                break
            max_u = random.randint(10, 100)
            n = random.randint(4, 12)
            if max_u < n - 1:
                continue
            inner = random.sample(range(1, max_u), min(n - 2, max_u - 1))
            cand = sorted([0] + inner + [max_u])
            try_set(cand)

    # Phase 7: LFSR-based structured candidates (inspiration from de Bruijn / LFSR idea)
    # The idea: use a linear feedback shift register to generate sequences with
    # good difference properties. We interpret the LFSR state as positions in U.
    if time.time() < deadline - 20:
        def lfsr_positions(taps, length, max_val=300):
            """Generate positions from an LFSR sequence.
            taps: tuple of bit positions (0-indexed from LSB)
            length: number of bits to process
            Returns a set of positions where the LFSR output is 1.
            """
            # Initialize state with a non-zero value
            state = 1
            positions = set()
            for i in range(length):
                # Output bit is LSB
                if state & 1:
                    positions.add(i)
                # Compute feedback
                fb = 0
                for t in taps:
                    fb ^= (state >> t) & 1
                state = (state >> 1) | (fb << (len(taps) - 1))
            return positions

        # Try various LFSR configurations
        # Common primitive polynomials for small registers
        lfsr_configs = [
            (3, 1),   # x^3 + x + 1
            (4, 3),   # x^4 + x + 1
            (5, 2),   # x^5 + x^2 + 1
            (6, 5),   # x^6 + x + 1
            (7, 3),   # x^7 + x^3 + 1
            (8, 4),   # x^8 + x^4 + x^3 + x^2 + 1
            (9, 4),   # x^9 + x^4 + 1
            (10, 7),  # x^10 + x^7 + 1
        ]
        
        for taps, _ in lfsr_configs:
            if time.time() > deadline - 10:
                break
            # Generate a sequence and extract positions
            for length in range(10, 60):
                if time.time() > deadline - 10:
                    break
                pos_set = lfsr_positions((taps,), length, max_val=300)
                # Add 0 and ensure it's a valid set
                pos_set.add(0)
                cand = sorted(pos_set)
                if len(cand) >= 3:
                    try_set(cand)

        # Also try using the LFSR to generate a sparse set
        for taps in [3, 4, 5, 6, 7, 8]:
            if time.time() > deadline - 10:
                break
            state = 1
            positions = [0]
            for i in range(1, 40):
                fb = 0
                for t in [taps, 1]:  # Simple 2-tap LFSR
                    fb ^= (state >> t) & 1
                state = (state >> 1) | (fb << (taps - 1))
                if state & 1:
                    positions.append(i)
            if len(positions) >= 3:
                try_set(sorted(positions))

    # Phase 8: Local search refinement (enhanced from current + inspiration)
    if time.time() < deadline - 15:
        current = list(best_set)
        no_improve = 0
        max_no_improve = 800  # Increased from 500

        while time.time() < deadline - 10 and no_improve < max_no_improve:
            improved = False

            # Try adding an element
            if len(current) < 15:
                max_cur = max(current)
                for new_el in range(1, max_cur + 15):  # Wider range
                    if new_el in current:
                        continue
                    cand = sorted(current + [new_el])
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        break

            # Try replacing an element
            if len(current) >= 3:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    for new_el in range(1, max(current) + 15):  # Wider range
                        if new_el in current:
                            continue
                        cand = list(current)
                        cand[i] = new_el
                        cand = sorted(set(cand))
                        if 0 not in cand:
                            continue
                        s = compute_score(cand)
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                            current = list(cand)
                            improved = True
                            break
                    if improved:
                        break

            # Try removing an element
            if len(current) > 3:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    cand = [x for j, x in enumerate(current) if j != i]
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        break

            # Try 2-element swap (inspired from other solutions)
            if not improved and len(current) >= 4:
                for i in range(1, len(current)):
                    for j in range(i + 1, len(current)):
                        if current[i] == 0 or current[j] == 0:
                            continue
                        for new_i in range(1, max(current) + 10):
                            if new_i in current:
                                continue
                            for new_j in range(1, max(current) + 10):
                                if new_j in current or new_j == new_i:
                                    continue
                                cand = list(current)
                                cand[i] = new_i
                                cand[j] = new_j
                                cand = sorted(set(cand))
                                if 0 not in cand or len(cand) < len(current):
                                    continue
                                s = compute_score(cand)
                                if s > best_score:
                                    best_score = s
                                    best_set = list(cand)
                                    current = list(cand)
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break

            if not improved:
                no_improve += 1
            else:
                no_improve = 0

    return best_set
