import numpy as np
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    def is_valid(coeffs):
        if len(coeffs) > 41:
            return False
        if len(coeffs) == 0:
            return False
        if coeffs[-1] == 0:
            return False
        nonzero_count = sum(1 for c in coeffs if c != 0)
        if nonzero_count != 12:
            return False
        if any(c < -64 or c > 64 for c in coeffs):
            return False
        return True

    def evaluate(coeffs):
        if not is_valid(coeffs):
            return float('inf')
        sq = np.convolve(coeffs, coeffs)
        return int(np.count_nonzero(sq))

    best_score = float('inf')
    best_poly = [1] * 12  # baseline: 12 ones, square has 23 nonzeros

    def try_poly(coeffs):
        nonlocal best_score, best_poly
        if not is_valid(coeffs):
            return
        score = evaluate(coeffs)
        if score < best_score:
            best_score = score
            best_poly = list(coeffs)

    # Fast sparse square count: given list of (pos, coeff) pairs, count nonzero in square
    def sparse_square_count(terms):
        """terms: list of (position, coefficient). Returns count of nonzero in square."""
        # Build dictionary of square coefficients
        sq = {}
        n = len(terms)
        for i in range(n):
            pi, ci = terms[i]
            for j in range(i, n):
                pj, cj = terms[j]
                val = ci * cj
                if i == j:
                    sq[pi + pj] = sq.get(pi + pj, 0) + val
                else:
                    sq[pi + pj] = sq.get(pi + pj, 0) + 2 * val
        # Count nonzero
        return sum(1 for v in sq.values() if v != 0)

    def state_to_terms(state):
        """state: tuple of (pos, sign, mag) -> list of (pos, coeff)"""
        return [(p, s * m) for p, s, m in state]

    def state_score_fast(state):
        """Fast score using sparse counting."""
        terms = state_to_terms(state)
        return sparse_square_count(terms)

    def state_to_coeffs(state):
        positions = [s[0] for s in state]
        max_deg = max(positions)
        coeffs = [0] * (max_deg + 1)
        for p, s, m in state:
            coeffs[p] = s * m
        while coeffs and coeffs[-1] == 0:
            coeffs.pop()
        return coeffs

    def valid_state(state):
        if len(state) != 12:
            return False
        positions = [s[0] for s in state]
        if len(set(positions)) != 12:
            return False
        if any(p < 0 or p > 40 for p in positions):
            return False
        for p, s, m in state:
            if s not in (-1, 1):
                return False
            if m < 1 or m > 64:
                return False
        return True

    def make_state(pos_sign_mag_list):
        sorted_list = sorted(pos_sign_mag_list, key=lambda x: x[0])
        return tuple(sorted_list)

    # Generate structured initial states
    initial_states = []

    # Strategy 1: Arithmetic progressions with various steps
    for d in range(1, 6):
        for a in range(0, 3):
            positions = [a + i * d for i in range(12)]
            if max(positions) > 40:
                continue
            # All ones
            initial_states.append(make_state([(p, 1, 1) for p in positions]))
            # Alternating signs
            initial_states.append(make_state([(p, 1 if i % 2 == 0 else -1, 1) for i, p in enumerate(positions)]))
            # Symmetric signs
            initial_states.append(make_state([(p, 1 if i < 6 else -1, 1) for i, p in enumerate(positions)]))

    # Strategy 2: P = A + x^k * A where A has 6 terms
    for k in range(6, 35):
        for d in range(1, 5):
            exps_A = [i * d for i in range(6)]
            if exps_A[-1] + k + exps_A[-1] > 40:
                continue
            # Check no overlap
            all_pos = exps_A + [e + k for e in exps_A]
            if len(set(all_pos)) != 12:
                continue
            # All ones
            initial_states.append(make_state([(p, 1, 1) for p in all_pos]))
            # A has all ones, x^k*A has all ones (same)
            # Try A with alternating, x^k*A with opposite
            initial_states.append(make_state([(p, 1 if i % 2 == 0 else -1, 1) for i, p in enumerate(all_pos)]))

    # Strategy 3: Two clusters of 6 consecutive
    for gap in range(3, 15):
        positions = list(range(6)) + list(range(gap, gap + 6))
        if max(positions) > 40:
            continue
        initial_states.append(make_state([(p, 1, 1) for p in positions]))
        initial_states.append(make_state([(p, 1 if p < 6 else -1, 1) for p in positions]))

    # Strategy 4: Three clusters of 4
    for g1 in range(3, 10):
        for g2 in range(3, 10):
            positions = list(range(4)) + list(range(g1, g1 + 4)) + list(range(g1 + g2, g1 + g2 + 4))
            if max(positions) > 40:
                continue
            initial_states.append(make_state([(p, 1, 1) for p in positions]))

    # Strategy 5: Random with small magnitudes
    for _ in range(30):
        positions = sorted(random.sample(range(41), 12))
        signs = [random.choice([-1, 1]) for _ in range(12)]
        mags = [random.choice([1, 1, 1, 2]) for _ in range(12)]
        initial_states.append(make_state([(positions[i], signs[i], mags[i]) for i in range(12)]))

    # Evaluate all initial states
    best_state = None
    best_state_score = float('inf')
    for state in initial_states:
        if not valid_state(state):
            continue
        cnt = state_score_fast(state)
        if cnt < best_state_score:
            best_state_score = cnt
            best_state = state
        coeffs = state_to_coeffs(state)
        if is_valid(coeffs):
            try_poly(coeffs)

    if best_state is None:
        best_state = make_state([(i, 1, 1) for i in range(12)])
        best_state_score = state_score_fast(best_state)

    current_state = best_state
    current_score = best_state_score

    # Simulated annealing with improved moves
    T0 = 5.0
    T_end = 0.01
    total_steps = 60000
    best_sa_score = current_score
    best_sa_state = current_state

    # Precompute: for each state, count how many sum collisions exist
    # A sum collision is when i+j = k for multiple pairs (i,j)
    # More collisions = more chance of cancellation

    for step in range(total_steps):
        if time.time() > deadline - 3:
            break

        progress = step / total_steps
        T = T0 * (T_end / T0) ** progress

        # Generate random move
        state_list = list(current_state)
        move_type = random.randint(1, 8)

        if move_type == 1:  # Move one position
            i = random.randint(0, 11)
            new_pos = random.randint(0, 40)
            if new_pos not in [s[0] for s in state_list if s[0] != state_list[i][0]]:
                state_list[i] = (new_pos, state_list[i][1], state_list[i][2])

        elif move_type == 2:  # Swap two positions
            i, j = random.sample(range(12), 2)
            pi, si, mi = state_list[i]
            pj, sj, mj = state_list[j]
            state_list[i] = (pj, si, mi)
            state_list[j] = (pi, sj, mj)

        elif move_type == 3:  # Flip one sign
            i = random.randint(0, 11)
            p, s, m = state_list[i]
            state_list[i] = (p, -s, m)

        elif move_type == 4:  # Flip two signs
            i, j = random.sample(range(12), 2)
            pi, si, mi = state_list[i]
            pj, sj, mj = state_list[j]
            state_list[i] = (pi, -si, mi)
            state_list[j] = (pj, -sj, mj)

        elif move_type == 5:  # Change one magnitude (small range)
            i = random.randint(0, 11)
            p, s, m = state_list[i]
            new_m = random.choice([1, 2, 3])
            state_list[i] = (p, s, new_m)

        elif move_type == 6:  # Change two magnitudes
            i, j = random.sample(range(12), 2)
            pi, si, mi = state_list[i]
            pj, sj, mj = state_list[j]
            state_list[i] = (pi, si, random.choice([1, 2]))
            state_list[j] = (pj, sj, random.choice([1, 2]))

        elif move_type == 7:  # Swap two magnitudes
            i, j = random.sample(range(12), 2)
            pi, si, mi = state_list[i]
            pj, sj, mj = state_list[j]
            state_list[i] = (pi, si, mj)
            state_list[j] = (pj, sj, mi)

        elif move_type == 8:  # Move position + flip sign
            i = random.randint(0, 11)
            new_pos = random.randint(0, 40)
            p, s, m = state_list[i]
            if new_pos not in [x[0] for x in state_list if x[0] != p]:
                state_list[i] = (new_pos, -s, m)

        new_state = make_state(state_list)
        if not valid_state(new_state):
            continue

        new_score = state_score_fast(new_state)
        delta = new_score - current_score

        if delta <= 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
            current_state = new_state
            current_score = new_score
            if new_score < best_sa_score:
                best_sa_score = new_score
                best_sa_state = new_state
                coeffs = state_to_coeffs(new_state)
                if is_valid(coeffs):
                    try_poly(coeffs)

    # Final local search: greedy hill climbing from best state
    if time.time() < deadline - 2:
        current_state = best_sa_state
        current_score = best_sa_score
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(12):
                if time.time() > deadline - 2:
                    break
                # Try moving position i to nearby positions
                old_pos = current_state[i][0]
                for new_pos in range(max(0, old_pos - 3), min(41, old_pos + 4)):
                    if new_pos == old_pos:
                        continue
                    if new_pos in [s[0] for s in current_state if s[0] != old_pos]:
                        continue
                    state_list = list(current_state)
                    state_list[i] = (new_pos, state_list[i][1], state_list[i][2])
                    new_state = make_state(state_list)
                    if not valid_state(new_state):
                        continue
                    new_score = state_score_fast(new_state)
                    if new_score < current_score:
                        current_state = new_state
                        current_score = new_score
                        improved = True
                        coeffs = state_to_coeffs(new_state)
                        if is_valid(coeffs):
                            try_poly(coeffs)
                        break
                if time.time() > deadline - 2:
                    break
                # Try flipping sign of i
                state_list = list(current_state)
                p, s, m = state_list[i]
                state_list[i] = (p, -s, m)
                new_state = make_state(state_list)
                new_score = state_score_fast(new_state)
                if new_score < current_score:
                    current_state = new_state
                    current_score = new_score
                    improved = True
                    coeffs = state_to_coeffs(new_state)
                    if is_valid(coeffs):
                        try_poly(coeffs)
                # Try changing magnitude of i
                for new_m in [1, 2, 3]:
                    if new_m == m:
                        continue
                    state_list = list(current_state)
                    state_list[i] = (p, s, new_m)
                    new_state = make_state(state_list)
                    new_score = state_score_fast(new_state)
                    if new_score < current_score:
                        current_state = new_state
                        current_score = new_score
                        improved = True
                        coeffs = state_to_coeffs(new_state)
                        if is_valid(coeffs):
                            try_poly(coeffs)
                        break

    # Ensure we have a valid result
    if not is_valid(best_poly):
        best_poly = [1] * 12

    return best_poly
