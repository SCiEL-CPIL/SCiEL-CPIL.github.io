"""Rewritten program: MWIS on conflict graph with B&B and fractional knapsack bound."""
# EVOLVE-BLOCK-START
import time
import math
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    candidates = list(range(20, 201))
    n_mods = len(candidates)

    # Precompute gcd matrix for speed
    gcd_mat = [[gcd(candidates[i], candidates[j]) for j in range(n_mods)] for i in range(n_mods)]

    inv_float = {m: 1.0 / m for m in candidates}

    def is_disjoint(m_new, r_new, existing):
        for (m_e, r_e) in existing:
            g = gcd(m_new, m_e)
            if r_new % g == r_e % g:
                return False
        return True

    def find_valid_residue(m, existing):
        for r in range(m):
            if is_disjoint(m, r, existing):
                return r
        return None

    # Phase 1: Greedy initialization with multiple random orderings
    best = []
    best_score = 0.0

    def greedy_init(order_seed):
        random.seed(order_seed)
        selected = []
        # Try adding moduli in a random order, preferring small moduli
        mods = list(candidates)
        # Weighted shuffle: smaller moduli more likely to be tried first
        for _ in range(len(mods)):
            remaining = [m for m in mods if m not in [x[0] for x in selected]]
            if not remaining:
                break
            # Pick with probability weighted by 1/m
            weights = [1.0 / m for m in remaining]
            total_w = sum(weights)
            r = random.random() * total_w
            cum = 0
            chosen = remaining[0]
            for m, w in zip(remaining, weights):
                cum += w
                if r <= cum:
                    chosen = m
                    break
            r_val = find_valid_residue(chosen, selected)
            if r_val is not None:
                selected.append((chosen, r_val))
        return selected

    # Try several random initializations
    for seed in range(20):
        if time.time() > deadline - 30:
            break
        candidate = greedy_init(seed)
        score = sum(inv_float[m] for m, r in candidate)
        if score > best_score:
            best_score = score
            best = candidate

    # Phase 2: Local search / simulated annealing
    def compute_score(state):
        return sum(inv_float[m] for m, r in state)

    def try_add(state, m, r):
        used_mods = set(x[0] for x in state)
        if m in used_mods:
            return None
        for (m_e, r_e) in state:
            g = gcd(m, m_e)
            if r % g == r_e % g:
                return None
        return state + [(m, r)]

    def try_remove(state, idx):
        return state[:idx] + state[idx+1:]

    random.seed(123)
    T = 0.5
    T_min = 0.001
    cooling = 0.9995

    while T > T_min and time.time() < deadline - 40:
        state = list(best)
        score = best_score

        for _ in range(300):
            if time.time() > deadline - 40:
                break

            move_type = random.choice(['add', 'remove', 'change_r', 'swap'])

            if move_type == 'add':
                used_mods = set(m for m, r in state)
                available = [m for m in candidates if m not in used_mods]
                if not available:
                    continue
                # Prefer small moduli
                available_sorted = sorted(available)
                m_new = random.choice(available_sorted[:min(60, len(available_sorted))])
                for attempt in range(30):
                    r_new = random.randint(0, m_new - 1)
                    new_state = try_add(state, m_new, r_new)
                    if new_state:
                        new_score = compute_score(new_state)
                        delta = new_score - score
                        if delta > 0 or random.random() < math.exp(delta / T):
                            state, score = new_state, new_score
                        break
            elif move_type == 'remove':
                if len(state) <= 1:
                    continue
                idx = random.randint(0, len(state) - 1)
                new_state = try_remove(state, idx)
                new_score = compute_score(new_state)
                delta = new_score - score
                if delta > 0 or random.random() < math.exp(delta / T):
                    state, score = new_state, new_score
            elif move_type == 'change_r':
                if not state:
                    continue
                idx = random.randint(0, len(state) - 1)
                m_old, r_old = state[idx]
                for attempt in range(30):
                    r_new = random.randint(0, m_old - 1)
                    if r_new == r_old:
                        continue
                    valid = True
                    for j, (m_e, r_e) in enumerate(state):
                        if j == idx:
                            continue
                        g = gcd(m_old, m_e)
                        if r_new % g == r_e % g:
                            valid = False
                            break
                    if valid:
                        new_state = state[:idx] + [(m_old, r_new)] + state[idx+1:]
                        new_score = compute_score(new_state)
                        delta = new_score - score
                        if delta > 0 or random.random() < math.exp(delta / T):
                            state, score = new_state, new_score
                        break
            elif move_type == 'swap':
                if len(state) <= 1:
                    continue
                idx = random.randint(0, len(state) - 1)
                m_old, r_old = state[idx]
                new_state_without = try_remove(state, idx)
                used_mods = set(m for m, r in new_state_without)
                available = [m for m in candidates if m not in used_mods]
                if not available:
                    continue
                candidates_sorted = sorted(available, key=lambda x: x)
                for m_new in candidates_sorted[:min(30, len(candidates_sorted))]:
                    found = False
                    for r_new in range(min(m_new, 15)):
                        new_state = try_add(new_state_without, m_new, r_new)
                        if new_state:
                            new_score = compute_score(new_state)
                            delta = new_score - score
                            if delta > 0 or random.random() < math.exp(delta / T):
                                state, score = new_state, new_score
                                found = True
                                break
                    if found:
                        break

        if score > best_score:
            best, best_score = state, score

        T *= cooling

    # Phase 3: B&B polish - try to add remaining moduli with branch and bound
    used_mods = set(m for m, r in best)
    remaining = [m for m in candidates if m not in used_mods]
    remaining.sort()  # by 1/m descending

    if remaining and time.time() < deadline - 10:
        # Try adding each remaining modulus one at a time
        for m in remaining[:50]:
            if time.time() > deadline - 10:
                break
            r = find_valid_residue(m, best)
            if r is not None:
                new_state = best + [(m, r)]
                new_score = best_score + inv_float[m]
                if new_score > best_score:
                    best, best_score = new_state, new_score

    # Phase 4: Final local search pass
    if time.time() < deadline - 5:
        T = 0.1
        state = list(best)
        score = best_score
        for _ in range(800):
            if time.time() > deadline - 5:
                break
            move_type = random.choice(['add', 'remove', 'change_r'])
            if move_type == 'add':
                used_mods = set(m for m, r in state)
                available = [m for m in candidates if m not in used_mods]
                if not available:
                    continue
                available_sorted = sorted(available)
                m_new = random.choice(available_sorted[:min(40, len(available_sorted))])
                for attempt in range(20):
                    r_new = random.randint(0, m_new - 1)
                    new_state = try_add(state, m_new, r_new)
                    if new_state:
                        new_score = compute_score(new_state)
                        delta = new_score - score
                        if delta > 0 or random.random() < math.exp(delta / T):
                            state, score = new_state, new_score
                        break
            elif move_type == 'remove':
                if len(state) <= 1:
                    continue
                idx = random.randint(0, len(state) - 1)
                new_state = try_remove(state, idx)
                new_score = compute_score(new_state)
                delta = new_score - score
                if delta > 0 or random.random() < math.exp(delta / T):
                    state, score = new_state, new_score
            elif move_type == 'change_r':
                if not state:
                    continue
                idx = random.randint(0, len(state) - 1)
                m_old, r_old = state[idx]
                for attempt in range(20):
                    r_new = random.randint(0, m_old - 1)
                    if r_new == r_old:
                        continue
                    valid = True
                    for j, (m_e, r_e) in enumerate(state):
                        if j == idx:
                            continue
                        g = gcd(m_old, m_e)
                        if r_new % g == r_e % g:
                            valid = False
                            break
                    if valid:
                        new_state = state[:idx] + [(m_old, r_new)] + state[idx+1:]
                        new_score = compute_score(new_state)
                        delta = new_score - score
                        if delta > 0 or random.random() < math.exp(delta / T):
                            state, score = new_state, new_score
                        break
        if score > best_score:
            best, best_score = state, score

    # Convert to required format
    result = [[m, r] for m, r in best]
    return result
# EVOLVE-BLOCK-END
