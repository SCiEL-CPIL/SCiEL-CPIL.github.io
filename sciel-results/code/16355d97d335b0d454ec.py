import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin for 90s limit

    best_score = float('inf')
    best_candidate = None

    def compute_square_nonzeros(exps, coeffs):
        """Efficient sparse square: count non-zero terms in P(x)^2."""
        square = defaultdict(int)
        n = len(exps)
        for i in range(n):
            ei = exps[i]
            ci = coeffs[i]
            square[2 * ei] += ci * ci
            for j in range(i + 1, n):
                ej = exps[j]
                cj = coeffs[j]
                square[ei + ej] += 2 * ci * cj
        return sum(1 for v in square.values() if v != 0)

    def try_candidate(exps, coeffs):
        """Validate and update best if improved."""
        nonlocal best_score, best_candidate
        if len(exps) != 12 or len(coeffs) != 12:
            return
        if len(set(exps)) != 12:
            return
        if max(exps) > 40 or min(exps) < 0:
            return
        if any(abs(c) > 64 for c in coeffs):
            return
        if any(c == 0 for c in coeffs):
            return

        score = compute_square_nonzeros(exps, coeffs)
        if score < best_score:
            best_score = score
            max_deg = max(exps)
            dense = [0] * (max_deg + 1)
            for e, c in zip(exps, coeffs):
                dense[e] = c
            if len(dense) <= 41 and dense[-1] != 0:
                best_candidate = dense
        return score

    def optimize_signs(exps, max_iter=200):
        """Local search for best sign assignment for given exponents."""
        if time.time() > deadline:
            return
        best_local_score = float('inf')
        best_local_signs = [1] * 12
        for _ in range(8):
            if time.time() > deadline:
                break
            signs = [1 if random.random() > 0.5 else -1 for _ in range(12)]
            current_score = compute_square_nonzeros(exps, signs)
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                improved = False
                for i in range(12):
                    old = signs[i]
                    signs[i] = -signs[i]
                    new_score = compute_square_nonzeros(exps, signs)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        signs[i] = old
                if not improved:
                    break
            if current_score < best_local_score:
                best_local_score = current_score
                best_local_signs = signs[:]
        try_candidate(exps, best_local_signs)

    def optimize_full(exps, coeffs, max_iters=3000):
        """Simulated annealing + deterministic polish on (exps, coeffs)."""
        if time.time() > deadline:
            return
        current_exps = exps[:]
        current_coeffs = coeffs[:]
        current_score = compute_square_nonzeros(current_exps, current_coeffs)
        try_candidate(current_exps, current_coeffs)

        temp = 1.0
        cooling = 0.9995
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            new_exps = current_exps[:]
            new_coeffs = current_coeffs[:]
            move = random.random()
            if move < 0.35:
                idx = random.randint(0, 11)
                new_coeffs[idx] = random.choice([-8, -4, -2, -1, 1, 2, 4, 8])
            elif move < 0.70:
                idx = random.randint(0, 11)
                delta = random.choice([-3, -2, -1, 1, 2, 3])
                new_exp = current_exps[idx] + delta
                if new_exp < 0 or new_exp > 40:
                    continue
                if new_exp in [current_exps[j] for j in range(12) if j != idx]:
                    continue
                new_exps[idx] = new_exp
            elif move < 0.85:
                idx = random.randint(0, 11)
                new_coeffs[idx] = random.randint(-64, 64)
                if new_coeffs[idx] == 0:
                    continue
            else:
                # Swap two exponents
                i, j = random.sample(range(12), 2)
                new_exps[i], new_exps[j] = new_exps[j], new_exps[i]
                if len(set(new_exps)) != 12:
                    continue

            new_score = compute_square_nonzeros(new_exps, new_coeffs)
            if new_score < current_score:
                current_exps, current_coeffs, current_score = new_exps, new_coeffs, new_score
                try_candidate(current_exps, current_coeffs)
            else:
                delta_e = new_score - current_score
                if random.random() < min(1.0, 2.718281828 ** (-delta_e / temp)):
                    current_exps, current_coeffs, current_score = new_exps, new_coeffs, new_score
            temp *= cooling

        # Deterministic polish: try all small coefficient changes
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_score = compute_square_nonzeros(current_exps, current_coeffs)
            for i in range(12):
                if time.time() > deadline:
                    break
                old_c = current_coeffs[i]
                for nc in [-1, 1, -2, 2, -3, 3, -4, 4, -8, 8]:
                    if nc == old_c:
                        continue
                    new_coeffs = current_coeffs[:]
                    new_coeffs[i] = nc
                    ns = compute_square_nonzeros(current_exps, new_coeffs)
                    if ns < current_score:
                        current_coeffs = new_coeffs
                        current_score = ns
                        improved = True
                        try_candidate(current_exps, current_coeffs)
                        break
                if improved:
                    break
                # Try exponent moves
                old_e = current_exps[i]
                for d in range(-4, 5):
                    if d == 0:
                        continue
                    ne = old_e + d
                    if ne < 0 or ne > 40:
                        continue
                    if ne in [current_exps[j] for j in range(12) if j != i]:
                        continue
                    new_exps = current_exps[:]
                    new_exps[i] = ne
                    ns = compute_square_nonzeros(new_exps, current_coeffs)
                    if ns < current_score:
                        current_exps = new_exps
                        current_score = ns
                        improved = True
                        try_candidate(current_exps, current_coeffs)
                        break
                if improved:
                    break

    def ab_decomposition_search(max_restarts=15):
        """A+B block separation: A in low degrees, B in high degrees."""
        nonlocal best_score
        for _ in range(max_restarts):
            if time.time() > deadline - 3:
                break
            a_start = random.randint(0, 4)
            b_start = random.randint(20, 28)
            a_exps = sorted(random.sample(range(a_start, a_start + 16), 6))
            b_exps = sorted(random.sample(range(b_start, b_start + 16), 6))
            if set(a_exps) & set(b_exps):
                continue
            exps = sorted(a_exps + b_exps)
            # Try sign patterns
            for pattern in range(4):
                coeffs = []
                for i in range(12):
                    if pattern == 0:
                        coeffs.append(1)
                    elif pattern == 1:
                        coeffs.append(1 if i % 2 == 0 else -1)
                    elif pattern == 2:
                        coeffs.append(1 if i < 6 else -1)
                    else:
                        coeffs.append(random.choice([-1, 1]))
                try_candidate(exps, coeffs)
            # Optimize with SA
            optimize_full(exps, [1] * 12, max_iters=800)

    def genetic_search(pop_size=40, gen=30):
        """Genetic algorithm with crossover to combine good features."""
        nonlocal best_score
        # Initialize population
        population = []
        for _ in range(pop_size):
            if time.time() > deadline:
                return
            exps = sorted(random.sample(range(0, 41), 12))
            coeffs = [random.choice([-1, 1]) for _ in range(12)]
            population.append((exps, coeffs, compute_square_nonzeros(exps, coeffs)))
            try_candidate(exps, coeffs)

        for g in range(gen):
            if time.time() > deadline - 2:
                break
            # Sort by score
            population.sort(key=lambda x: x[2])
            # Elitism: keep top 5
            new_pop = population[:5]
            while len(new_pop) < pop_size:
                # Crossover: pick two parents, mix exponents and coeffs
                p1, p2 = random.sample(population, 2)
                child_exps = p1[0][:]
                child_coeffs = p1[1][:]
                # Mix some exponents from p2
                for i in random.sample(range(12), 3):
                    child_exps[i] = p2[0][i]
                    child_coeffs[i] = p2[1][i]
                if len(set(child_exps)) != 12 or max(child_exps) > 40:
                    # Fallback: random child
                    child_exps = sorted(random.sample(range(0, 41), 12))
                    child_coeffs = [random.choice([-1, 1]) for _ in range(12)]
                child_score = compute_square_nonzeros(child_exps, child_coeffs)
                # Mutation
                if random.random() < 0.5:
                    idx = random.randint(0, 11)
                    child_coeffs[idx] = random.choice([-2, -1, 1, 2, 4])
                    child_score = compute_square_nonzeros(child_exps, child_coeffs)
                new_pop.append((child_exps, child_coeffs, child_score))
                try_candidate(child_exps, child_coeffs)
            population = new_pop

    # --- Main search ---

    # 1. A+B decomposition (strong structural prior)
    ab_decomposition_search(max_restarts=12)

    # 2. Systematic exponent patterns with sign optimization
    candidate_exps = []
    for start in range(0, 10):
        for step in range(1, 5):
            exps = [start + i * step for i in range(12)]
            if max(exps) <= 40:
                candidate_exps.append(exps)
    # Two interleaved APs
    for s1 in range(0, 6):
        for d1 in range(1, 3):
            for s2 in range(8, 20):
                for d2 in range(1, 3):
                    e1 = [s1 + i * d1 for i in range(6)]
                    e2 = [s2 + i * d2 for i in range(6)]
                    exps = sorted(e1 + e2)
                    if len(set(exps)) == 12 and max(exps) <= 40:
                        candidate_exps.append(exps)

    seen = set()
    unique_exps = []
    for e in candidate_exps:
        k = tuple(e)
        if k not in seen:
            seen.add(k)
            unique_exps.append(e)

    for exps in unique_exps:
        if time.time() > deadline - 5:
            break
        optimize_signs(exps, max_iter=100)

    # 3. Genetic search for broader exploration
    if time.time() < deadline - 10:
        genetic_search(pop_size=30, gen=20)

    # 4. Final polish on best candidate if found
    if best_candidate is not None and time.time() < deadline - 3:
        # Reconstruct from best_candidate
        exps = [i for i, c in enumerate(best_candidate) if c != 0]
        coeffs = [c for c in best_candidate if c != 0]
        if len(exps) == 12:
            optimize_full(exps, coeffs, max_iters=500)

    # Fallback
    if best_candidate is None:
        best_candidate = [1] * 12

    return best_candidate
