import numpy as np
import time
import random

def _sieve_primes(limit):
    """Return list of primes <= limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:limit+1:step] = bytearray(len(range(start, limit+1, step)))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    primes = _sieve_primes(1000)
    n_primes = len(primes)
    N_max = 100000
    
    rng = np.random.default_rng(12345)
    
    def build_counts(residues, N):
        counts = np.zeros(N + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = int(residues[i])
            if r == 0:
                if p <= N:
                    counts[p::p] += 1
            else:
                if r <= N:
                    counts[r::p] += 1
        return counts

    def get_score(counts, N):
        """Returns (L, margin) where L is prefix length and margin is min coverage in [1,L]."""
        failures_mask = counts[1:N+1] < 2
        if not np.any(failures_mask):
            L = N
            # margin: min count in [1, L]
            margin = int(np.min(counts[1:L+1])) if L > 0 else 0
            return L, margin
        L = int(np.argmax(failures_mask)) + 1
        if L <= 1:
            return L, 0
        margin = int(np.min(counts[1:L+1]))
        return L, margin

    def apply_move(counts, residues, i, r_old, r_new, N):
        p = primes[i]
        if r_old == 0:
            if p <= N:
                counts[p::p] -= 1
        else:
            if r_old <= N:
                counts[r_old::p] -= 1
        if r_new == 0:
            if p <= N:
                counts[p::p] += 1
        else:
            if r_new <= N:
                counts[r_new::p] += 1

    def revert_move(counts, residues, i, r_old, r_new, N):
        p = primes[i]
        if r_new == 0:
            if p <= N:
                counts[p::p] -= 1
        else:
            if r_new <= N:
                counts[r_new::p] -= 1
        if r_old == 0:
            if p <= N:
                counts[p::p] += 1
        else:
            if r_old <= N:
                counts[r_old::p] += 1

    def pareto_dominates(a, b):
        """a = (L_a, margin_a), b = (L_b, margin_b). True if a dominates b."""
        if a[0] >= b[0] and a[1] >= b[1] and (a[0] > b[0] or a[1] > b[1]):
            return True
        return False

    def local_search(residues, N_cur, max_steps=200):
        """Safe-move local search: only accept moves that don't decrease L."""
        counts = build_counts(residues, N_cur)
        L_cur, margin_cur = get_score(counts, N_cur)
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            if L_cur == N_cur:
                break
            
            x_fail = L_cur
            
            # Find primes that don't cover x_fail
            non_covering = []
            for i in range(n_primes):
                p = primes[i]
                if x_fail % p != residues[i]:
                    non_covering.append(i)
            
            if not non_covering:
                break
            
            # Weighted selection: prefer smaller primes
            weights = np.array([1.0 / (primes[i] + 1) for i in non_covering])
            weights = weights / weights.sum()
            idx = int(rng.choice(len(non_covering), p=weights))
            i = non_covering[idx]
            p = primes[i]
            r_old = int(residues[i])
            r_new = int(x_fail % p)
            
            apply_move(counts, residues, i, r_old, r_new, N_cur)
            L_new, margin_new = get_score(counts, N_cur)
            
            # Safe move: only accept if L doesn't decrease
            if L_new >= L_cur:
                residues[i] = r_new
                L_cur = L_new
                margin_cur = margin_new
            else:
                revert_move(counts, residues, i, r_old, r_new, N_cur)
        
        return residues, L_cur, margin_cur

    def evolve_individual(residues, N_cur):
        """Evolve a single individual with local search + occasional kick."""
        residues, L_cur, margin_cur = local_search(residues, N_cur, max_steps=150)
        return residues, L_cur, margin_cur

    # Initialize population
    POP_SIZE = 24
    
    def init_individual(idx):
        if idx == 0:
            return np.array([rng.integers(0, p) for p in primes], dtype=np.int64)
        elif idx == 1:
            return np.zeros(n_primes, dtype=np.int64)
        elif idx == 2:
            return np.array([p // 2 for p in primes], dtype=np.int64)
        elif idx == 3:
            # Greedy: for each prime, pick residue covering most singly-covered numbers
            residues = np.zeros(n_primes, dtype=np.int64)
            N_init = 3000
            counts = np.zeros(N_init + 1, dtype=np.int32)
            for i in range(n_primes):
                p = primes[i]
                best_r = 0
                best_gain = -1
                # Sample residues
                candidates = set([0, 1, p // 2])
                for _ in range(min(10, p)):
                    candidates.add(int(rng.integers(0, p)))
                for r in candidates:
                    if r == 0:
                        if p <= N_init:
                            gain = np.sum(counts[p::p] == 1)
                        else:
                            gain = 0
                    else:
                        if r <= N_init:
                            gain = np.sum(counts[r::p] == 1)
                        else:
                            gain = 0
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
                residues[i] = best_r
                if best_r == 0:
                    if p <= N_init:
                        counts[p::p] += 1
                else:
                    if best_r <= N_init:
                        counts[best_r::p] += 1
            return residues
        elif idx == 4:
            return np.array([(i * 7919 + p * 104729) % p for i, p in enumerate(primes)], dtype=np.int64)
        elif idx == 5:
            return np.array([(i * 13 + 5) % p for i, p in enumerate(primes)], dtype=np.int64)
        else:
            return np.array([rng.integers(0, p) for p in primes], dtype=np.int64)

    population = [init_individual(i) for i in range(POP_SIZE)]
    
    # Evaluate initial population
    N_cur = 5000
    scores = []
    for ind in population:
        L, margin = evolve_individual(ind.copy(), N_cur)
        scores.append((L, margin))
    
    best_L = max(s[0] for s in scores)
    best_residues = None
    for i, s in enumerate(scores):
        if s[0] == best_L:
            best_residues = population[i].copy()
            break
    
    # Evolutionary loop
    gen = 0
    stall_generations = 0
    MAX_STALL = 15
    
    while time.time() < deadline - 10.0:
        gen += 1
        
        # Check if we can expand the window
        if best_L >= N_cur and N_cur < N_max:
            N_cur = min(N_cur * 2, N_max)
            # Re-evaluate all with new window
            for i in range(POP_SIZE):
                L, margin = evolve_individual(population[i].copy(), N_cur)
                scores[i] = (L, margin)
            best_L = max(s[0] for s in scores)
            for i, s in enumerate(scores):
                if s[0] == best_L:
                    best_residues = population[i].copy()
                    break
            stall_generations = 0
            continue
        
        # Pareto front selection
        pareto_front = []
        for i in range(POP_SIZE):
            dominated = False
            for j in range(POP_SIZE):
                if i != j and pareto_dominates(scores[j], scores[i]):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(i)
        
        # If no Pareto front found (shouldn't happen), use all
        if not pareto_front:
            pareto_front = list(range(POP_SIZE))
        
        # Create next generation
        new_population = []
        new_scores = []
        
        # Elitism: keep best individuals
        sorted_idx = sorted(range(POP_SIZE), key=lambda i: (-scores[i][0], -scores[i][1]))
        elite_count = max(2, POP_SIZE // 4)
        for i in sorted_idx[:elite_count]:
            new_population.append(population[i].copy())
            new_scores.append(scores[i])
        
        # Fill rest with offspring
        while len(new_population) < POP_SIZE:
            # Selection: tournament from Pareto front
            if len(pareto_front) >= 2:
                a = pareto_front[int(rng.integers(len(pareto_front)))]
                b = pareto_front[int(rng.integers(len(pareto_front)))]
            else:
                a = pareto_front[0]
                b = pareto_front[0]
            
            # Crossover: uniform
            child = population[a].copy()
            mask = rng.random(n_primes) > 0.5
            child[mask] = population[b][mask]
            
            # Mutation
            n_mut = int(rng.integers(1, 4))
            for _ in range(n_mut):
                i = int(rng.integers(n_primes))
                p = primes[i]
                child[i] = int(rng.integers(0, p))
            
            # Local search
            child, L, margin = evolve_individual(child, N_cur)
            new_population.append(child)
            new_scores.append((L, margin))
        
        population = new_population
        scores = new_scores
        
        # Track best
        gen_best_L = max(s[0] for s in scores)
        if gen_best_L > best_L:
            best_L = gen_best_L
            for i, s in enumerate(scores):
                if s[0] == best_L:
                    best_residues = population[i].copy()
                    break
            stall_generations = 0
        else:
            stall_generations += 1
        
        # If stalled, do a kick on the best individual
        if stall_generations >= MAX_STALL:
            # Find best individual
            best_idx = int(np.argmax([s[0] for s in scores]))
            # Kick: random perturbation
            kick_count = int(rng.integers(3, 8))
            for _ in range(kick_count):
                i = int(rng.integers(n_primes))
                p = primes[i]
                population[best_idx][i] = int(rng.integers(0, p))
            # Local search after kick
            population[best_idx], L, margin = evolve_individual(population[best_idx], N_cur)
            scores[best_idx] = (L, margin)
            if L > best_L:
                best_L = L
                best_residues = population[best_idx].copy()
            stall_generations = 0
    
    # Final verification with full N_max
    if best_residues is not None and time.time() < deadline - 2.0:
        N_verify = min(best_L * 2, N_max) if best_L > 0 else N_max
        counts_v = build_counts(best_residues, N_verify)
        final_L, final_margin = get_score(counts_v, N_verify)
        # Update best if needed
        if final_L > best_L:
            best_L = final_L
    
    if best_residues is None:
        best_residues = np.array([rng.integers(0, p) for p in primes], dtype=np.int64)
    
    return best_residues.tolist()
