"""Rewritten program: exact MIP (PuLP/CBC) for max-weight independent set on the
pairwise-coprime conflict graph, with a fast greedy fallback. Deadline-guarded."""
import math
import time

N = 256
DEADLINE = 60.0  # evaluator wall-clock limit; keep safety margin

# ---- precompute primes and prime factors once (module load, cheap) ----
def _sieve_primes(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, math.isqrt(n) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]

PRIMES = _sieve_primes(N)

def _prime_factors(x):
    fs = set()
    for p in PRIMES:
        if p * p > x:
            break
        if x % p == 0:
            fs.add(p)
            while x % p == 0:
                x //= p
    if x > 1:
        fs.add(x)
    return fs

PFACT = {a: _prime_factors(a) for a in range(1, N + 1)}

def construct():
    t0 = time.time()

    # ---- greedy fallback (always valid, computed fast) ----
    def greedy():
        # start with primes + 1 (a valid independent set)
        selected = {1} | set(PRIMES)
        # try to swap a prime for a multiple that is coprime to everything else
        improved = True
        while improved:
            improved = False
            for p in sorted(PRIMES):
                if p not in selected:
                    continue
                others = [a for a in selected if a != p]
                # candidate composites whose only prime factor is p
                cands = [m for m in range(1, N + 1)
                         if PFACT[m] == {p} and m != p]
                best = None
                for m in cands:
                    if all(math.gcd(m, o) == 1 for o in others):
                        if best is None or m > best:
                            best = m
                if best is not None and best > p:
                    selected.discard(p)
                    selected.add(best)
                    improved = True
        return selected

    greedy_set = greedy()

    # ---- exact MIP solve (PuLP + CBC) ----
    try:
        import pulp
        deadline = t0 + DEADLINE - 8.0  # safety margin
        prob = pulp.LpProblem("max_coprime_sum", pulp.LpMaximize)
        x = {a: pulp.LpVariable(f"x_{a}", cat="Binary") for a in range(1, N + 1)}
        prob += pulp.lpSum(a * x[a] for a in range(1, N + 1))
        # conflict constraints: for each prime p, at most one multiple of p selected
        for p in PRIMES:
            mults = [a for a in range(1, N + 1) if p in PFACT[a]]
            if len(mults) > 1:
                prob += pulp.lpSum(x[a] for a in mults) <= 1
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=int(max(1, deadline - time.time())))
        status = prob.solve(solver)
        if pulp.LpStatus[status] == "Optimal":
            chosen = [a for a in range(1, N + 1) if x[a].value() is not None and x[a].value() > 0.5]
            # validate
            ok = len(chosen) > 0
            if ok:
                for i in range(len(chosen)):
                    for j in range(i + 1, len(chosen)):
                        if math.gcd(chosen[i], chosen[j]) != 1:
                            ok = False
                            break
                    if not ok:
                        break
            if ok:
                chosen = [int(a) for a in chosen]
                return chosen
    except Exception:
        pass

    # ---- fallback ----
    return [int(a) for a in sorted(greedy_set)]
