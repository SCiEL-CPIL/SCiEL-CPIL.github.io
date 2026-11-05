"""Egyptian fraction of 5/31 minimizing the LARGEST denominator.
Combines bounded DFS over early terms (structural idea) with exact greedy
tail completion (refinement machinery) and an explicit wall-clock deadline guard."""
import time
from fractions import Fraction

TARGET = Fraction(5, 31)
MAX_DEN = 10000000
MAX_TERMS = 256


def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin under the 90 s kill

    best = {"max": MAX_DEN, "sol": None}

    def valid(sol):
        if not sol or len(sol) > MAX_TERMS:
            return False
        if len(set(sol)) != len(sol):
            return False
        if any(d < 1 or d > MAX_DEN for d in sol):
            return False
        return sum(Fraction(1, d) for d in sol) == TARGET

    # --- Greedy tail completion (exact, distinct, bounded) ---
    def greedy_complete(r, used, min_den):
        vals = []
        while r > 0 and len(vals) < MAX_TERMS:
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d < min_den:
                d = min_den
            while d in used:
                d += 1
            if d > MAX_DEN:
                return None
            vals.append(d)
            used = used | {d}
            r -= Fraction(1, d)
        if r == 0:
            return vals
        return None

    # --- Bounded DFS over the first few terms, greedy tail for the rest ---
    def dfs(r, used, min_den, current, current_max):
        if time.time() > deadline:
            return
        if r == 0:
            if current_max < best["max"]:
                best["max"] = current_max
                best["sol"] = list(current)
            return
        if len(current) >= MAX_TERMS - 1:
            return
        # Prune: cannot beat current best
        if current_max >= best["max"]:
            return
        # Lower bound on next denominator: 1/d <= r  ->  d >= ceil(1/r)
        lo = max(min_den, (r.denominator + r.numerator - 1) // r.numerator)
        # Upper bound: keep the remainder representable by (MAX_TERMS - len) terms.
        # Feasibility: r - 1/d >= 0  ->  d >= ceil(1/r) (already enforced).
        # Also need r <= (remaining terms) / min_den, i.e. r * min_den <= remaining.
        remaining_terms = MAX_TERMS - len(current)
        if r * min_den > remaining_terms + 1:
            return
        # Try a bounded window of candidate denominators around the greedy choice.
        d = lo
        # Cap the branching: explore up to a window of candidates.
        limit = lo + 400
        while d <= limit:
            if d in used:
                d += 1
                continue
            if d > MAX_DEN:
                break
            new_r = r - Fraction(1, d)
            if new_r < 0:
                break  # further d only make new_r more negative
            current.append(d)
            dfs(new_r, used | {d}, d + 1, current, max(current_max, d))
            current.pop()
            if time.time() > deadline:
                return
            d += 1

    # Seed with a pure-greedy solution so best is finite from the start.
    used = set()
    r = TARGET
    sol = []
    while r > 0 and len(sol) < MAX_TERMS:
        d = (r.denominator + r.numerator - 1) // r.numerator
        while d in used:
            d += 1
        sol.append(d)
        used.add(d)
        r -= Fraction(1, d)
    if r == 0 and valid(sol):
        best["max"] = max(sol)
        best["sol"] = sol

    # Run the bounded DFS from the root.
    dfs(TARGET, set(), 1, [], 0)

    # If DFS found nothing better, fall back to the greedy solution.
    if best["sol"] is None or not valid(best["sol"]):
        # Absolute fallback: recompute greedy.
        used = set()
        r = TARGET
        sol = []
        while r > 0 and len(sol) < MAX_TERMS:
            d = (r.denominator + r.numerator - 1) // r.numerator
            while d in used:
                d += 1
            sol.append(d)
            used.add(d)
            r -= Fraction(1, d)
        best["sol"] = sol

    return best["sol"]
