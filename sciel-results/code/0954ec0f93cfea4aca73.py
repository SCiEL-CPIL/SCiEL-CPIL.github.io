"""Direct construction: factor 12! into distinct factors > 12, minimizing the max factor.

12! = 2^10 * 3^5 * 5^2 * 7 * 11 = 479001600.

Strategy: build a candidate factor set with exponent arithmetic, verify product == 12!,
all factors distinct and > 12, at most 12 factors. Then locally try to lower the max
factor by replacing one factor with a smaller valid divisor, keeping distinctness.
"""
import math
import time

T12 = math.factorial(12)  # 479001600
PRIMES = [2, 3, 5, 7, 11]
EXP = {2: 10, 3: 5, 5: 2, 7: 1, 11: 1}

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def exp_to_int(e):
    r = 1
    for p in PRIMES:
        if e.get(p, 0):
            r *= p ** e[p]
    return r


def int_to_exp(n):
    e = {}
    for p in PRIMES:
        c = 0
        while n % p == 0:
            n //= p
            c += 1
        if c:
            e[p] = c
    if n != 1:
        e[n] = 1
    return e


def sub(a, b):
    r = dict(a)
    for p, c in b.items():
        r[p] = r.get(p, 0) - c
        if r[p] == 0:
            del r[p]
    return r


def valid_solution(factors):
    if not (1 <= len(factors) <= 12):
        return False
    if len(set(factors)) != len(factors):
        return False
    if any(f < 13 for f in factors):
        return False
    p = 1
    for f in factors:
        p *= f
    return p == T12


def try_build(candidates):
    """Given a list of candidate factors (ints > 12, distinct), return the subset whose
    product is exactly T12 (using exponent arithmetic) if one exists, else None."""
    cand = []
    seen = set()
    for f in candidates:
        if f in seen:
            continue
        seen.add(f)
        cand.append((f, int_to_exp(f)))

    # Try to find a subset whose exponents sum to EXP via DFS.
    best = []

    def dfs(i, remaining, chosen):
        if not best and time.time() > deadline:
            return
        if remaining == 0:
            # all exponents zero
            if not best or max(chosen) < max(best):
                best[:] = list(chosen)
            return
        if i == len(cand):
            return
        f, e = cand[i]
        # skip
        dfs(i + 1, remaining, chosen)
        # take, if it fits
        ok = True
        for p, c in e.items():
            if remaining.get(p, 0) < c:
                ok = False
                break
        if ok:
            new_rem = sub(remaining, e)
            chosen.append(f)
            dfs(i + 1, new_rem, chosen)
            chosen.pop()

    # remaining as a tuple key is not hashable easily; use a frozenset-of-items
    def key(rem):
        return tuple(sorted(rem.items()))

    start = dict(EXP)
    dfs(0, start, [])
    return best or None


deadline = time.time() + 80  # safety margin under 90s


def build_candidate_sets():
    """Generate explicit factor sets that multiply to 12!."""
    sets = []

    # Set A: derived from grouping prime powers.
    # 2^10 -> 32, 2^6 ; 3^5 -> 27, 3^2 ; 5^2 -> 25 ; 7 -> 7 (too small, combine)
    # Let's just propose specific numbers and verify.

    # Proposal 1: {32, 27, 25, 21, 33, 100}
    # 32=2^5, 27=3^3, 25=5^2, 21=3*7, 33=3*11, 100=2^2*5^2
    # 2:5+2=7 (need 10) no.
    # Instead, construct directly:
    # 2^10 = 1024 -> too large alone? 1024 is fine as a factor (>12).
    # Let's aim for a set:
    # {1024, 27, 25, 7, 11} -> 7 and 11 < 13, invalid.
    # Combine 7 with something: 7*3=21, 11*3=33.
    # 2^10=1024, 3^2*7=63, 3^3=27, 5^2=25, 11 -> 11 invalid.
    # 11*3=33, then 3 left: 3^5 total, used 3^2(63)+3^3(27)+3(33)=3^6, too many.
    #
    # Let me just brute-force a small explicit search at import time with a tight bound.
    return sets


def direct_search():
    """Direct DFS: pick factors in increasing order; each must divide the remaining
    product; all > 12; distinct. Minimize the max factor."""
    best = []
    best_max = T12

    def dfs(remaining, last, chosen):
        nonlocal best, best_max
        if time.time() > deadline:
            return
        if remaining == 1:
            if chosen:
                m = max(chosen)
                if m < best_max:
                    best_max = m
                    best = list(chosen)
            return
        # remaining itself as last factor
        if remaining > last and remaining >= 13:
            m = max(max(chosen) if chosen else 0, remaining)
            if m < best_max:
                best_max = m
                best = chosen + [remaining]
        # try divisors d in (last, sqrt(remaining)]
        lim = math.isqrt(remaining)
        start = max(last + 1, 13)
        for d in range(start, lim + 1):
            if time.time() > deadline:
                return
            if remaining % d == 0:
                c = remaining // d
                if c > d:  # need next factors > d; c==d impossible (distinct)
                    chosen.append(d)
                    dfs(c, d, chosen)
                    chosen.pop()

    dfs(T12, 0, [])
    return best


def local_improve(factors, deadline):
    """Try to reduce the max factor by splitting it into two smaller distinct factors."""
    if not factors:
        return factors
    factors = sorted(factors)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        cur_max = factors[-1]
        others = factors[:-1]
        # Try to split cur_max into a*b where a,b > 12, a < b, a*b = cur_max
        # and a,b not in others
        for a in range(13, math.isqrt(cur_max) + 1):
            if time.time() > deadline:
                break
            if cur_max % a == 0:
                b = cur_max // a
                if b > a and a >= 13 and b >= 13:
                    if a not in others and b not in others:
                        new_factors = sorted(others + [a, b])
                        if valid_solution(new_factors):
                            if max(new_factors) < cur_max:
                                factors = new_factors
                                improved = True
                                break
        if not improved:
            # Try 2-for-1: merge two smallest factors into one if result < cur_max
            # This doesn't help minimize max, so skip
            pass
    return factors


def solve_ilp():
    """Solve the factorization problem using Integer Linear Programming."""
    if not HAS_SCIPY:
        return None
    
    # 1. Generate all divisors of T12 that are > 12
    divisors = []
    # Iterate up to sqrt(T12) to find pairs
    limit = math.isqrt(T12)
    for i in range(1, limit + 1):
        if T12 % i == 0:
            d1 = i
            d2 = T12 // i
            if d1 > 12:
                divisors.append(d1)
            if d2 > 12 and d2 != d1:
                divisors.append(d2)
    
    # Remove duplicates and sort
    divisors = sorted(set(divisors))
    N = len(divisors)
    
    if N == 0:
        return None

    # Precompute prime exponents for each divisor
    # primes: 2, 3, 5, 7, 11
    # exp_matrix[i][j] is the exponent of primes[j] in divisors[i]
    exp_matrix = []
    for d in divisors:
        row = []
        for p in PRIMES:
            count = 0
            while d % p == 0:
                d //= p
                count += 1
            row.append(count)
        exp_matrix.append(row)
    
    # 2. Formulate ILP
    # Variables: x_0 ... x_{N-1} (binary), x_N (continuous, M)
    # Total variables: N + 1
    # Indices: 0..N-1 for x_i, N for M
    
    # Objective: Minimize M
    # c = [0, ..., 0, 1]
    c = [0] * N + [1]
    
    # Constraints:
    # 1. M >= d_i * x_i  =>  -M + d_i * x_i <= 0
    #    For each i: [d_i, ..., 0, ..., 0, -1] dot vars <= 0
    # 2. Sum of exponents for each prime p_j is EXP[p_j]
    #    For each prime j: sum_i exp_matrix[i][j] * x_i = EXP[PRIMES[j]]
    # 3. Sum x_i <= 12
    # 4. x_i in {0, 1}
    
    # Build A_ub, b_ub for inequality constraints (A_ub @ vars <= b_ub)
    # Build A_eq, b_eq for equality constraints (A_eq @ vars == b_eq)
    
    A_ub = []
    b_ub = []
    
    # Constraints 1: M >= d_i x_i
    for i in range(N):
        row = [0] * (N + 1)
        row[i] = divisors[i]
        row[N] = -1
        A_ub.append(row)
        b_ub.append(0)
        
    # Constraint 3: Sum x_i <= 12
    row_sum = [1] * N + [0]
    A_ub.append(row_sum)
    b_ub.append(12)
    
    # Constraints 2: Equality on prime exponents
    A_eq = []
    b_eq = []
    for j, p in enumerate(PRIMES):
        row = [0] * (N + 1)
        for i in range(N):
            row[i] = exp_matrix[i][j]
        A_eq.append(row)
        b_eq.append(EXP[p])
        
    # Bounds
    # x_i in [0, 1] for i in 0..N-1
    # M in [0, inf)
    bounds = Bounds(lb=[0] * (N + 1), ub=[1] * N + [math.inf])
    
    # Integrality
    integrality = [1] * N + [0]
    
    # Solve
    try:
        res = milp(
            c=c,
            constraints=(
                LinearConstraint(A_ub, -math.inf, b_ub),
                LinearConstraint(A_eq, b_eq, b_eq)
            ),
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": 30}  # Leave time for verification
        )
        
        if res.status == 0:  # Optimal
            x = res.x
            selected = []
            for i in range(N):
                if x[i] > 0.5:
                    selected.append(divisors[i])
            if selected:
                return selected
        else:
            return None
    except Exception:
        return None

def construct():
    global deadline
    deadline = time.time() + 85  # reset safety margin, leave 5s buffer
    
    # Try ILP first
    best = solve_ilp()
    best_max = max(best) if best else T12
    
    # Fallback to known-good explicit set if ILP fails
    if not best:
        candidate = [144, 64, 77, 25, 27]
        if valid_solution(candidate):
            best = list(candidate)
            best_max = max(candidate)
        else:
            best = []
            best_max = T12

    # Local improvement: try to split the largest factor into two smaller ones
    if best and time.time() < deadline:
        best = local_improve(best, deadline)
        best_max = max(best) if best else T12

    # Full direct DFS if time remains and we haven't found a very good solution
    # ILP should be optimal, so DFS is only for safety if ILP is buggy/absent
    if best and max(best) > 200 and time.time() < deadline - 5:
        found = direct_search()
        if found and max(found) < best_max:
            best = found
            best_max = max(found)

    # Final verification
    if best and valid_solution(best):
        return best
    
    # Emergency fallback
    return [T12]
