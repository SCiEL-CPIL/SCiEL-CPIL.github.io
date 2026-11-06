import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # Safety margin

    moduli = list(range(2, 129))
    N = len(moduli)  # 127 moduli: 2 to 128 inclusive

    # Precompute gcd table
    gcd_table = [[0]*129 for _ in range(129)]
    for i in range(2, 129):
        for j in range(i, 129):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def check_pair_disjoint(r1, m1, r2, m2):
        """Check if residue class r1 mod m1 is disjoint from r2 mod m2.
        They are disjoint iff r1 mod g != r2 mod g where g = gcd(m1, m2)."""
        g = gcd_table[m1][m2]
        if g == 1:
            return False  # gcd=1 means they intersect (by CRT)
        return (r1 % g) != (r2 % g)

    # ---- Deterministic constructions to get good initial solutions ----

    def construct_multiples_of(p):
        """Construct a solution using all multiples of p in [2,128] with distinct residues mod p."""
        multiples = [m for m in moduli if m % p == 0]
        k = len(multiples)
        p_eff = min(p, k)
        if p_eff <= 1:
            return []
        # Assign distinct residues 0, 1, ..., p_eff-1 to the multiples
        result = []
        for i, m in enumerate(multiples[:p_eff]):
            result.append([i % p, m])
        return result

    def construct_single_prime():
        """Find the best single-prime construction."""
        best = []
        for p in range(2, 129):
            if time.time() > deadline:
                break
            sol = construct_multiples_of(p)
            if len(sol) > len(best):
                best = sol
        return best

    def construct_greedy_order(order, deterministic=True):
        """Greedy: try to add moduli in the given order, picking a valid residue."""
        selected = []
        for m in order:
            if not selected:
                selected.append((0, m))
                continue
            found = False
            if deterministic:
                for r in range(m):
                    ok = True
                    for r_k, m_k in selected:
                        if not check_pair_disjoint(r, m, r_k, m_k):
                            ok = False
                            break
                    if ok:
                        selected.append((r, m))
                        found = True
                        break
            else:
                valid_r = []
                for r in range(m):
                    ok = True
                    for r_k, m_k in selected:
                        if not check_pair_disjoint(r, m, r_k, m_k):
                            ok = False
                            break
                    if ok:
                        valid_r.append(r)
                if valid_r:
                    r = random.choice(valid_r)
                    selected.append((r, m))
                    found = True
            if not found:
                continue
        return [[r, m] for r, m in selected]

    # ---- Local search / hill climbing to improve a solution ----

    def try_add_unselected(sol):
        """Try to add any unselected modulus with a valid residue."""
        selected_set = set(m for _, m in sol)
        new_sol = list(sol)
        for m in moduli:
            if m in selected_set:
                continue
            for r in range(m):
                ok = True
                for r_k, m_k in new_sol:
                    if not check_pair_disjoint(r, m, r_k, m_k):
                        ok = False
                        break
                if ok:
                    new_sol.append([r, m])
                    selected_set.add(m)
                    break
        return new_sol

    def local_search(sol, max_rounds=50):
        """Hill climbing: try swapping to add more classes."""
        sol = list(sol)
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            # Try adding each unselected modulus
            new_sol = try_add_unselected(sol)
            if len(new_sol) > len(sol):
                sol = new_sol
                improved = True
            if not improved:
                # Try 1-for-1 swap: remove one, try to add two
                for i in range(len(sol)):
                    if time.time() > deadline:
                        break
                    sol_copy = sol[:i] + sol[i+1:]
                    new_sol = try_add_unselected(sol_copy)
                    if len(new_sol) > len(sol):
                        sol = new_sol
                        improved = True
                        break
            if not improved:
                break
        return sol

    # ---- Main optimization ----

    best_solution = []

    # 1. Single-prime constructions
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
        if time.time() > deadline:
            break
        sol = construct_multiples_of(p)
        if len(sol) > len(best_solution):
            best_solution = sol

    # 2. Greedy with various orderings
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    orders = [
        list(range(2, 129)),
        list(range(128, 1, -1)),
        sorted(moduli, key=lambda x: -num_divisors(x)),
        sorted(moduli, key=lambda x: x.bit_count(), reverse=True),
        sorted(moduli, key=lambda x: -x.bit_count()),
    ]

    for order in orders:
        if time.time() > deadline:
            break
        sol = construct_greedy_order(order, deterministic=True)
        if len(sol) > len(best_solution):
            best_solution = sol

    # 3. Randomized restarts with greedy
    for _ in range(200):
        if time.time() > deadline:
            break
        random_order = moduli[:]
        random.shuffle(random_order)
        sol = construct_greedy_order(random_order, deterministic=False)
        if len(sol) > len(best_solution):
            best_solution = sol

    # 4. Local search from best solution
    best_solution = local_search(best_solution, max_rounds=100)

    # 5. More randomized restarts with local search
    for _ in range(100):
        if time.time() > deadline:
            break
        random_order = moduli[:]
        random.shuffle(random_order)
        sol = construct_greedy_order(random_order, deterministic=False)
        sol = local_search(sol, max_rounds=30)
        if len(sol) > len(best_solution):
            best_solution = sol

    # 6. Try to further improve with more aggressive local search
    if time.time() < deadline - 5.0:
        best_solution = local_search(best_solution, max_rounds=200)

    # Final safety: ensure non-empty
    if not best_solution:
        best_solution = [[0, 2]]

    return [[r, m] for r, m in best_solution]
