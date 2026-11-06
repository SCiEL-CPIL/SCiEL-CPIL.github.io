import math
import time
import random

def construct():
    deadline = time.time() + 140  # 150s wall-clock limit, keep 10s margin

    max_m = 200
    min_m = 20

    # ---------- Precomputation ----------
    # gcd table
    gcd_table = [[0] * (max_m + 1) for _ in range(max_m + 1)]
    for i in range(min_m, max_m + 1):
        for j in range(min_m, max_m + 1):
            gcd_table[i][j] = math.gcd(i, j)

    # prime factors
    prime_factors = [[] for _ in range(max_m + 1)]
    for p in range(2, max_m + 1):
        if not prime_factors[p]:
            for multiple in range(p, max_m + 1, p):
                prime_factors[multiple].append(p)

    # ---------- Fast validity (O(1) per pair via residue masks) ----------
    # For a class (m, r), the set of "forbidden" residues for a modulus m2 is
    # {x in [0, g) : x == r mod g} where g = gcd(m, m2). We store, for the
    # current solution, for each modulus m2 (20..200), the set of residue
    # classes mod gcd(m, m2) that are already taken.
    # Faster: maintain for each modulus m2 a set of (g, r mod g) taken, where
    # g = gcd(m, m2) for each selected m. Actually the constraint is pairwise,
    # so we maintain: for each selected class (m_i, r_i), and for each candidate
    # modulus m2, the forbidden residue is r_i mod gcd(m_i, m2).
    # We'll just do the O(n) check but with the precomputed gcd table; n is small.

    def is_valid(m, r, selected, skip_idx=None):
        """Check if (m, r) is valid against selected classes."""
        for idx in range(len(selected)):
            if skip_idx is not None and idx == skip_idx:
                continue
            m2, r2 = selected[idx]
            g = gcd_table[m][m2]
            if r % g == r2 % g:
                return False
        return True

    def calc_score(selected):
        return sum(1.0 / m for m, _ in selected)

    def greedy_fill(selected, deadline):
        """Try to add more classes to the selected list, smallest m first."""
        for m in range(min_m, max_m + 1):
            if time.time() > deadline:
                break
            if any(m == m_sel for m_sel, _ in selected):
                continue
            for r in range(m):
                if is_valid(m, r, selected):
                    selected.append([m, r])
                    break
        return selected

    def local_search_polish(selected, deadline):
        """Intensive local search: try all single moves, accept improvements,
        iterate until no improvement or deadline."""
        current_sol = [list(x) for x in selected]
        current_score = calc_score(current_sol)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # 1) Try adding each missing modulus (smallest first for max gain)
            for m in range(min_m, max_m + 1):
                if time.time() > deadline:
                    return current_sol, current_score
                if any(m == m_sel for m_sel, _ in current_sol):
                    continue
                for r in range(m):
                    if is_valid(m, r, current_sol):
                        current_sol.append([m, r])
                        current_score += 1.0 / m
                        improved = True
                        break
            # 2) Try changing residue of each class to enable a new addition
            for idx in range(len(current_sol)):
                if time.time() > deadline:
                    return current_sol, current_score
                m_chg, old_r = current_sol[idx]
                found = False
                for new_r in range(m_chg):
                    if new_r == old_r:
                        continue
                    if not is_valid(m_chg, new_r, current_sol, skip_idx=idx):
                        continue
                    current_sol[idx] = [m_chg, new_r]
                    added = False
                    for m_add in range(min_m, max_m + 1):
                        if any(m_add == m_sel for m_sel, _ in current_sol):
                            continue
                        for r_add in range(m_add):
                            if is_valid(m_add, r_add, current_sol):
                                current_sol.append([m_add, r_add])
                                current_score += 1.0 / m_add
                                added = True
                                break
                        if added:
                            break
                    if added:
                        improved = True
                        found = True
                        break
                    else:
                        current_sol[idx] = [m_chg, old_r]
                if found:
                    continue
                # 3) Try removing this class and re-adding a better one
                m_rem, r_rem = current_sol[idx]
                current_sol.pop(idx)
                current_score -= 1.0 / m_rem
                best_add = None
                best_add_score = -1.0
                for m_add in range(min_m, max_m + 1):
                    if m_add == m_rem:
                        continue
                    if any(m_add == m_sel for m_sel, _ in current_sol):
                        continue
                    for r_add in range(m_add):
                        if is_valid(m_add, r_add, current_sol):
                            if 1.0 / m_add > best_add_score:
                                best_add_score = 1.0 / m_add
                                best_add = [m_add, r_add]
                            break
                if best_add is not None and best_add_score > 1.0 / m_rem:
                    current_sol.append(best_add)
                    current_score += best_add_score
                    improved = True
                else:
                    current_sol.insert(idx, [m_rem, r_rem])
                    current_score += 1.0 / m_rem
        return current_sol, current_score

    def simulated_annealing(initial_solution, deadline):
        current_sol = [list(x) for x in initial_solution]
        current_score = calc_score(current_sol)
        best_sol = [list(x) for x in current_sol]
        best_score = current_score

        T = 0.5
        cooling = 0.9995
        min_T = 1e-8

        while time.time() < deadline:
            if T < min_T:
                break

            move_type = random.random()
            new_sol = [list(x) for x in current_sol]
            new_score = current_score
            valid_move = False

            if move_type < 0.50:
                # Add a modulus - prefer smaller moduli for higher contribution
                m_add = random.randint(min_m, max_m)
                if random.random() < 0.3:
                    m_add = random.randint(min_m, min_m + 30)
                if any(m == m_add for m, _ in new_sol):
                    continue
                for _ in range(8):
                    r_add = random.randint(0, m_add - 1)
                    if is_valid(m_add, r_add, new_sol):
                        new_sol.append([m_add, r_add])
                        new_score += 1.0 / m_add
                        valid_move = True
                        break
            elif move_type < 0.75:
                # Remove a modulus - prefer removing large moduli (low contribution)
                if not new_sol:
                    continue
                weights = [1.0 / new_sol[i][0] for i in range(len(new_sol))]
                total_w = sum(weights)
                r = random.random() * total_w
                cumulative = 0
                idx = len(new_sol) - 1
                for i in range(len(new_sol)):
                    cumulative += weights[i]
                    if r <= cumulative:
                        idx = i
                        break
                m_rem, r_rem = new_sol[idx]
                new_sol.pop(idx)
                new_score -= 1.0 / m_rem
                valid_move = True
            elif move_type < 0.90:
                # Change residue of a random class
                if not new_sol:
                    continue
                idx = random.randint(0, len(new_sol) - 1)
                m_chg, old_r = new_sol[idx]
                for _ in range(10):
                    new_r = random.randint(0, m_chg - 1)
                    if new_r == old_r:
                        continue
                    if is_valid(m_chg, new_r, new_sol, skip_idx=idx):
                        new_sol[idx] = [m_chg, new_r]
                        valid_move = True
                        break
            else:
                # Swap: remove one class, try to add another (different m)
                if not new_sol:
                    continue
                idx = random.randint(0, len(new_sol) - 1)
                m_old, r_old = new_sol[idx]
                new_sol.pop(idx)
                new_score -= 1.0 / m_old
                for _ in range(15):
                    m_new = random.randint(min_m, max_m)
                    if m_new == m_old:
                        continue
                    if any(m == m_new for m, _ in new_sol):
                        continue
                    r_new = random.randint(0, m_new - 1)
                    if is_valid(m_new, r_new, new_sol):
                        new_sol.append([m_new, r_new])
                        new_score += 1.0 / m_new
                        valid_move = True
                        break

            if not valid_move:
                continue

            delta = new_score - current_score
            if delta > 0 or (delta > -5.0 and random.random() < math.exp(delta / T)):
                current_sol = new_sol
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_sol = [list(x) for x in current_sol]

            T *= cooling

        return best_sol

    def initial_solution(deadline):
        """Build initial solution: greedy smallest-first, then fill."""
        selected = []
        for m in range(min_m, max_m + 1):
            if time.time() > deadline:
                break
            for r in range(m):
                if is_valid(m, r, selected):
                    selected.append([m, r])
                    break
        selected = greedy_fill(selected, deadline)
        return selected

    # ---------- Main search loop ----------
    best_solution = initial_solution(deadline - 20)
    best_solution, best_score = local_search_polish(best_solution, deadline - 15)

    # Simulated annealing from polished initial
    result = simulated_annealing(best_solution, deadline - 8)
    result, sa_score = local_search_polish(result, deadline - 3)
    if sa_score > best_score:
        best_score = sa_score
        best_solution = result

    # Restart a few times if time permits
    restart_deadline = deadline - 5
    for _ in range(4):
        if time.time() > restart_deadline:
            break
        sol = initial_solution(deadline - 12)
        sol, s1 = local_search_polish(sol, deadline - 8)
        sol = simulated_annealing(sol, deadline - 5)
        sol, s2 = local_search_polish(sol, deadline - 2)
        if s2 > best_score:
            best_score = s2
            best_solution = sol

    # Final exhaustive add-polish
    final = [list(x) for x in best_solution]
    changed = True
    while changed and time.time() < deadline - 1:
        changed = False
        for m in range(min_m, max_m + 1):
            if any(m == m_sel for m_sel, _ in final):
                continue
            for r in range(m):
                if is_valid(m, r, final):
                    final.append([m, r])
                    changed = True
                    break
    return final

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    score = sum(1.0 / m for m, _ in res)
    print(f"Score: {score}")
    print(f"Time: {end - start:.2f}s")
    print(f"Classes: {len(res)}")
    valid = True
    for i in range(len(res)):
        for j in range(i + 1, len(res)):
            m1, r1 = res[i]
            m2, r2 = res[j]
            g = math.gcd(m1, m2)
            if r1 % g == r2 % g:
                valid = False
                print(f"Invalid: {res[i]} and {res[j]}")
    print(f"Valid: {valid}")
