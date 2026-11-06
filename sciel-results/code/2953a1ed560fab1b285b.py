import math
import time

def construct():
    deadline = time.time() + 140

    max_m = 200
    gcd_table = {}
    for i in range(20, max_m + 1):
        for j in range(20, max_m + 1):
            gcd_table[(i, j)] = math.gcd(i, j)

    moduli_ordered = list(range(20, 201))
    weights = [1.0 / m for m in moduli_ordered]

    suffix_weights = [0.0] * (len(moduli_ordered) + 1)
    for i in range(len(moduli_ordered) - 1, -1, -1):
        suffix_weights[i] = suffix_weights[i + 1] + weights[i]

    best_score = 0.0
    best_solution = []

    def is_compatible(m_new, r_new, selected):
        for m_sel, r_sel in selected:
            g = gcd_table.get((m_new, m_sel), gcd_table.get((m_sel, m_new)))
            if r_new % g == r_sel % g:
                return False
        return True

    def find_any_valid_r(m, selected):
        for r in range(m):
            if is_compatible(m, r, selected):
                return r
        return None

    def greedy_complete(idx, selected, current_score):
        sel = [x[:] for x in selected]
        for i in range(idx, len(moduli_ordered)):
            m = moduli_ordered[i]
            r = find_any_valid_r(m, sel)
            if r is not None:
                sel.append([m, r])
                current_score += 1.0 / m
        return current_score, sel

    def bb_search(idx, selected, current_score, depth_remaining):
        nonlocal best_score, best_solution

        if time.time() > deadline - 2:
            return

        if depth_remaining <= 0:
            gs, gsel = greedy_complete(idx, selected, current_score)
            if gs > best_score:
                best_score = gs
                best_solution = [x[:] for x in gsel]
            return

        m = moduli_ordered[idx]
        ub = current_score + suffix_weights[idx]
        if ub <= best_score + 1e-12:
            return

        # Skip
        bb_search(idx + 1, selected, current_score, depth_remaining - 1)

        # Try residues
        for r in range(m):
            if is_compatible(m, r, selected):
                selected.append([m, r])
                bb_search(idx + 1, selected, current_score + 1.0 / m, depth_remaining - 1)
                selected.pop()

    # Initial greedy
    greedy_selected = []
    for m in moduli_ordered:
        r = find_any_valid_r(m, greedy_selected)
        if r is not None:
            greedy_selected.append([m, r])
    best_score = sum(1.0 / m for m, _ in greedy_selected)
    best_solution = [x[:] for x in greedy_selected]

    # B&B with limited depth (branch on first ~30 moduli, greedy complete rest)
    bb_search(0, [], 0.0, 30)

    return best_solution

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
        for j in range(i+1, len(res)):
            m1, r1 = res[i]
            m2, r2 = res[j]
            g = math.gcd(m1, m2)
            if r1 % g == r2 % g:
                valid = False
                print(f"Invalid: {res[i]} and {res[j]}")
    print(f"Valid: {valid}")
