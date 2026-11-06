import time
import random
import math
import sys

def construct():
    """
    Find a 5-coloring of K_n with no monochromatic C4, maximizing n.
    Combines algebraic GF(25) construction (n=26) with incremental
    energy local search using best-of-4 moves for n>26.
    """
    deadline = time.time() + 115

    def time_left():
        return deadline - time.time()

    def edge_idx(i, j, n):
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def compute_energy(adj, n):
        total = 0
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    common = adj[c][i] & adj[c][j]
                    k = common.bit_count()
                    if k >= 2:
                        total += k * (k - 1) // 2
        return total

    # --- Algebraic GF(25) construction for n=26 ---
    def gf25_construction():
        p = 5
        # GF(25) elements as (a, b) = a + b*alpha, alpha^2 = 3
        elements = [(a, b) for a in range(p) for b in range(p)]

        def gf_mul(x, y):
            a1, b1 = x
            a2, b2 = y
            a = (a1 * a2 + 3 * b1 * b2) % p
            b = (a1 * b2 + a2 * b1) % p
            return (a, b)

        def gf_pow(x, n):
            result = (1, 0)
            base = x
            while n > 0:
                if n % 2 == 1:
                    result = gf_mul(result, base)
                base = gf_mul(base, base)
                n //= 2
            return result

        # Find primitive element (order 24)
        primitive = None
        for elem in elements:
            if elem == (0, 0):
                continue
            if gf_pow(elem, 24) == (1, 0):
                is_primitive = True
                for d in [1, 2, 3, 4, 6, 8, 12]:
                    if gf_pow(elem, d) == (1, 0):
                        is_primitive = False
                        break
                if is_primitive:
                    primitive = elem
                    break

        if primitive is None:
            return None

        # Generate powers
        powers = []
        current = (1, 0)
        for _ in range(24):
            powers.append(current)
            current = gf_mul(current, primitive)

        log_map = {elem: i for i, elem in enumerate(powers)}

        n = 26
        colors = []

        for i in range(n):
            for j in range(i + 1, n):
                if i == 25 or j == 25:
                    other = j if i == 25 else i
                    if other == 0:
                        colors.append(0)
                    else:
                        c = log_map[elements[other]] % 4
                        colors.append(c + 1)
                else:
                    elem_i = elements[i]
                    elem_j = elements[j]
                    if elem_i == (0, 0) or elem_j == (0, 0):
                        colors.append(0)
                    else:
                        log_i = log_map[elem_i]
                        log_j = log_map[elem_j]
                        c = (log_i - log_j) % 4
                        colors.append(c + 1)

        # Validate
        adj = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        if compute_energy(adj, n) == 0:
            return {'n': n, 'colors': colors}
        return None

    # Try algebraic construction first
    best_n = 1
    best_colors = [0]

    gf_result = gf25_construction()
    if gf_result is not None:
        best_n = gf_result['n']
        best_colors = gf_result['colors']

    # --- Local search for n > 26 (or if algebraic fails, for n >= 8) ---
    start_n = best_n + 1 if best_n >= 8 else 8
    for target_n in range(start_n, 29):
        if time_left() < 10:
            break

        n = target_n
        num_edges = n * (n - 1) // 2

        edge_list = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_list.append((i, j, edge_idx(i, j, n)))

        found = False
        max_restarts = 150

        for restart in range(max_restarts):
            if time_left() < 5:
                break

            colors = [random.randint(0, 4) for _ in range(num_edges)]

            adj = [[0] * n for _ in range(5)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx]
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
                    idx += 1

            energy = compute_energy(adj, n)

            if energy == 0:
                best_n = n
                best_colors = colors[:]
                found = True
                break

            best_energy = energy
            best_colors_local = colors[:]

            temp = 2.0
            cooling = 0.99995
            min_temp = 0.05
            max_steps = 500000
            stuck_count = 0
            max_stuck = 20000

            for step in range(max_steps):
                if time_left() < 3:
                    break
                if temp < min_temp:
                    break
                if stuck_count > max_stuck:
                    break

                # Best-of-4 move selection
                best_delta = float('inf')
                best_ei = -1
                best_u = best_v = best_c_old = best_c_new = -1

                for _ in range(4):
                    ei = random.randint(0, num_edges - 1)
                    u, v, e_idx = edge_list[ei]
                    c_old = colors[e_idx]
                    c_new = random.randint(0, 4)
                    if c_new == c_old:
                        continue

                    delta = 0

                    # Remove from c_old
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        mask_w = adj[c_old][w]
                        if mask_w & (1 << v):
                            k = (adj[c_old][u] & mask_w).bit_count()
                            delta -= (k - 1)
                        if mask_w & (1 << u):
                            k = (adj[c_old][v] & mask_w).bit_count()
                            delta -= (k - 1)

                    # Add to c_new
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        mask_w = adj[c_new][w]
                        if mask_w & (1 << v):
                            k = (adj[c_new][u] & mask_w).bit_count()
                            delta += k
                        if mask_w & (1 << u):
                            k = (adj[c_new][v] & mask_w).bit_count()
                            delta += k

                    if delta < best_delta:
                        best_delta = delta
                        best_ei = ei
                        best_u = u
                        best_v = v
                        best_c_old = c_old
                        best_c_new = c_new

                if best_ei < 0:
                    continue

                new_energy = energy + best_delta

                if best_delta <= 0 or random.random() < math.exp(-best_delta / temp):
                    energy = new_energy

                    adj[best_c_old][best_u] &= ~(1 << best_v)
                    adj[best_c_old][best_v] &= ~(1 << best_u)
                    adj[best_c_new][best_u] |= (1 << best_v)
                    adj[best_c_new][best_v] |= (1 << best_u)
                    colors[best_ei] = best_c_new

                    if energy < best_energy:
                        best_energy = energy
                        best_colors_local = colors[:]
                        stuck_count = 0
                        if energy == 0:
                            best_n = n
                            best_colors = colors[:]
                            found = True
                            break
                else:
                    stuck_count += 1

                temp *= cooling

            if found:
                break

    return {'n': best_n, 'colors': best_colors}


if __name__ == '__main__':
    result = construct()
    print(f"Found n = {result['n']}")
