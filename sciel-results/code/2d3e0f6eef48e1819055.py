"""Structural change: direct simulated annealing on a flat edge-color array for target n,
using the known n=32 Greenwood-Gleason (Clebsch) coloring as a seed for larger lattices.
Energy = count of monochromatic triangles. Incremental energy update per edge flip.
Bitset-based neighbor tracking for fast delta computation."""
import random
import time

def construct():
    deadline = time.time() + 130.0  # safety margin under 150s

    # ---- Greenwood-Gleason / Clebsch n=32 coloring ----
    def mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r

    def make_seed_32():
        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
        n = 32
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if i // 16 == j // 16:
                    c = log[(i % 16) ^ (j % 16)] % 3
                else:
                    c = 3
                colors.append(c)
        return colors

    # ---- Edge index helper: for n vertices, edge (i,j) with i<j is at position ----
    def edge_idx(i, j, n):
        # Number of edges before row i: sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
        # Plus offset within row i: j - i - 1
        return i * (n - 1) - i * (i + 1) // 2 + j - i - 1

    # ---- Build bitsets from flat color array ----
    def build_bitsets(colors, n):
        # bitsets[c*n + i] has bit j set iff edge (i,j) has colour c
        bs = [0] * (4 * n)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                idx += 1
                bs[c * n + i] |= (1 << j)
                bs[c * n + j] |= (1 << i)
        return bs

    # ---- Count total monochromatic triangles ----
    def count_triangles(bs, n):
        total = 0
        for c in range(4):
            base = c * n
            for i in range(n):
                ni = bs[base + i]
                # Count pairs (j,k) in ni such that edge(j,k) is colour c
                # = (1/2) * sum_{j in ni} popcount(ni & bs[base+j])
                s = 0
                temp = ni
                while temp:
                    lsb = temp & (-temp)
                    j = lsb.bit_length() - 1
                    s += (ni & bs[base + j]).bit_count()
                    temp ^= lsb
                total += s // 2
        return total

    # ---- Incremental delta for flipping edge (i,j) from old_c to new_c ----
    def flip_delta(bs, n, i, j, old_c, new_c):
        if old_c == new_c:
            return 0
        # Triangles broken: vertices k where (i,k) and (j,k) are both old_c
        broken = (bs[old_c * n + i] & bs[old_c * n + j]).bit_count()
        # Triangles formed: vertices k where (i,k) and (j,k) are both new_c
        formed = (bs[new_c * n + i] & bs[new_c * n + j]).bit_count()
        return formed - broken

    # ---- Apply flip ----
    def apply_flip(bs, n, i, j, old_c, new_c):
        bs[old_c * n + i] &= ~(1 << j)
        bs[old_c * n + j] &= ~(1 << i)
        bs[new_c * n + i] |= (1 << j)
        bs[new_c * n + j] |= (1 << i)

    # ---- Simulated annealing on flat color array ----
    def sa_search(colors, n, time_limit):
        bs = build_bitsets(colors, n)
        energy = count_triangles(bs, n)
        if energy == 0:
            return colors, 0

        best_energy = energy
        best_colors = list(colors)
        best_bs = list(bs)

        T = 3.0
        T_min = 0.05
        cooling = 0.9997
        steps_since_best = 0
        no_improve_patience = 80000

        while time.time() < time_limit and energy > 0:
            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i

            old_c = colors[edge_idx(i, j, n)]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue

            delta = flip_delta(bs, n, i, j, old_c, new_c)

            if delta < 0 or (delta == 0 and random.random() < 0.3) or random.random() < (2.718281828 ** (-delta / T)):
                apply_flip(bs, n, i, j, old_c, new_c)
                colors[edge_idx(i, j, n)] = new_c
                energy += delta

                if energy < best_energy:
                    best_energy = energy
                    best_colors = list(colors)
                    best_bs = list(bs)
                    steps_since_best = 0
                else:
                    steps_since_best += 1

                # Reheat if stuck
                if steps_since_best > no_improve_patience:
                    T = max(T, 2.0)
                    steps_since_best = 0

            T *= cooling
            if T < T_min:
                T = T_min

        if best_energy < energy:
            return best_colors, best_energy
        return colors, energy

    # ---- Try to build a valid coloring for target n ----
    def try_build(target_n, time_budget):
        """Try to find a valid 4-coloring of K_target_n. Returns (colors, success)."""
        if target_n == 32:
            return make_seed_32(), True

        # Initialize: seed the first 32 vertices with Clebsch, rest random
        seed32 = make_seed_32()
        colors = list(seed32)
        # Add edges involving new vertices (32..target_n-1)
        for i in range(32, target_n):
            for j in range(i + 1, target_n):
                colors.append(random.randint(0, 3))
        # Edges between seed and new vertices
        # Need to insert them in correct lexicographic order
        # Actually, let's just build the full array properly
        colors = []
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if i < 32 and j < 32:
                    colors.append(seed32[edge_idx(i, j, 32)])
                else:
                    colors.append(random.randint(0, 3))

        # Run SA
        colors, energy = sa_search(colors, target_n, time.time() + time_budget)
        if energy == 0:
            return colors, True
        return colors, False

    # ---- Main search loop ----
    best_n = 32
    best_colors = make_seed_32()

    # Verify seed
    bs32 = build_bitsets(best_colors, 32)
    assert count_triangles(bs32, 32) == 0

    # Try increasing n from 33 upward
    for target_n in range(33, 63):
        if time.time() > deadline - 10:
            break

        # Allocate time budget: more time for harder targets
        remaining = deadline - time.time() - 5
        if remaining < 5:
            break
        # Give each target a fraction of remaining time
        time_budget = min(remaining * 0.3, 20.0)

        success = False
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            colors, ok = try_build(target_n, time_budget)
            if ok:
                best_n = target_n
                best_colors = colors
                success = True
                break

        if not success:
            # Try a couple more with slightly different init
            for attempt in range(2):
                if time.time() > deadline - 5:
                    break
                # Random restart for this n
                colors = [random.randint(0, 3) for _ in range(target_n * (target_n - 1) // 2)]
                colors, energy = sa_search(colors, target_n, min(remaining * 0.15, 10.0))
                if energy == 0:
                    best_n = target_n
                    best_colors = colors
                    success = True
                    break

    # Final: if we have a valid coloring, try to push one more vertex
    if time.time() < deadline - 8 and best_n < 62:
        next_n = best_n + 1
        colors = list(best_colors)
        # Add edges for new vertex
        for i in range(best_n):
            colors.append(random.randint(0, 3))
        # Wait, the flat array ordering is tricky when adding a vertex at the end.
        # Let's just do a full SA from a perturbed best.
        if len(colors) != next_n * (next_n - 1) // 2:
            # Rebuild properly
            colors = []
            for i in range(next_n):
                for j in range(i + 1, next_n):
                    if i < best_n and j < best_n:
                        colors.append(best_colors[edge_idx(i, j, best_n)])
                    else:
                        colors.append(random.randint(0, 3))
        colors, energy = sa_search(colors, next_n, deadline - time.time() - 3)
        if energy == 0:
            best_n = next_n
            best_colors = colors

    return {'n': best_n, 'colors': best_colors}
