import time
import random
import sys

def construct():
    deadline = time.time() + 145.0

    def verify(n, colors):
        m = n * (n - 1) // 2
        if len(colors) != m:
            return False
        masks = [[0] * 4 for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                if c < 0 or c > 3:
                    return False
                masks[i][c] |= (1 << j)
                masks[j][c] |= (1 << i)
                idx += 1
        for u in range(n):
            for c in range(4):
                mask_u = masks[u][c]
                if mask_u == 0:
                    continue
                temp = mask_u
                while temp:
                    lsb = temp & (-temp)
                    v = lsb.bit_length() - 1
                    if masks[v][c] & mask_u:
                        return False
                    temp ^= lsb
        return True

    def build_index(n):
        idx = [[0] * n for _ in range(n)]
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                idx[i][j] = k
                idx[j][i] = k
                k += 1
        return idx

    def solve_min_conflicts(n, time_budget, seed=None):
        if seed is not None:
            random.seed(seed)
        start_time = time.time()
        m = n * (n - 1) // 2
        idx = build_index(n)

        for restart in range(200):
            if time.time() > start_time + time_budget:
                break

            # Initialize random colors
            colors = [random.randint(0, 3) for _ in range(m)]

            # Build masks
            masks = [[0] * 4 for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx[i][j]]
                    masks[i][c] |= (1 << j)
                    masks[j][c] |= (1 << i)

            # Calculate initial triangle count
            tri_count = 0
            for u in range(n):
                for c in range(4):
                    mask_u = masks[u][c]
                    if mask_u == 0:
                        continue
                    temp = mask_u
                    while temp:
                        lsb = temp & (-temp)
                        v = lsb.bit_length() - 1
                        tri_count += (masks[v][c] & mask_u).bit_count()
                        temp ^= lsb
            tri_count //= 3

            if tri_count == 0:
                return colors

            # Min-Conflicts Loop
            steps = 0
            max_steps = 500000
            while tri_count > 0 and steps < max_steps and time.time() < start_time + time_budget - 2:
                steps += 1

                # Find a random triangle
                # We can pick a random vertex u, random color c, random neighbor v in mask[u][c]
                # Then check if there's a w in mask[v][c] & mask[u][c]
                found = False
                attempts = 0
                while not found and attempts < 200:
                    attempts += 1
                    u = random.randint(0, n - 1)
                    c = random.randint(0, 3)
                    mask_u = masks[u][c]
                    if mask_u == 0:
                        continue
                    # Pick random v in mask_u
                    v = random.randint(0, n - 1)
                    if u == v:
                        continue
                    if not (masks[u][c] >> v) & 1:
                        continue
                    # Check if triangle exists: common = masks[v][c] & masks[u][c]
                    common = masks[v][c] & mask_u
                    if common:
                        # Found triangle (u, v, w) where w is in common
                        # Pick random w from common
                        lsb = common & (-common)
                        w = lsb.bit_length() - 1
                        found = True

                if not found:
                    # Should not happen if tri_count > 0, but just in case
                    break

                # We have triangle (u, v, w) with color c.
                # Pick a random edge from (u,v), (u,w), (v,w)
                edge_choice = random.randint(0, 2)
                if edge_choice == 0:
                    eu, ev = u, v
                elif edge_choice == 1:
                    eu, ev = u, w
                else:
                    eu, ev = v, w

                if eu > ev:
                    eu, ev = ev, eu

                e_idx = idx[eu][ev]
                old_c = colors[e_idx]

                # Find best new color
                # We want to minimize the number of triangles created.
                # The cost of changing (eu, ev) to new_c is the number of common neighbors in color new_c.
                # Note: if new_c == old_c, we don't change.
                best_new_c = old_c
                min_cost = float('inf')
                for new_c in range(4):
                    if new_c == old_c:
                        continue
                    # Calculate potential triangles
                    # masks[eu][new_c] and masks[ev][new_c] do NOT include each other yet for new_c
                    # So this is the number of triangles (eu, ev, x) that would be created.
                    created = (masks[eu][new_c] & masks[ev][new_c]).bit_count()
                    if created < min_cost:
                        min_cost = created
                        best_new_c = new_c

                # Apply change
                if best_new_c != old_c:
                    # Update masks
                    masks[eu][old_c] &= ~(1 << ev)
                    masks[ev][old_c] &= ~(1 << eu)
                    masks[eu][best_new_c] |= (1 << ev)
                    masks[ev][best_new_c] |= (1 << eu)

                    # Update tri_count
                    # destroyed = number of triangles removed by removing edge from old_c
                    # created = number of triangles added by adding edge to new_c
                    destroyed = (masks[eu][old_c] & masks[ev][old_c]).bit_count()
                    created = min_cost
                    tri_count += (created - destroyed)

                    colors[e_idx] = best_new_c

                    if tri_count == 0:
                        return colors

                # If best_new_c == old_c, we couldn't improve by changing this edge.
                # To escape local minima, occasionally kick the search.
                if random.random() < 0.05:
                    # Random kick: change a random edge to a random color
                    eu = random.randint(0, n - 1)
                    ev = random.randint(0, n - 1)
                    if eu == ev:
                        continue
                    if eu > ev:
                        eu, ev = ev, eu
                    e_idx = idx[eu][ev]
                    old_c = colors[e_idx]
                    new_c = random.randint(0, 3)
                    if new_c == old_c:
                        continue

                    destroyed = (masks[eu][old_c] & masks[ev][old_c]).bit_count()
                    created = (masks[eu][new_c] & masks[ev][new_c]).bit_count()
                    tri_count += (created - destroyed)

                    masks[eu][old_c] &= ~(1 << ev)
                    masks[ev][old_c] &= ~(1 << eu)
                    masks[eu][new_c] |= (1 << ev)
                    masks[ev][new_c] |= (1 << eu)
                    colors[e_idx] = new_c

        return None

    # Fallback n=32
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    fallback_colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            fallback_colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)

    # Try n=50 first
    if time.time() < deadline - 100:
        result = solve_min_conflicts(50, 100)
        if result:
            if verify(50, result):
                return {'n': 50, 'colors': result}

    # Try n=49
    if time.time() < deadline - 50:
        result = solve_min_conflicts(49, 50)
        if result:
            if verify(49, result):
                return {'n': 49, 'colors': result}

    # Try n=48
    if time.time() < deadline - 20:
        result = solve_min_conflicts(48, 20)
        if result:
            if verify(48, result):
                return {'n': 48, 'colors': result}

    return {'n': 32, 'colors': fallback_colors}
