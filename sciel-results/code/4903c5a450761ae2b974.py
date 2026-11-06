import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def get_idx(i, j, n):
        return i * (2 * n - i - 1) // 2 + (j - i - 1)

    def idx_to_ij(idx, n):
        # Binary search for i such that base(i) <= idx < base(i) + (n-i-1)
        lo, hi = 0, n - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            base = mid * (2 * n - mid - 1) // 2
            if base <= idx:
                lo = mid
            else:
                hi = mid - 1
        i = lo
        base = i * (2 * n - i - 1) // 2
        j = i + 1 + (idx - base)
        return i, j

    def build_common_counts(colors, n):
        """Build common[c][u][v] = number of common neighbors of u,v in color c graph."""
        adj = [[0] * n for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                idx = get_idx(i, j, n)
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)

        common = [[[0] * n for _ in range(n)] for _ in range(5)]
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common[c][u][v] = common[c][v][u] = (adj[c][u] & adj[c][v]).bit_count()
        return adj, common

    def count_violations(common, n):
        v = 0
        for c in range(5):
            for u in range(n):
                for w in range(u + 1, n):
                    cnt = common[c][u][w]
                    if cnt > 1:
                        v += cnt - 1
        return v

    def greedy_construct(n):
        """Greedy: assign each edge a color that minimizes new violations."""
        num_edges = n * (n - 1) // 2
        colors = [-1] * num_edges
        adj = [[0] * n for _ in range(5)]
        common = [[[0] * n for _ in range(n)] for _ in range(5)]

        for i in range(n):
            for j in range(i + 1, n):
                idx = get_idx(i, j, n)
                best_c = 0
                best_delta = float('inf')
                for c in range(5):
                    # If we add (i,j) to color c, what's the new common count for affected pairs?
                    # Pairs affected: (i, w) for w in N_c(j) and (j, w) for w in N_c(i)
                    # Also pair (i,j) itself: common count becomes |N_c(i) ∩ N_c(j)| (excluding i,j which aren't self-adj)
                    # Actually adding (i,j) to c:
                    #   For pair (i,j): new common = |N_c(i) ∩ N_c(j)| (these are vertices k with (i,k) and (j,k) in c)
                    #   For pair (i,w) where w in N_c(j): j is now in N_c(i), so if j was already in N_c(w), common increases
                    #     Wait: j is in N_c(w) iff (w,j) in c. Since w in N_c(j) means (j,w) in c, yes.
                    #     So common[i][w] increases by 1 for each w in N_c(j) such that w is also in N_c(w)... no.
                    #     Let me think more carefully.
                    #
                    # Adding edge (i,j) to color c:
                    #   N_c(i) gains j, N_c(j) gains i.
                    #   For pair (u,v) where u,v != i,j:
                    #     common[u][v] changes only if i or j is a common neighbor of u,v.
                    #     i is common neighbor of u,v iff (u,i) in c AND (v,i) in c.
                    #     j is common neighbor of u,v iff (u,j) in c AND (v,j) in c.
                    #     So common[u][v] increases by [i is common] + [j is common].
                    #   For pair (i,v), v != j:
                    #     N_c(i) gains j. common[i][v] = |N_c(i) ∩ N_c(v)|.
                    #     j is in N_c(v) iff (j,v) in c. So common[i][v] increases by 1 iff (j,v) in c.
                    #   For pair (j,v), v != i:
                    #     Similarly, common[j][v] increases by 1 iff (i,v) in c.
                    #   For pair (i,j):
                    #     common[i][j] = |N_c(i) ∩ N_c(j)|. This is the count of vertices k with (i,k) and (j,k) in c.
                    #     This doesn't change when we add (i,j) since i∉N_c(i) and j∉N_c(j).
                    #
                    # So delta for adding (i,j) to color c:
                    #   delta = max(0, common[i][j] - 1)  # pair (i,j) contribution (was 0 before since not an edge... wait)
                    #   Actually before adding, (i,j) is not in color c, so pair (i,j) still has common count
                    #   |N_c(i) ∩ N_c(j)|. The violation is max(0, common-1) regardless of whether (i,j) is an edge.
                    #   So pair (i,j) contribution doesn't change!
                    #
                    #   For v != i, j:
                    #     pair (i,v): delta += [1 if (j,v) in c else 0] contributes to common[i][v]
                    #       new_common = common[i][v] + (1 if (j,v) in c else 0)
                    #       delta += max(0, new_common-1) - max(0, common[i][v]-1)
                    #     pair (j,v): similarly
                    #   For u,v != i,j:
                    #     pair (u,v): delta += [1 if (u,i) in c and (v,i) in c else 0] + [1 if (u,j) in c and (v,j) in c else 0]
                    #     This is expensive. Let's skip for now and use a simpler heuristic.

                    # Simplified: just check the immediate effect on pairs involving i and j
                    delta = 0
                    # Pair (i,j): common count unchanged, but let's compute it
                    # Actually the violation for pair (i,j) is already counted in common[i][j].
                    # Adding (i,j) doesn't change common[i][j]. So no delta from pair (i,j).

                    # For v != i, j:
                    for v in range(n):
                        if v == i or v == j:
                            continue
                        # Pair (i, v)
                        inc_i = 1 if (adj[c][v] >> j) & 1 else 0
                        old_cv = common[c][i][v]
                        new_cv = old_cv + inc_i
                        delta += max(0, new_cv - 1) - max(0, old_cv - 1)

                        # Pair (j, v)
                        inc_j = 1 if (adj[c][v] >> i) & 1 else 0
                        old_cv2 = common[c][j][v]
                        new_cv2 = old_cv2 + inc_j
                        delta += max(0, new_cv2 - 1) - max(0, old_cv2 - 1)

                    # For pairs (u,v) not involving i or j:
                    # u,v both != i,j. common[u][v] increases by:
                    #   1 if (u,i) in c and (v,i) in c  [i becomes a common neighbor]
                    #   1 if (u,j) in c and (v,j) in c  [j becomes a common neighbor]
                    # This is O(n^2) which is too slow for greedy. Skip for now.
                    # The greedy will be approximate.

                    if delta < best_delta:
                        best_delta = delta
                        best_c = c

                colors[idx] = best_c
                adj[best_c][i] |= (1 << j)
                adj[best_c][j] |= (1 << i)
                # Update common counts
                for v in range(n):
                    if v == i or v == j:
                        continue
                    inc_i = 1 if (adj[best_c][v] >> j) & 1 else 0
                    common[best_c][i][v] += inc_i
                    common[best_c][v][i] += inc_i
                    inc_j = 1 if (adj[best_c][v] >> i) & 1 else 0
                    common[best_c][j][v] += inc_j
                    common[best_c][v][j] += inc_j
                # For pairs (u,v) not involving i,j:
                for u in range(n):
                    if u == i or u == j:
                        continue
                    for v in range(u + 1, n):
                        if v == i or v == j:
                            continue
                        inc = 0
                        if (adj[best_c][u] >> i) & 1 and (adj[best_c][v] >> i) & 1:
                            inc += 1
                        if (adj[best_c][u] >> j) & 1 and (adj[best_c][v] >> j) & 1:
                            inc += 1
                        if inc:
                            common[best_c][u][v] += inc
                            common[best_c][v][u] += inc

        return colors

    def local_search(colors, n, max_iters, start_temp, cooling, deadline):
        num_edges = n * (n - 1) // 2
        adj, common = build_common_counts(colors, n)
        current_violations = count_violations(common, n)

        if current_violations == 0:
            return True, colors

        temp = start_temp
        best_violations = current_violations
        best_colors = colors[:]

        for it in range(max_iters):
            if time.time() > deadline - 3.0:
                break
            if current_violations == 0:
                return True, colors

            idx = random.randint(0, num_edges - 1)
            i, j = idx_to_ij(idx, n)
            old_color = colors[idx]
            new_color = random.randint(0, 4)
            if new_color == old_color:
                continue

            delta = 0

            # Remove from old_color
            c = old_color
            # Pair (i,j): common unchanged
            for v in range(n):
                if v == i or v == j:
                    continue
                # Pair (i, v): removing j from N_c(i)
                dec_i = 1 if (adj[c][v] >> j) & 1 else 0
                old_cv = common[c][i][v]
                new_cv = old_cv - dec_i
                delta += max(0, new_cv - 1) - max(0, old_cv - 1)

                # Pair (j, v): removing i from N_c(j)
                dec_j = 1 if (adj[c][v] >> i) & 1 else 0
                old_cv2 = common[c][j][v]
                new_cv2 = old_cv2 - dec_j
                delta += max(0, new_cv2 - 1) - max(0, old_cv2 - 1)

            # Pairs (u,v) not involving i,j: i and j removed as potential common neighbors
            for u in range(n):
                if u == i or u == j:
                    continue
                for v in range(u + 1, n):
                    if v == i or v == j:
                        continue
                    dec = 0
                    if (adj[c][u] >> i) & 1 and (adj[c][v] >> i) & 1:
                        dec += 1
                    if (adj[c][u] >> j) & 1 and (adj[c][v] >> j) & 1:
                        dec += 1
                    if dec:
                        old_cv3 = common[c][u][v]
                        new_cv3 = old_cv3 - dec
                        delta += max(0, new_cv3 - 1) - max(0, old_cv3 - 1)

            # Apply removal
            adj[c][i] &= ~(1 << j)
            adj[c][j] &= ~(1 << i)
            for v in range(n):
                if v == i or v == j:
                    continue
                dec_i = 1 if (adj[c][v] >> j) & 1 else 0
                common[c][i][v] -= dec_i
                common[c][v][i] -= dec_i
                dec_j = 1 if (adj[c][v] >> i) & 1 else 0
                common[c][j][v] -= dec_j
                common[c][v][j] -= dec_j
            for u in range(n):
                if u == i or u == j:
                    continue
                for v in range(u + 1, n):
                    if v == i or v == j:
                        continue
                    dec = 0
                    if (adj[c][u] >> i) & 1 and (adj[c][v] >> i) & 1:
                        dec += 1
                    if (adj[c][u] >> j) & 1 and (adj[c][v] >> j) & 1:
                        dec += 1
                    if dec:
                        common[c][u][v] -= dec
                        common[c][v][u] -= dec

            # Add to new_color
            c = new_color
            for v in range(n):
                if v == i or v == j:
                    continue
                inc_i = 1 if (adj[c][v] >> j) & 1 else 0
                old_cv = common[c][i][v]
                new_cv = old_cv + inc_i
                delta += max(0, new_cv - 1) - max(0, old_cv - 1)

                inc_j = 1 if (adj[c][v] >> i) & 1 else 0
                old_cv2 = common[c][j][v]
                new_cv2 = old_cv2 + inc_j
                delta += max(0, new_cv2 - 1) - max(0, old_cv2 - 1)

            for u in range(n):
                if u == i or u == j:
                    continue
                for v in range(u + 1, n):
                    if v == i or v == j:
                        continue
                    inc = 0
                    if (adj[c][u] >> i) & 1 and (adj[c][v] >> i) & 1:
                        inc += 1
                    if (adj[c][u] >> j) & 1 and (adj[c][v] >> j) & 1:
                        inc += 1
                    if inc:
                        old_cv3 = common[c][u][v]
                        new_cv3 = old_cv3 + inc
                        delta += max(0, new_cv3 - 1) - max(0, old_cv3 - 1)

            # Apply addition
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            for v in range(n):
                if v == i or v == j:
                    continue
                inc_i = 1 if (adj[c][v] >> j) & 1 else 0
                common[c][i][v] += inc_i
                common[c][v][i] += inc_i
                inc_j = 1 if (adj[c][v] >> i) & 1 else 0
                common[c][j][v] += inc_j
                common[c][v][j] += inc_j
            for u in range(n):
                if u == i or u == j:
                    continue
                for v in range(u + 1, n):
                    if v == i or v == j:
                        continue
                    inc = 0
                    if (adj[c][u] >> i) & 1 and (adj[c][v] >> i) & 1:
                        inc += 1
                    if (adj[c][u] >> j) & 1 and (adj[c][v] >> j) & 1:
                        inc += 1
                    if inc:
                        common[c][u][v] += inc
                        common[c][v][u] += inc

            new_violations = current_violations + delta

            if delta < 0 or (delta > 0 and random.random() < math.exp(-delta / temp)):
                current_violations = new_violations
                colors[idx] = new_color
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
            else:
                # Revert: undo addition to new_color, redo removal from old_color
                c = new_color
                adj[c][i] &= ~(1 << j)
                adj[c][j] &= ~(1 << i)
                for v in range(n):
                    if v == i or v == j:
                        continue
                    dec_i = 1 if (adj[c][v] >> j) & 1 else 0
                    common[c][i][v] -= dec_i
                    common[c][v][i] -= dec_i
                    dec_j = 1 if (adj[c][v] >> i) & 1 else 0
                    common[c][j][v] -= dec_j
                    common[c][v][j] -= dec_j
                for u in range(n):
                    if u == i or u == j:
                        continue
                    for v in range(u + 1, n):
                        if v == i or v == j:
                            continue
                        dec = 0
                        if (adj[c][u] >> i) & 1 and (adj[c][v] >> i) & 1:
                            dec += 1
                        if (adj[c][u] >> j) & 1 and (adj[c][v] >> j) & 1:
                            dec += 1
                        if dec:
                            common[c][u][v] -= dec
                            common[c][v][u] -= dec

                c = old_color
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                for v in range(n):
                    if v == i or v == j:
                        continue
                    inc_i = 1 if (adj[c][v] >> j) & 1 else 0
                    common[c][i][v] += inc_i
                    common[c][v][i] += inc_i
                    inc_j = 1 if (adj[c][v] >> i) & 1 else 0
                    common[c][j][v] += inc_j
                    common[c][v][j] += inc_j
                for u in range(n):
                    if u == i or u == j:
                        continue
                    for v in range(u + 1, n):
                        if v == i or v == j:
                            continue
                        inc = 0
                        if (adj[c][u] >> i) & 1 and (adj[c][v] >> i) & 1:
                            inc += 1
                        if (adj[c][u] >> j) & 1 and (adj[c][v] >> j) & 1:
                            inc += 1
                        if inc:
                            common[c][u][v] += inc
                            common[c][v][u] += inc

            temp *= cooling

        if current_violations == 0:
            return True, colors
        if best_violations < 20:
            return False, best_colors
        return False, None

    # Strategy: greedy construct + local search, multi-start
    best_n = 1
    best_colors = [0]

    # Try increasing n
    for target_n in [16, 18, 20, 22, 24, 25, 26]:
        if time.time() > deadline - 15.0:
            break

        found = False
        num_restarts = 3 if target_n <= 20 else 2

        for restart in range(num_restarts):
            if time.time() > deadline - 5.0:
                break

            # Greedy initial construction
            colors = greedy_construct(target_n)

            # Local search
            max_iters = 15000 if target_n <= 20 else 10000
            start_temp = 2.0
            cooling = 0.9997

            success, result_colors = local_search(colors, target_n, max_iters, start_temp, cooling, deadline)

            if success:
                best_n = target_n
                best_colors = result_colors
                found = True
                break

            # If we got a good partial solution, try another restart with it as seed
            if result_colors is not None and not found:
                # Keep trying
                pass

        if not found and target_n <= 18:
            # If we can't even do 16 or 18, break
            if target_n <= 16:
                break

    # Fallback: ensure we have a valid small coloring
    if best_n < 11:
        # Simple valid coloring for small n
        n = 11
        num_edges = n * (n - 1) // 2
        colors = []
        adj = [[0] * n for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                c = (i + j) % 5
                colors.append(c)
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        # Verify
        common = [[[0] * n for _ in range(n)] for _ in range(5)]
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common[c][u][v] = common[c][v][u] = (adj[c][u] & adj[c][v]).bit_count()
        v = 0
        for c in range(5):
            for u in range(n):
                for w in range(u + 1, n):
                    if common[c][u][w] > 1:
                        v += common[c][u][w] - 1
        if v == 0:
            best_n = n
            best_colors = colors
        else:
            # Random search for small n
            for attempt in range(100):
                colors = [random.randint(0, 4) for _ in range(num_edges)]
                adj = [[0] * n for _ in range(5)]
                for i in range(n):
                    for j in range(i + 1, n):
                        idx = get_idx(i, j, n)
                        c = colors[idx]
                        adj[c][i] |= (1 << j)
                        adj[c][j] |= (1 << i)
                v = 0
                for c in range(5):
                    for u in range(n):
                        for w in range(u + 1, n):
                            cnt = (adj[c][u] & adj[c][w]).bit_count()
                            if cnt > 1:
                                v += cnt - 1
                if v == 0:
                    best_n = n
                    best_colors = colors
                    break
            else:
                # Last resort: n=1
                best_n = 1
                best_colors = [0]

    return {'n': best_n, 'colors': best_colors}
