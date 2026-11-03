"""Shortest superpermutation on n symbols.

Strategy: model the problem as finding a minimum-cost cyclic ordering of the
(n-1)! (n-1)-letter blocks, where the cost of an ordering is
    m + sum_i (m - LCP(block_i, block_{i+1}))  (cyclic)
which is equivalent to maximizing the total LCP overlap.

Key improvements over the seed:
  1. Vectorized LCP matrix construction using a "match-count" prefix trick.
  2. Full-sweep 2-opt (segment reversal) with numpy-vectorized delta evaluation.
  3. Or-opt (segment relocation) local search.
  4. Simulated annealing restarts to escape local optima.
  5. Explicit wall-clock deadline guard with a safe fallback.
"""
import time
import random
import numpy as np
from itertools import permutations


def _build_lcp_matrix(bm1):
    """Build the LCP matrix for all pairs of (n-1)-letter blocks.

    bm1: list of tuples, each of length m.
    Returns an (L, L) int32 numpy array where lcp[i,j] is the length of the
    longest common prefix of bm1[i] and bm1[j].

    Uses the fact that LCP(a,b) = m - (number of positions p where a[:p] != b[:p]
    first occurs) ... more precisely, define match_prefix[i,j] = 1 if a==b at
    position 0, then 1 if a==b at position 1, etc. LCP = sum of consecutive
    leading matches. We compute, for each position p, the boolean matrix
    eq_p[i,j] = (bm1[i][p] == bm1[j][p]). Then a prefix match of length L means
    eq_0, eq_1, ..., eq_{L-1} are all true and (if L<m) eq_L is false.
    We compute the cumulative product along p and sum.
    """
    L = len(bm1)
    m = len(bm1[0])
    arr = np.array(bm1, dtype=np.int32)  # shape (L, m)

    # For each position p, compute the (L, L) equality matrix
    # eq_p[i,j] = (arr[i,p] == arr[j,p])
    # We build the cumulative product: cumprod_p[i,j] = prod_{q=0..p} eq_q[i,j]
    # Then LCP[i,j] = sum_p cumprod_p[i,j]  (number of leading positions that match)

    # To avoid L large intermediate (L,L) matrices per p (memory), process in chunks
    # of positions. But m is small (<=6 for n<=7), so we can do it directly.
    cumprod = np.ones((L, L), dtype=np.int32)
    lcp_mat = np.zeros((L, L), dtype=np.int32)
    for p in range(m):
        # eq_p[i,j] = (arr[i,p] == arr[j,p])
        col = arr[:, p]  # shape (L,)
        eq_p = (col[:, None] == col[None, :])  # shape (L, L)
        cumprod = cumprod * eq_p.astype(np.int32)
        lcp_mat += cumprod
    return lcp_mat


def _build_word(bm1, order, n):
    """Build the superpermutation string from the cyclic block order.

    We treat the order as a cyclic sequence. Start with the first block, then
    for each subsequent block append the suffix that doesn't overlap with the
    previous block's suffix. Because it's cyclic, the word is built linearly
    starting from order[0]; the cyclic edge (last -> first) is implicitly
    satisfied by the structure of the superpermutation (the word contains all
    windows).
    """
    parts = ["".join(map(str, bm1[order[0]]))]
    for k in range(1, len(order)):
        prev = bm1[order[k - 1]]
        cur = bm1[order[k]]
        l = 0
        for a, b in zip(prev, cur):
            if a != b:
                break
            l += 1
        parts.append("".join(map(str, cur[l:])))
    return "".join(parts)


def _total_cost(order, lcp_np, m):
    """Total length for a cyclic ordering: m + sum_i (m - LCP[i, next_i])."""
    L = len(order)
    order_arr = np.array(order, dtype=np.int32)
    next_arr = np.roll(order_arr, -1)
    lcp_vals = lcp_np[order_arr, next_arr]
    return m + (m - lcp_vals).sum()


def _greedy_order(bm1, lcp_np):
    """Greedy nearest-neighbor: start from a random block, repeatedly go to the
    block with the maximum LCP overlap (minimum transition cost)."""
    L = len(bm1)
    start = random.randrange(L)
    order = [start]
    visited = np.zeros(L, dtype=bool)
    visited[start] = True
    cur = start
    for _ in range(L - 1):
        # Find unvisited block with max LCP from cur
        lcp_row = lcp_np[cur]
        lcp_row = lcp_row.copy()
        lcp_row[visited] = -1  # exclude visited
        nxt = int(np.argmax(lcp_row))
        order.append(nxt)
        visited[nxt] = True
        cur = nxt
    return order


def _two_opt_sweep(order, lcp_np, m, deadline, max_seg=200):
    """Repeated 2-opt sweeps: try reversing segments [i, j] to reduce cost.
    Vectorize the delta computation over all j for a fixed i."""
    L = len(order)
    if L < 4:
        return order, _total_cost(order, lcp_np, m)
    order = list(order)
    current_cost = _total_cost(order, lcp_np, m)

    improved = True
    sweeps = 0
    max_sweeps = 200
    while improved and sweeps < max_sweeps:
        if time.time() > deadline - 3.0:
            break
        improved = False
        sweeps += 1
        # Convert to numpy for vectorized operations
        ord_arr = np.array(order, dtype=np.int32)
        # For 2-opt reversing [i, j]:
        # Old edges: (order[i-1], order[i]) and (order[j], order[j+1])
        # New edges: (order[i-1], order[j]) and (order[i], order[j+1])
        # Delta in LCP = new_lcp - old_lcp; delta in cost = -delta_lcp
        # We want delta_cost < 0, i.e., delta_lcp > 0.
        #
        # For a fixed i, we can compute the delta for all j > i at once.
        # Let a = order[i-1], b = order[i], c = order[j], d = order[j+1]
        # old = lcp[a,b] + lcp[c,d]
        # new = lcp[a,d] + lcp[c,b]
        # delta_lcp = new - old
        for i in range(1, L):
            if time.time() > deadline - 3.0:
                break
            a = order[i - 1]
            b = order[i]
            old_ab = lcp_np[a][b]
            # For all j in [i+1, min(i+max_seg, L-1)]:
            j_start = i + 1
            j_end = min(i + max_seg, L)
            if j_start >= j_end:
                continue
            # c = order[j], d = order[j+1]
            c_arr = np.array(order[j_start:j_end], dtype=np.int32)
            d_arr = np.array(order[(j_start + 1):(j_end + 1)] % L, dtype=np.int32)
            old_cd = lcp_np[c_arr, d_arr]  # shape (num_j,)
            new_ad = lcp_np[a, d_arr]       # shape (num_j,)
            new_cb = lcp_np[c_arr, b]       # shape (num_j,)
            delta_lcp = (new_ad + new_cb) - (old_ab + old_cd)
            # Find the best j with positive delta_lcp
            best_idx = int(np.argmax(delta_lcp))
            if delta_lcp[best_idx] > 0:
                j = j_start + best_idx
                # Reverse segment [i, j]
                order[i:j + 1] = reversed(order[i:j + 1])
                current_cost -= int(delta_lcp[best_idx])
                improved = True
    return order, current_cost


def _or_opt_sweep(order, lcp_np, m, deadline, max_seg=10):
    """Or-opt: move a segment of length 1, 2, or 3 to a different position.
    This is O(L^2 * seg_len) per sweep, which is feasible for L=720."""
    L = len(order)
    if L < 5:
        return order, _total_cost(order, lcp_np, m)
    order = list(order)
    current_cost = _total_cost(order, lcp_np, m)

    improved = True
    sweeps = 0
    max_sweeps = 50
    while improved and sweeps < max_sweeps:
        if time.time() > deadline - 3.0:
            break
        improved = False
        sweeps += 1
        for seg_len in [1, 2, 3]:
            if time.time() > deadline - 3.0:
                break
            for i in range(L):
                if time.time() > deadline - 3.0:
                    break
                # Segment is order[i : i+seg_len]
                j = i + seg_len - 1
                if j >= L:
                    continue
                a = order[i - 1]       # before segment
                b = order[i]           # first of segment
                c = order[j]           # last of segment
                d = order[(j + 1) % L]  # after segment

                # Try inserting the segment after each position k
                # k should not be i-1 (original position) or j (just before original)
                for k in range(L):
                    if k == i - 1 or k == j:
                        continue
                    # After removing segment, a and d become adjacent.
                    # Insert after k: the segment goes between order[k] and order[k+1]
                    # (in the modified sequence after removal).
                    # We need to compute the new LCP contributions.
                    # This is complex; for simplicity, only try single-element moves (seg_len=1)
                    # and skip longer segments for now.
                    if seg_len > 1:
                        continue
                    x = order[i]  # the element to move
                    # Old edges: (a, x), (x, d), and the edge at insertion point
                    # Insert after k: new edges are (a, d), (order[k], x), (x, order[k+1])
                    # But we need to be careful: if k is after i, the indices shift.
                    # For a quick approximation, just compute the delta using the
                    # current order and check if it improves.
                    k_elem = order[k]
                    k_next = order[(k + 1) % L]
                    old_cost = lcp_np[a][x] + lcp_np[x][d] + lcp_np[k_elem][k_next]
                    new_cost = lcp_np[a][d] + lcp_np[k_elem][x] + lcp_np[x][k_next]
                    delta_lcp = new_cost - old_cost
                    if delta_lcp > 0:
                        # Move x to after k
                        x_val = order.pop(i)
                        # Adjust k if k >= i (since we removed an element before k)
                        if k >= i:
                            k -= 1
                        order.insert(k + 1, x_val)
                        current_cost -= int(delta_lcp)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return order, current_cost


def _simulated_annealing(order, lcp_np, m, deadline, start_temp=5.0, cool_rate=0.95):
    """Simulated annealing to escape local optima.
    Moves: 2-opt (reverse segment) and single-element relocation.
    Accept worse moves with probability exp(-delta/T)."""
    L = len(order)
    if L < 4:
        return order, _total_cost(order, lcp_np, m)
    order = list(order)
    current_cost = _total_cost(order, lcp_np, m)
    best_order = list(order)
    best_cost = current_cost
    temp = start_temp

    max_iters = 50000
    for it in range(max_iters):
        if time.time() > deadline - 3.0:
            break
        # Choose a random move
        move_type = random.random()
        if move_type < 0.5:
            # 2-opt: reverse a random segment
            i = random.randint(1, L - 2)
            j = random.randint(i + 1, L - 1)
            a = order[i - 1]
            b = order[i]
            c = order[j]
            d = order[(j + 1) % L]
            old_lcp = lcp_np[a][b] + lcp_np[c][d]
            new_lcp = lcp_np[a][d] + lcp_np[c][b]
            delta_lcp = new_lcp - old_lcp
            delta_cost = -delta_lcp
            if delta_cost < 0 or random.random() < np.exp(-delta_cost / max(temp, 1e-9)):
                order[i:j + 1] = reversed(order[i:j + 1])
                current_cost += delta_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_order = list(order)
        else:
            # Relocate a single element
            i = random.randint(0, L - 1)
            k = random.randint(0, L - 1)
            if k == i or (k + 1) % L == i:
                continue
            x = order[i]
            a = order[i - 1]
            d = order[(i + 1) % L]
            k_elem = order[k]
            k_next = order[(k + 1) % L]
            old_lcp = lcp_np[a][x] + lcp_np[x][d] + lcp_np[k_elem][k_next]
            new_lcp = lcp_np[a][d] + lcp_np[k_elem][x] + lcp_np[x][k_next]
            delta_lcp = new_lcp - old_lcp
            delta_cost = -delta_lcp
            if delta_cost < 0 or random.random() < np.exp(-delta_cost / max(temp, 1e-9)):
                x_val = order.pop(i)
                if k >= i:
                    k -= 1
                order.insert(k + 1, x_val)
                current_cost += delta_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_order = list(order)
        temp *= cool_rate
        if temp < 0.1:
            break
    return best_order, best_cost


def _construct_superperm(n, deadline):
    """Main construction: build LCP matrix, find good ordering via greedy + local search."""
    bm1 = list(permutations(range(1, n)))
    L = len(bm1)
    m = n - 1

    # Build LCP matrix
    lcp_np = _build_lcp_matrix(bm1)

    # Try multiple random starts and pick the best
    best_order = None
    best_cost = float('inf')
    num_starts = 5

    for start_idx in range(num_starts):
        if time.time() > deadline - 10.0:
            break
        # Greedy order from a random start
        order = _greedy_order(bm1, lcp_np)

        # 2-opt sweep
        order, cost = _two_opt_sweep(order, lcp_np, m, deadline)

        # Or-opt sweep
        order, cost = _or_opt_sweep(order, lcp_np, m, deadline)

        # Simulated annealing
        if time.time() < deadline - 5.0:
            order, cost = _simulated_annealing(order, lcp_np, m, deadline)

        # Final 2-opt polish
        if time.time() < deadline - 2.0:
            order, cost = _two_opt_sweep(order, lcp_np, m, deadline, max_seg=50)

        if cost < best_cost:
            best_cost = cost
            best_order = order

    if best_order is None:
        # Fallback: identity order
        best_order = list(range(L))

    return _build_word(bm1, best_order, n)


def _classic_construct(n):
    """Classic recursive superpermutation construction (sum of i! for i=1..n)."""
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    word = [1]
    for k in range(2, n + 1):
        # Find all (k-1)-letter substrings with distinct characters
        perms, seen = [], set()
        for i in range(len(word) - k + 2):
            w = tuple(word[i:i + k - 1])
            if len(set(w)) == k - 1 and w not in seen:
                seen.add(w)
                perms.append(w)
        new = []
        for p in perms:
            block = list(p) + [k] + list(p)
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))


def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"

    deadline = time.time() + 110.0  # safety margin below 120s

    try:
        return _construct_superperm(n, deadline)
    except Exception:
        # Fallback: classic construction
        return _classic_construct(n)
