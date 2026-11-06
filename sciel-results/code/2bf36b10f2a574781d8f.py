import time
import random
import sys

def construct():
    n = 40
    start_time = time.time()
    deadline = 115.0  # 5s buffer before 120s limit

    def time_left():
        return deadline - (time.time() - start_time)

    # Bitset adjacency
    def can_add(u, v, adj_state):
        """
        Check if adding edge (u,v) creates a C6.
        C6: u-v-x-y-z-w-u where all 6 vertices distinct.
        x in N(v)\{u}, w in N(u)\{v}, y in N(x)\{u,v,w}, z in N(w)\{u,v,x}, y~z, y!=z.
        """
        Nv = adj_state[v] & ~(1 << u)
        Nu = adj_state[u] & ~(1 << v)

        # Iterate over the smaller neighbor set for the outer loop
        if Nv.bit_count() <= Nu.bit_count():
            x_mask = Nv
            while x_mask:
                x = (x_mask & -x_mask).bit_length() - 1
                x_mask &= x_mask - 1
                w_mask = Nu & ~(1 << x)
                while w_mask:
                    w = (w_mask & -w_mask).bit_length() - 1
                    w_mask &= w_mask - 1
                    # Need y in N(x)\{u,v,w}, z in N(w)\{u,v,x}, y~z, y!=z
                    S_x = adj_state[x] & ~(1 << u) & ~(1 << v) & ~(1 << w)
                    S_w = adj_state[w] & ~(1 << u) & ~(1 << v) & ~(1 << x)
                    # Check if there's an edge between S_x and S_w with y != z
                    y_mask = S_x
                    while y_mask:
                        y = (y_mask & -y_mask).bit_length() - 1
                        y_mask &= y_mask - 1
                        if adj_state[y] & (S_w & ~(1 << y)):
                            return False
        else:
            x_mask = Nu
            while x_mask:
                x = (x_mask & -x_mask).bit_length() - 1
                x_mask &= x_mask - 1
                w_mask = Nv & ~(1 << x)
                while w_mask:
                    w = (w_mask & -w_mask).bit_length() - 1
                    w_mask &= w_mask - 1
                    S_x = adj_state[x] & ~(1 << u) & ~(1 << v) & ~(1 << w)
                    S_w = adj_state[w] & ~(1 << u) & ~(1 << v) & ~(1 << x)
                    y_mask = S_x
                    while y_mask:
                        y = (y_mask & -y_mask).bit_length() - 1
                        y_mask &= y_mask - 1
                        if adj_state[y] & (S_w & ~(1 << y)):
                            return False
        return True

    def get_edges(adj_state):
        edges = []
        for i in range(n):
            mask = adj_state[i]
            while mask:
                j = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if i < j:
                    edges.append([i, j])
        return edges

    def edge_count(adj_state):
        return sum(a.bit_count() for a in adj_state) // 2

    random.seed(42)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    best_edges = 0
    best_adj = [0] * n

    # Phase 1: Multiple Random Greedy Trials
    for trial in range(120):
        if time_left() < 15:
            break

        adj_try = [0] * n
        ec = 0

        shuffled = all_pairs[:]
        random.shuffle(shuffled)

        for u, v in shuffled:
            if time_left() < 5:
                break
            if can_add(u, v, adj_try):
                adj_try[u] |= (1 << v)
                adj_try[v] |= (1 << u)
                ec += 1

        if ec > best_edges:
            best_edges = ec
            best_adj = adj_try[:]

    # Fallback
    if best_edges == 0:
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u = 5 * k + a
                    v = 5 * k + b
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)
        best_edges = 80

    # Phase 2: Local Search with 1-for-2 moves and Tabu
    adj = best_adj[:]
    current_edges = best_edges
    best_local_edges = best_edges
    best_local_adj = best_adj[:]

    tabu = {}
    tabu_iteration = 0
    no_improve = 0

    while time_left() > 8:
        tabu_iteration += 1
        improved = False

        # 1. Try to add a random non-edge (0-for-1 move)
        for _ in range(30):
            if time_left() < 8:
                break
            u, v = random.choice(all_pairs)
            if adj[u] & (1 << v):
                continue
            if can_add(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges += 1
                improved = True
                no_improve = 0
                if current_edges > best_local_edges:
                    best_local_edges = current_edges
                    best_local_adj = adj[:]
                break

        if improved:
            continue

        no_improve += 1

        # 2. Try 1-for-2 moves: Remove one edge, add two
        # Build edge list
        edge_list = []
        for i in range(n):
            mask = adj[i]
            while mask:
                j = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if i < j:
                    edge_list.append((i, j))

        if not edge_list:
            break

        # Try several candidate removals
        for _ in range(8):
            if time_left() < 8:
                break

            # Prefer removing edges incident to high-degree vertices (they block more additions)
            # Weighted random selection
            if random.random() < 0.6 and edge_list:
                # Pick a random edge but bias toward high-degree endpoints
                candidates = []
                for e in edge_list:
                    u_e, v_e = e
                    deg = (adj[u_e].bit_count() + adj[v_e].bit_count())
                    candidates.append((deg, e))
                candidates.sort(key=lambda x: x[0], reverse=True)
                # Pick from top candidates with some randomness
                top_k = min(10, len(candidates))
                u_rem, v_rem = random.choice(candidates[:top_k])[1]
            else:
                u_rem, v_rem = random.choice(edge_list)

            # Check tabu
            key = (min(u_rem, v_rem), max(u_rem, v_rem))
            if tabu.get(key, 0) > tabu_iteration - 60:
                continue

            # Temporarily remove edge
            adj[u_rem] &= ~(1 << v_rem)
            adj[v_rem] &= ~(1 << u_rem)

            added_count = 0
            added_edges = []

            # Try to add two edges
            for _ in range(200):
                if time_left() < 8:
                    break
                u_add, v_add = random.choice(all_pairs)
                if adj[u_add] & (1 << v_add):
                    continue
                if (u_add, v_add) in added_edges:
                    continue
                if can_add(u_add, v_add, adj):
                    adj[u_add] |= (1 << v_add)
                    adj[v_add] |= (1 << u_add)
                    added_edges.append((u_add, v_add))
                    added_count += 1
                    if added_count == 2:
                        break

            if added_count == 2:
                # Net +1 edge
                current_edges += 1
                tabu[key] = tabu_iteration
                improved = True
                no_improve = 0
                if current_edges > best_local_edges:
                    best_local_edges = current_edges
                    best_local_adj = adj[:]
                break
            elif added_count == 1:
                # Net 0, but might help future additions. Accept if stuck.
                if no_improve > 50:
                    tabu[key] = tabu_iteration
                    improved = True
                    break
                else:
                    # Undo
                    adj[u_add] &= ~(1 << v_add)
                    adj[v_add] &= ~(1 << u_add)
                    adj[u_rem] |= (1 << v_rem)
                    adj[v_rem] |= (1 << u_rem)
            else:
                # No additions possible, undo removal
                adj[u_rem] |= (1 << v_rem)
                adj[v_rem] |= (1 << u_rem)

        if not improved:
            # 3. Simple edge removal to escape (annealing-like)
            if no_improve > 80 and current_edges > 50:
                if edge_list:
                    u_rem, v_rem = random.choice(edge_list)
                    key = (min(u_rem, v_rem), max(u_rem, v_rem))
                    if tabu.get(key, 0) <= tabu_iteration - 60:
                        adj[u_rem] &= ~(1 << v_rem)
                        adj[v_rem] &= ~(1 << u_rem)
                        current_edges -= 1
                        tabu[key] = tabu_iteration
                        no_improve = 0

        # Periodic restart from best if stuck too long
        if no_improve > 200:
            adj = best_local_adj[:]
            current_edges = best_local_edges
            no_improve = 0
            tabu = {}

    # Final: try to add any remaining edges
    for u, v in all_pairs:
        if time_left() < 3:
            break
        if not (adj[u] & (1 << v)):
            if can_add(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges += 1

    if current_edges > best_local_edges:
        best_local_edges = current_edges
        best_local_adj = adj[:]

    if best_local_edges > best_edges:
        best_edges = best_local_edges
        best_adj = best_local_adj

    return get_edges(best_adj)
