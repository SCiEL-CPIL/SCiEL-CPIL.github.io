import time
import random
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 118

    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    all_edges_set = set(all_edges)

    adj = [0] * n

    def is_valid_add(u, v):
        if (adj[u] >> v) & 1:
            return False
        if (adj[u] & adj[v]).bit_count() > 2:
            return False
        temp = adj[u]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            if ((adj[w] & adj[v]) | (1 << u)).bit_count() > 2:
                return False
            temp ^= lsb
        temp = adj[v]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            if ((adj[u] & adj[w]) | (1 << v)).bit_count() > 2:
                return False
            temp ^= lsb
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def edge_count():
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def get_missing():
        return [(u, v) for u, v in all_edges if not ((adj[u] >> v) & 1)]

    def get_present():
        return [(u, v) for u, v in all_edges if (adj[u] >> v) & 1]

    best_count = 0
    best_adj = [0] * n

    def update_best():
        nonlocal best_count, best_adj
        c = edge_count()
        if c > best_count:
            best_count = c
            best_adj = adj[:]

    def greedy_init():
        nonlocal adj
        adj = [0] * n
        edges = all_edges[:]
        random.shuffle(edges)
        for u, v in edges:
            if is_valid_add(u, v):
                add_edge(u, v)

    def k4_init():
        """Start from 5 disjoint K4s (known valid 30-edge solution)"""
        nonlocal adj
        adj = [0] * n
        for k in range(5):
            base = k * 4
            for a in range(4):
                for b in range(a + 1, 4):
                    add_edge(base + a, base + b)

    def polish():
        nonlocal adj
        improved = True
        while improved:
            improved = False
            for u, v in all_edges:
                if not ((adj[u] >> v) & 1):
                    if is_valid_add(u, v):
                        add_edge(u, v)
                        improved = True

    def polish_order():
        """Polish with a fixed deterministic order for reproducibility"""
        nonlocal adj
        improved = True
        while improved:
            improved = False
            for u, v in all_edges:
                if not ((adj[u] >> v) & 1):
                    if is_valid_add(u, v):
                        add_edge(u, v)
                        improved = True

    def two_opt_move():
        """Remove one present edge, then greedily add as many missing edges as possible.
        Accept if net gain > 0."""
        nonlocal adj
        present = get_present()
        if not present:
            return False
        u1, v1 = random.choice(present)
        remove_edge(u1, v1)
        old_count = edge_count()
        # Greedily add missing edges in random order
        missing = get_missing()
        random.shuffle(missing)
        for u, v in missing:
            if is_valid_add(u, v):
                add_edge(u, v)
        new_count = edge_count()
        if new_count > old_count:
            return True
        else:
            # Roll back: remove all newly added edges, re-add original
            # Easier: just rebuild from scratch isn't efficient. Instead,
            # we can't easily rollback. Let's use a different approach:
            # Save state before, restore after.
            pass
        return False

    def two_opt_with_save():
        """Two-opt with explicit state saving for rollback."""
        nonlocal adj
        present = get_present()
        if not present:
            return False
        u1, v1 = random.choice(present)
        saved_adj = adj[:]
        remove_edge(u1, v1)
        old_count = edge_count()
        missing = get_missing()
        random.shuffle(missing)
        for u, v in missing:
            if is_valid_add(u, v):
                add_edge(u, v)
        new_count = edge_count()
        if new_count > old_count:
            return True
        else:
            adj = saved_adj
            return False

    def single_removal_dive():
        """Try removing each present edge one at a time, then polish. Keep best."""
        nonlocal adj
        present = get_present()
        if not present:
            return
        best_local = edge_count()
        best_local_adj = adj[:]
        for u1, v1 in present:
            if time.time() > deadline - 5:
                break
            saved_adj = adj[:]
            remove_edge(u1, v1)
            polish()
            c = edge_count()
            if c > best_local:
                best_local = c
                best_local_adj = adj[:]
            adj = saved_adj
        if best_local > edge_count():
            adj = best_local_adj[:]
            update_best()

    def run_sa():
        nonlocal adj, best_count, best_adj
        # Initialize
        if random.random() < 0.5:
            k4_init()
        else:
            greedy_init()
        polish()
        update_best()

        cur_count = edge_count()
        T = 1.0
        alpha = 0.99995
        min_T = 0.01

        while time.time() < deadline - 10:
            present = get_present()
            missing = get_missing()

            move_type = random.random()
            if move_type < 0.4 and missing:
                # Try to add a random missing edge
                u, v = random.choice(missing)
                if is_valid_add(u, v):
                    add_edge(u, v)
                    cur_count += 1
                    update_best()
            elif move_type < 0.6 and present:
                # Remove a random present edge with some probability
                u, v = random.choice(present)
                if random.random() < math.exp(-1.0 / max(T, min_T)):
                    remove_edge(u, v)
                    cur_count -= 1
            elif move_type < 0.8 and present and missing:
                # Swap: remove one present, try to add one missing
                u1, v1 = random.choice(present)
                u2, v2 = random.choice(missing)
                remove_edge(u1, v1)
                if is_valid_add(u2, v2):
                    add_edge(u2, v2)
                    cur_count = edge_count()
                    update_best()
                else:
                    add_edge(u1, v1)
            else:
                # Two-opt with save
                if two_opt_with_save():
                    cur_count = edge_count()
                    update_best()

            T = max(T * alpha, min_T)

        # Final polish
        polish()
        update_best()

    # Run multiple SA restarts
    num_restarts = 8
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        run_sa()
        # Try single removal dive on the best found so far
        if time.time() < deadline - 15:
            adj = best_adj[:]
            single_removal_dive()
            polish()
            update_best()

    # Final global polish
    adj = best_adj[:]
    polish()
    update_best()

    res = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                res.append([i, j])
    return res

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
    adj = [0] * 20
    for u, v in e:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    valid = True
    for i in range(20):
        for j in range(i + 1, 20):
            if (adj[i] & adj[j]).bit_count() > 2:
                valid = False
                print(f"Violation: {i}, {j} have {(adj[i] & adj[j]).bit_count()} common neighbors")
    print("Valid" if valid else "Invalid")
