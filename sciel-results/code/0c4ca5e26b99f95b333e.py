import time
import random
import math


def construct():
    t0 = time.time()
    DEADLINE = t0 + 140.0  # safety margin under 150 s kill

    def gen_instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict(edges):
        m = len(edges)
        if m == 0:
            return [], []
        n = max(max(u, v) for u, v in edges) + 1
        adj = [set() for _ in range(n)]
        inc = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        for i, (u, v) in enumerate(edges):
            inc[u].add(i)
            inc[v].add(i)
        conflict = [set() for _ in range(m)]
        for i, (u, v) in enumerate(edges):
            forb = adj[u] | adj[v]
            s = set()
            for w in forb:
                s |= inc[w]
            s.discard(i)
            conflict[i] = s
        return conflict, adj

    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    instances = []
    for spec in specs:
        edges = gen_instance(*spec)
        conflict, adj = build_conflict(edges)
        instances.append((edges, conflict))

    # Global SA over all instances
    # state: list of lists col[inst][edge_idx]
    # energy: sum over instances of (num_colours + PENALTY * num_conflicts)
    # We want to minimize num_colours, but conflicts must be 0 in final.
    # During SA, allow conflicts with penalty to escape local minima.

    PENALTY = 1000  # large penalty for any conflict
    m_list = [len(edges) for edges, _ in instances]
    total_m = sum(m_list)

    # Initial greedy per instance (DSATUR-like: order by conflict degree desc)
    col = []
    for idx, (edges, conflict) in enumerate(instances):
        m = len(edges)
        order = sorted(range(m), key=lambda i: -len(conflict[i]))
        c = [-1] * m
        for i in order:
            used = set()
            for j in conflict[i]:
                cj = c[j]
                if cj >= 0:
                    used.add(cj)
            k = 0
            while k in used:
                k += 1
            c[i] = k
        col.append(c)

    def count_colours(c):
        return len(set(c))

    def count_conflicts(c, conflict):
        """Count number of conflicting pairs in the colouring."""
        cnt = 0
        classes = {}
        for i, k in enumerate(c):
            classes.setdefault(k, set()).add(i)
        for k, es in classes.items():
            for i in es:
                for j in es:
                    if i != j and j in conflict[i]:
                        cnt += 1
        return cnt // 2  # each pair counted twice

    def energy(col_list):
        e = 0
        for idx, (edges, conflict) in enumerate(instances):
            c = col_list[idx]
            e += count_colours(c) + PENALTY * count_conflicts(c, conflict)
        return e

    def remap_colours(c):
        present = sorted(set(c))
        remap = {old: new for new, old in enumerate(present)}
        return [remap[k] for k in c]

    # Simulated Annealing
    rng = random.Random(42)
    current = [c[:] for c in col]
    current_e = energy(current)
    best = [c[:] for c in current]
    best_e = current_e

    T_start = 50.0
    T_end = 0.1
    # Number of steps: run until deadline - 2s
    steps_budget = 200000  # will be limited by time

    t_sa_start = time.time()
    sa_time_budget = DEADLINE - t_sa_start - 3.0  # leave 3s for finalization
    if sa_time_budget < 5:
        sa_time_budget = 5

    # Adaptive: estimate steps per second
    # For now, just loop with time check
    step = 0
    no_improve = 0
    max_no_improve = 5000

    while time.time() < t_sa_start + sa_time_budget:
        step += 1
        # Cool
        T = T_start * (T_end / T_start) ** (step / max(1, steps_budget))
        if T < T_end:
            T = T_end

        # Pick random instance and edge
        inst = rng.randrange(len(instances))
        m = m_list[inst]
        if m == 0:
            continue
        e_idx = rng.randrange(m)
        conflict = instances[inst][1]
        c = current[inst]
        old_k = c[e_idx]

        # Pick new colour: try to reuse an existing colour or create new
        # Get used colours in this instance
        used_cols = set(c)
        # Candidate colours: 0..max(used_cols)+1
        max_c = max(used_cols) + 1 if used_cols else 1
        new_k = rng.randrange(max_c + 1)

        if new_k == old_k:
            continue

        # Compute delta energy
        # Remove e_idx from old_k class, add to new_k class
        # Conflicts: check if e_idx conflicts with any edge in new_k class
        # and if removing e_idx from old_k class resolves conflicts

        # For efficiency, just compute full local delta by checking conflicts
        # Delta = (new_conflicts - old_conflicts) + (new_num_cols - old_num_cols) * 1
        # But num_cols change is complex. Let's just compute penalty delta.

        # Count conflicts involving e_idx in old class
        old_conf = 0
        for j in conflict[e_idx]:
            if c[j] == old_k:
                old_conf += 1

        # Count conflicts involving e_idx in new class
        new_conf = 0
        for j in conflict[e_idx]:
            if c[j] == new_k:
                new_conf += 1

        # Penalty delta
        d_pen = (new_conf - old_conf) * PENALTY

        # Num colours delta:
        # If old_k class becomes empty and new_k class was non-empty: -1
        # If old_k class was non-empty and new_k class was empty: +1
        # Else: 0
        old_class_size = sum(1 for x in c if x == old_k)
        new_class_size = sum(1 for x in c if x == new_k)

        d_cols = 0
        if old_class_size == 1 and new_class_size == 0:
            d_cols = -1
        elif old_class_size > 1 and new_class_size == 0:
            d_cols = 0
        elif old_class_size == 1 and new_class_size > 0:
            d_cols = 0
        elif old_class_size > 1 and new_class_size > 0:
            d_cols = 0
        else:
            # Should not happen
            pass

        d_e = d_pen + d_cols

        if d_e <= 0 or rng.random() < math.exp(-d_e / T):
            c[e_idx] = new_k
            current_e += d_e

            # Check for improvement
            # Only update best if no conflicts (penalty = 0)
            if current_e < best_e:
                # Verify no conflicts
                total_conf = 0
                for idx2, (edges2, conflict2) in enumerate(instances):
                    total_conf += count_conflicts(current[idx2], conflict2)
                if total_conf == 0:
                    best_e = current_e
                    best = [c2[:] for c2 in current]
                    no_improve = 0
                else:
                    # Don't update best if conflicts exist
                    pass
            else:
                no_improve += 1

            if no_improve > max_no_improve:
                # Restart: re-heat
                T_start = max(T * 10, 10.0)
                no_improve = 0
                # Optionally: random restart of current state
                # For now, just continue

    # Final: ensure best is valid (no conflicts)
    # If best has conflicts, fall back to greedy
    total_conf = 0
    for idx2, (edges2, conflict2) in enumerate(instances):
        total_conf += count_conflicts(best[idx2], conflict2)

    if total_conf > 0:
        # Use the last valid greedy or re-greedy
        for idx2, (edges2, conflict2) in enumerate(instances):
            m = len(edges2)
            order = sorted(range(m), key=lambda i: -len(conflict2[i]))
            c = [-1] * m
            for i in order:
                used = set()
                for j in conflict2[i]:
                    cj = c[j]
                    if cj >= 0:
                        used.add(cj)
                k = 0
                while k in used:
                    k += 1
                c[i] = k
            best[idx2] = c

    # Remap colours to be compact
    out = []
    for idx2, c in enumerate(best):
        out.append(remap_colours(c))

    return out
