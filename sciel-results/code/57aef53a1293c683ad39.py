"""Max K(2,2,2)-free triple system on 16 points via structural seed + local search."""
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 120s limit
    random.seed(42)

    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_triples = len(all_triples)  # 560
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

    # 15 pairings of 6 indices into 3 pairs
    pairings = [
        [(0,1),(2,3),(4,5)],
        [(0,1),(2,4),(3,5)],
        [(0,1),(2,5),(3,4)],
        [(0,2),(1,3),(4,5)],
        [(0,2),(1,4),(3,5)],
        [(0,2),(1,5),(3,4)],
        [(0,3),(1,2),(4,5)],
        [(0,3),(1,4),(2,5)],
        [(0,3),(1,5),(2,4)],
        [(0,4),(1,2),(3,5)],
        [(0,4),(1,3),(2,5)],
        [(0,4),(1,5),(2,3)],
        [(0,5),(1,2),(3,4)],
        [(0,5),(1,3),(2,4)],
        [(0,5),(1,4),(2,3)]
    ]

    # triple_configs[i] = list of config indices that contain triple i
    triple_configs = [[] for _ in range(num_triples)]
    config_triples = []  # per config: list of 8 triple indices

    for vertices in itertools.combinations(range(N), 6):
        for p in pairings:
            a1, a2 = vertices[p[0][0]], vertices[p[0][1]]
            b1, b2 = vertices[p[1][0]], vertices[p[1][1]]
            c1, c2 = vertices[p[2][0]], vertices[p[2][1]]

            t_indices = []
            for a in (a1, a2):
                for b in (b1, b2):
                    for c in (c1, c2):
                        t = tuple(sorted((a, b, c)))
                        t_indices.append(triple_to_idx[t])

            config_idx = len(config_triples)
            config_triples.append(t_indices)
            for idx in t_indices:
                triple_configs[idx].append(config_idx)

    num_configs = len(config_triples)
    config_counts = [0] * num_configs
    in_set = [False] * num_triples

    def can_add(i):
        for c in triple_configs[i]:
            if config_counts[c] == 7:
                return False
        return True

    def do_add(i):
        in_set[i] = True
        for c in triple_configs[i]:
            config_counts[c] += 1

    def do_remove(i):
        in_set[i] = False
        for c in triple_configs[i]:
            config_counts[c] -= 1

    # --- Structural seed: remove all triples containing any of 4 forbidden vertices ---
    # This guarantees K(2,2,2)-free: any K(2,2,2) needs 6 distinct vertices,
    # but we only have 12 "active" vertices, and all triples use only active vertices.
    # Actually simpler: no triple uses a forbidden vertex, so no config (which needs
    # 6 vertices all appearing in triples) can be satisfied.
    forbidden = set(random.sample(range(N), 4))
    active = [v for v in range(N) if v not in forbidden]

    for i, t in enumerate(all_triples):
        if all(v in active for v in t):
            do_add(i)

    current_size = sum(in_set)
    best_size = current_size
    best_state = in_set[:]
    best_counts = config_counts[:]

    # --- Local Search ---
    no_improve_count = 0

    while time.time() < deadline:
        improved = False

        # Phase 1: Try to add any single missing triple (random order)
        add_indices = [i for i in range(num_triples) if not in_set[i]]
        random.shuffle(add_indices)
        for i in add_indices:
            if can_add(i):
                do_add(i)
                current_size += 1
                improved = True
                break

        if improved:
            no_improve_count = 0
            if current_size > best_size:
                best_size = current_size
                best_state = in_set[:]
                best_counts = config_counts[:]
            continue

        # Phase 2: Try 1-out, k-in swaps
        in_indices = [i for i in range(num_triples) if in_set[i]]
        if not in_indices:
            break

        # Try several random removals
        attempts = min(50, len(in_indices))
        for _ in range(attempts):
            if time.time() > deadline:
                break
            i_out = random.choice(in_indices)
            do_remove(i_out)
            current_size -= 1

            added_count = 0
            added_indices = []
            add_indices = [i for i in range(num_triples) if not in_set[i]]
            random.shuffle(add_indices)
            for i in add_indices:
                if can_add(i):
                    do_add(i)
                    added_count += 1
                    added_indices.append(i)
                    current_size += 1

            if added_count > 1:
                improved = True
                if current_size > best_size:
                    best_size = current_size
                    best_state = in_set[:]
                    best_counts = config_counts[:]
            else:
                for i in added_indices:
                    do_remove(i)
                    current_size -= 1
                do_add(i_out)
                current_size += 1

            if improved:
                break

        if not improved:
            no_improve_count += 1
            # Occasionally try 2-out, 3-in swaps for deeper search
            if no_improve_count % 5 == 0:
                in_indices = [i for i in range(num_triples) if in_set[i]]
                if len(in_indices) >= 2:
                    for _ in range(20):
                        if time.time() > deadline:
                            break
                        out_pair = random.sample(in_indices, 2)
                        for i in out_pair:
                            do_remove(i)
                            current_size -= 1

                        added_count = 0
                        added_indices = []
                        add_indices = [i for i in range(num_triples) if not in_set[i]]
                        random.shuffle(add_indices)
                        for i in add_indices:
                            if can_add(i):
                                do_add(i)
                                added_count += 1
                                added_indices.append(i)
                                current_size += 1

                        if added_count > 2:
                            improved = True
                            if current_size > best_size:
                                best_size = current_size
                                best_state = in_set[:]
                                best_counts = config_counts[:]
                        else:
                            for i in added_indices:
                                do_remove(i)
                                current_size -= 1
                            for i in out_pair:
                                do_add(i)
                                current_size += 1

                        if improved:
                            break

    # Restore best state
    in_set = best_state
    config_counts = best_counts

    return [list(all_triples[i]) for i in range(num_triples) if in_set[i]]
