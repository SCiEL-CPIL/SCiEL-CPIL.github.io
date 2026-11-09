import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin before 300s limit

    # We'll try to build a 42-vertex graph by extending Paley(37)
    q = 37
    n_core = 37
    n_total = 42

    # Build Paley graph on 37 vertices
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * n_total

    for i in range(n_core):
        for j in range(i + 1, n_core):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # Precompute all 5-subsets that involve at least one new vertex (37-41)
    # This is a significant optimization: we only need to check these
    subsets = []
    for combo in itertools.combinations(range(n_total), 5):
        if any(v >= n_core for v in combo):
            subsets.append(combo)

    def count_violations(adj):
        k5_count = 0
        ik5_count = 0
        for v0, v1, v2, v3, v4 in subsets:
            mask = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)

            # Check K5
            is_k5 = True
            if (adj[v0] & mask) != (mask ^ (1 << v0)):
                is_k5 = False
            elif (adj[v1] & mask) != (mask ^ (1 << v1)):
                is_k5 = False
            elif (adj[v2] & mask) != (mask ^ (1 << v2)):
                is_k5 = False
            elif (adj[v3] & mask) != (mask ^ (1 << v3)):
                is_k5 = False
            elif (adj[v4] & mask) != (mask ^ (1 << v4)):
                is_k5 = False

            if is_k5:
                k5_count += 1

            # Check Independent 5
            is_ik5 = True
            if (adj[v0] & mask) != 0:
                is_ik5 = False
            elif (adj[v1] & mask) != 0:
                is_ik5 = False
            elif (adj[v2] & mask) != 0:
                is_ik5 = False
            elif (adj[v3] & mask) != 0:
                is_ik5 = False
            elif (adj[v4] & mask) != 0:
                is_ik5 = False

            if is_ik5:
                ik5_count += 1

        return k5_count + ik5_count

    best_violations = float('inf')
    best_adj = None

    # Try multiple random initializations with local search
    for attempt in range(100):
        if time.time() > deadline:
            break

        # Randomize new vertices
        adj_temp = adj[:]
        for i in range(n_core, n_total):
            # Clear new vertex connections
            for j in range(n_total):
                if i != j:
                    adj_temp[i] &= ~(1 << j)
                    adj_temp[j] &= ~(1 << i)

            # Randomly connect to core
            for k in range(n_core):
                if random.random() < 0.5:
                    adj_temp[i] |= (1 << k)
                    adj_temp[k] |= (1 << i)

            # Randomly connect to other new vertices
            for j in range(i + 1, n_total):
                if random.random() < 0.5:
                    adj_temp[i] |= (1 << j)
                    adj_temp[j] |= (1 << i)

        current_violations = count_violations(adj_temp)

        if current_violations < best_violations:
            best_violations = current_violations
            best_adj = adj_temp[:]

        if best_violations == 0:
            break

        # Local Search / Simulated Annealing
        current_adj = best_adj[:]
        current_v = best_violations
        temp = 1.0

        for step in range(3000):
            if time.time() > deadline:
                break

            i = random.randint(0, n_total - 1)
            j = random.randint(0, n_total - 1)
            if i == j:
                continue

            # Flip edge
            current_adj[i] ^= (1 << j)
            current_adj[j] ^= (1 << i)

            new_v = count_violations(current_adj)

            if new_v < current_v or random.random() < pow(2.718, -(new_v - current_v) / temp):
                current_v = new_v
                if new_v < best_violations:
                    best_violations = new_v
                    best_adj = current_adj[:]
                    if best_violations == 0:
                        break
            else:
                # Revert
                current_adj[i] ^= (1 << j)
                current_adj[j] ^= (1 << i)

            temp *= 0.9995

    # If we found a valid 42-vertex graph, return it
    if best_violations == 0 and best_adj is not None:
        edges = []
        for i in range(n_total):
            for j in range(i + 1, n_total):
                if (best_adj[i] >> j) & 1:
                    edges.append((i, j))
        return n_total, edges
    else:
        # Fallback to Paley 37
        edges = []
        for i in range(n_core):
            for j in range(i + 1, n_core):
                if (j - i) % q in residues:
                    edges.append((i, j))
        return n_core, edges
