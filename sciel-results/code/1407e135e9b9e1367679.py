"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import deque

def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    adj = np.array(adjacency, dtype=np.int8)
    best_labels = None
    best_cut = -1

    def compute_cut(labels):
        diff = labels[:, None] != labels[None, :]
        return int(np.sum(adj * diff)) // 2

    def try_labels(labels):
        nonlocal best_labels, best_cut
        cut = compute_cut(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # --- Spectral initialization (Goemans-Williamson style) ---
    def spectral_init():
        degree = adj.sum(axis=1).astype(np.float64)
        D = np.diag(degree)
        L = D - adj.astype(np.float64)
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            k = min(5, n - 1)
            if k >= 2:
                Y = eigvecs[:, 1:k+1]
                norms = np.linalg.norm(Y, axis=1, keepdims=True)
                norms[norms < 1e-10] = 1.0
                Y = Y / norms
                u = np.random.randn(k)
                u = u / np.linalg.norm(u)
                proj = Y @ u
                labels = (proj > 0).astype(np.int8)
            else:
                labels = np.random.randint(0, 2, size=n)
        except:
            labels = np.random.randint(0, 2, size=n)
        return labels

    # --- Cerny Tabu Search ---
    def cerny_tabu_search(labels, tabu_len=10, max_steps=3000, aspiration=True):
        labels = labels.copy()
        current_cut = compute_cut(labels)
        best_local_cut = current_cut
        best_local_labels = labels.copy()

        # Precompute neighbor lists
        neighbors = [np.where(adj[v] == 1)[0] for v in range(n)]

        # Precompute sum_same and sum_opp for each vertex
        sum_same = np.zeros(n, dtype=np.int32)
        sum_opp = np.zeros(n, dtype=np.int32)
        for v in range(n):
            same_mask = labels == labels[v]
            opp_mask = labels != labels[v]
            sum_same[v] = int(adj[v, same_mask].sum())
            sum_opp[v] = int(adj[v, opp_mask].sum())

        # Tabu list: store (vertex, step_index) for recent flips
        tabu = deque()
        current_step = 0

        for step in range(max_steps):
            if time.time() > deadline - 10.0:
                break

            current_step += 1

            # Remove expired tabu entries
            while tabu and (current_step - tabu[0][1]) >= tabu_len:
                tabu.popleft()

            # Build tabu set for quick lookup
            tabu_set = set(v for v, _ in tabu)

            # Find best non-tabu move (or best tabu move if aspiration applies)
            best_delta = -np.inf
            best_v = -1
            best_is_tabu = False

            for v in range(n):
                if v in tabu_set:
                    continue
                delta = sum_same[v] - sum_opp[v]
                if delta > best_delta:
                    best_delta = delta
                    best_v = v
                    best_is_tabu = False

            # If all moves are tabu, pick the best tabu move (aspiration)
            if best_v == -1:
                for v in range(n):
                    delta = sum_same[v] - sum_opp[v]
                    if delta > best_delta:
                        best_delta = delta
                        best_v = v
                        best_is_tabu = True

            if best_v == -1:
                break

            # Aspiration: accept tabu move if it improves best
            if best_is_tabu and not aspiration:
                # Pick a random non-tabu vertex if available, else break
                non_tabu = [v for v in range(n) if v not in tabu_set]
                if non_tabu:
                    best_v = random.choice(non_tabu)
                    best_delta = sum_same[best_v] - sum_opp[best_v]
                    best_is_tabu = False
                else:
                    break

            # Perform the flip
            v = best_v
            old_label = labels[v]
            new_label = 1 - old_label
            labels[v] = new_label
            current_cut += best_delta

            # Update tabu list
            tabu.append((v, current_step))

            # Update sum_same and sum_opp
            sum_same[v], sum_opp[v] = sum_opp[v], sum_same[v]

            for u in neighbors[v]:
                if labels[u] == new_label:
                    sum_same[u] += 1
                    sum_opp[u] -= 1
                else:
                    sum_same[u] -= 1
                    sum_opp[u] += 1

            # Track best
            if current_cut > best_local_cut:
                best_local_cut = current_cut
                best_local_labels = labels.copy()

        return best_local_labels, best_local_cut

    # --- Greedy 1-flip local search (fast, vectorized per pass) ---
    def greedy_1flip(labels, max_passes=15):
        labels = labels.copy()
        for _ in range(max_passes):
            if time.time() > deadline - 2.0:
                break
            changed = False
            same_count = (adj * (labels[None, :] == labels[:, None])).sum(axis=1)
            opp_count = (adj * (labels[None, :] != labels[:, None])).sum(axis=1)
            flip_mask = same_count > opp_count
            if not np.any(flip_mask):
                break
            labels[flip_mask] = 1 - labels[flip_mask]
            changed = True
        return labels

    # --- Multi-start with diverse initializations + Cerny Tabu ---
    # Phase 1: Spectral init + Cerny Tabu
    for r in range(10):
        if time.time() > deadline - 30.0:
            break
        if r % 2 == 0:
            labels = spectral_init()
        else:
            labels = np.random.randint(0, 2, size=n)
        tabu_len = 8 + (r % 5) * 2  # Vary tabu length: 8,10,12,14,16
        max_steps = 2500
        ts_labels, ts_cut = cerny_tabu_search(labels, tabu_len=tabu_len, max_steps=max_steps)
        try_labels(ts_labels)

        # Polish with greedy
        if time.time() < deadline - 10.0:
            polished = greedy_1flip(ts_labels, max_passes=8)
            try_labels(polished)

    # Phase 2: More Cerny Tabu runs with different parameters
    for r in range(8):
        if time.time() > deadline - 15.0:
            break

        if r % 3 == 0:
            labels = spectral_init()
        elif r % 3 == 1:
            labels = np.random.randint(0, 2, size=n)
        else:
            labels = np.arange(n) % 2
            flip_mask = np.random.random(n) < 0.3
            labels[flip_mask] = 1 - labels[flip_mask]

        tabu_len = 12 + (r % 4) * 3  # 12,15,18,21
        max_steps = 2000
        ts_labels, ts_cut = cerny_tabu_search(labels, tabu_len=tabu_len, max_steps=max_steps)
        try_labels(ts_labels)

        if time.time() < deadline - 10.0:
            polished = greedy_1flip(ts_labels, max_passes=10)
            try_labels(polished)

    # Final greedy polish on best
    if best_labels is not None and time.time() < deadline - 5.0:
        final = greedy_1flip(best_labels, max_passes=15)
        try_labels(final)

    # Final safety net
    if best_labels is None:
        labels = np.arange(n) % 2
        return labels

    return best_labels
# EVOLVE-BLOCK-END
