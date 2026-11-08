import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])

    # Compute features
    deg = adj.sum(axis=1)
    sum_deg = adj @ deg
    
    # Triangle counts
    try:
        tri = np.diag(adj @ adj @ adj)
    except:
        tri = np.zeros(n)

    # Normalize features
    d_norm = deg / (deg.max() + 1e-9)
    s_norm = sum_deg / (sum_deg.max() + 1e-9)
    t_norm = tri / (tri.max() + 1e-9)

    # Precompute neighbor lists for faster simulation
    neighbor_lists = [np.where(adj[i])[0] for i in range(n)]

    def simulate_greedy(scores):
        order = np.argsort(scores)[::-1]
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                selected[v] = True
                count += 1
                neighbors = neighbor_lists[v]
                if len(neighbors) > 0:
                    selected[neighbors] = True
        return count

    candidates = []

    # 1. Seed: Lowest degree first
    candidates.append(-deg)

    # 2. High sum of neighbor degrees
    candidates.append(sum_deg)

    # 3. Linear combinations
    for lam in [1, 2, 5, 10, 20, 50, 100, 200, 500]:
        candidates.append(sum_deg - lam * deg)
    for lam in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
        candidates.append(-deg + lam * sum_deg)

    # 4. Average neighbor degree
    candidates.append(sum_deg / (deg + 1))

    # 5. Iterative relaxation scores (bipartition-inspired)
    # Initialize with low degree
    scores_relax = -d_norm.copy()
    for _ in range(10):
        if time.time() > deadline - 10.0:
            break
        # Update: score_i = 1 / (1 + sum_{j in N(i)} score_j)
        # This encourages independent sets
        new_scores = np.zeros(n)
        for i in range(n):
            s = 0.0
            for j in neighbor_lists[i]:
                s += scores_relax[j]
            new_scores[i] = 1.0 / (1.0 + s)
        scores_relax = new_scores
        candidates.append(scores_relax.copy())

    # 6. Random linear combinations
    np.random.seed(42)
    for _ in range(50):
        w1 = np.random.uniform(-1, 1)
        w2 = np.random.uniform(-1, 1)
        w3 = np.random.uniform(-0.5, 0.5)
        if abs(w1) + abs(w2) + abs(w3) < 0.1:
            w2 = 1.0
        scores = w1 * d_norm + w2 * s_norm + w3 * t_norm
        candidates.append(scores)

    # 7. Local search (random swaps) on best candidate
    best_score = -np.inf
    best_scores = -deg.copy()

    for scores in candidates:
        if time.time() > deadline - 5.0:
            break
        count = simulate_greedy(scores)
        if count > best_score:
            best_score = count
            best_scores = scores.copy()

    # Local search: perturb best_scores and see if it improves
    if time.time() < deadline - 5.0:
        current_scores = best_scores.copy()
        current_count = best_score
        np.random.seed(43)
        for _ in range(1000):
            if time.time() > deadline - 5.0:
                break
            # Pick a random vertex and perturb its score
            v = np.random.randint(n)
            original = current_scores[v]
            current_scores[v] = original + np.random.normal(0, 1.0)
            new_count = simulate_greedy(current_scores)
            if new_count > current_count:
                current_count = new_count
                best_score = new_count
                best_scores = current_scores.copy()
            else:
                current_scores[v] = original

    return best_scores
