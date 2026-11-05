"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    # Convert to numpy bool/int matrix for fast vectorized ops
    adj = np.asarray(adjacency, dtype=np.int8)
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    # Deadline guard: kill after ~140s to be safe under 150s limit
    start_time = time.time()
    DEADLINE = start_time + 140.0

    def time_left():
        return DEADLINE - time.time()

    E = int(np.sum(adj)) // 2

    def cut_size_v2(labels):
        x = 2 * labels.astype(np.float64) - 1
        xTadjx = x @ adj @ x
        return 0.5 * E - 0.25 * xTadjx

    def local_search(labels, max_rounds=50):
        """Hill-climbing local search with incremental adjx update."""
        best_labels = labels.copy()
        best_cut = cut_size_v2(best_labels)
        x = 2 * labels.astype(np.float64) - 1
        adjx = adj @ x
        for rnd in range(max_rounds):
            if time_left() < 2.0:
                break
            gains = x * adjx
            max_gain = np.max(gains)
            if max_gain > 0:
                v = int(np.argmax(gains))
                # Flip vertex v: x[v] -> -x[v]
                # adjx[u] changes by -2 * x[v] * adj[u,v] for all u
                # Since adj is symmetric, adjx -= 2 * x[v] * adj[:, v]
                adjx -= 2 * x[v] * adj[:, v]
                x[v] = -x[v]
                labels[v] = 1 - labels[v]
            else:
                break
        final_cut = cut_size_v2(labels)
        if final_cut > best_cut:
            best_labels = labels.copy()
            best_cut = final_cut
        return best_labels, best_cut

    def simulated_annealing(labels, T_start=10.0, T_end=0.1, cooling_rate=0.995, max_iters=2000):
        """Simulated annealing with incremental adjx update."""
        best_labels = labels.copy()
        best_cut = cut_size_v2(best_labels)
        current_cut = best_cut
        T = T_start
        rng = np.random.default_rng()
        x = 2 * labels.astype(np.float64) - 1
        adjx = adj @ x
        
        for it in range(max_iters):
            if time_left() < 5.0:
                break
            if T <= T_end:
                break
            
            v = rng.integers(0, n)
            gain = x[v] * adjx[v]
            
            if gain > 0 or rng.random() < np.exp(gain / T):
                # Flip vertex v
                adjx -= 2 * x[v] * adj[:, v]
                x[v] = -x[v]
                labels[v] = 1 - labels[v]
                current_cut += gain / 2.0
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            
            T *= cooling_rate
        
        return best_labels, best_cut

    def multi_start(num_starts, labels_init=None):
        """Try multiple random starts and return the best result."""
        best_labels = None
        best_cut = -1
        for s in range(num_starts):
            if time_left() < 3.0:
                break
            if s == 0 and labels_init is not None:
                labels = labels_init.copy()
            else:
                labels = np.random.randint(0, 2, size=n)
            labels, cut = local_search(labels, max_rounds=100)
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
        return best_labels, best_cut

    # Strategy: 
    # 1. Start with alternating, run local search
    # 2. Try several random restarts with local search
    # 3. Use simulated annealing to escape local optima
    # 4. Polish with local search

    # Initial: alternating
    labels_alt = np.arange(n) % 2
    best_labels, best_cut = local_search(labels_alt, max_rounds=150)

    # Random restarts with local search
    num_restarts = 20
    labels_rs, cut_rs = multi_start(num_restarts)
    if cut_rs > best_cut:
        best_labels = labels_rs
        best_cut = cut_rs

    # Simulated annealing phase: use SA from best solution and from random starts
    if time_left() > 20.0:
        # SA from the current best
        sa_labels, sa_cut = simulated_annealing(best_labels.copy(), T_start=5.0, T_end=0.5, cooling_rate=0.998, max_iters=5000)
        if sa_cut > best_cut:
            best_labels = sa_labels
            best_cut = sa_cut
        # Polish after SA
        if time_left() > 10.0:
            polish_labels, polish_cut = local_search(best_labels, max_rounds=200)
            if polish_cut > best_cut:
                best_labels = polish_labels
                best_cut = polish_cut

    # More SA from different random starts if time permits
    if time_left() > 30.0:
        for trial in range(8):
            if time_left() < 15.0:
                break
            random_labels = np.random.randint(0, 2, size=n)
            sa_labels, sa_cut = simulated_annealing(random_labels, T_start=8.0, T_end=0.3, cooling_rate=0.997, max_iters=8000)
            if sa_cut > best_cut:
                best_labels = sa_labels
                best_cut = sa_cut
            # Polish after each SA
            if time_left() > 10.0:
                polish_labels, polish_cut = local_search(best_labels, max_rounds=150)
                if polish_cut > best_cut:
                    best_labels = polish_labels
                    best_cut = polish_cut

    # Final polish: run local search multiple times on the best
    if time_left() > 5.0:
        for _ in range(15):
            if time_left() < 2.0:
                break
            labels_polish, cut_polish = local_search(best_labels, max_rounds=150)
            if cut_polish > best_cut:
                best_cut = cut_polish
                best_labels = labels_polish
            else:
                break

    return best_labels.astype(int)
# EVOLVE-BLOCK-END
