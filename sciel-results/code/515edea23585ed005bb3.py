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

    # Deadline guard: kill after ~110s to be safe under 120s limit
    start_time = time.time()
    DEADLINE = start_time + 110.0

    def time_left():
        return DEADLINE - time.time()

    def cut_size(labels):
        # labels: array of 0/1 of length n
        # Cut size = sum over edges where endpoints differ
        # Using matrix: cut = sum(adj[i,j] * (labels[i] != labels[j]))
        # Vectorized: for each row, count neighbors with different label
        # Efficient: cut = 0.5 * (sum over i of sum over j of adj[i,j]*(1 - 2*labels[i])*(1 - 2*labels[j]))
        # Or simpler: labels as +1/-1, then cut = 0.25 * sum(adj * (x_i - x_j)^2) ... 
        # Let's use: x = 2*labels - 1 (so 0->-1, 1->+1)
        # cut = 0.25 * sum_{i,j} adj[i,j] * (x_i - x_j)^2
        # But simpler: cut = sum over i<j of adj[i,j] * (labels[i] != labels[j])
        # Vectorized approach:
        # For each vertex v, degree to opposite side
        # cut = 0.5 * sum_v (adj[v] @ (labels != labels[v]).astype(int))
        # Actually: cut = 0.5 * sum_v sum_u adj[v,u] * (labels[v] != labels[u])
        # Let's compute: for each v, count of neighbors with different label
        diff = (labels[:, None] != labels[None, :])
        return np.sum(adj * diff) // 2

    def cut_size_fast(labels):
        # Faster: use matrix multiplication
        # Let x = 2*labels - 1, so x in {-1, +1}
        # cut = 0.25 * (n * sum(x^2) - sum(x)^2 ... no
        # Actually: sum_{i,j} adj[i,j] * (1 - x_i*x_j)/2 = (sum(adj) - sum(adj * outer(x,x))) / 2
        # But we only want i<j, so divide by 2 more? No, adj is symmetric with 0 diagonal.
        # sum_{i,j} adj[i,j] * (labels[i] != labels[j]) counts each cut edge twice.
        # So cut = 0.5 * sum_{i,j} adj[i,j] * (labels[i] != labels[j])
        # In terms of x: labels[i] != labels[j] <=> x_i * x_j = -1
        # So cut = 0.5 * sum_{i,j} adj[i,j] * (1 - x_i*x_j)/2 = 0.25 * (sum(adj) - sum_{i,j} adj[i,j] * x_i * x_j)
        x = 2 * labels.astype(np.float64) - 1
        # sum_{i,j} adj[i,j] * x_i * x_j = x^T adj x
        xTadjx = x @ adj @ x
        total_edges = np.sum(adj) // 2  # since symmetric
        return (total_edges * 2 - xTadjx) / 4  # wait, let me re-derive

    # Let me re-derive cleanly:
    # cut = number of edges crossing = sum_{i<j} adj[i,j] * [labels[i] != labels[j]]
    # = 0.5 * sum_{i != j} adj[i,j] * [labels[i] != labels[j]]  (since adj symmetric, adj[i,i]=0)
    # With x_i = 2*labels[i] - 1 in {-1,1}:
    # [labels[i] != labels[j]] = (1 - x_i * x_j) / 2
    # So cut = 0.5 * sum_{i != j} adj[i,j] * (1 - x_i * x_j) / 2
    #        = 0.25 * sum_{i != j} adj[i,j] * (1 - x_i * x_j)
    #        = 0.25 * (sum_{i != j} adj[i,j] - sum_{i != j} adj[i,j] * x_i * x_j)
    # sum_{i != j} adj[i,j] = 2 * E where E = total number of edges
    # sum_{i != j} adj[i,j] * x_i * x_j = x^T adj x (since adj[i,i]=0, the diagonal doesn't contribute)
    # So cut = 0.25 * (2E - x^T adj x) = 0.5 * E - 0.25 * x^T adj x

    E = int(np.sum(adj)) // 2

    def cut_size_v2(labels):
        x = 2 * labels.astype(np.float64) - 1
        xTadjx = x @ adj @ x
        return 0.5 * E - 0.25 * xTadjx

    def local_search(labels, max_rounds=50):
        """Hill-climbing local search. Returns best labels found."""
        best_labels = labels.copy()
        best_cut = cut_size_v2(best_labels)
        for rnd in range(max_rounds):
            if time_left() < 2.0:
                break
            improved = False
            for v in range(n):
                if time_left() < 1.0:
                    break
                # Compute gain from flipping v
                # Current contribution of v to cut: sum of adj[v,u] for u with labels[u] != labels[v]
                # If we flip v, the new contribution becomes sum of adj[v,u] for u with labels[u] == labels[v] (before flip)
                # Gain = (new_contribution - old_contribution)
                # old_contribution = sum_{u: labels[u] != labels[v]} adj[v,u]
                # new_contribution = sum_{u: labels[u] == labels[v]} adj[v,u]
                # Note: this doesn't double count because we're computing v's contribution to the total cut.
                # Total cut = sum_v (contribution of v) / 2 ... no.
                # Actually cut = sum_{i<j} ... = 0.5 * sum_i sum_j adj[i,j] [labels[i]!=labels[j]]
                # The term for vertex v in sum_i sum_j is: sum_j adj[v,j] [labels[v]!=labels[j]]
                # When we flip v, this term changes from old to new.
                # The change in total sum_i sum_j is (new - old), so change in cut is 0.5 * (new - old).
                old_contrib = adj[v][labels != labels[v]].sum()
                new_contrib = adj[v][labels == labels[v]].sum()
                if new_contrib > old_contrib:
                    labels[v] = 1 - labels[v]
                    improved = True
            if not improved:
                break
        # Recompute best
        final_cut = cut_size_v2(labels)
        if final_cut > best_cut:
            best_labels = labels.copy()
            best_cut = final_cut
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
            labels, cut = local_search(labels)
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
        return best_labels, best_cut

    # Strategy: 
    # 1. Start with alternating, run local search
    # 2. Try several random restarts
    # 3. If time permits, try more sophisticated approaches

    # Initial: alternating
    labels_alt = np.arange(n) % 2
    best_labels, best_cut = local_search(labels_alt, max_rounds=100)

    # Random restarts
    num_restarts = 20
    labels_rs, cut_rs = multi_start(num_restarts)
    if cut_rs > best_cut:
        best_labels = labels_rs
        best_cut = cut_rs

    # If time permits, try a few more restarts or a different heuristic
    if time_left() > 10.0:
        # Try starting from the best found and doing more aggressive search
        # Or try different number of restarts
        labels_rs2, cut_rs2 = multi_start(30)
        if cut_rs2 > best_cut:
            best_labels = labels_rs2
            best_cut = cut_rs2

    # Final polish: run local search a few more times on the best
    if time_left() > 3.0:
        for _ in range(5):
            if time_left() < 1.0:
                break
            labels_polish, cut_polish = local_search(best_labels, max_rounds=50)
            if cut_polish > best_cut:
                best_cut = cut_polish
                best_labels = labels_polish
            else:
                break

    return best_labels.astype(int)
# EVOLVE-BLOCK-END
