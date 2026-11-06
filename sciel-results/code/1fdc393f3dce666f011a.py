import numpy as np
import time

def priority(adjacency):
    t0 = time.time()
    deadline = t0 + 115.0

    n = adjacency.shape[0]
    A = np.asarray(adjacency, dtype=bool)
    A_f = A.astype(float)
    deg = A.sum(axis=1).astype(float)
    has_edge = deg > 0
    BIG = 1e9

    # Start with inverse degree as initial safety score
    s = 1.0 / (deg + 1.0)
    s[~has_edge] = BIG

    # Damped fixed-point iteration: s[i] = 1 / (1 + sum_{j in nbr(i)} s[j])
    # Use damping to ensure smooth convergence
    alpha = 0.5
    for it in range(300):
        if time.time() > deadline:
            break
        nbr_sum = A_f @ s
        new_s = 1.0 / (1.0 + nbr_sum)
        new_s[~has_edge] = BIG
        # Damped update
        s = alpha * new_s + (1.0 - alpha) * s
        s[~has_edge] = BIG
        # Check convergence
        if np.max(np.abs(new_s - s)) < 1e-10:
            break

    # Now apply a maximum-entropy-inspired refinement:
    # Model the greedy process as sequential: a vertex's "effective loss"
    # depends on how many of its neighbors will be picked before it.
    # Approximate via soft-min of neighbor scores to capture the "risk"
    # of being adjacent to high-scoring vertices.
    # Use log-sum-exp style soft-max to get a smooth proxy for max neighbor score.

    # Compute soft-max of neighbor scores (higher = more contested)
    # For numerical stability, use the standard trick
    eps = 1e-12
    s_safe = np.where(has_edge, s, 0.0)

    # For each vertex, compute soft-max over neighbors
    # Use: soft_max(s_nbrs) = log(sum(exp(s_nbrs / tau))) * tau
    # But we need to be careful: we want a smooth "maximum" that's high
    # when any neighbor has high score.
    # Use temperature-scaled log-sum-exp
    tau = 1.0
    # Multiply by 1/tau for the log-sum-exp
    s_scaled = s_safe / tau

    # For each vertex i, soft-max over neighbors j:
    # We can compute: for each i, sum_j A[i,j] * exp(s_j / tau)
    # Then soft_max_i = tau * log(sum_j A[i,j] * exp(s_j/tau))
    # To avoid overflow, subtract max per vertex first

    # Compute per-vertex max of neighbor scores for stability
    # max_nbr[i] = max_{j in nbr(i)} s[j]
    # Use: A_f @ s, but we need max not sum. Use a trick:
    # For boolean A, max over neighbors can be approximated by:
    # Actually, let's compute it directly using a loop-free approach:
    # max_nbr[i] = max of s[j] for j where A[i,j]=1
    # We can do this with: for each i, s[nbrs(i)].max()
    # But that's O(n*deg). For n=200, this is fine.
    max_nbr = np.zeros(n)
    for i in range(n):
        if has_edge[i]:
            nbrs = np.where(A[i])[0]
            max_nbr[i] = s[nbrs].max()

    # Soft-max: tau * (log-sum-exp(s_nbrs / tau))
    # = tau * (max + log(sum(exp((s_nbrs - max)/tau))))
    soft_max = np.zeros(n)
    for i in range(n):
        if has_edge[i]:
            nbrs = np.where(A[i])[0]
            s_nbrs = s[nbrs]
            m = np.max(s_nbrs)
            if m > 1e-12:
                soft_max[i] = tau * (m + np.log(np.sum(np.exp((s_nbrs - m) / tau))))
            else:
                soft_max[i] = 0.0

    # The "risk" of vertex i is proportional to soft_max of its neighbors' scores.
    # Vertices with high-risk neighbors should have slightly lower priority.
    # Normalize risk
    risk_max = np.max(soft_max)
    if risk_max > 1e-12:
        risk = soft_max / risk_max
    else:
        risk = np.zeros(n)

    # Adjust scores: reduce score for high-risk vertices
    # Use a moderate coefficient
    penalty = 0.1 * risk
    scores = s - penalty
    scores[~has_edge] = BIG

    # Final polish: add tiny tie-breaker for lower degree
    deg_max = deg.max()
    if deg_max > 0:
        scores = scores + 1e-6 * (-deg / deg_max)

    return scores
