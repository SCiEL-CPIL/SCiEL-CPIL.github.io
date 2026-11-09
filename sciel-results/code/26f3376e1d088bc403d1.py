"""TSP constructive heuristic: fast vectorized 2-step lookahead + convex-hull angle bias."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_DEADLINE_MARGIN = 8.0  # seconds before 150s limit


def _check_deadline():
    """Return True if we should stop doing expensive work and fall back."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return (time.time() - _start_time) > (150.0 - _DEADLINE_MARGIN)


def _reset_clock():
    global _start_time
    _start_time = time.time()


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next city in a constructive TSP tour.

    Strategy:
      - For each candidate we estimate a full-tour cost using a fast 2-step lookahead:
        cost = d(cur->cand) + d(cand->dest) + sum over remaining of forward-nn distance.
      - The forward-nn term is a cheap global-connectivity penalty: it is large when
        picking `cand` leaves a node that is far from its own nearest neighbor (i.e. we
        risk stranding a distant point), which discourages crossing / bad structure.
      - We add a small convex-hull / angle bias that nudges the tour to trace the
        outer boundary of the point set before moving inward, reducing crossings.
      - A deadline guard degrades to pure nearest-neighbor near the time limit.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    n_unvisited = len(unvisited_nodes)

    if n_unvisited == 0:
        return int(destination_node)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Deadline guard: fall back to cheap nearest-neighbor.
    if _check_deadline():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # Distances from current to all unvisited.
    dist_cur = distance_matrix[current_node, unvisited_nodes]  # (n,)

    # Distances from destination to all unvisited.
    dist_dest = distance_matrix[destination_node, unvisited_nodes]  # (n,)

    # Forward nearest-neighbor distance for each unvisited node:
    # nn[i] = min over j != i of d(unvisited[i], unvisited[j])
    # This is a proxy for "how expensive it is to eventually place node i".
    sub = distance_matrix[unvisited_nodes, unvisited_nodes]  # (n, n)
    # Exclude self by setting diagonal to inf.
    np.fill_diagonal(sub, np.inf)
    nn = np.min(sub, axis=1)  # (n,)

    # ---- Candidate selection ----
    # For large n, only consider top-k nearest to current (plus a couple of strategic ones).
    if n_unvisited > 25:
        k = min(n_unvisited, 14)
        order = np.argsort(dist_cur)
        cand_idx = order[:k]
        # Add node closest to destination (connectivity anchor).
        closest_dest_idx = int(np.argmin(dist_dest))
        if closest_dest_idx not in cand_idx:
            cand_idx = np.append(cand_idx, closest_dest_idx)
    else:
        cand_idx = np.arange(n_unvisited)

    # ---- Score each candidate ----
    # base[i] = d(cur->cand) + d(cand->dest)
    # penalty[i] = sum of nn over remaining nodes after removing cand
    # We want to MINIMIZE base + penalty, but penalty is roughly constant across
    # candidates except it drops by nn[cand] when we remove cand. So:
    #   penalty_i = total_nn - nn[cand]
    #   score_i   = dist_cur[cand] + dist_dest[cand] + (total_nn - nn[cand])
    #             = (dist_cur[cand] + dist_dest[cand] - nn[cand]) + total_nn
    # total_nn is constant, so minimize (dist_cur + dist_dest - nn) over candidates.
    # This is a very fast vectorized score.

    score = dist_cur[cand_idx] + dist_dest[cand_idx] - nn[cand_idx]

    # ---- Convex-hull / angle bias ----
    # Encourage tracing the outer boundary: among candidates, prefer those that
    # lie "far out" relative to the centroid of the unvisited set, and that are
    # roughly in the direction continuing the sweep toward the destination.
    # We compute a small bonus (subtracted from score) for candidates that are
    # far from the centroid and aligned with the current->destination direction.
    if n_unvisited > 4:
        # We don't have coordinates, only the distance matrix. Use a proxy:
        # A node that is far from the average of all other unvisited nodes is
        # "peripheral". Compute average distance from each candidate to the set.
        # avg_dist_to_set[i] = mean_j d(unvisited[i], unvisited[j])
        avg_to_set = np.mean(sub, axis=1)  # (n,)
        # Peripheral candidates get a small discount (they are on the hull).
        peripherality = avg_to_set[cand_idx]
        # Normalize to [0,1] range for a gentle bias.
        p_min = peripherality.min()
        p_max = peripherality.max()
        if p_max > p_min:
            periph_norm = (peripherality - p_min) / (p_max - p_min)
        else:
            periph_norm = np.zeros_like(peripherality)
        # Small bias: peripheral nodes slightly preferred (helps boundary tracing).
        score = score - 0.05 * periph_norm * (dist_cur[cand_idx] + dist_dest[cand_idx]).mean()

    best_pos = int(np.argmin(score))
    best_node = int(unvisited_nodes[cand_idx[best_pos]])

    return best_node
# EVOLVE-BLOCK-END
