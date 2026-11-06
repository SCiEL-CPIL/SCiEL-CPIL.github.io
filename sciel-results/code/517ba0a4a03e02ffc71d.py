"""Improved program for Maximum Independent Set greedy scoring.
SEARCH approach: generate many diverse candidate priority orders via multiple
heuristic strategies and local search, keeping the best within time budget.
Added: spectral relaxation for initial scoring, improved local search with
faster evaluation, and better heuristic ordering.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import deque


def _greedy_size(adjacency_bool, order):
    """Compute independent set size given a vertex order (indices sorted by priority, ascending)."""
    n = len(order)
    available = np.ones(n, dtype=bool)
    count = 0
    for v in order:
        if available[v]:
            count += 1
            available[v] = False
            available[adjacency_bool[v]] = False
    return count


def _greedy_size_fast(A, order, n, neighbor_sets):
    """Faster greedy size using precomputed neighbor sets."""
    available = [True] * n
    count = 0
    for v in order:
        if available[v]:
            count += 1
            available[v] = False
            for u in neighbor_sets[v]:
                available[u] = False
    return count


def _greedy_size_incremental(A, order, pos1, pos2):
    """Fast evaluation of a single swap at positions pos1 and pos2 in the order.
    Only re-simulates from min(pos1, pos2) onward.
    """
    n = len(order)
    lo = min(pos1, pos2)
    if lo >= n:
        return -1

    available = np.ones(n, dtype=bool)
    count = 0
    for i in range(lo):
        v = order[i]
        if available[v]:
            count += 1
            available[v] = False
            available[A[v]] = False

    v1, v2 = order[pos1], order[pos2]
    for i in range(lo, n):
        if i == pos1:
            v = v2
        elif i == pos2:
            v = v1
        else:
            v = order[i]
        if available[v]:
            count += 1
            available[v] = False
            available[A[v]] = False
    return count


def _spectral_scores(A, n, rng, num_iters=50):
    """Generate scores using power iteration on a modified adjacency matrix.
    
    We want to find a vector x where vertices that are 'far apart' (not adjacent)
    get high scores. We use a relaxation approach: start with random vector,
    repeatedly apply a transformation that penalizes adjacency.
    
    The idea: if two vertices are adjacent, we want their scores to be anti-correlated.
    We use a matrix B = -A + c*I where c is chosen so B has positive eigenvalues,
    then do power iteration to find the dominant eigenvector.
    
    Actually, a simpler approach: use the Laplacian or a modified adjacency.
    Let's try: score = 1 - (degree-normalized adjacency influence)
    """
    # Simple spectral approach: use the second eigenvector of the Laplacian
    # or power iteration on a modified matrix
    
    # Approach: create a "non-adjacency" matrix and find its dominant eigenvector
    # Non-adjacency matrix: J - I - A where J is all-ones
    # But this is dense. For n=200, it's fine.
    
    # Alternative: use power iteration on (I - A * alpha) where alpha controls
    # the penalty for adjacency
    
    alpha = 0.8  # penalty strength
    M = np.eye(n) - alpha * A.astype(float)
    
    # Power iteration to find dominant eigenvector
    x = rng.random(n)
    x = x / (np.linalg.norm(x) + 1e-10)
    
    for _ in range(num_iters):
        x_new = M @ x
        norm = np.linalg.norm(x_new)
        if norm < 1e-10:
            break
        x = x_new / norm
    
    # The dominant eigenvector gives us a score, but we want to maximize
    # independent set size. Let's use the absolute value or a transformed version.
    
    # Actually, let's try a different approach: use the Fiedler vector (second
    # eigenvector of the Laplacian) which captures community structure.
    
    # Simpler: just use degree-based with spectral refinement
    # Combine degree with spectral information
    
    # Use the spectral score as a perturbation
    deg = A.sum(axis=1).astype(float)
    deg_norm = deg / (deg.max() + 1e-10)
    
    # Spectral component: vertices with high x get slightly boosted
    x_mean = x.mean()
    x_std = x.std() + 1e-10
    x_norm = (x - x_mean) / x_std
    
    # Combined score: low degree + some spectral component
    # We want to select vertices that are low-degree and have spectral structure
    combined = deg_norm - 0.3 * x_norm
    
    return combined


def _generate_heuristic_orders(A, deg, n, rng):
    """Generate a diverse set of candidate orders using different heuristics."""
    orders = []
    
    # 1. Ascending degree (classic greedy for MIS)
    orders.append(np.argsort(deg, kind='stable'))
    
    # 2. Ascending degree with random tie-breaking
    noise = rng.random(n) * 0.5
    orders.append(np.argsort(deg + noise, kind='stable'))
    
    # 3. Pure random
    orders.append(rng.permutation(n))
    
    # 4. Descending degree (inverse, sometimes works for sparse graphs)
    orders.append(np.argsort(-deg, kind='stable'))
    
    # 5. Based on average neighbor degree (prefer vertices with low-degree neighbors)
    if n > 0:
        avg_nbr_deg = np.zeros(n)
        for v in range(n):
            nbrs = np.where(A[v])[0]
            if len(nbrs) > 0:
                avg_nbr_deg[v] = deg[nbrs].mean()
            else:
                avg_nbr_deg[v] = 0.0
        # Prefer vertices with low degree and low avg neighbor degree
        combined = deg + 0.5 * avg_nbr_deg
        orders.append(np.argsort(combined, kind='stable'))
    
    # 6. Core number approximation (peel low-degree vertices first)
    core_order = _core_order(A, deg, n)
    orders.append(core_order)
    
    # 7. Weighted random with degree bias
    weights = np.exp(-deg / max(deg.max(), 1))
    weights /= weights.sum()
    perm = rng.choice(n, size=n, replace=False, p=weights)
    orders.append(perm)
    
    # 8. Multiple random perturbations of degree order
    base_order = np.argsort(deg, kind='stable')
    for _ in range(3):
        perturbed = base_order.copy()
        for _ in range(n // 10):
            i, j = rng.integers(0, n, size=2)
            perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
        orders.append(perturbed)
    
    # 9. Spectral-based ordering
    spectral_scores = _spectral_scores(A, n, rng, num_iters=30)
    orders.append(np.argsort(spectral_scores, kind='stable'))
    
    # 10. Boost low-degree vertices with high-degree neighbors
    nbr_max_deg = np.zeros(n)
    for v in range(n):
        nbrs = np.where(A[v])[0]
        if len(nbrs) > 0:
            nbr_max_deg[v] = deg[nbrs].max()
        else:
            nbr_max_deg[v] = 0.0
    boost_score = deg - 0.3 * nbr_max_deg
    orders.append(np.argsort(boost_score, kind='stable'))

    # 11. Effective degree: degree weighted by neighbor degree
    eff_deg = np.zeros(n)
    for v in range(n):
        nbrs = np.where(A[v])[0]
        if len(nbrs) > 0:
            eff_deg[v] = deg[v] * (1 + deg[nbrs].mean() / max(deg[v], 1))
        else:
            eff_deg[v] = 0
    orders.append(np.argsort(eff_deg, kind='stable'))

    # 12. 2-hop neighborhood size
    two_hop = np.zeros(n)
    for v in range(n):
        nbrs = np.where(A[v])[0]
        if len(nbrs) == 0:
            two_hop[v] = 0
        else:
            two_hop_nbrs = set()
            for u in nbrs:
                two_hop_nbrs.update(np.where(A[u])[0])
            two_hop_nbrs.discard(v)
            two_hop_nbrs.difference_update(nbrs)
            two_hop[v] = len(two_hop_nbrs)
    orders.append(np.argsort(two_hop, kind='stable'))
    
    # 13. Minimum fill-in heuristic: prefer vertices that connect few pairs of non-adjacent neighbors
    min_fill = np.zeros(n)
    for v in range(n):
        nbrs = np.where(A[v])[0]
        if len(nbrs) > 1:
            fill = 0
            for i in range(len(nbrs)):
                for j in range(i+1, len(nbrs)):
                    if not A[nbrs[i], nbrs[j]]:
                        fill += 1
            min_fill[v] = fill
        else:
            min_fill[v] = 0
    orders.append(np.argsort(min_fill, kind='stable'))
    
    # 14. Degree squared penalty: prefer very low degree vertices
    deg_sq = deg ** 2
    orders.append(np.argsort(deg_sq, kind='stable'))
    
    # 15. Combined: degree + 0.3*avg_nbr_deg + 0.1*two_hop
    avg_nbr_deg2 = np.zeros(n)
    for v in range(n):
        nbrs = np.where(A[v])[0]
        if len(nbrs) > 0:
            avg_nbr_deg2[v] = deg[nbrs].mean()
        else:
            avg_nbr_deg2[v] = 0.0
    combined2 = deg + 0.3 * avg_nbr_deg2 + 0.1 * two_hop
    orders.append(np.argsort(combined2, kind='stable'))

    return orders


def _core_order(A, deg, n):
    """Generate order based on core decomposition (peel lowest degree first)."""
    order = np.zeros(n, dtype=int)
    remaining = set(range(n))
    deg_copy = deg.copy().astype(int)
    
    for pos in range(n):
        if not remaining:
            break
        # Find vertex with minimum degree among remaining
        min_deg = min(deg_copy[v] for v in remaining)
        candidates = [v for v in remaining if deg_copy[v] == min_deg]
        v = random.choice(candidates)
        order[pos] = v
        remaining.remove(v)
        for u in np.where(A[v])[0]:
            if u in remaining:
                deg_copy[u] -= 1
    
    return order


def _local_search(A, start_order, n, rng, deadline, neighbor_sets=None):
    """Perform local search starting from start_order using incremental evaluation.
    Returns best order and size.
    """
    if neighbor_sets is None:
        neighbor_sets = [np.where(A[v])[0] for v in range(n)]
    
    current_order = start_order.copy()
    current_size = _greedy_size(A, current_order)
    best_order = current_order.copy()
    best_size = current_size
    no_improve = 0
    max_no_improve = 500

    while time.time() < deadline:
        if no_improve > max_no_improve:
            break

        # Try multiple random swaps per iteration to increase diversity
        for _ in range(3):
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i == j:
                continue

            new_size = _greedy_size_incremental(A, current_order, i, j)

            if new_size > current_size:
                current_order[i], current_order[j] = current_order[j], current_order[i]
                current_size = new_size
                no_improve = 0
                if current_size > best_size:
                    best_size = current_size
                    best_order = current_order.copy()
                break
            else:
                no_improve += 1
        else:
            # No improvement in this batch
            no_improve += 1

    return best_order, best_size


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n = adjacency.shape[0]
    A = adjacency.astype(bool)
    
    # Precompute degrees and neighbor sets
    deg = A.sum(axis=1).astype(float)
    neighbor_sets = [np.where(A[v])[0] for v in range(n)]
    
    rng = np.random.default_rng(12345)
    
    # Generate diverse candidate orders
    candidates = _generate_heuristic_orders(A, deg, n, rng)
    
    best_size = -1
    best_order = None
    
    # Evaluate all initial candidates
    for order in candidates:
        if time.time() > deadline - 20:
            break
        size = _greedy_size(A, order)
        if size > best_size:
            best_size = size
            best_order = order.copy()
    
    if best_order is None:
        best_order = np.argsort(deg, kind='stable')
        best_size = _greedy_size(A, best_order)
    
    # Multi-start local search
    ls_deadline = deadline - 5.0
    
    # Do local search from best initial candidate
    best_order, best_size = _local_search(A, best_order, n, rng, ls_deadline, neighbor_sets)

    # If time remains, do additional restarts from other candidates
    if time.time() < ls_deadline - 10:
        other_starts = [candidates[0], candidates[2], candidates[9], candidates[11], candidates[12], candidates[14]] if len(candidates) >= 15 else [candidates[0]]
        for start in other_starts:
            if time.time() > ls_deadline - 5:
                break
            o, s = _local_search(A, start, n, rng, ls_deadline, neighbor_sets)
            if s > best_size:
                best_size = s
                best_order = o

    # Final polish: try adjacent swaps around the best solution
    if time.time() < deadline - 2:
        for _ in range(500):
            if time.time() > deadline - 2:
                break
            i = rng.integers(0, n - 1)
            new_size = _greedy_size_incremental(A, best_order, i, i + 1)
            if new_size > best_size:
                best_order[i], best_order[i + 1] = best_order[i + 1], best_order[i]
                best_size = new_size
    
    # Convert best_order to priority vector
    # Higher priority means selected earlier.
    # Vertex at index 0 in order gets highest priority.
    priority_vec = np.zeros(n)
    for idx, v in enumerate(best_order):
        priority_vec[v] = n - idx
    
    return priority_vec.astype(float)
# EVOLVE-BLOCK-END
