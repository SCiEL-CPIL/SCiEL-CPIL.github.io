import time
import random
import math
from collections import defaultdict

N = 64

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin

    num_points = N * N
    coords = [(i // N, i % N) for i in range(num_points)]

    # Precompute squared distances using numpy
    try:
        import numpy as np
        xs = np.array([c[0] for c in coords], dtype=np.int32)
        ys = np.array([c[1] for c in coords], dtype=np.int32)
        dist2 = (xs[:, None] - xs[None, :])**2 + (ys[:, None] - ys[None, :])**2
        dist2 = dist2.tolist()
    except ImportError:
        dist2 = [[0]*num_points for _ in range(num_points)]
        for i in range(num_points):
            xi, yi = coords[i]
            for j in range(num_points):
                xj, yj = coords[j]
                dx = xi - xj
                dy = yi - yj
                dist2[i][j] = dx*dx + dy*dy

    def get_dist(i, j):
        return dist2[i][j]

    def validate_set(point_set):
        """Check if a set of points has no isosceles triangles."""
        if len(point_set) <= 2:
            return True
        for p in point_set:
            dists = set()
            for q in point_set:
                if p == q:
                    continue
                d = get_dist(p, q)
                if d in dists:
                    return False
                dists.add(d)
        return True

    def greedy_construction(order):
        """
        Greedily add points in the given order if they don't create an isosceles triangle.
        Returns the resulting set.
        """
        selected = set()
        dist_sets = {}
        for p in order:
            can_add = True
            new_dists = set()
            for q in selected:
                d = get_dist(p, q)
                if d in new_dists:
                    can_add = False
                    break
                new_dists.add(d)
                if d in dist_sets.get(q, set()):
                    can_add = False
                    break
            if can_add:
                selected.add(p)
                dist_sets[p] = new_dists
                for q in selected:
                    if q != p:
                        if q not in dist_sets:
                            dist_sets[q] = set()
                        dist_sets[q].add(get_dist(p, q))
        return selected, dist_sets

    def try_add_point(p, selected, dist_sets):
        """Try to add point p to the selected set. Returns True if successful."""
        new_dists = set()
        for q in selected:
            d = get_dist(p, q)
            if d in new_dists:
                return False
            new_dists.add(d)
            if d in dist_sets.get(q, set()):
                return False
        if len(new_dists) == len(selected):
            selected.add(p)
            dist_sets[p] = new_dists
            for q in selected:
                if q != p:
                    if q not in dist_sets:
                        dist_sets[q] = set()
                    dist_sets[q].add(get_dist(p, q))
            return True
        return False

    def try_swap(p_remove, p_add, selected, dist_sets):
        """
        Try to swap p_remove with p_add in the selected set.
        Returns True if successful.
        """
        if p_remove not in selected or p_add in selected:
            return False

        # Check if p_add can be added to selected - {p_remove}
        new_dists = set()
        for q in selected:
            if q == p_remove:
                continue
            d = get_dist(p_add, q)
            if d in new_dists:
                return False
            new_dists.add(d)
            if d in dist_sets.get(q, set()):
                return False

        # Perform the swap
        selected.remove(p_remove)
        if p_remove in dist_sets:
            del dist_sets[p_remove]
        for q in selected:
            if q in dist_sets:
                dist_sets[q].discard(get_dist(p_remove, q))

        selected.add(p_add)
        dist_sets[p_add] = new_dists
        for q in selected:
            if q != p_add and q in dist_sets:
                dist_sets[q].add(get_dist(p_add, q))
        return True

    def local_search(selected, dist_sets, order, time_limit=30.0):
        """
        Perform local search to improve the selected set.
        Combines addition attempts and swap attempts.
        """
        start_ls = time.time()
        improved = True
        max_iterations = 500
        iteration = 0

        while improved and iteration < max_iterations:
            if time.time() > time_limit + start_ls:
                break
            improved = False
            iteration += 1

            # First, try to add any point
            for p in order:
                if p in selected:
                    continue
                if try_add_point(p, selected, dist_sets):
                    improved = True

            # Then, try swaps
            if not improved:
                selected_list = list(selected)
                for p_remove in selected_list:
                    if len(selected) <= 2:
                        break
                    for p_add in order:
                        if p_add in selected or p_add == p_remove:
                            continue
                        if try_swap(p_remove, p_add, selected, dist_sets):
                            improved = True
                            break
                    if improved:
                        break

        return selected, dist_sets

    # Try multiple different orderings
    best_set = set()
    best_dist_sets = {}

    rng = random.Random(42)

    # Ordering 1: Random
    order1 = list(range(num_points))
    rng.shuffle(order1)
    set1, dist1 = greedy_construction(order1)
    set1, dist1 = local_search(set1, dist1, order1, time_limit=20.0)
    if len(set1) > len(best_set):
        best_set = set1
        best_dist_sets = dist1

    # Ordering 2: Sort by x-coordinate, then y
    order2 = sorted(range(num_points), key=lambda i: (coords[i][0], coords[i][1]))
    set2, dist2_local = greedy_construction(order2)
    set2, dist2_local = local_search(set2, dist2_local, order2, time_limit=20.0)
    if len(set2) > len(best_set):
        best_set = set2
        best_dist_sets = dist2_local

    # Ordering 3: Sort by distance from center
    cx, cy = N/2.0, N/2.0
    order3 = sorted(range(num_points), key=lambda i: (coords[i][0]-cx)**2 + (coords[i][1]-cy)**2)
    set3, dist3 = greedy_construction(order3)
    set3, dist3 = local_search(set3, dist3, order3, time_limit=20.0)
    if len(set3) > len(best_set):
        best_set = set3
        best_dist_sets = dist3

    # Ordering 4: Corner-biased
    order4 = list(range(num_points))
    corner_scores = []
    for i in range(num_points):
        x, y = coords[i]
        min_corner_dist = min(
            x*x + y*y,
            (N-1-x)**2 + y*y,
            x*x + (N-1-y)**2,
            (N-1-x)**2 + (N-1-y)**2
        )
        corner_scores.append((min_corner_dist, i))
    corner_scores.sort()
    order4 = [i for _, i in corner_scores]
    set4, dist4 = greedy_construction(order4)
    set4, dist4 = local_search(set4, dist4, order4, time_limit=20.0)
    if len(set4) > len(best_set):
        best_set = set4
        best_dist_sets = dist4

    # Ordering 5: Spiral from center
    order5 = []
    visited = set()
    cx, cy = N//2, N//2
    for r in range(max(N//2, 1) + 1):
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                if max(abs(dx), abs(dy)) == r:
                    x, y = cx+dx, cy+dy
                    if 0 <= x < N and 0 <= y < N:
                        idx = y*N + x
                        if idx not in visited:
                            order5.append(idx)
                            visited.add(idx)
    for i in range(num_points):
        if i not in visited:
            order5.append(i)
    set5, dist5 = greedy_construction(order5)
    set5, dist5 = local_search(set5, dist5, order5, time_limit=20.0)
    if len(set5) > len(best_set):
        best_set = set5
        best_dist_sets = dist5

    # Simulated Annealing / Random Walk on the best set found
    # Use the best set as a starting point and try random swaps
    if len(best_set) > 0 and time.time() < deadline - 30:
        sa_start = time.time()
        sa_deadline = sa_start + 60.0
        current_set = set(best_set)
        current_dist = {p: set(d) for p, d in best_dist_sets.items()}
        best_sa_size = len(best_set)
        best_sa_set = set(best_set)
        best_sa_dist = {p: set(d) for p, d in best_dist_sets.items()}

        # Try random swaps for a while
        swap_attempts = 0
        max_swaps = 5000
        while swap_attempts < max_swaps and time.time() < sa_deadline:
            swap_attempts += 1
            if len(current_set) < 2:
                break

            # Pick a random point to remove
            p_remove = random.choice(list(current_set))

            # Pick a random point to add
            p_add = None
            for _ in range(20):
                rand_idx = random.randint(0, num_points - 1)
                if rand_idx not in current_set:
                    p_add = rand_idx
                    break
            if p_add is None:
                continue

            # Try the swap
            if try_swap(p_remove, p_add, current_set, current_dist):
                # Try to add more points
                for p in random.sample(range(num_points), min(100, num_points)):
                    if p not in current_set:
                        try_add_point(p, current_set, current_dist)

                if len(current_set) > best_sa_size:
                    best_sa_size = len(current_set)
                    best_sa_set = set(current_set)
                    best_sa_dist = {p: set(d) for p, d in current_dist.items()}
                    # Reset swap attempts to continue searching
                    swap_attempts = 0
                    max_swaps = 5000

        if best_sa_size > len(best_set):
            best_set = best_sa_set
            best_dist_sets = best_sa_dist

    # Validate the best set
    if not validate_set(best_set):
        # Fallback to a small valid set
        best_set = {0, N+2, 4*N+1}

    return [(i // N, i % N) for i in best_set]

# EVOLVE-BLOCK-END
