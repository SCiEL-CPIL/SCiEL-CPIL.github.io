"""Optimized program for spectral graph lifts using local search."""
import time
import numpy as np


def sign_edges(graph, budget):
    deadline = time.time() + 82.0  # 8s safety margin before 90s kill

    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2

    # Precompute edge list as a list of (u, v) tuples for fast indexing
    edge_list = [(u, v) for u, v in edges]

    def build_signed_matrix(signs):
        A = np.zeros((n, n), dtype=np.float64)
        for i in range(num_edges):
            u, v = edge_list[i]
            s = float(signs[i])
            A[u, v] = s
            A[v, u] = s
        return A

    def spectral_radius(signs):
        A = build_signed_matrix(signs)
        evals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(evals)))

    def is_connected_lift(signs):
        # Build lift graph and check connectivity
        # Lift has 2n vertices: (v,0) and (v,1)
        # For edge (u,v) with sign s:
        #   +1: (u,0)-(v,0) and (u,1)-(v,1)
        #   -1: (u,0)-(v,1) and (u,1)-(v,0)
        # Use BFS/DFS on the lift
        parent = np.full(2 * n, -1, dtype=np.int32)
        for i in range(num_edges):
            u, v = edge_list[i]
            s = signs[i]
            if s == 1:
                # (u,0)-(v,0) and (u,1)-(v,1)
                p_u0 = u
                p_v0 = v + n
                p_u1 = u + n
                p_v1 = v
            else:
                # (u,0)-(v,1) and (u,1)-(v,0)
                p_u0 = v + n
                p_v0 = u + n
                p_u1 = v
                p_v1 = u
            # Union-find approach: just check connectivity via BFS
        # Simpler: BFS from vertex 0 in the lift
        visited = np.zeros(2 * n, dtype=bool)
        stack = [0]
        visited[0] = True
        while stack:
            node = stack.pop()
            # node is in 0..2n-1; base vertex = node % n, side = node // n
            base_v = node % n
            side = node // n
            # Find all edges incident to base_v
            for i in range(num_edges):
                u, v = edge_list[i]
                if u == base_v:
                    other = v
                elif v == base_v:
                    other = u
                else:
                    continue
                s = signs[i]
                if s == 1:
                    new_side = side
                else:
                    new_side = 1 - side
                neighbor = other + new_side * n
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        return bool(np.all(visited))

    def is_valid(signs):
        if not is_connected_lift(signs):
            return False
        # Check 3-regular: each vertex in lift must have degree 3
        # In the base graph, each vertex has degree 3 (2 cycle + 1 matching)
        # In the lift, each (v, side) connects to 3 neighbors
        # This is automatically satisfied if the base is 3-regular
        return True

    # Generate initial random valid signing
    def random_valid_signing(rng):
        """Find a random valid signing by trying random sign vectors."""
        for _ in range(5000):
            signs = rng.choice([-1, 1], size=num_edges)
            if is_valid(signs):
                return signs
        # Fallback: all +1
        return np.ones(num_edges, dtype=np.int8)

    # Local search: flip edges to reduce spectral radius
    def local_search(signs, deadline):
        current_signs = signs.copy()
        current_radius = spectral_radius(current_signs)

        # Track best
        best_signs = current_signs.copy()
        best_radius = current_radius

        # Edge-flipping local search
        # For each edge, try flipping and check if it improves
        # Use simulated annealing style: accept if better, occasionally accept worse
        temp = 0.1
        cooling = 0.999

        while time.time() < deadline:
            # Try random edge flips
            improved = False
            for _ in range(num_edges * 2):
                if time.time() >= deadline:
                    break
                idx = np.random.randint(num_edges)
                trial = current_signs.copy()
                trial[idx] = -trial[idx]

                if not is_valid(trial):
                    continue

                trial_radius = spectral_radius(trial)

                if trial_radius < current_radius:
                    current_signs = trial
                    current_radius = trial_radius
                    improved = True
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_signs = current_signs.copy()
                else:
                    # Accept worse with some probability (simulated annealing)
                    if np.random.random() < np.exp(-(trial_radius - current_radius) / temp):
                        current_signs = trial
                        current_radius = trial_radius

            if not improved:
                temp *= cooling

        return best_signs, best_radius

    # Main search: multiple restarts
    rng = np.random.default_rng(42)
    best_signs_overall = None
    best_radius_overall = np.inf

    # First, get a valid starting point
    initial_signs = random_valid_signing(rng)
    if initial_signs is None:
        # Fallback: all +1 should be valid for prism graphs
        initial_signs = np.ones(num_edges, dtype=np.int8)

    # Run local search with multiple restarts
    num_restarts = 0
    while time.time() < deadline:
        # Start from current best or a new random point
        if num_restarts == 0:
            start_signs = initial_signs
        else:
            # Perturb the best solution
            start_signs = best_signs_overall.copy()
            # Flip a few random edges
            num_flips = min(num_edges // 3, 10)
            flip_indices = rng.choice(num_edges, size=num_flips, replace=False)
            start_signs[flip_indices] = -start_signs[flip_indices]
            # Ensure validity; if not valid, revert
            if not is_valid(start_signs):
                start_signs = best_signs_overall.copy()

        signs, radius = local_search(start_signs, deadline)

        if radius < best_radius_overall:
            best_radius_overall = radius
            best_signs_overall = signs

        num_restarts += 1

        # Check if we've made significant progress
        if num_restarts >= 5 and best_radius_overall < 1.0:
            # We've found something good, do a few more refinements
            if num_restarts >= 15:
                break

    # Final polish: try single-edge flips on the best solution
    if best_signs_overall is not None and time.time() < deadline:
        polished = best_signs_overall.copy()
        polished_radius = best_radius_overall
        for _ in range(50):
            if time.time() >= deadline:
                break
            improved = False
            for idx in range(num_edges):
                trial = polished.copy()
                trial[idx] = -trial[idx]
                if not is_valid(trial):
                    continue
                trial_radius = spectral_radius(trial)
                if trial_radius < polished_radius - 1e-12:
                    polished = trial
                    polished_radius = trial_radius
                    improved = True
                    break
            if not improved:
                break
        best_signs_overall = polished
        best_radius_overall = polished_radius

    if best_signs_overall is None:
        best_signs_overall = initial_signs

    return best_signs_overall
