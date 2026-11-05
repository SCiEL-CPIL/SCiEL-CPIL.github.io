import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def count_c6_fast(adj, n, color):
        """Count C6 in a color class using matrix multiplication."""
        if n < 6:
            return 0
        # Build adjacency for this color
        A = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j] == color:
                    A[i][j] = 1
                    A[j][i] = 1
        if not np.any(A):
            return 0
        A2 = A @ A
        A4 = A2 @ A2
        A6 = A4 @ A2
        return int(np.trace(A6))

    def total_c6(adj, n):
        """Total number of monochromatic C6 across all 4 colors."""
        total = 0
        for c in range(4):
            total += count_c6_fast(adj, n, c)
        return total

    def is_valid(adj, n):
        """Check if coloring is valid (no monochromatic C6)."""
        for c in range(4):
            if count_c6_fast(adj, n, c) > 0:
                return False
        return True

    def local_search(adj, n, max_iters=5000):
        """
        Local search: minimize total monochromatic C6.
        Try flipping individual edges to reduce the count.
        """
        current_count = total_c6(adj, n)
        if current_count == 0:
            return True

        # Edge list
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        edge_indices = list(range(len(edges)))

        # Precompute for each edge, which colors are possible
        # For local search, we try changing an edge's color to another color
        # and see if it reduces total C6

        for iteration in range(max_iters):
            if time.time() > deadline:
                break

            improved = False
            # Randomly shuffle edge order for diversity
            random.shuffle(edge_indices)

            for eidx in edge_indices:
                if time.time() > deadline:
                    break

                i, j = edges[eidx]
                old_color = adj[i][j]

                # Try each other color
                for new_color in range(4):
                    if new_color == old_color:
                        continue

                    adj[i][j] = new_color
                    adj[j][i] = new_color

                    new_count = total_c6(adj, n)

                    if new_count < current_count:
                        current_count = new_count
                        improved = True
                        if current_count == 0:
                            adj[i][j] = new_color
                            adj[j][i] = new_color
                            return True
                        break
                    else:
                        # Revert
                        adj[i][j] = old_color
                        adj[j][i] = old_color

            if not improved:
                # Stuck in local minimum, do a random perturbation
                # Flip a few random edges
                for _ in range(min(5, len(edges))):
                    eidx = random.choice(edge_indices)
                    i, j = edges[eidx]
                    new_color = random.randint(0, 3)
                    adj[i][j] = new_color
                    adj[j][i] = new_color
                current_count = total_c6(adj, n)
                if current_count == 0:
                    return True

        return current_count == 0

    def extract_colors(adj, n):
        """Extract color list in lexicographic order."""
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(int(adj[i][j]))
        return colors

    # Try n from 17 down to 10
    best_result = None

    for n in range(17, 9, -1):
        if time.time() > deadline:
            break

        found = False
        num_restarts = 2000 if n <= 15 else 500

        for restart in range(num_restarts):
            if time.time() > deadline:
                break

            # Random initial coloring
            adj = np.random.randint(0, 4, size=(n, n), dtype=np.int8)
            # Make symmetric
            for i in range(n):
                for j in range(i + 1, n):
                    adj[j][i] = adj[i][j]
            # Diagonal is 0 (irrelevant)
            np.fill_diagonal(adj, 0)

            # Run local search
            if local_search(adj, n, max_iters=3000):
                # Verify
                if is_valid(adj, n):
                    colors = extract_colors(adj, n)
                    best_result = {'n': n, 'colors': colors}
                    found = True
                    break

        if found:
            break

    if best_result is not None:
        return best_result

    # Fallback: K9 cyclic construction
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}
