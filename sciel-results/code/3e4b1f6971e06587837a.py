import random
import time
import numpy as np

def construct():
    n = 20
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(edges)

    def is_k33_free_add(u, v, M):
        A = M[v] & ~(1 << u)
        B = M[u] & ~(1 << v)

        a_list = []
        temp = A
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            a_list.append(idx)
            temp ^= lsb

        if len(a_list) < 2:
            return True

        for i in range(len(a_list)):
            for j in range(i + 1, len(a_list)):
                a1 = a_list[i]
                a2 = a_list[j]
                common = M[a1] & M[a2] & B
                if common & (common - 1):
                    return False
        return True

    def count_edges(M):
        return sum(bin(M[i]).count("1") for i in range(n)) // 2

    def spectral_init():
        """Initialize using spectral method: create dense random-like graph."""
        # Generate random symmetric matrix
        A = np.random.rand(n, n)
        A = (A + A.T) / 2.0
        np.fill_diagonal(A, 0)

        # Get top eigenvector
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        top_vec = eigenvectors[:, -1]

        # Threshold at mean to create dense graph
        threshold = np.mean(top_vec)
        M = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if top_vec[i] * top_vec[j] > threshold:
                    if is_k33_free_add(i, j, M):
                        M[i] |= (1 << j)
                        M[j] |= (1 << i)
        return M

    def greedy_fill(M):
        """Try to add edges in random order."""
        shuffled = edges[:]
        random.shuffle(shuffled)
        for u, v in shuffled:
            if not (M[u] & (1 << v)):
                if is_k33_free_add(u, v, M):
                    M[u] |= (1 << v)
                    M[v] |= (1 << u)
        return M

    def local_search(M, deadline):
        """Local search: try removing edges and re-adding others."""
        current = M[:]
        best_count = count_edges(current)

        while time.time() < deadline:
            improved = False

            # Try removing an edge and see if we can add multiple new ones
            edge_list = []
            for i in range(n):
                temp = current[i]
                while temp:
                    lsb = temp & -temp
                    j = lsb.bit_length() - 1
                    if i < j:
                        edge_list.append((i, j))
                    temp ^= lsb

            random.shuffle(edge_list)

            for u, v in edge_list:
                if time.time() >= deadline:
                    break

                # Remove edge (u, v)
                trial = current[:]
                trial[u] &= ~(1 << v)
                trial[v] &= ~(1 << u)

                # Try to greedily add back edges in random order
                shuffled = edges[:]
                random.shuffle(shuffled)
                for a, b in shuffled:
                    if not (trial[a] & (1 << b)):
                        if is_k33_free_add(a, b, trial):
                            trial[a] |= (1 << b)
                            trial[b] |= (1 << a)

                new_count = count_edges(trial)
                if new_count > best_count:
                    current = trial
                    best_count = new_count
                    improved = True

            if not improved:
                # Try single edge removals followed by greedy fill
                for u, v in edge_list:
                    if time.time() >= deadline:
                        break
                    trial = current[:]
                    trial[u] &= ~(1 << v)
                    trial[v] &= ~(1 << u)
                    trial = greedy_fill(trial)
                    new_count = count_edges(trial)
                    if new_count > best_count:
                        current = trial
                        best_count = new_count
                        improved = True

            if not improved:
                break

        return current, best_count

    best_M = [0] * n
    best_count = 0

    start_time = time.time()
    deadline = start_time + 110.0

    # First, try spectral initialization
    while time.time() < deadline:
        # Spectral init
        M = spectral_init()
        count = count_edges(M)
        if count > best_count:
            best_count = count
            best_M = M[:]

        # Greedy fill on spectral init
        M = greedy_fill(M)
        count = count_edges(M)
        if count > best_count:
            best_count = count
            best_M = M[:]

        # Local search
        if time.time() < deadline - 5.0:
            M, count = local_search(M, deadline - 2.0)
            if count > best_count:
                best_count = count
                best_M = M[:]

        # Also try pure random greedy for comparison
        M = [0] * n
        shuffled = edges[:]
        random.shuffle(shuffled)
        for u, v in shuffled:
            if is_k33_free_add(u, v, M):
                M[u] |= (1 << v)
                M[v] |= (1 << u)
        count = count_edges(M)
        if count > best_count:
            best_count = count
            best_M = M[:]

    # Convert best_M to list of edges
    result = []
    for i in range(n):
        temp = best_M[i]
        while temp:
            lsb = temp & -temp
            j = lsb.bit_length() - 1
            if i < j:
                result.append([i, j])
            temp ^= lsb
    return result
