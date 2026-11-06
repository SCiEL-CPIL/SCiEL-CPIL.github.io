import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    # Fallback: Paley graph on 17 vertices
    def get_paley_17():
        squares = {x * x % 17 for x in range(1, 17)}
        edges = []
        for i in range(17):
            for j in range(i + 1, 17):
                if (j - i) % 17 in squares:
                    edges.append([i, j])
        return {'n': 17, 'edges': edges}

    best_result = get_paley_17()

    # Target n values to try, descending
    targets = [36, 35, 34, 33, 32, 31, 30, 28, 25, 22, 20]

    for n in targets:
        if time.time() > deadline:
            break

        # Try to find a valid graph for this n
        result = try_find_graph(n, deadline)
        if result:
            # Verify the result before accepting
            if is_valid(result):
                best_result = result
                # If we found 36 or 35, we can stop as it's optimal or near-optimal
                if n >= 35:
                    return best_result
                # Continue to try for larger n? No, we iterate descending.
                # If we find 34, we should check 35 first.
                # My loop is [36, 35, ...]. So if 36 fails, we try 35.
                # If we find 35, we return it because 36 already failed.
                # Actually, if we find 35, it's a great score.
                # We should probably return immediately if n >= 35 to save time?
                # But what if 36 was possible but we just failed to find it?
                # Given the difficulty, finding 35 is a huge win.
                # Let's return if n >= 35.
                return best_result

    return best_result

def is_valid(graph):
    n = graph['n']
    edges = graph['edges']
    # Build adjacency matrix
    A = [[0]*n for _ in range(n)]
    for u, v in edges:
        A[u][v] = 1
        A[v][u] = 1

    # Check K4
    for i in range(n):
        for j in range(i+1, n):
            if A[i][j]:
                for k in range(j+1, n):
                    if A[i][k] and A[j][k]:
                        for l in range(k+1, n):
                            if A[i][l] and A[j][l] and A[k][l]:
                                return False

    # Check Independent 6-set
    # Equivalent to K6 in complement
    for i in range(n):
        for j in range(i+1, n):
            if not A[i][j]:
                for k in range(j+1, n):
                    if not A[i][k] and not A[j][k]:
                        for l in range(k+1, n):
                            if not A[i][l] and not A[j][l] and not A[k][l]:
                                for m in range(l+1, n):
                                    if not A[i][m] and not A[j][m] and not A[k][m] and not A[l][m]:
                                        for p in range(m+1, n):
                                            if not A[i][p] and not A[j][p] and not A[k][p] and not A[l][p] and not A[m][p]:
                                                return False
    return True

def try_find_graph(n, deadline):
    # Heuristic search using local search with bitmask acceleration
    # Start with random graph
    # p=0.5 is a good start

    max_iters = 10000 # Limit iterations per n to avoid hanging
    iters = 0

    # Try multiple restarts
    for restart in range(10):
        if time.time() > deadline:
            return None
        if iters > max_iters:
            break

        # Initialize
        A = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    A[i][j] = 1
                    A[j][i] = 1

        # Local Search
        for _ in range(max_iters):
            if time.time() > deadline:
                return None

            # Find a violation
            viol, vtype = find_violation(A, n)
            if viol is None:
                # Valid graph found
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if A[i][j]:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}

            # Flip an edge in the violation
            if vtype == 'K4':
                # viol is a tuple (i,j,k,l)
                i,j,k,l = viol
                # Pick a random edge to remove
                edges_to_flip = [(i,j), (i,k), (i,l), (j,k), (j,l), (k,l)]
            else: # I6
                # viol is a tuple (i,j,k,l,m,p)
                i,j,k,l,m,p = viol
                # Pick a random non-edge to add
                edges_to_flip = [(i,j), (i,k), (i,l), (i,m), (i,p), (j,k), (j,l), (j,m), (j,p), (k,l), (k,m), (k,p), (l,m), (l,p), (m,p)]

            u, v = random.choice(edges_to_flip)
            A[u][v] = 1 - A[u][v]
            A[v][u] = A[u][v]

            iters += 1

    return None

def find_violation(A, n):
    # Find one K4 or one Independent 6-set
    # Optimization: Use bitsets for speed

    # Precompute neighbor bitmasks
    neighbors = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if A[i][j]:
                mask |= (1 << j)
        neighbors[i] = mask

    # Find K4
    # Iterate over edges (i, j)
    for i in range(n):
        # Only consider j > i to avoid duplicates? No, we just need any K4.
        # But for speed, we can restrict.
        # Actually, let's just iterate all i, and j > i.
        for j in range(i + 1, n):
            if A[i][j]:
                # Common neighbors of i and j
                common = neighbors[i] & neighbors[j]
                # Remove i and j from common
                common &= ~(1 << i)
                common &= ~(1 << j)

                # Iterate over common neighbors k
                k_list = []
                temp = common
                while temp:
                    # Get lowest set bit
                    lsb = temp & -temp
                    k = lsb.bit_length() - 1
                    k_list.append(k)
                    temp ^= lsb

                # Now check for K3 in common neighbors?
                # We need a triangle in the subgraph induced by common neighbors.
                # Which means we need k, l in common such that A[k][l] is true.
                # And l must be in common.
                # So we just need to check if there is an edge between any two vertices in k_list.
                # Wait, if k and l are both in common, they are neighbors of i and j.
                # If A[k][l] is true, then i,j,k,l form a K4.
                # So we just need to find an edge in the induced subgraph of k_list.

                for idx1, k in enumerate(k_list):
                    # Check neighbors of k that are in k_list
                    # We can use bitmask again
                    # But k_list is small on average?
                    # For n=36, degree ~18, common ~9.
                    # So k_list size ~9.
                    # Checking edges among 9 vertices is 36 checks.
                    # 36 * 18 (avg common size) * 630 (edges) ~ 400k ops.
                    # This is fast.

                    # Optimization: use bitmask
                    k_mask = neighbors[k]
                    # We want to see if k_mask intersects with the bitmask of k_list
                    # But we need to exclude k itself?
                    # And we need to ensure the neighbor is in k_list.
                    # Let's build a bitmask for k_list?
                    # That's expensive if done every time.
                    # Just loop over k_list[idx1+1:]
                    for l in k_list[idx1+1:]:
                        if A[k][l]:
                            return (i, j, k, l), 'K4'

    # Find Independent 6-set
    # This is equivalent to finding a K6 in the complement graph.
    # Complement neighbor bitmask: comp_neighbors[i] = ~neighbors[i] & ((1<<n)-1)
    # But we need to exclude i itself.
    # comp_neighbors[i] = (~neighbors[i] & ((1<<n)-1)) & ~(1<<i)

    comp_neighbors = [0] * n
    all_mask = (1 << n) - 1
    for i in range(n):
        comp_neighbors[i] = (~neighbors[i] & all_mask) & ~(1 << i)

    # Use a recursive backtracking to find a clique of size 6 in the complement graph.
    # This is much faster than checking all 6-subsets if we prune well.

    def find_clique_size_6():
        # Order vertices by degree in complement descending
        degrees = [bin(comp_neighbors[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: degrees[x], reverse=True)

        # We will use a recursive function
        # State: current_clique (list of vertices), candidates (list of vertices that can extend the clique)
        # To speed up, we can use bitmasks for candidates too?
        # But list is easier to manage for small sizes.

        def backtrack(clique, candidates):
            if len(clique) == 6:
                return clique
            if not candidates:
                return None

            # Pruning
            if len(candidates) + len(clique) < 6:
                return None

            # Try to add a vertex from candidates
            # Heuristic: try vertices with high degree in the induced subgraph of candidates
            # For simplicity, iterate through candidates.
            # We can optimize by picking the first valid one?
            # Or random?
            # Let's iterate.

            for idx, v in enumerate(candidates):
                # Check if v is connected to all in clique in complement
                valid = True
                for u in clique:
                    if not (comp_neighbors[u] & (1 << v)):
                        valid = False
                        break

                if valid:
                    # New candidates are those in candidates[idx+1:] that are connected to v in complement
                    new_candidates = []
                    for w in candidates[idx+1:]:
                        if comp_neighbors[v] & (1 << w):
                            new_candidates.append(w)

                    res = backtrack(clique + [v], new_candidates)
                    if res:
                        return res
            return None

        return backtrack([], order)

    clique = find_clique_size_6()
    if clique:
        return tuple(clique), 'I6'

    return None, None
