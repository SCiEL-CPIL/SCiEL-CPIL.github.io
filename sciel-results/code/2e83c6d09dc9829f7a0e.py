"""Rewritten program: local search for K4-free graphs minimizing max triangle-free induced subgraph size."""
import random
import time

N = 18
DEADLINE = 115.0  # seconds; evaluator kills at 150s

def construct():
    start = time.time()
    deadline = start + DEADLINE

    # Precompute all 4-subsets as tuples of 4 vertices
    all_quads = []
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                for d in range(c+1, N):
                    all_quads.append((a, b, c, d))
    n_quads = len(all_quads)

    # Precompute all 3-subsets (triangles to check)
    all_tris = []
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                all_tris.append((a, b, c))

    # Precompute all 2-subsets (edges)
    all_edges = []
    for a in range(N):
        for b in range(a+1, N):
            all_edges.append((a, b))
    edge_index = {(a, b): i for i, (a, b) in enumerate(all_edges)}
    n_edges = len(all_edges)

    def make_adj(edge_mask):
        """Build adjacency bitmasks from edge mask (int where bit i means edge all_edges[i] present)."""
        adj = [0] * N
        for i, (a, b) in enumerate(all_edges):
            if edge_mask >> i & 1:
                adj[a] |= 1 << b
                adj[b] |= 1 << a
        return adj

    def is_k4_free(adj):
        """Check if graph with adjacency adj is K4-free."""
        for (a, b, c, d) in all_quads:
            if ((adj[a] >> b & 1) and (adj[a] >> c & 1) and (adj[a] >> d & 1) and
                (adj[b] >> c & 1) and (adj[b] >> d & 1) and
                (adj[c] >> d & 1)):
                return False
        return True

    def is_k4_free_from_mask(edge_mask):
        """Check K4-free directly from edge mask without building adj."""
        for (a, b, c, d) in all_quads:
            e_ab = edge_index[(a, b)]
            e_ac = edge_index[(a, c)]
            e_ad = edge_index[(a, d)]
            e_bc = edge_index[(b, c)]
            e_bd = edge_index[(b, d)]
            e_cd = edge_index[(c, d)]
            if (edge_mask >> e_ab & 1 and edge_mask >> e_ac & 1 and edge_mask >> e_ad & 1 and
                edge_mask >> e_bc & 1 and edge_mask >> e_bd & 1 and
                edge_mask >> e_cd & 1):
                return False
        return True

    def largest_triangle_free(adj):
        """Find the largest subset of vertices inducing a triangle-free subgraph."""
        # We'll use a greedy + local search approach, plus some randomization
        best_size = 0

        # Try multiple random orderings
        for trial in range(50):
            # Greedy: repeatedly pick a vertex whose neighbors in the current set are triangle-free-compatible
            # Simpler: build a candidate set and check
            # Actually, let's use a different approach: iterate over all subsets of size >= best_size+1
            # That's 2^18 = 262144, which is fast.
            pass

        # Exact: check all subsets in descending order of size
        # For size s from N down to best_size+1, check if any s-subset is triangle-free
        # Use bitmasks for vertex subsets
        # A subset S is triangle-free iff for every (a,b,c) in all_tris, not all three are in S
        # We can precompute for each triangle a bitmask of vertices it contains
        tri_masks = [((1 << a) | (1 << b) | (1 << c)) for (a, b, c) in all_tris]

        # Iterate over all 2^18 subsets
        # For each, check if it contains any triangle
        # Optimization: use a recursive / DP approach
        # Actually 2^18 = 262144, and for each we check ~816 triangles. 262144 * 816 ~ 214M, too slow in Python.
        
        # Better: for each subset, we can precompute whether it's triangle-free using inclusion or
        # use the fact that we can build a table: for each vertex v, the set of triangles containing v.
        # Alternative: iterate over subsets and for each, check if any triangle is fully contained.
        # We can do this with bit operations: a subset S (bitmask) contains triangle T iff (S & tri_mask) == tri_mask.
        
        # Let's use a smarter approach: find maximum triangle-free induced subgraph via
        # branch and bound on the complement (max independent set in triangle graph? no).
        
        # Actually, let's just do it with a reasonable heuristic + verification.
        # For n=18, let's use a greedy algorithm with random restarts, and verify.
        
        def is_tri_free_subset(S_mask):
            for tm in tri_masks:
                if (S_mask & tm) == tm:
                    return False
            return True

        # Greedy with random restarts
        best = 0
        rng = random.Random(42 + trial if 'trial' in dir() else 42)
        
        # Try all single-vertex additions greedily
        for _ in range(200):
            order = list(range(N))
            rng.shuffle(order)
            S = 0
            for v in order:
                # Check if adding v keeps S triangle-free
                new_S = S | (1 << v)
                # Check if any triangle is fully in new_S but not in S
                # This means a triangle with two vertices in S and one at v, or all three new (impossible since S was tri-free)
                # So check: for each triangle containing v, are the other two in S?
                v_triangles = [tm for tm in tri_masks if (tm >> v) & 1]
                ok = True
                for tm in v_triangles:
                    others = tm ^ (1 << v)
                    if (S & others) == others:
                        ok = False
                        break
                if ok:
                    S = new_S
            size = bin(S).count('1')
            if size > best:
                best = size

        return best

    # Initial graph: complete tripartite K6,6,6
    def initial_edge_mask():
        mask = 0
        parts = [range(0, 6), range(6, 12), range(12, 18)]
        for i in range(3):
            for j in range(i+1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        mask |= 1 << edge_index[(a, b)]
        return mask

    best_mask = initial_edge_mask()
    best_score = largest_triangle_free(make_adj(best_mask))
    print(f"Initial score: {best_score}")

    def score_of_mask(mask):
        adj = make_adj(mask)
        return largest_triangle_free(adj)

    # Local search
    current_mask = best_mask
    current_score = best_score
    no_improve_count = 0
    max_no_improve = 5000

    while time.time() < deadline - 5:
        improved = False
        # Try flipping each edge
        for ei, (a, b) in enumerate(all_edges):
            if time.time() > deadline - 3:
                break
            new_mask = current_mask ^ (1 << ei)
            # Check K4-free
            if not is_k4_free_from_mask(new_mask):
                continue
            new_score = score_of_mask(new_mask)
            if new_score < current_score:
                current_mask = new_mask
                current_score = new_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_mask = current_mask
                    print(f"Improved to {best_score} at time {time.time()-start:.1f}s")
                no_improve_count = 0
                break  # restart search from new state
            elif new_score == current_score:
                # Accept with some probability for exploration
                if random.random() < 0.1:
                    current_mask = new_mask
                    no_improve_count += 1
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Restart with random perturbation
                perturbed = current_mask
                for _ in range(random.randint(5, 20)):
                    ei = random.randint(0, n_edges-1)
                    new_mask = perturbed ^ (1 << ei)
                    if is_k4_free_from_mask(new_mask):
                        perturbed = new_mask
                current_mask = perturbed
                current_score = score_of_mask(current_mask)
                no_improve_count = 0

    # Final: return the best found
    # Ensure the graph is valid (K4-free)
    if not is_k4_free_from_mask(best_mask):
        # Fallback to initial
        best_mask = initial_edge_mask()

    edges = []
    for i, (a, b) in enumerate(all_edges):
        if best_mask >> i & 1:
            edges.append([a, b])
    return edges
