import time
import sys
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def _time_left():
        return deadline - time.time()

    def _is_valid_c6(n, colors_list):
        """
        Exact check for C6 in any color class.
        colors_list: list of colors for edges (i,j) i<j in lexicographic order.
        Returns True if valid (no C6), False otherwise.
        """
        # Reconstruct adjacency matrices for each color
        # Using numpy for speed in matrix multiplication for trace?
        # No, for exact check, we need to know if a specific cycle exists.
        # The trace of A^6 being 0 is sufficient for no C6?
        # If tr(A^6) == 0, then there are no closed walks of length 6, so no C6.
        # If tr(A^6) > 0, there might be C6s or other walks.
        # So we can use tr(A^6) == 0 as a necessary condition for validity.
        # But it's not sufficient (e.g., a triangle with a tail? No, simple graph.
        # In a simple graph, can tr(A^6) be non-zero without a C6?
        # Yes, e.g., K_4 has triangles. Walks of length 6 can bounce on triangles.
        # So we need an exact check if tr(A^6) > 0?
        # Actually, if we minimize tr(A^6) to 0, we are guaranteed no C6.
        # If we get a state with tr(A^6) > 0, it might still be valid (no C6) but have other walks.
        # However, for the purpose of the search, minimizing tr(A^6) is a good proxy.
        # At the end, we must verify exact validity.
        
        # Exact verification:
        # Build adj lists for each color
        adj = [[set() for _ in range(n)] for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors_list[idx]
                adj[c][i].add(j)
                adj[c][j].add(i)
                idx += 1
        
        # Check for C6 in each color
        for c in range(4):
            # For each edge (u,v), check if there is a path of length 4 from u to v in G-c - {u,v}?
            # Or just check all 6-tuples?
            # C(n, 6) is small.
            # Precompute all 6-combinations?
            # Or use the property: A C6 exists iff there is a path of length 5 between two adjacent vertices?
            # Let's just do a direct search for C6 using the "two neighbors" method.
            # For each u, for each pair of neighbors v, w of u in color c:
            #   Check if there is a path of length 3 from v to w in color c,
            #   using vertices not in {u, v, w}.
            #   If yes, u-v-...-w-u is a C6?
            #   Path v-x-y-w. Cycle u-v-x-y-w-u. Length 5 edges?
            #   u-v (1), v-x (2), x-y (3), y-w (4), w-u (5).
            #   This is a C5. We need C6.
            #   So we need a path of length 4 from v to w.
            #   v-x-y-z-w. Cycle u-v-x-y-z-w-u. Length 6 edges.
            
            for u in range(n):
                neighbors = list(adj[c][u])
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        v = neighbors[i]
                        w = neighbors[j]
                        # Check path of length 4 from v to w, avoiding u, v, w
                        # v - a - b - c - w
                        # a != u, v, w
                        # b != u, v, w, a
                        # c != u, v, w, a, b
                        
                        # BFS/DFS limited depth 4
                        # Level 0: {v}
                        # Level 1: N(v) \ {u, v, w}
                        # Level 2: N(Level 1) \ {u, v, w} \ Level 0 \ Level 1
                        # Level 3: N(Level 2) \ {u, v, w} \ Level 0 \ Level 1 \ Level 2
                        # Level 4: N(Level 3) \ {u, v, w} \ Level 0 \ Level 1 \ Level 2 \ Level 3
                        # If w in Level 4, return False
                        
                        l0 = {v}
                        l1 = set()
                        for x in adj[c][v]:
                            if x != u and x != v and x != w:
                                l1.add(x)
                        if not l1:
                            continue
                        l2 = set()
                        for x in l1:
                            for y in adj[c][x]:
                                if y != u and y != v and y != w and y not in l1:
                                    l2.add(y)
                        if not l2:
                            continue
                        l3 = set()
                        for y in l2:
                            for z in adj[c][y]:
                                if z != u and z != v and z != w and z not in l1 and z not in l2:
                                    l3.add(z)
                        if not l3:
                            continue
                        l4 = set()
                        for z in l3:
                            for w2 in adj[c][z]:
                                if w2 != u and w2 != v and w2 != w and w2 not in l1 and w2 not in l2 and w2 not in l3:
                                    l4.add(w2)
                        if w in l4:
                            return False
        return True

    def _sa_search(n, seed, time_limit):
        """
        Simulated Annealing to minimize sum of tr(A_c^6).
        """
        rng = np.random.RandomState(seed)
        
        # Initialize random coloring
        # colors[i][j] for i<j
        colors = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                colors[i, j] = rng.randint(4)
                colors[j, i] = colors[i, j]
        
        # Build initial adjacency matrices
        A = [np.zeros((n, n), dtype=np.int8) for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[i, j]
                A[c][i, j] = 1
                A[c][j, i] = 1
        
        # Precompute A^6? No, update dynamically.
        # Cost function: sum of tr(A_c^6)
        # tr(A^6) = np.trace(np.linalg.matrix_power(A, 6))
        # This is O(n^3 log 6) or O(n^3 * 6) if manual.
        # np.trace(A @ A @ A @ A @ A @ A) is 5 matrix multiplications.
        # For n=18, 18x18 matrix mult is very fast.
        
        def calc_cost():
            total = 0
            for c in range(4):
                # A_c^6
                # Use matrix multiplication
                # A2 = A @ A
                # A4 = A2 @ A2
                # A6 = A4 @ A2
                # This is 3 mults.
                A_c = A[c]
                A2 = A_c @ A_c
                A4 = A2 @ A2
                A6 = A4 @ A2
                total += np.trace(A6)
            return total
        
        current_cost = calc_cost()
        best_cost = current_cost
        best_colors = colors.copy()
        
        T = 10.0
        T_min = 0.1
        cooling = 0.995
        
        while _time_left() > time_limit:
            # Pick random edge
            i = rng.randint(n)
            j = rng.randint(n)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old_c = colors[i, j]
            new_c = rng.choice([c for c in range(4) if c != old_c])
            
            # Apply move
            A[old_c][i, j] = 0
            A[old_c][j, i] = 0
            A[new_c][i, j] = 1
            A[new_c][j, i] = 1
            colors[i, j] = new_c
            colors[j, i] = new_c
            
            new_cost = calc_cost()
            delta = new_cost - current_cost
            
            if delta < 0 or rng.rand() < np.exp(-delta / T):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_colors = colors.copy()
                    if best_cost == 0:
                        break
            else:
                # Revert
                A[new_c][i, j] = 0
                A[new_c][j, i] = 0
                A[old_c][i, j] = 1
                A[old_c][j, i] = 1
                colors[i, j] = old_c
                colors[j, i] = old_c
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        # Verify best solution
        # Extract colors list
        col_list = []
        for i in range(n):
            for j in range(i + 1, n):
                col_list.append(int(best_colors[i, j]))
        
        if _is_valid_c6(n, col_list):
            return {'n': n, 'colors': col_list}
        else:
            return None

    # Try n=17 first, as it is known to exist
    # Then try n=18 if time permits
    
    # Allocate time: 60s for n=17, 60s for n=18
    # But we want to maximize n. So try 18 first?
    # If 18 is hard, we might waste time.
    # Let's try 18 with 70s limit. If it fails, try 17 with remaining time.
    
    result_18 = None
    if _time_left() > 70:
        result_18 = _sa_search(18, 1, 65)
    
    if result_18 is not None:
        return result_18
    
    # Try 17
    result_17 = None
    if _time_left() > 20:
        # Use multiple seeds if time allows
        seeds = [1, 2, 3, 4, 5]
        for s in seeds:
            if _time_left() < 10:
                break
            res = _sa_search(17, s, 15)
            if res is not None:
                result_17 = res
                break
    
    if result_17 is not None:
        return result_17
        
    # Fallback: return a trivial valid coloring for small n
    # n=1 is always valid
    return {'n': 1, 'colors': []}
