import time
import random
import numpy as np

def construct():
    """
    Finds a 5-coloring of K_n without monochromatic C4.
    Uses NumPy-accelerated Simulated Annealing.
    """
    deadline = time.time() + 110.0
    
    def verify(colors, n):
        """
        Verifies if the coloring has no monochromatic C4.
        colors: flat list of length n*(n-1)//2
        """
        # Reconstruct color matrix
        c_mat = np.zeros((n, n), dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c_mat[i, j] = colors[idx]
                c_mat[j, i] = colors[idx]
                idx += 1
        
        # For each color c, check if any subgraph contains C4
        # A C4 exists if there are two vertices u, v with >= 2 common neighbors in color c.
        for c in range(5):
            # Get adjacency matrix for color c
            adj = (c_mat == c).astype(np.int8)
            np.fill_diagonal(adj, 0)
            
            # Number of common neighbors for each pair (u, v) is (adj @ adj)[u, v]
            # If any (adj @ adj)[u, v] >= 2, then there is a C4.
            common_neighbors = adj @ adj
            if np.any(common_neighbors >= 2):
                return False
        return True

    def solve_sa(n, time_limit):
        total_edges = n * (n - 1) // 2
        if total_edges == 0:
            return []

        # Initialize colors randomly
        # color is n x n, only i < j used, but we keep symmetry for matrix ops
        color = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                color[i, j] = random.randint(0, 4)
                color[j, i] = color[i, j]

        # K[c][u][v] = number of common neighbors of u and v in color c
        # Initialize K
        K = np.zeros((5, n, n), dtype=np.int32)
        
        # Efficient initialization of K using matrix multiplication
        # For each color c, let A_c be the adjacency matrix of color c.
        # K_c = A_c @ A_c
        # But we only care about off-diagonal elements and u != v.
        # Actually, K[c][u][v] counts x such that color[u][x]==c and color[v][x]==c.
        # This is exactly (A_c @ A_c)[u][v].
        
        for c in range(5):
            A_c = (color == c).astype(np.int32)
            np.fill_diagonal(A_c, 0)
            K[c] = A_c @ A_c
        
        # Calculate initial cost
        # Cost is number of pairs (u, v) with K[c][u][v] >= 2
        # We sum over all c and all u < v
        # Note: K is symmetric.
        cost = 0
        for c in range(5):
            # Create mask for u < v
            # We can just sum the whole matrix and divide by 2, but careful with diagonal (which is 0)
            # K[c][u][u] is degree of u in color c. We don't care about u==v.
            # So sum(K[c]) / 2 gives sum over u < v.
            # Actually, just sum the upper triangle to be safe.
            cost += np.sum(np.triu(K[c], 1) >= 2)

        if cost == 0:
            res_colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    res_colors.append(int(color[i, j]))
            return res_colors

        best_cost = cost
        best_color = color.copy()
        
        T = 1.0
        cooling_rate = 0.999
        start_time = time.time()
        
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        while time.time() - start_time < time_limit:
            # Pick random edge
            e_idx = random.randint(0, len(edges) - 1)
            u, v = edges[e_idx]
            old_c = int(color[u, v])
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
            
            # Calculate delta cost
            # We need to update K for pairs (u, x) and (v, x) for all x != u, v
            # Delta K for color c:
            # If we change color of (u, v) from old_c to new_c:
            # For any x:
            #   K[old_c][u][x] changes by -1 if color[v][x] == old_c
            #   K[old_c][v][x] changes by -1 if color[u][x] == old_c
            #   K[new_c][u][x] changes by +1 if color[v][x] == new_c
            #   K[new_c][v][x] changes by +1 if color[u][x] == new_c
            
            delta = 0
            
            # We need to iterate over x. This is O(n) per move.
            # For n=26, n=26. 26 * 4 checks. Very fast.
            
            # Vectorized approach might be faster, but let's stick to loop for clarity and correctness first.
            # We can use numpy to get the rows.
            
            # Let's just do the loop.
            for x in range(n):
                if x == u or x == v:
                    continue
                
                # Update for old_c
                if color[v, x] == old_c:
                    old_k = K[old_c, u, x]
                    new_k = old_k - 1
                    if old_k >= 2 and new_k < 2:
                        delta -= 1
                    elif old_k < 2 and new_k >= 2:
                        delta += 1
                    K[old_c, u, x] = new_k
                    K[old_c, x, u] = new_k

                if color[u, x] == old_c:
                    old_k = K[old_c, v, x]
                    new_k = old_k - 1
                    if old_k >= 2 and new_k < 2:
                        delta -= 1
                    elif old_k < 2 and new_k >= 2:
                        delta += 1
                    K[old_c, v, x] = new_k
                    K[old_c, x, v] = new_k

                # Update for new_c
                if color[v, x] == new_c:
                    old_k = K[new_c, u, x]
                    new_k = old_k + 1
                    if old_k >= 2 and new_k < 2:
                        delta -= 1
                    elif old_k < 2 and new_k >= 2:
                        delta += 1
                    K[new_c, u, x] = new_k
                    K[new_c, x, u] = new_k

                if color[u, x] == new_c:
                    old_k = K[new_c, v, x]
                    new_k = old_k + 1
                    if old_k >= 2 and new_k < 2:
                        delta -= 1
                    elif old_k < 2 and new_k >= 2:
                        delta += 1
                    K[new_c, v, x] = new_k
                    K[new_c, x, v] = new_k

            # Acceptance criterion
            if delta < 0 or random.random() < np.exp(-delta / T):
                color[u, v] = new_c
                color[v, u] = new_c
                cost += delta
                if cost < best_cost:
                    best_cost = cost
                    best_color = color.copy()
                    if best_cost == 0:
                        res_colors = []
                        for i in range(n):
                            for j in range(i + 1, n):
                                res_colors.append(int(color[i, j]))
                        return res_colors
            else:
                # Revert changes
                # We must revert the K updates we just made.
                for x in range(n):
                    if x == u or x == v:
                        continue
                    
                    if color[v, x] == old_c:
                        K[old_c, u, x] += 1
                        K[old_c, x, u] += 1

                    if color[u, x] == old_c:
                        K[old_c, v, x] += 1
                        K[old_c, x, v] += 1

                    if color[v, x] == new_c:
                        K[new_c, u, x] -= 1
                        K[new_c, x, u] -= 1

                    if color[u, x] == new_c:
                        K[new_c, v, x] -= 1
                        K[new_c, x, v] -= 1
                
            T *= cooling_rate
            if T < 0.01:
                T = 1.0 # Reheat

        # Return best found if it's valid (cost 0)
        if best_cost == 0:
            res_colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    res_colors.append(int(best_color[i, j]))
            return res_colors
        return None

    best_n, best_colors = 11, None
    
    # Fallback: Try to find a small valid coloring
    # n=11 is small, SA should find it instantly
    # We iterate from high to low
    
    search_plan = [
        (26, 5.0),
        (25, 5.0),
        (24, 5.0),
        (23, 5.0),
        (22, 5.0),
        (21, 5.0),
        (20, 5.0),
        (19, 5.0),
        (18, 5.0),
        (17, 5.0),
        (16, 5.0),
        (15, 5.0),
        (14, 5.0),
        (13, 5.0),
        (12, 5.0),
        (11, 5.0),
    ]
    
    for n, t_alloc in search_plan:
        if time.time() > deadline - 5.0:
            break
            
        # Run SA
        res = solve_sa(n, t_alloc)
        
        if res is not None:
            # Verify the result
            if verify(res, n):
                return {'n': n, 'colors': res}
            else:
                # Should not happen if solve_sa is correct, but safety check
                pass
        
        # If we found a valid one, we return immediately in the loop above.
        # If not, we continue to smaller n.
        
    # If we get here, we didn't find a valid one for any n in the plan?
    # This is unlikely for n=11.
    # Let's force a valid small coloring if all else fails.
    # A trivial valid coloring: Random might fail, but for n=11, it's easy.
    # Let's just return a hardcoded valid coloring for n=11 if we reach here.
    # Or just run a very short SA for n=11 again.
    
    res = solve_sa(11, 1.0)
    if res is not None and verify(res, 11):
        return {'n': 11, 'colors': res}
        
    # Absolute fallback: n=1 is always valid
    return {'n': 1, 'colors': []}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
