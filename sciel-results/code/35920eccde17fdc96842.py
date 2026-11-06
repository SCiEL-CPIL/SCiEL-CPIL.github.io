import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def count_violations(n, colors):
        """Count C4 violations using NumPy adjacency matrices."""
        total_edges = n * (n - 1) // 2
        assert len(colors) == total_edges
        
        adj = np.zeros((5, n, n), dtype=np.uint8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c, i, j] = 1
                adj[c, j, i] = 1
                idx += 1
        
        cost = 0
        for c in range(5):
            # Common neighbors: (A @ A.T) gives number of common neighbors
            common = adj[c] @ adj[c].T
            # For each pair (u,v), if common[u,v] >= 2, that's a C4
            # We need to count pairs where common[u,v] >= 2
            # common is symmetric, so count upper triangle
            upper = np.triu(common, k=1)
            cost += int(np.sum(upper >= 2))
        return cost

    def get_seed_solution():
        n = 11
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = min(j - i, n - (j - i))
                colors.append(d - 1)
        return n, colors

    best_n, best_colors = get_seed_solution()

    def solve_local_search(n, time_limit):
        """
        Fast local search using NumPy for adjacency operations.
        Returns a valid coloring list or None if not found within time_limit.
        """
        total_edges = n * (n - 1) // 2
        colors = np.random.randint(0, 5, size=total_edges).tolist()
        
        # Build adjacency matrices
        adj = np.zeros((5, n, n), dtype=np.uint8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c, i, j] = 1
                adj[c, j, i] = 1
                idx += 1
        
        # Precompute common neighbor counts
        common = np.zeros((5, n, n), dtype=np.int16)
        for c in range(5):
            common[c] = adj[c] @ adj[c].T
        
        def get_cost():
            cost = 0
            for c in range(5):
                upper = np.triu(common[c], k=1)
                cost += int(np.sum(upper >= 2))
            return cost
        
        current_cost = get_cost()
        if current_cost == 0:
            return colors
        
        search_start = time.time()
        stagnation_counter = 0
        best_cost_so_far = current_cost
        best_colors_snapshot = colors[:]
        
        # Precompute edge index mapping
        edge_idx = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_idx[(i, j)] = idx
                idx += 1
        
        # Precompute for each vertex, list of neighbors per color
        # This helps with delta computation
        
        while time.time() - search_start < time_limit:
            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            c_old = adj[0, i, j] if False else None
            # Get old color from colors array
            idx_ij = edge_idx[(i, j)]
            c_old = colors[idx_ij]
            
            best_new_color = -1
            best_cost = current_cost
            
            for c_new in range(5):
                if c_new == c_old:
                    continue
                
                delta = 0
                
                # Compute delta for removing edge (i,j) from c_old
                # and adding to c_new
                
                # For c_old: removing edge (i,j)
                # For each w, check if removing (i,j) affects C4s involving w
                # A C4 in c_old involving i,j exists if there's w such that
                # adj[c_old][i][w] and adj[c_old][j][w] (w is common neighbor)
                # After removal, common neighbors of i and j in c_old decrease by 1
                # if w was connected to both.
                
                # More carefully: the cost contribution from pairs (u,v) where
                # common[u][v] >= 2. When we remove edge (i,j) from c_old:
                # - For pair (i,j): common_old[i][j] decreases by 1 (if it was >= 2, cost decreases by 1)
                # - For pair (i,w): if adj[c_old][j][w] was 1, common_old[i][w] decreases by 1
                #   (if it was >= 2, cost decreases by 1; if it was 2, now 1, no change in cost)
                # - For pair (j,w): if adj[c_old][i][w] was 1, common_old[j][w] decreases by 1
                
                # Let's compute the delta more carefully
                # Current common counts before change:
                # common_old[i][j] = number of common neighbors of i and j in c_old
                # After removal: common_new[i][j] = common_old[i][j] - 1 (since j was a neighbor of i)
                # Wait, no. adj[c_old][i][j] = 1 means i and j are adjacent in c_old.
                # common[c_old][i][j] = sum_k adj[c_old][i][k] * adj[c_old][j][k]
                # This counts common neighbors, NOT including each other.
                # So removing edge (i,j) doesn't change common[c_old][i][j] directly.
                # But it changes common[c_old][i][w] for all w:
                #   common_new[i][w] = common_old[i][w] - adj[c_old][j][w]
                # Similarly for j.
                
                # Cost change for c_old:
                # For pair (i,w) where w != i,j:
                #   If common_old[i][w] >= 2 and common_old[i][w] - adj[c_old][j][w] < 2: cost decreases by 1
                #   i.e., if common_old[i][w] == 2 and adj[c_old][j][w] == 1: cost decreases by 1
                #   If common_old[i][w] >= 3: no change
                #   If common_old[i][w] <= 1: no change
                # Similarly for (j,w).
                # For pair (i,j): common doesn't change (as argued above).
                
                # Let me just recompute common for affected colors and compare costs.
                # This is O(n^2) per color, but with NumPy it's fast.
                
                # Save current adj state
                adj_save = adj[c_old].copy()
                
                # Apply change: remove (i,j) from c_old, add to c_new
                adj[c_old, i, j] = 0
                adj[c_old, j, i] = 0
                adj[c_new, i, j] = 1
                adj[c_new, j, i] = 1
                
                # Recompute common for c_old and c_new
                common_new_cold = adj[c_old] @ adj[c_old].T
                common_new_cnew = adj[c_new] @ adj[c_new].T
                
                cost_old_cold = int(np.sum(np.triu(common[c_old], k=1) >= 2))
                cost_new_cold = int(np.sum(np.triu(common_new_cold, k=1) >= 2))
                
                cost_old_cnew = int(np.sum(np.triu(common[c_new], k=1) >= 2))
                cost_new_cnew = int(np.sum(np.triu(common_new_cnew, k=1) >= 2))
                
                delta = (cost_new_cold - cost_old_cold) + (cost_new_cnew - cost_old_cnew)
                
                # Revert
                adj[c_old, i, j] = 1
                adj[c_old, j, i] = 1
                adj[c_new, i, j] = 0
                adj[c_new, j, i] = 0
                
                new_cost = current_cost + delta
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_new_color = c_new
            
            # Apply best move
            if best_new_color != -1:
                # Update adj
                adj[c_old, i, j] = 0
                adj[c_old, j, i] = 0
                adj[best_new_color, i, j] = 1
                adj[best_new_color, j, i] = 1
                
                # Update common
                common[c_old] = adj[c_old] @ adj[c_old].T
                common[best_new_color] = adj[best_new_color] @ adj[best_new_color].T
                
                colors[idx_ij] = best_new_color
                
                current_cost = best_cost
                stagnation_counter = 0
                
                if current_cost < best_cost_so_far:
                    best_cost_so_far = current_cost
                    best_colors_snapshot = colors[:]
                
                if current_cost == 0:
                    return colors
            else:
                stagnation_counter += 1
                # Perturbation if stuck
                if stagnation_counter > 50:
                    # Make a few random moves
                    for _ in range(3):
                        ri = random.randint(0, n - 1)
                        rj = random.randint(0, n - 1)
                        if ri == rj:
                            continue
                        if ri > rj:
                            ri, rj = rj, ri
                        ridx = edge_idx[(ri, rj)]
                        rc_old = colors[ridx]
                        rc_new = random.randint(0, 4)
                        if rc_new == rc_old:
                            continue
                        
                        adj[rc_old, ri, rj] = 0
                        adj[rc_old, rj, ri] = 0
                        adj[rc_new, ri, rj] = 1
                        adj[rc_new, rj, ri] = 1
                        
                        common[rc_old] = adj[rc_old] @ adj[rc_old].T
                        common[rc_new] = adj[rc_new] @ adj[rc_new].T
                        
                        colors[ridx] = rc_new
                        
                    current_cost = get_cost()
                    stagnation_counter = 0

        return None

    # Strategy: Try to find valid colorings for n=26 down to 23.
    targets = [26, 25, 24, 23, 22, 21, 20]
    
    for n in targets:
        if time.time() > deadline - 5:
            break
            
        t_limit = min(18.0, deadline - time.time() - 2)
        if t_limit < 1.0:
            break
            
        sol = solve_local_search(n, t_limit)
        
        if sol is not None:
            if count_violations(n, sol) == 0:
                best_n = n
                best_colors = sol
                break
            else:
                pass
                
    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    res = construct()
    print(f"n={res['n']}")
