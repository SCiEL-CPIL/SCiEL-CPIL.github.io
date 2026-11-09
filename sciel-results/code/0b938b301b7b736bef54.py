import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Fallback solution: n=11 cyclic
    def get_fallback():
        n = 11
        colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
        return {'n': n, 'colors': colors}

    # Objective function: Count C4 violations
    def count_violations(color_matrix):
        """
        color_matrix: (n, n) array of colors (0..4)
        Returns: number of pairs (i,j) with >= 2 common neighbors in any color.
        """
        n = color_matrix.shape[0]
        violations = 0
        for c in range(5):
            # Get adjacency matrix for color c
            # A[i,j] = 1 if color_matrix[i,j] == c
            A = (color_matrix == c).astype(np.int32)
            # Compute common neighbors
            # B = A @ A
            # B[i,j] is number of common neighbors of i and j in color c
            B = A @ A
            # For i != j, if B[i,j] >= 2, it's a violation (part of a C4)
            # We count each C4 multiple times, but we just want to minimize this count.
            # Actually, we want to find if ANY C4 exists.
            # A C4 exists if there is i != j with B[i,j] >= 2.
            # We sum the excess: sum(max(0, B[i,j]-1))
            # This is a good proxy for the number of C4s.
            np.fill_diagonal(B, 0)
            # Count pairs with B[i,j] >= 2
            # Since B is symmetric, we can just sum over all i,j and divide by 2?
            # Or just sum (B - 1) * (B >= 2)
            excess = (B - 1) * (B >= 2)
            violations += np.sum(excess)
        return int(violations)

    def local_search(n, max_time=50):
        # Initialize random coloring
        # For each pair (i,j), assign random color 0..4
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        
        current_colors = np.random.randint(0, 5, size=(n, n))
        # Ensure symmetry
        current_colors = (current_colors + current_colors.T) // 2
        # Diagonal is 0, doesn't matter
        
        # Calculate initial violations
        current_viol = count_violations(current_colors)
        
        best_colors = current_colors.copy()
        best_viol = current_viol
        
        t_start = time.time()
        step = 0
        while time.time() < t_start + max_time:
            # Pick a random edge
            i, j = edges[random.randint(0, len(edges) - 1)]
            old_color = current_colors[i, j]
            new_color = random.randint(0, 4)
            if old_color == new_color:
                continue
            
            # Try change
            current_colors[i, j] = new_color
            current_colors[j, i] = new_color
            
            new_viol = count_violations(current_colors)
            
            if new_viol < current_viol:
                current_viol = new_viol
                if new_viol < best_viol:
                    best_viol = new_viol
                    best_colors = current_colors.copy()
                    if best_viol == 0:
                        break
            else:
                # Revert
                current_colors[i, j] = old_color
                current_colors[j, i] = old_color
                
            step += 1
            # Optional: Simulated annealing step?
            # For now, greedy.
            
        return best_colors, best_viol

    # Try n=26
    if time.time() < deadline:
        colors, viol = local_search(26, max_time=60)
        if viol == 0:
            # Convert to list format
            n = 26
            color_list = []
            for i in range(n):
                for j in range(i + 1, n):
                    color_list.append(int(colors[i, j]))
            return {'n': n, 'colors': color_list}

    # Try n=25
    if time.time() < deadline:
        colors, viol = local_search(25, max_time=40)
        if viol == 0:
            n = 25
            color_list = []
            for i in range(n):
                for j in range(i + 1, n):
                    color_list.append(int(colors[i, j]))
            return {'n': n, 'colors': color_list}

    # Try n=24
    if time.time() < deadline:
        colors, viol = local_search(24, max_time=20)
        if viol == 0:
            n = 24
            color_list = []
            for i in range(n):
                for j in range(i + 1, n):
                    color_list.append(int(colors[i, j]))
            return {'n': n, 'colors': color_list}

    # Fallback
    return get_fallback()
