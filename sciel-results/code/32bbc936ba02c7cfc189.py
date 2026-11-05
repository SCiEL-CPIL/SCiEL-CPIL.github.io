import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 108.0  # Safety margin for 120s limit (using 108s)
    
    def count_c6_total(A):
        """Count total C6 across all 4 color classes using matrix powers."""
        total = 0
        for c in range(4):
            A_c = A[c]
            # A_c^5[i,j] counts walks of length 5 from i to j
            # C6 count = (1/12) * trace(A_c^6)
            # But trace(A^6) counts each 6-cycle 12 times (6 starting points * 2 directions)
            # So C6 count = trace(A^6) / 12
            A2 = A_c @ A_c
            A4 = A2 @ A2
            A6 = A4 @ A2
            total += np.trace(A6) // 12
        return total
    
    def is_valid(A, n):
        """Check if coloring has no monochromatic C6."""
        for c in range(4):
            A_c = A[c]
            A2 = A_c @ A_c
            A4 = A2 @ A2
            A6 = A4 @ A2
            if np.trace(A6) > 0:
                return False
        return True
    
    def local_search(n, max_iters=50000, initial_temp=1.0):
        """Simulated annealing to minimize C6 count."""
        # Initialize randomly
        A = [np.zeros((n, n), dtype=np.int32) for _ in range(4)]
        for i in range(n):
            for j in range(i+1, n):
                c = random.randint(0, 3)
                A[c][i][j] = 1
                A[c][j][i] = 1
        
        current_cost = count_c6_total(A)
        if current_cost == 0:
            return A, 0
        
        best_A = [A[c].copy() for c in range(4)]
        best_cost = current_cost
        
        temp = initial_temp
        cooling = 0.9995
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random edge and flip its color
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                continue
            
            # Find current color
            old_c = -1
            for c in range(4):
                if A[c][i][j] == 1:
                    old_c = c
                    break
            if old_c == -1:
                continue
            
            # Pick new color
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
            
            # Apply flip
            A[old_c][i][j] = 0
            A[old_c][j][i] = 0
            A[new_c][i][j] = 1
            A[new_c][j][i] = 1
            
            new_cost = count_c6_total(A)
            delta = new_cost - current_cost
            
            if delta <= 0 or random.random() < np.exp(-delta / temp):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_A = [A[c].copy() for c in range(4)]
                    if best_cost == 0:
                        return best_A, 0
            else:
                # Revert
                A[new_c][i][j] = 0
                A[new_c][j][i] = 0
                A[old_c][i][j] = 1
                A[old_c][j][i] = 1
            
            temp *= cooling
        
        return best_A, best_cost
    
    def extract_colors(A, n):
        """Extract color list in lexicographic order."""
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                for c in range(4):
                    if A[c][i][j] == 1:
                        colors.append(c)
                        break
        return colors
    
    # Try n from 17 down to 10
    best_result = None
    best_n = 9
    
    for n in range(17, 9, -1):
        if time.time() > deadline:
            break
        
        # Try multiple restarts for this n
        found = False
        for restart in range(50):
            if time.time() > deadline:
                break
            
            A, cost = local_search(n, max_iters=20000)
            if cost == 0:
                # Verify
                if is_valid(A, n):
                    colors = extract_colors(A, n)
                    best_result = {'n': n, 'colors': colors}
                    best_n = n
                    found = True
                    break
        
        if found:
            # Try to go higher? No, we're going down. 
            # Actually we want max n, so if we found n=17, that's great.
            # If we found n=15, we should have tried 17 and 16 first.
            # Since we're iterating 17 down to 10, the first success is the best.
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: K9 cyclic construction
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}
