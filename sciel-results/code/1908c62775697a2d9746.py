# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    
    def shannon_entropy(p):
        p = p[p > 0]
        if len(p) == 0:
            return 0.0
        return -np.sum(p * np.log2(p))
    
    def compute_score(P):
        # H_{-1}: x - y
        p_minus1 = np.zeros(2*n - 1)
        for y in range(n):
            for x in range(n):
                if P[y, x] > 0:
                    p_minus1[x - y + n - 1] += P[y, x]
        H_minus1 = shannon_entropy(p_minus1)
        
        # H_0: x
        p_x = np.sum(P, axis=0)
        H_0 = shannon_entropy(p_x)
        
        # H_1: x + y
        p_xy = np.zeros(2*n - 1)
        for y in range(n):
            for x in range(n):
                if P[y, x] > 0:
                    p_xy[x + y] += P[y, x]
        H_1 = shannon_entropy(p_xy)
        
        # H_2: x + 2y
        p_x2y = np.zeros(3*n - 1)
        for y in range(n):
            for x in range(n):
                if P[y, x] > 0:
                    p_x2y[x + 2*y] += P[y, x]
        H_2 = shannon_entropy(p_x2y)
        
        M = max(H_0, H_1, H_2)
        if M <= 1e-9:
            return -1.0
            
        return H_minus1 / M
    
    # Strategy: Maximize D = H_{-1} while minimizing M = max(H_0, H_1, H_2)
    
    # D is maximized when x-y is as uniform as possible over its support.
    # The support of x-y is [-9, 9]. There are 19 possible values.
    # To maximize entropy, we want the distribution of x-y to be as flat as possible.
    # This suggests placing mass on cells such that each diagonal k=x-y has roughly equal mass.
    
    # M is minimized if x, x+y, and x+2y are concentrated on few values.
    # This suggests aligning cells so that they share x, x+y, or x+2y values.
    
    # The previous approach tried to align x+2y to specific values based on k mod 3.
    # Let's try a different structure:
    # Instead of just 1 cell per diagonal, what if we use a "thick" diagonal?
    # Or what if we optimize the weights directly?
    
    # Let's try a grid search over a parametric family of distributions.
    # Family: P[y, x] = exp(-alpha * (x - target_x(y))^2 - beta * (y - target_y(x))^2) ? No, too complex.
    
    # Let's stick to the "one cell per diagonal" idea but allow multiple cells if it helps M.
    # Actually, the previous search was limited to C in [5, 25]. Let's expand the search range
    # and also try to allow multiple cells per diagonal if it reduces M significantly.
    
    # However, allowing multiple cells per diagonal increases the support of x-y,
    # which might increase H_{-1} if the distribution becomes flatter, but it also
    # increases the complexity of M.
    
    # Let's try a different parametric approach:
    # Define a function f(k) that gives the x-coordinate for diagonal k.
    # y = x - k.
    # Constraints: 0 <= x < 10, 0 <= y < 10 => max(0, k) <= x <= min(9, 9+k).
    
    # We want to choose x_k for each k in [-9, 9] to maximize D/M.
    # D will be log2(19) if we put equal weight on all 19 diagonals.
    # So D is fixed at log2(19) if we use one cell per diagonal with equal weight.
    # Then we just need to minimize M.
    
    # Let's do a local search / simulated annealing on the choice of x_k.
    # Start with the best from the previous exhaustive search, or a heuristic start.
    
    # Heuristic start:
    # To minimize H_0 (x), we want x_k to be constant for as many k as possible.
    # To minimize H_1 (x+y = 2x-k), we want 2x_k - k to be constant.
    # To minimize H_2 (x+2y = x+2(x-k) = 3x-2k), we want 3x_k - 2k to be constant.
    
    # Let's try to satisfy x+2y = C for all k.
    # 3x_k - 2k = C => x_k = (C + 2k)/3.
    # This requires C + 2k to be divisible by 3.
    # C + 2k ≡ C - k (mod 3).
    # If C ≡ 0 (mod 3), then k ≡ 0 (mod 3) works.
    # If C ≡ 1 (mod 3), then k ≡ 1 (mod 3) works.
    # If C ≡ 2 (mod 3), then k ≡ 2 (mod 3) works.
    
    # This is exactly what the previous program did. It found the best C0, C1, C2.
    # Let's re-run that search but with a wider range and maybe refine with local search.
    
    best_P = None
    best_score = -1
    
    # Search C in [0, 27]
    for C0 in range(0, 28):
        for C1 in range(0, 28):
            for C2 in range(0, 28):
                P_curr = np.zeros((n, n), dtype=np.float64)
                valid = True
                
                for k in range(-9, 10):
                    r = k % 3
                    if r == 0:
                        C = C0
                    elif r == 1:
                        C = C1
                    else:
                        C = C2
                    
                    num_x = C + 2 * k
                    num_y = C - k
                    
                    if num_x % 3 != 0 or num_y % 3 != 0:
                        valid = False
                        break
                    
                    x = num_x // 3
                    y = num_y // 3
                    
                    if x < 0 or x >= n or y < 0 or y >= n:
                        valid = False
                        break
                    
                    P_curr[y, x] += 1.0 / 19.0
                
                if not valid:
                    continue
                
                score = compute_score(P_curr)
                if score > best_score:
                    best_score = score
                    best_P = P_curr.copy()
    
    # Now, let's try to improve by perturbing the solution.
    # We can move cells slightly to reduce M further.
    # But moving a cell changes the diagonal k, which changes D.
    # If we keep one cell per diagonal, D is fixed.
    # So we just need to find x_k that minimizes M.
    
    # Let's do a local search on x_k.
    if best_P is not None:
        # Extract current x_k
        x_k = np.zeros(19, dtype=int)
        for i, k in enumerate(range(-9, 10)):
            # Find the cell in P_curr for this k
            # P_curr[y, x] > 0 and x - y == k
            found = False
            for y in range(n):
                for x in range(n):
                    if P_curr[y, x] > 0 and x - y == k:
                        x_k[i] = x
                        found = True
                        break
                if found:
                    break
        
        # Local search
        for iteration in range(1000):
            improved = False
            for i in range(19):
                k = -9 + i
                current_x = x_k[i]
                current_y = current_x - k
                
                # Try moving x to neighbors
                for dx in [-1, 1]:
                    new_x = current_x + dx
                    new_y = new_x - k
                    if 0 <= new_x < n and 0 <= new_y < n:
                        # Check if this cell is already occupied by another diagonal
                        # In our "one cell per diagonal" model, we assume distinct cells.
                        # But different k can map to the same (x,y) only if k is same.
                        # So we just need to check if (new_x, new_y) is valid.
                        
                        # Update P
                        P_test = P_curr.copy()
                        P_curr[current_y, current_x] = 0
                        P_test[new_y, new_x] += 1.0 / 19.0
                        
                        # Wait, if we move a cell, we must ensure it doesn't collide with another cell.
                        # Since each k is unique, and (x,y) determines k uniquely,
                        # two different k's cannot map to the same (x,y).
                        # So the new cell (new_x, new_y) has k' = new_x - new_y = k.
                        # So it's the same diagonal. No collision with other diagonals.
                        
                        score_test = compute_score(P_test)
                        if score_test > best_score:
                            best_score = score_test
                            best_P = P_test.copy()
                            x_k[i] = new_x
                            improved = True
                            break
            
            if not improved:
                break
    
    if best_P is None:
        return np.ones((n, n), dtype=np.float64) / (n * n)
        
    return best_P

# EVOLVE-BLOCK-END
