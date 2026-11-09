# EVOLVE-BLOCK-START
import numpy as np

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    P = np.zeros((n, n), dtype=np.float64)
    
    # Diagonals k = x - y in [-9, 9]
    ks = range(-9, 10)
    
    # We assign each k to a residue class mod 3
    # and try to align x + 2y to a small set of values.
    
    # Let's try to optimize C0, C1, C2 for residues 0, 1, 2
    # C_r is the target value for x + 2y for k in class r
    
    best_P = None
    best_score = -1
    
    # Search space for C_r
    # x+2y ranges from 0 to 27.
    # Valid C_r should allow x,y in [0,9] for most k.
    # x = (C+2k)/3, y = (C-k)/3
    # 0 <= C+2k <= 27
    # 0 <= C-k <= 27
    
    # We'll iterate over a reasonable range of C values
    # Since k is in [-9, 9], C needs to be around 10-20 to keep y positive for large k.
    # Let's try C in [5, 25]
    
    for C0 in range(5, 26):
        for C1 in range(5, 26):
            for C2 in range(5, 26):
                P_curr = np.zeros((n, n), dtype=np.float64)
                valid = True
                # Map residue to C
                # k % 3 in Python: -1 % 3 = 2, 0 % 3 = 0, 1 % 3 = 1
                # Classes:
                # 0: k = -9, -6, -3, 0, 3, 6, 9 (7 values)
                # 1: k = -8, -5, -2, 1, 4, 7 (6 values)
                # 2: k = -7, -4, -1, 2, 5, 8 (6 values)
                
                for k in ks:
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
                
                # Calculate Entropies
                def shannon_entropy(p):
                    p = p[p > 0]
                    if len(p) == 0:
                        return 0.0
                    return -np.sum(p * np.log2(p))
                
                # H_{-1}: x - y
                # Already uniform by construction (1/19 each)
                H_minus1 = np.log2(19)
                
                # H_0: x
                p_x = np.sum(P_curr, axis=0)
                H_0 = shannon_entropy(p_x)
                
                # H_1: x + y
                p_xy = np.zeros(2*n - 1)
                for y in range(n):
                    for x in range(n):
                        if P_curr[y, x] > 0:
                            p_xy[x + y] += P_curr[y, x]
                H_1 = shannon_entropy(p_xy)
                
                # H_2: x + 2y
                p_x2y = np.zeros(3*n - 1)
                for y in range(n):
                    for x in range(n):
                        if P_curr[y, x] > 0:
                            p_x2y[x + 2*y] += P_curr[y, x]
                H_2 = shannon_entropy(p_x2y)
                
                M = max(H_0, H_1, H_2)
                if M <= 1e-9:
                    continue
                    
                score = H_minus1 / M
                if score > best_score:
                    best_score = score
                    best_P = P_curr.copy()
                    
    if best_P is None:
        # Fallback to uniform
        return np.ones((n, n), dtype=np.float64) / (n * n)
        
    return best_P

# EVOLVE-BLOCK-END
