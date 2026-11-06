import time
import numpy as np

def construct():
    deadline = time.time() + 80
    N = 128
    
    # Precompute all 4-term APs
    aps_list = []
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            aps_list.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    aps = np.array(aps_list, dtype=np.int32)
    M = len(aps)
    
    def get_violations(colors):
        c0 = colors[aps[:, 0]]
        c1 = colors[aps[:, 1]]
        c2 = colors[aps[:, 2]]
        c3 = colors[aps[:, 3]]
        all_same = (c0 == c1) & (c1 == c2) & (c2 == c3)
        two_distinct = (
            ((c0 == c1) & (c2 == c3) & (c0 != c2)) |
            ((c0 == c2) & (c1 == c3) & (c0 != c1)) |
            ((c0 == c3) & (c1 == c2) & (c0 != c1))
        )
        return np.sum(all_same | two_distinct)
    
    def count_colors(colors):
        return len(np.unique(colors))
    
    best_colors = None
    best_key = (float('inf'), float('inf'))  # (violations, num_colors)
    
    rng = np.random.RandomState(42)
    
    for trial in range(100):
        if time.time() > deadline - 5:
            break
        
        # Initialize: try different strategies
        if trial % 3 == 0:
            colors = rng.randint(0, 3, size=N)
        elif trial % 3 == 1:
            # Pattern-based init
            colors = np.zeros(N, dtype=np.int32)
            for i in range(N):
                colors[i] = i % 3
        else:
            # Another pattern
            colors = np.zeros(N, dtype=np.int32)
            for i in range(N):
                colors[i] = (i * 2) % 3
        
        violations = get_violations(colors)
        
        if violations == 0:
            nc = count_colors(colors)
            key = (0, nc)
            if key < best_key:
                best_key = key
                best_colors = colors.copy()
            if nc <= 3:
                break
            continue
        
        # Local search
        temp = 2.0
        for step in range(3000):
            if time.time() > deadline - 5:
                break
            if violations == 0:
                break
            
            idx = rng.randint(N)
            old = colors[idx]
            new = (old + 1 + rng.randint(0, 1)) % 3
            colors[idx] = new
            nv = get_violations(colors)
            
            if nv <= violations:
                violations = nv
                temp = max(0.05, temp * 0.995)
            else:
                if rng.random() < np.exp(-(nv - violations) / max(temp, 0.05)):
                    violations = nv
                    temp = max(0.05, temp * 0.98)
                else:
                    colors[idx] = old
        
        nc = count_colors(colors)
        key = (violations, nc)
        if key < best_key:
            best_key = key
            best_colors = colors.copy()
    
    if best_colors is not None:
        return best_colors.tolist()
    
    # Fallback
    colors = [i % 3 for i in range(N)]
    return colors
