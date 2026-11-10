import numpy as np
import time

N = 256

def _build_aps():
    """Build list of all 4-term arithmetic progressions in 1..256 (0-indexed)."""
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    return np.array(aps, dtype=np.int32)

APS = _build_aps()

def _calculate_cost(colors):
    """Calculate the number of monochromatic 4-APs."""
    c = colors[APS]
    mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
    return np.sum(mono)

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    return np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer for safety
    
    best_colors = None
    best_cost = float('inf')
    
    # Initial candidates
    candidates = [
        _thue_morse(N),
        np.random.randint(0, 2, N, dtype=np.int8),
        np.array([(i // 2) % 2 for i in range(N)], dtype=np.int8),
        np.array([1 if (i % 4 < 2) else 0 for i in range(N)], dtype=np.int8),
    ]
    
    # Add a few random ones
    for _ in range(10):
        candidates.append(np.random.randint(0, 2, N, dtype=np.int8))
    
    # Evaluate initial candidates
    for c in candidates:
        cost = _calculate_cost(c)
        if cost < best_cost:
            best_cost = cost
            best_colors = c.copy()
    
    # Simulated Annealing
    current_colors = best_colors.copy()
    current_cost = best_cost
    
    T = 10.0
    T_min = 0.01
    alpha = 0.9995
    
    steps = 0
    while time.time() < deadline:
        # Pick a random index to flip
        i = np.random.randint(N)
        
        # Flip
        current_colors[i] = 1 - current_colors[i]
        
        # Calculate new cost
        new_cost = _calculate_cost(current_colors)
        
        delta = new_cost - current_cost
        
        if delta < 0:
            # Accept
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_colors = current_colors.copy()
        else:
            # Accept with probability exp(-delta/T)
            if T > 0:
                prob = np.exp(-delta / T)
                if np.random.random() < prob:
                    current_cost = new_cost
                else:
                    # Revert
                    current_colors[i] = 1 - current_colors[i]
            else:
                # Revert
                current_colors[i] = 1 - current_colors[i]
        
        T *= alpha
        
        steps += 1
        
        # Periodic reheat and kick to escape local minima
        if steps % 2000 == 0 and T < 0.1:
            # Random kick: flip 5 random bits
            for _ in range(5):
                k = np.random.randint(N)
                current_colors[k] = 1 - current_colors[k]
            current_cost = _calculate_cost(current_colors)
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = current_colors.copy()
            T = 5.0 # Reheat
    
    return list(best_colors)
