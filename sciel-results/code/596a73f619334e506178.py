import numpy as np
import time

N = 50

def compute_score(f):
    """Compute ||f||^2 / (||f||_1 * ||f*f||_inf) for nonnegative f."""
    f = np.asarray(f, dtype=float)
    l1 = np.sum(f)
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(f ** 2)
    # Autoconvolution (full)
    conv = np.convolve(f, f, mode='full')
    sup_norm = np.max(conv)
    if sup_norm <= 0:
        return 0.0
    return l2_sq / (l1 * sup_norm)

def greedy_sidon(n_elements, n_positions, max_iter=500000):
    """Greedy construction of a Sidon-like set minimizing max collision count."""
    positions = set()
    collision_counts = np.zeros(n_positions * 2 - 1, dtype=int)
    # Index: lag + (n_positions-1) for the full convolution offset
    
    def add_pos(p):
        positions.add(p)
        # Add all pairs involving p
        for q in positions:
            lag = p - q
            idx = lag + (n_positions - 1)
            collision_counts[idx] += 1
        # Self pair
        collision_counts[p - p + (n_positions - 1)] += 1  # already counted above if p in positions
    
    # Actually, let's be more careful. For each pair (a,b) with a,b in positions,
    # the lag a-b gets +1. Let's just track it properly.
    
    positions = []
    # collision_counts[idx] where idx = lag + offset, offset = n_positions-1
    offset = n_positions - 1
    collision_counts = np.zeros(2 * n_positions - 1, dtype=int)
    
    for _ in range(n_elements):
        best_p = -1
        best_max = n_elements + 1
        for p in range(n_positions):
            if p in positions:
                continue
            # Compute new collision counts if we add p
            # For each existing q in positions, lag = p - q and q - p
            # Also p - p = 0
            new_max = 0
            for q in positions:
                lag1 = p - q
                lag2 = q - p
                idx1 = lag1 + offset
                idx2 = lag2 + offset
                c1 = collision_counts[idx1] + 1
                c2 = collision_counts[idx2] + 1
                new_max = max(new_max, c1, c2)
            # Self pair at lag 0
            idx0 = 0 + offset
            c0 = collision_counts[idx0] + 1
            new_max = max(new_max, c0)
            
            if new_max < best_max:
                best_max = new_max
                best_p = p
        
        if best_p == -1:
            # All positions taken, pick randomly
            available = [p for p in range(n_positions) if p not in positions]
            if not available:
                break
            best_p = available[0]
        
        positions.append(best_p)
        # Update collision counts
        for q in positions[:-1]:  # all previously added
            lag1 = best_p - q
            lag2 = q - best_p
            collision_counts[lag1 + offset] += 1
            collision_counts[lag2 + offset] += 1
        # Self pair
        collision_counts[0 + offset] += 1
    
    return positions

def local_search_improve(positions, n_positions, max_iter=100000, time_limit=10.0):
    """Local search: try swapping one element to reduce max collision."""
    positions = list(positions)
    n = len(positions)
    offset = n_positions - 1
    
    def get_max_collision(pos_list):
        cc = np.zeros(2 * n_positions - 1, dtype=int)
        for i in range(len(pos_list)):
            for j in range(len(pos_list)):
                lag = pos_list[i] - pos_list[j]
                cc[lag + offset] += 1
        return np.max(cc)
    
    current_max = get_max_collision(positions)
    pos_set = set(positions)
    
    start_time = time.time()
    iters = 0
    while iters < max_iter and (time.time() - start_time) < time_limit:
        iters += 1
        improved = False
        for i in range(n):
            old_p = positions[i]
            for new_p in range(n_positions):
                if new_p == old_p or new_p in pos_set:
                    continue
                # Try swap
                positions[i] = new_p
                pos_set.discard(old_p)
                pos_set.add(new_p)
                new_max = get_max_collision(positions)
                if new_max < current_max:
                    current_max = new_max
                    improved = True
                    break
                else:
                    # Revert
                    positions[i] = old_p
                    pos_set.discard(new_p)
                    pos_set.add(old_p)
            if improved:
                break
        if not improved:
            # Try random perturbations to escape local minima
            for _ in range(100):
                i = np.random.randint(n)
                old_p = positions[i]
                new_p = np.random.randint(n_positions)
                if new_p in pos_set:
                    continue
                positions[i] = new_p
                pos_set.discard(old_p)
                pos_set.add(new_p)
                new_max = get_max_collision(positions)
                if new_max < current_max:
                    current_max = new_max
                    break
                else:
                    positions[i] = old_p
                    pos_set.discard(new_p)
                    pos_set.add(old_p)
    
    return positions, current_max

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 10s margin
    
    # Start with greedy Sidon construction
    positions = greedy_sidon(N, N)
    
    # Local search to improve
    if time.time() < deadline - 30:
        positions, max_coll = local_search_improve(positions, N, max_iter=50000, time_limit=20.0)
    
    # Build the height vector: equal heights on selected positions
    f = np.zeros(N)
    for p in positions:
        f[p] = 1.0 / N
    
    score = compute_score(f)
    
    # Try additional local search with time remaining
    if time.time() < deadline - 10:
        # More aggressive local search
        positions, max_coll = local_search_improve(positions, N, max_iter=200000, time_limit=min(25.0, deadline - time.time() - 5))
        f = np.zeros(N)
        for p in positions:
            f[p] = 1.0 / N
        new_score = compute_score(f)
        if new_score > score:
            score = new_score
    
    # Convert to list
    return [float(x) for x in f]

# Test
if __name__ == "__main__":
    f = construct()
    print("Score:", compute_score(f))
    print("Nonzero count:", sum(1 for x in f if x > 0))
