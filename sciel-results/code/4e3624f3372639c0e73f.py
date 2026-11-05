"""Minimum maximal no-isosceles set on 16x16 grid."""
N = 16
N2 = N * N

# EVOLVE-BLOCK-START
import itertools
import random
import time
import math

# Precompute point coordinates and index mapping
points = list(itertools.product(range(N), repeat=2))
point_to_idx = {p: i for i, p in enumerate(points)}

# Precompute squared distances
dist2 = [[0] * N2 for _ in range(N2)]
for i in range(N2):
    xi, yi = points[i]
    for j in range(i + 1, N2):
        xj, yj = points[j]
        dx = xi - xj
        dy = yi - yj
        d = dx * dx + dy * dy
        dist2[i][j] = d
        dist2[j][i] = d

def _is_iso_idx(i, j, k):
    d1 = dist2[i][j]
    d2 = dist2[i][k]
    d3 = dist2[j][k]
    return d1 == d2 or d1 == d3 or d2 == d3

# Precompute iso_with[i][j] as a bitset of k such that i, j, k form an isosceles triangle
iso_with = [[0] * N2 for _ in range(N2)]
for i in range(N2):
    for j in range(N2):
        if i == j:
            continue
        mask = 0
        for k in range(N2):
            if k == i or k == j:
                continue
            if _is_iso_idx(i, j, k):
                mask |= (1 << k)
        iso_with[i][j] = mask

def is_maximal(S_list):
    """
    S_list: list of indices in the set
    Returns True if S_list is a maximal no-isosceles set
    """
    S_bits = 0
    for idx in S_list:
        S_bits |= (1 << idx)
    
    # Check each point not in S
    for q in range(N2):
        if (S_bits >> q) & 1:
            continue
        
        blocked = False
        for a in S_list:
            if (iso_with[q][a] & S_bits):
                blocked = True
                break
        
        if not blocked:
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    best_size = float('inf')
    best_set = []
    
    def greedy_construct(order):
        chosen = []
        S_bits = 0
        for p in order:
            valid = True
            for a in chosen:
                if (iso_with[p][a] & S_bits):
                    valid = False
                    break
            if valid:
                chosen.append(p)
                S_bits |= (1 << p)
        return chosen

    def shrink(S_list):
        """Try removing single points while maintaining maximality."""
        S_list = list(S_list)
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            indices = list(range(len(S_list)))
            random.shuffle(indices)
            for idx in indices:
                p = S_list[idx]
                candidate = S_list[:idx] + S_list[idx + 1:]
                if is_maximal(candidate):
                    S_list = candidate
                    changed = True
                    break
        return S_list

    def shrink_pairs(S_list):
        """Try removing pairs of points to escape local minima."""
        S_list = list(S_list)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            n = len(S_list)
            if n < 2:
                break
            for _ in range(min(30, n * (n - 1) // 2)):
                i, j = random.sample(range(n), 2)
                candidate = [S_list[k] for k in range(n) if k != i and k != j]
                if is_maximal(candidate):
                    S_list = candidate
                    improved = True
                    break
        return S_list

    def local_search(S_list):
        """Combined local search: shrink, shrink_pairs, and swaps."""
        S_list = list(S_list)
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            
            # Try to remove single points
            new_S = shrink(S_list)
            if len(new_S) < len(S_list):
                S_list = new_S
                improved = True
                continue
            
            # Try to remove pairs
            new_S = shrink_pairs(S_list)
            if len(new_S) < len(S_list):
                S_list = new_S
                improved = True
                continue
            
            # Try swaps
            S_bits = 0
            for idx in S_list:
                S_bits |= (1 << idx)
            
            if len(S_list) == 0:
                break
                
            p = random.choice(S_list)
            q = random.randint(0, N2 - 1)
            while (S_bits >> q) & 1:
                q = random.randint(0, N2 - 1)
                
            S_minus_p_bits = S_bits ^ (1 << p)
            S_minus_p_list = [x for x in S_list if x != p]
            
            valid_swap = True
            for a in S_minus_p_list:
                if (iso_with[q][a] & S_minus_p_bits):
                    valid_swap = False
                    break
                    
            if valid_swap:
                new_S = S_minus_p_list + [q]
                new_S = shrink(new_S)
                new_S = shrink_pairs(new_S)
                if len(new_S) < len(S_list):
                    S_list = new_S
                    improved = True
                    break
        return S_list

    def simulated_annealing(S_list):
        """Simulated annealing to escape local minima."""
        S_list = list(S_list)
        current_size = len(S_list)
        best_local = S_list[:]
        best_local_size = current_size
        
        temp = 100.0
        cooling = 0.995
        
        while temp > 0.1 and time.time() < deadline - 15:
            S_bits = 0
            for idx in S_list:
                S_bits |= (1 << idx)
            
            # Random move: remove a point, add a random point
            if len(S_list) > 0:
                p = random.choice(S_list)
                S_minus_p_list = [x for x in S_list if x != p]
                S_minus_p_bits = S_bits ^ (1 << p)
                
                q = random.randint(0, N2 - 1)
                while (S_minus_p_bits >> q) & 1:
                    q = random.randint(0, N2 - 1)
                
                valid = True
                for a in S_minus_p_list:
                    if (iso_with[q][a] & S_minus_p_bits):
                        valid = False
                        break
                
                if valid:
                    new_S = S_minus_p_list + [q]
                    new_size = len(new_S)
                    delta = new_size - current_size
                    
                    # Accept if better or with probability exp(-delta/temp)
                    if delta <= 0 or random.random() < math.exp(-delta / temp):
                        S_list = new_S
                        current_size = new_size
                        if current_size < best_local_size:
                            best_local_size = current_size
                            best_local = S_list[:]
                
                # Try removing a point (if maximal)
                if len(S_list) > 0 and random.random() < 0.3:
                    p = random.choice(S_list)
                    candidate = [x for x in S_list if x != p]
                    if is_maximal(candidate):
                        S_list = candidate
                        current_size = len(S_list)
                        if current_size < best_local_size:
                            best_local_size = current_size
                            best_local = S_list[:]
            
            temp *= cooling
        
        return best_local

    # Run multiple restarts with different strategies
    restart_count = 0
    while time.time() < deadline - 20:
        restart_count += 1
        strategy = random.random()
        
        if strategy < 0.25:
            # Random order
            order = list(range(N2))
            random.shuffle(order)
        elif strategy < 0.5:
            # Prefer corners and edges (points with fewer isosceles relationships)
            # Compute a heuristic score for each point
            scores = [0] * N2
            for i in range(N2):
                # Count how many other points form isosceles triangles with i
                # This is a proxy for "conflict"
                conflict = 0
                for j in range(N2):
                    if i != j:
                        # Check if i and j have many common isosceles partners
                        pass
                # Use distance from center as a simple heuristic
                x, y = points[i]
                cx, cy = N // 2 - 0.5, N // 2 - 0.5
                scores[i] = abs(x - cx) + abs(y - cy)
            
            order = sorted(range(N2), key=lambda i: (-scores[i], random.random()))
        elif strategy < 0.75:
            # Spiral from center
            cx, cy = N // 2, N // 2
            order = sorted(range(N2), key=lambda i: (abs(points[i][0] - cx) + abs(points[i][1] - cy), random.random()))
        else:
            # Random with some structure
            order = list(range(N2))
            random.shuffle(order)
        
        S = greedy_construct(order)
        
        # Apply local search
        S = local_search(S)
        
        # Apply simulated annealing
        S = simulated_annealing(S)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S[:]
            
            # Final polish
            if time.time() < deadline - 10:
                best_set = shrink(best_set)
                best_set = shrink_pairs(best_set)
                if len(best_set) < best_size:
                    best_size = len(best_set)
        
        # If we've found a very good solution, do more local search
        if best_size <= 20 and time.time() < deadline - 30:
            for _ in range(3):
                if time.time() >= deadline - 30:
                    break
                S2 = simulated_annealing(best_set)
                if len(S2) < best_size:
                    best_size = len(S2)
                    best_set = S2[:]
                    best_set = shrink(best_set)
                    best_set = shrink_pairs(best_set)
                    if len(best_set) < best_size:
                        best_size = len(best_set)
    
    # Convert indices back to points
    return [points[i] for i in best_set]

# EVOLVE-BLOCK-END
