"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Returns a tournament on n vertices where every set of 4 vertices has a common dominator.
    Primary: Paley tournament on p=67 (known valid).
    Secondary: attempts a local search for smaller n (31-35) within a strict short time budget.
    """
    start_time = time.time()
    # Safety margin: Return well before 120s kill.
    # We want to spend max 10s on search, so total time < 15s.
    search_deadline = start_time + 10.0 

    # 1. Precompute the known valid Paley tournament on 67 vertices
    p_best = 67
    squares_best = [False] * p_best
    for x in range(1, p_best):
        squares_best[(x * x) % p_best] = True

    orientation_best = []
    for i in range(p_best):
        for j in range(i + 1, p_best):
            diff = (j - i) % p_best
            if squares_best[diff]:
                orientation_best.append(0)
            else:
                orientation_best.append(1)

    best_result = {'n': p_best, 'orientation': orientation_best}
    best_n_found = p_best

    # 2. Local search for smaller n
    # We will try a few small n values. If we don't find a solution quickly, we return n=67.
    
    def build_adjacency(n, orientation):
        """Build out-neighborhood bitsets from orientation list."""
        out_neighbors = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                bit = orientation[idx]
                idx += 1
                if bit == 0:
                    out_neighbors[i] |= (1 << j)
                else:
                    out_neighbors[j] |= (1 << i)
        return out_neighbors

    def count_uncovered_fast(out_neighbors, n):
        """
        Count 4-sets without a common dominator.
        Uses in-neighborhood intersection logic.
        D(x) = {v : v -> x}.
        4-set {a,b,c,d} is covered if D(a) & D(b) & D(c) & D(d) != 0.
        """
        # Build in-neighborhoods
        in_neighbors = [0] * n
        for v in range(n):
            on = out_neighbors[v]
            temp = on
            while temp:
                lb = temp & (-temp)
                idx = lb.bit_length() - 1
                in_neighbors[idx] |= (1 << v)
                temp ^= lb
        
        uncovered = 0
        for a in range(n):
            for b in range(a + 1, n):
                ab = in_neighbors[a] & in_neighbors[b]
                if ab == 0:
                    # No vertex dominates both a and b
                    remaining = n - b - 1
                    if remaining >= 2:
                        uncovered += remaining * (remaining - 1) // 2
                else:
                    for c in range(b + 1, n):
                        abc = ab & in_neighbors[c]
                        if abc == 0:
                            remaining = n - c - 1
                            if remaining >= 1:
                                uncovered += remaining
                        else:
                            for d in range(c + 1, n):
                                if (abc & in_neighbors[d]) == 0:
                                    uncovered += 1
        return uncovered

    def random_orientation(n):
        m = n * (n - 1) // 2
        return [random.randint(0, 1) for _ in range(m)]

    # Try searching for n=31, 32, 33, 34, 35
    # Use a very fast local search: first-improvement, limited steps
    for n in [31, 32, 33, 34, 35]:
        if time.time() > search_deadline:
            break
            
        # Time budget for this n
        n_deadline = time.time() + 2.0 # 2 seconds per n
        
        # Start with a random orientation
        orient = random_orientation(n)
        out_neigh = build_adjacency(n, orient)
        fitness = count_uncovered_fast(out_neigh, n)
        
        if fitness == 0:
            return {'n': n, 'orientation': orient}
        
        best_orient = orient[:]
        best_fitness = fitness
        m = n * (n - 1) // 2
        
        # Simple local search: try random flips
        steps = 0
        max_steps = 5000 # Limit steps to ensure we don't hang
        while steps < max_steps and time.time() < n_deadline:
            steps += 1
            # Pick a random bit to flip
            idx = random.randint(0, m - 1)
            old_bit = orient[idx]
            orient[idx] = 1 - old_bit
            
            out_neigh = build_adjacency(n, orient)
            new_fitness = count_uncovered_fast(out_neigh, n)
            
            if new_fitness < fitness:
                fitness = new_fitness
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_orient = orient[:]
                    if fitness == 0:
                        return {'n': n, 'orientation': orient}
            else:
                # Revert
                orient[idx] = old_bit
                out_neigh = build_adjacency(n, orient)
                
        if best_fitness == 0:
            return {'n': n, 'orientation': best_orient}

    # If no smaller n found, return the best known (67)
    return best_result

# EVOLVE-BLOCK-END
