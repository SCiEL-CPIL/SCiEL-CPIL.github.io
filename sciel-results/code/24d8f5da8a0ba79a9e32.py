import time
import random
import math
import sys

def construct():
    """
    Find a 5-coloring of K_n with no monochromatic C4, maximizing n.
    
    Uses a simulated annealing approach with bitset-based C4 counting.
    """
    deadline = time.time() + 115  # 115 seconds, leaving 5s margin
    
    def time_left():
        return deadline - time.time()
    
    def validate(colors, n):
        """Check if no color class contains a C4 using bitsets."""
        # adj[c][i] is a bitmask of neighbors of i in color c
        adj = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        for c in range(5):
            for i in range(n):
                for j in range(i+1, n):
                    # Count common neighbors
                    common = adj[c][i] & adj[c][j]
                    if bin(common).count('1') >= 2:
                        return False
        return True

    def count_c4s(colors, n):
        """Count total number of C4s using bitsets."""
        adj = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        total = 0
        for c in range(5):
            for i in range(n):
                for j in range(i+1, n):
                    common = adj[c][i] & adj[c][j]
                    cnt = bin(common).count('1')
                    if cnt >= 2:
                        total += cnt - 1
        return total

    def get_idx(i, j, n):
        if i > j:
            i, j = j, i
        return i * n - i * (i + 1) // 2 + (j - i - 1)

    best_n = 1
    best_colors = [0]
    
    # Try to find a valid coloring for increasing n
    # We prioritize n=26
    for target_n in [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8]:
        if time_left() < 5:
            break
            
        n = target_n
        num_edges = n * (n - 1) // 2
        
        found = False
        max_attempts = 50 if n == 26 else 100
        
        for attempt in range(max_attempts):
            if time_left() < 3:
                break
                
            # Random initialization
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            
            # Check if valid
            if validate(colors, n):
                best_n = n
                best_colors = colors[:]
                found = True
                break
            
            # Simulated Annealing / Local Search
            energy = count_c4s(colors, n)
            best_energy = energy
            best_local_colors = colors[:]
            
            temp = 1.0
            cooling = 0.999
            min_temp = 0.01
            steps = 20000 if n >= 25 else 10000
            
            for step in range(steps):
                if time_left() < 3:
                    break
                
                # Pick random edge
                i = random.randint(0, n - 2)
                j = random.randint(i + 1, n - 1)
                ei = get_idx(i, j, n)
                
                old_c = colors[ei]
                new_c = random.randint(0, 4)
                if new_c == old_c:
                    continue
                
                # Update bitsets for delta energy? 
                # For simplicity, recompute energy. 
                # Optimization: Only recompute if we accept?
                # No, we need delta.
                # Recomputing count_c4s is O(n^2 * 5). 
                # For n=26, 26^2 * 5 = 3380 ops. 
                # 20000 steps * 3380 = 67M ops. Feasible in 120s?
                # Python is slow. 67M ops might take 5-10s.
                # So 50 attempts * 10s = 500s. Too slow.
                
                # We need a faster delta update or fewer steps.
                # Let's reduce steps and rely on more attempts.
                
                colors[ei] = new_c
                new_energy = count_c4s(colors, n)
                delta = new_energy - energy
                
                if delta <= 0 or random.random() < math.exp(-delta / temp):
                    energy = new_energy
                    if energy < best_energy:
                        best_energy = energy
                        best_local_colors = colors[:]
                        if energy == 0:
                            break
                else:
                    colors[ei] = old_c
                
                temp *= cooling
                if temp < min_temp:
                    break
            
            if best_energy == 0:
                best_n = n
                best_colors = best_local_colors[:]
                found = True
                break
        
        if found:
            break

    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    result = construct()
    print(f"Found n = {result['n']}")
