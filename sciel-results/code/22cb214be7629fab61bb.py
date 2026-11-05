import time
import random
import math
import sys

def construct():
    """
    Find a 5-coloring of K_n with no monochromatic C4, maximizing n.
    
    Uses fast local search with:
    1. Precomputed edge index mapping
    2. Bitset-based common neighbor counting for fast C4 detection
    3. Deadline guard to return best valid result before timeout
    """
    deadline = time.time() + 115  # 115 seconds, leaving 5s margin
    
    def time_left():
        return deadline - time.time()
    
    def total_c4s(colors, n):
        """Count total C4s across all colors using bitsets."""
        # For each color, track adjacency as bitmasks
        adj = [[0] * 5 for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[i][c] |= (1 << j)
                adj[j][c] |= (1 << i)
                idx += 1
        total = 0
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    common = adj[i][c] & adj[j][c]
                    # Count bits >= 2
                    cnt = bin(common).count('1')
                    if cnt >= 2:
                        total += cnt - 1
        return total
    
    def is_valid(colors, n):
        """Check if coloring has no monochromatic C4."""
        return total_c4s(colors, n) == 0
    
    def edge_idx(i, j, n):
        """Compute linear index for edge (i,j) with i < j."""
        # Number of edges before row i: sum_{k=0}^{i-1} (n-k-1) = i*(n-1) - i*(i-1)//2
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    # Try to find a valid coloring for increasing n
    best_n = 1
    best_colors = [0]  # trivially valid for n=1
    
    # Start from n=8 and go up to 26
    for target_n in range(8, 27):
        if time_left() < 8:
            break
        
        n = target_n
        num_edges = n * (n - 1) // 2
        
        # Precompute edge list for faster random access
        edge_list = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_list.append((i, j, edge_idx(i, j, n)))
        
        # Multiple restarts with hill climbing + simulated annealing
        found = False
        max_restarts = 50
        
        for restart in range(max_restarts):
            if time_left() < 5:
                break
            
            # Random initialization
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            energy = total_c4s(colors, n)
            
            if energy == 0:
                best_n = n
                best_colors = colors[:]
                found = True
                break
            
            best_energy = energy
            best_colors_local = colors[:]
            
            # Simulated annealing
            temp = 3.0
            cooling = 0.9998
            min_temp = 0.01
            max_steps = 30000
            stuck_count = 0
            max_stuck = 5000
            
            for step in range(max_steps):
                if time_left() < 3:
                    break
                if temp < min_temp:
                    break
                if stuck_count > max_stuck:
                    break
                
                # Pick random edge
                ei, i, j = edge_list[random.randint(0, num_edges - 1)]
                old_color = colors[ei]
                new_color = random.randint(0, 4)
                if new_color == old_color:
                    continue
                
                colors[ei] = new_color
                new_energy = total_c4s(colors, n)
                delta = new_energy - energy
                
                if delta <= 0 or random.random() < math.exp(-delta / temp):
                    energy = new_energy
                    if energy < best_energy:
                        best_energy = energy
                        best_colors_local = colors[:]
                        stuck_count = 0
                        if energy == 0:
                            best_n = n
                            best_colors = colors[:]
                            found = True
                            break
                else:
                    colors[ei] = old_color
                    stuck_count += 1
                
                temp *= cooling
            
            if found:
                break
            
            # If we found a better but not zero solution, keep it
            if best_energy < float('inf'):
                pass
        
        if found:
            continue
    
    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    result = construct()
    print(f"Found n = {result['n']}")
