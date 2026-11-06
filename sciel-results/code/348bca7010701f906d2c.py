import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def count_violations(n, colors):
        """
        Counts the number of pairs of vertices (u, v) in each color class
        that have at least 2 common neighbors.
        Returns total violations.
        """
        neighbors = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
                idx += 1
        
        cost = 0
        for c in range(5):
            for u in range(n):
                nu = neighbors[c][u]
                if nu == 0:
                    continue
                for v in range(u + 1, n):
                    nv = neighbors[c][v]
                    common = nu & nv
                    if common & (common - 1):
                        cost += 1
        return cost

    def get_seed_solution():
        # Fallback seed, n=11
        n = 11
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = min(j - i, n - (j - i))
                colors.append(d - 1)
        return n, colors

    best_n, best_colors = get_seed_solution()

    def solve_local_search(n, time_limit):
        """
        Simulated Annealing local search.
        """
        total_edges = n * (n - 1) // 2
        # Initialize with random colors
        colors = [random.randint(0, 4) for _ in range(total_edges)]
        
        color_matrix = [[-1] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                color_matrix[i][j] = c
                color_matrix[j][i] = c
                idx += 1
        
        neighbors = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
                idx += 1
        
        def get_cost():
            cost = 0
            for c in range(5):
                for u in range(n):
                    nu = neighbors[c][u]
                    if nu == 0:
                        continue
                    for v in range(u + 1, n):
                        nv = neighbors[c][v]
                        common = nu & nv
                        if common & (common - 1):
                            cost += 1
            return cost
        
        current_cost = get_cost()
        if current_cost == 0:
            return colors
        
        search_start = time.time()
        current_temp = 10.0
        cooling_rate = 0.995
        min_temp = 0.01
        
        # Precompute edge indices for fast access
        # idx(i, j) = i * (2*n - i - 1) // 2 + j - i - 1
        
        while time.time() - search_start < time_limit and current_temp > min_temp:
            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            c_old = color_matrix[i][j]
            
            # Pick a new color (different from old)
            c_new = random.randint(0, 4)
            if c_new == c_old:
                c_new = (c_new + 1) % 5
            
            # Calculate delta cost
            delta = 0
            
            # Cost change for removing edge (i,j) from c_old
            # We need to check if removing it reduces violations for pairs involving i or j
            # Actually, the cost function counts pairs (u,v) with >=2 common neighbors.
            # Changing color of (i,j) affects:
            # 1. The set of common neighbors for pairs (i, w) and (j, w) in c_old and c_new.
            
            # This incremental update is complex to get right quickly.
            # For N=26, full recalculation is O(N^2 * 5) = 26^2 * 5 ~ 3000 ops.
            # 100k steps in 10s -> 300M ops. Python is slow (~10M ops/s).
            # So 300M ops will take 30s. Too slow.
            # Must use incremental update.
            
            # Let's use the incremental update from Inspiration 1, which was efficient.
            # It calculates delta by checking neighbors of i and j.
            
            # Remove (i,j) from c_old
            # For any w, if w is neighbor of i in c_old, and w is neighbor of j in c_old...
            # Actually, the violation is defined on pairs of vertices (u,v).
            # A violation exists if |N(u) ∩ N(v)| >= 2.
            
            # Let's stick to the Inspiration 1 logic for delta, as it worked.
            
            # c_old removals
            for w in range(n):
                if w == i or w == j:
                    continue
                if neighbors[c_old][j] & (1 << w):
                    # i and w are connected in c_old? No, we are removing (i,j).
                    # This logic in Inspiration 1 was:
                    # if neighbors[c_old][j] & (1 << w): # j and w connected in c_old
                    #    k = bin(neighbors[c_old][i] & neighbors[c_old][w]).count('1')
                    #    delta -= (k - 1)
                    # This seems to be counting how many common neighbors i and w have, minus 1?
                    # This logic is tricky. I will use a safer, slightly slower incremental check
                    # or just full recalculation if N is small enough.
                    # For N=26, full recalculation might be acceptable if we limit steps.
                    # But SA needs many steps.
                    
                    # Let's try a different approach: 
                    # Just do full recalculation every K steps? No.
                    
                    # I will implement the incremental delta carefully.
                    # Cost = sum_{u < v} [ |N(u) ∩ N(v)| >= 2 ]
                    
                    # When we change color of (i,j) from c_old to c_new:
                    # 1. In c_old, (i,j) is removed.
                    #    This affects pairs (i, w) and (j, w) for all w.
                    #    For pair (i, w): N(i) loses j. N(w) unchanged (unless w=j, but w!=j).
                    #    Common neighbors of i and w: previously C_old. Now C_old \ {j} (if j was in C_old).
                    #    j is in N(i) and N(w) if (j,w) is in c_old.
                    #    So if (j,w) in c_old, then j was a common neighbor.
                    #    If |C_old| >= 2, and we remove one, it might drop to 1 (no violation) or stay >=2.
                    
                    # This is too complex to debug in short time.
                    # I will use the "Full Recalculation" but optimize it with NumPy?
                    # NumPy is allowed.
                    
                    pass # Skip incremental, use full recalc with NumPy if possible?
            
            # Given the complexity and time, I will fall back to a simpler, robust local search
            # that uses full recalculation but is highly optimized in Python using bit operations.
            # For N=26, full recalc is ~ 5 * (26 * 25 / 2) = 1625 bit-and-ops.
            # Very fast.
            # 100,000 steps * 1625 ops = 160M ops.
            # Python can do ~10-50M simple ops/sec.
            # So 100k steps might take 3-15 seconds.
            # This is acceptable for a few restarts.
            
            # Let's implement a simple Hill Climbing with Full Recalc.
            # It is less likely to get stuck than SA if we add random restarts.
            
            # I will replace the inner loop with a full recalculation.
            pass

    # Strategy:
    # 1. Try n=26 with multiple restarts of Hill Climbing (Full Recalc).
    # 2. If fail, try n=25, etc.
    
    targets = [26, 25, 24, 23, 22, 21]
    
    for n in targets:
        if time.time() > deadline - 5:
            break
        
        # Allocate time based on n
        # 26 gets 50s, 25 gets 20s, etc.
        if n == 26:
            t_limit = 50.0
        elif n == 25:
            t_limit = 20.0
        elif n == 24:
            t_limit = 10.0
        else:
            t_limit = 5.0
            
        t_limit = min(t_limit, deadline - time.time() - 2)
        if t_limit < 1.0:
            break
            
        # Run multiple restarts
        found = False
        search_start = time.time()
        while time.time() - search_start < t_limit:
            # Random init
            total_edges = n * (n - 1) // 2
            colors = [random.randint(0, 4) for _ in range(total_edges)]
            
            # Build neighbor bitsets
            neighbors = [[0] * n for _ in range(5)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx]
                    neighbors[c][i] |= (1 << j)
                    neighbors[c][j] |= (1 << i)
                    idx += 1
            
            def get_cost():
                cost = 0
                for c in range(5):
                    for u in range(n):
                        nu = neighbors[c][u]
                        if nu == 0: continue
                        for v in range(u + 1, n):
                            nv = neighbors[c][v]
                            common = nu & nv
                            if common & (common - 1):
                                cost += 1
                return cost
            
            current_cost = get_cost()
            if current_cost == 0:
                found = True
                break
            
            # Hill Climbing
            stagnant = 0
            while time.time() - search_start < t_limit:
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j: continue
                if i > j: i, j = j, i
                
                c_old = colors[i * (2 * n - i - 1) // 2 + j - i - 1]
                c_new = random.randint(0, 4)
                if c_new == c_old: continue
                
                # Apply move
                idx = i * (2 * n - i - 1) // 2 + j - i - 1
                neighbors[c_old][i] &= ~(1 << j)
                neighbors[c_old][j] &= ~(1 << i)
                neighbors[c_new][i] |= (1 << j)
                neighbors[c_new][j] |= (1 << i)
                colors[idx] = c_new
                
                new_cost = get_cost()
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    stagnant = 0
                    if current_cost == 0:
                        found = True
                        break
                else:
                    # Revert
                    neighbors[c_new][i] &= ~(1 << j)
                    neighbors[c_new][j] &= ~(1 << i)
                    neighbors[c_old][i] |= (1 << j)
                    neighbors[c_old][j] |= (1 << i)
                    colors[idx] = c_old
                    stagnant += 1
                    
                if stagnant > 100:
                    break # Restart
            
            if found:
                break
        
        if found:
            best_n = n
            best_colors = colors
            break
            
    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    res = construct()
    print(f"n={res['n']}")
