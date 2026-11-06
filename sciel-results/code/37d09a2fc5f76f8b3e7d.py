import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin
    
    def fast_local_search(n, time_limit):
        """
        Fast local search using bitmasks for neighbor sets.
        """
        num_edges = n * (n - 1) // 2
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Initial random coloring
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        
        # neighbors[c][i] is a bitmask of neighbors of i in color c
        # n <= 29, so we need at least 29 bits. Python ints are arbitrary precision.
        neighbors = [[0] * n for _ in range(5)]
        
        for idx, (i, j) in enumerate(edges):
            c = colors[idx]
            neighbors[c][i] |= (1 << j)
            neighbors[c][j] |= (1 << i)
            
        def calc_violations():
            """
            Calculate total violations.
            Violation for pair (u,v) in color c is C(k,2) where k is common neighbors.
            """
            violations = 0
            for c in range(5):
                for u in range(n):
                    mask_u = neighbors[c][u]
                    for v in range(u + 1, n):
                        # Common neighbors
                        common = mask_u & neighbors[c][v]
                        k = common.bit_count()
                        if k >= 2:
                            violations += k * (k - 1) // 2
            return violations

        current_violations = calc_violations()
        if current_violations == 0:
            return colors
        
        best_violations = current_violations
        best_colors = colors[:]
        
        temperature = 1.0
        cooling_rate = 0.9995  # Slower cooling
        min_temp = 0.01
        
        # Precompute edge indices for quick lookup?
        # We just need to iterate edges.
        
        while time.time() < time_limit:
            # Pick a random edge
            e_idx = random.randint(0, num_edges - 1)
            i, j = edges[e_idx]
            old_color = colors[e_idx]
            
            # Pick a new color
            new_color = random.randint(0, 4)
            if new_color == old_color:
                continue
            
            # Calculate delta in violations
            delta = 0
            
            # 1. Remove (i,j) from old_color
            c = old_color
            # For each v in N_c(j) \ {i}, i and v lose common neighbor j?
            # No.
            # If we remove j from N_c(i):
            # For any v, if j is in N_c(v), then j is a common neighbor of i and v.
            # Removing j from N_c(i) means j is no longer a common neighbor of i and v.
            # So k_iv decreases by 1.
            # Change in C(k,2) is -(k-1).
            
            mask_j = neighbors[c][j]
            # Iterate over set bits in mask_j
            v = 0
            m = mask_j
            while m:
                if m & 1:
                    if v != i:
                        # Current k is bit_count(N_c(i) & N_c(v))
                        # This k includes j?
                        # Yes, because j is in N_c(i) (currently) and j is in N_c(v) (since v is in N_c(j)).
                        # So new k is k-1.
                        # Delta -= (k-1)
                        k = (neighbors[c][i] & neighbors[c][v]).bit_count()
                        delta -= (k - 1)
                m >>= 1
                v += 1
                
            # Similarly for v in N_c(i) \ {j}, j and v lose common neighbor i.
            mask_i = neighbors[c][i]
            v = 0
            m = mask_i
            while m:
                if m & 1:
                    if v != j:
                        k = (neighbors[c][j] & neighbors[c][v]).bit_count()
                        delta -= (k - 1)
                m >>= 1
                v += 1
            
            # Apply removal
            neighbors[c][i] &= ~(1 << j)
            neighbors[c][j] &= ~(1 << i)
            
            # 2. Add (i,j) to new_color
            c = new_color
            # For each v in N_c(j) \ {i}, i and v gain common neighbor j.
            # k_iv increases by 1.
            # Change in C(k,2) is +k.
            # k is the count BEFORE adding.
            
            mask_j = neighbors[c][j]
            v = 0
            m = mask_j
            while m:
                if m & 1:
                    if v != i:
                        k = (neighbors[c][i] & neighbors[c][v]).bit_count()
                        delta += k
                m >>= 1
                v += 1
                
            # For each v in N_c(i) \ {j}, j and v gain common neighbor i.
            mask_i = neighbors[c][i]
            v = 0
            m = mask_i
            while m:
                if m & 1:
                    if v != j:
                        k = (neighbors[c][j] & neighbors[c][v]).bit_count()
                        delta += k
                m >>= 1
                v += 1
            
            # Apply addition
            neighbors[c][i] |= (1 << j)
            neighbors[c][j] |= (1 << i)
            
            colors[e_idx] = new_color
            
            new_violations = current_violations + delta
            
            # Acceptance criterion
            if delta < 0 or random.random() < pow(2.71828, -delta / temperature):
                current_violations = new_violations
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_colors = colors[:]
                    if best_violations == 0:
                        return best_colors
            else:
                # Revert
                # Undo addition to new_color
                neighbors[c][i] &= ~(1 << j)
                neighbors[c][j] &= ~(1 << i)
                
                # Undo removal from old_color (i.e., add back)
                c = old_color
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
                
                colors[e_idx] = old_color
            
            temperature *= cooling_rate
            if temperature < min_temp:
                temperature = min_temp
                
        return None

    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try n=26 with multiple restarts
    # We have ~110s.
    # Let's try 26 with 50s. If fail, try 25 with 30s, etc.
    
    target_ns = [26, 25, 24, 23, 22, 21, 20]
    
    for n in target_ns:
        if time.time() > deadline - 5:
            break
        
        # Allocate time
        if n == 26:
            time_budget = 50.0
        elif n == 25:
            time_budget = 30.0
        else:
            time_budget = 10.0
            
        # Try a few restarts for the larger n
        restarts = 3 if n == 26 else 1
        found = False
        for r in range(restarts):
            remaining = time_limit = min(time_budget, deadline - time.time() - 5)
            if remaining <= 1:
                break
            res = fast_local_search(n, remaining)
            if res is not None:
                best_n = n
                best_colors = res
                found = True
                break
            time_budget -= remaining
            
        if found:
            break
            
    return {'n': best_n, 'colors': best_colors}
