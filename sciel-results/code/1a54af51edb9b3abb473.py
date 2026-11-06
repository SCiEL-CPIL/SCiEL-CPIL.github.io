import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin for safety

    def popcount(x):
        return bin(x).count('1')

    def get_idx(i, j, n):
        # Lexicographic order: (0,1), (0,2), ..., (0,n-1), (1,2), ...
        # Index for (i,j) with i<j
        # Number of edges before i: i * (2*n - i - 1) // 2
        # Offset within i: j - i - 1
        return i * (2 * n - i - 1) // 2 + (j - i - 1)

    def build_adjacency(n, colors):
        N = [[0] * n for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[get_idx(i, j, n)]
                N[c][i] |= (1 << j)
                N[c][j] |= (1 << i)
        return N

    def count_violations(n, colors, N):
        violations = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[get_idx(i, j, n)]
                if popcount(N[c][i] & N[c][j]) >= 2:
                    violations += 1
        return violations

    def is_violating(u, v, c, N):
        return popcount(N[c][u] & N[c][v]) >= 2

    def sa_search(n, max_time=5.0, seed=None):
        if seed is not None:
            random.seed(seed)
        
        n_edges = n * (n - 1) // 2
        if n_edges == 0:
            return [], 0
            
        # Initialize
        best_colors = None
        best_violations = float('inf')
        
        # Structured init: random
        colors = [random.randint(0, 4) for _ in range(n_edges)]
        N = build_adjacency(n, colors)
        violations = count_violations(n, colors, N)
        
        if violations == 0:
            return colors, 0
            
        T = 1.0
        cooling = 0.995
        current_time = time.time()
        
        # Precompute edge list indices for faster access?
        # Just using i, j is fine.
        
        while time.time() - current_time < max_time:
            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            idx = get_idx(i, j, n)
            c_old = colors[idx]
            
            # Pick new color
            c_new = random.randint(0, 4)
            if c_new == c_old:
                continue
            
            # Calculate Delta
            delta = 0
            
            # 1. Edge (i, j) itself
            old_viol_ij = 1 if is_violating(i, j, c_old, N) else 0
            
            # New status for (i, j) with color c_new:
            # Common neighbors in c_new: N[c_new][i] & N[c_new][j]
            new_viol_ij = 1 if popcount(N[c_new][i] & N[c_new][j]) >= 2 else 0
            
            delta += new_viol_ij - old_viol_ij
            
            # 2. Edges (i, w) for w != j
            for w in range(n):
                if w == i or w == j:
                    continue
                idx_ew = get_idx(min(i, w), max(i, w), n)
                c_ew = colors[idx_ew]
                
                if c_ew == c_old:
                    # N[c_old][i] loses j.
                    old_common = N[c_old][i] & N[c_old][w]
                    old_viol = 1 if popcount(old_common) >= 2 else 0
                    new_common = old_common & ~(1 << j)
                    new_viol = 1 if popcount(new_common) >= 2 else 0
                    delta += new_viol - old_viol
                    
                elif c_ew == c_new:
                    # N[c_new][i] gains j.
                    old_common = N[c_new][i] & N[c_new][w]
                    old_viol = 1 if popcount(old_common) >= 2 else 0
                    if N[c_new][w] & (1 << j):
                        new_common = old_common | (1 << j)
                    else:
                        new_common = old_common
                    new_viol = 1 if popcount(new_common) >= 2 else 0
                    delta += new_viol - old_viol

            # 3. Edges (j, w) for w != i
            for w in range(n):
                if w == i or w == j:
                    continue
                idx_ew = get_idx(min(j, w), max(j, w), n)
                c_ew = colors[idx_ew]
                
                if c_ew == c_old:
                    # N[c_old][j] loses i.
                    old_common = N[c_old][j] & N[c_old][w]
                    old_viol = 1 if popcount(old_common) >= 2 else 0
                    new_common = old_common & ~(1 << i)
                    new_viol = 1 if popcount(new_common) >= 2 else 0
                    delta += new_viol - old_viol
                    
                elif c_ew == c_new:
                    # N[c_new][j] gains i.
                    old_common = N[c_new][j] & N[c_new][w]
                    old_viol = 1 if popcount(old_common) >= 2 else 0
                    if N[c_new][w] & (1 << i):
                        new_common = old_common | (1 << i)
                    else:
                        new_common = old_common
                    new_viol = 1 if popcount(new_common) >= 2 else 0
                    delta += new_viol - old_viol

            # Acceptance
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                # Apply move
                N[c_old][i] &= ~(1 << j)
                N[c_old][j] &= ~(1 << i)
                N[c_new][i] |= (1 << j)
                N[c_new][j] |= (1 << i)
                colors[idx] = c_new
                violations += delta
                
                if violations == 0:
                    return colors, 0
                    
                if violations < best_violations:
                    best_violations = violations
                    best_colors = colors[:]
            
            T *= cooling
            if T < 0.01:
                T = 0.01

        return best_colors, best_violations

    best_n = 11
    # Fallback coloring for n=11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try to find larger n
    # Start from 12 up to 26
    # We will try to solve for n=12, then 13, etc.
    # If we fail to find a valid coloring for n, we keep the best valid one found so far.
    
    # To maximize the chance of finding higher n, we should spend time on the current target n.
    # If we find n=12, we try n=13.
    
    for n in range(12, 27):
        if time.time() > deadline:
            break
            
        # Try a few restarts for this n
        # If we find a valid one, update best and move to next n.
        # If we don't, we might still have time to try again or move on?
        # Moving on is risky if n=12 is hard. But n=13 is harder.
        # Let's give 10s to each n.
        
        found_valid = False
        start_n_time = time.time()
        
        while time.time() - start_n_time < 10.0 and time.time() < deadline:
            # Run SA with a short time slice
            # We use different seeds implicitly by not resetting random state globally, 
            # but sa_search uses its own random calls.
            colors, viol = sa_search(n, max_time=2.0)
            
            if viol == 0:
                best_n = n
                best_colors = colors
                found_valid = True
                break
        
        if not found_valid:
            # If we couldn't find n, maybe we should try harder?
            # Or just accept that we might not reach this n.
            # Given the difficulty, if we fail 12, 13 is unlikely.
            # But let's not break immediately, maybe a longer run later?
            # No, time is limited.
            # If we fail n, we likely fail n+1.
            # So we can break?
            # Actually, if we fail 12, we might still have best_n=11.
            # Let's just continue the loop but with less time?
            # Or break?
            # If we break, we return 11.
            # If we continue, we might waste time on 13 which is harder.
            # Let's break if we fail to find a valid coloring for the current n.
            # This assumes monotonic difficulty.
            pass

    return {'n': best_n, 'colors': best_colors}
