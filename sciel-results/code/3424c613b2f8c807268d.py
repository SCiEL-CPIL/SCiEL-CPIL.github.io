import time
import random

def construct():
    N = 128
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # Precompute APs ending at each index x (1-indexed)
    # APs are (i, j, k, x) such that i, j, k, x form a 4-AP
    # i = x - 3d, j = x - 2d, k = x - d
    aps_ending = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for d in range(1, (x - 1) // 3 + 1):
            i = x - 3 * d
            if i >= 1:
                j = i + d
                k = i + 2 * d
                aps_ending[x].append((i, j, k))
    
    def solve(k):
        # colors[1..N], colors[0] is unused or dummy
        colors = [None] * (N + 1)
        
        # Symmetry breaking: c[1] = 0
        colors[1] = 0
        
        # We can also try to fix c[2] if it helps, but let's start with just c[1]
        
        def backtrack(x):
            if time.time() > deadline:
                return None
            
            if x > N:
                return colors[1:].copy()
            
            # Try each color
            # Heuristic: try colors 0..k-1
            # We can shuffle if we want, but deterministic is fine for now
            # To break symmetry further, if x==2, we can restrict? 
            # No, c[2] can be 0 or 1 (if k>=2).
            
            for c in range(k):
                # Check constraints for APs ending at x
                valid = True
                for (i, j, k_idx) in aps_ending[x]:
                    ci = colors[i]
                    cj = colors[j]
                    ck = colors[k_idx]
                    # ci, cj, ck are assigned
                    # Check if {ci, cj, ck, c} has >= 3 distinct colors
                    # This is equivalent to: NOT (all same) AND NOT (exactly 2 distinct)
                    # Faster check:
                    # Count distinct in {ci, cj, ck, c}
                    # If len(set([ci, cj, ck, c])) < 3: invalid
                    # Optimization: 
                    # If ci == cj == ck == c: invalid (1 distinct)
                    # If two are same and the other two are same (but different from first pair): invalid (2 distinct)
                    # If three are same and one is different: invalid (2 distinct)
                    
                    # Simple set check is fast enough for small N
                    if len({ci, cj, ck, c}) < 3:
                        valid = False
                        break
                
                if valid:
                    colors[x] = c
                    res = backtrack(x + 1)
                    if res is not None:
                        return res
                    colors[x] = None
            
            return None
        
        return backtrack(2)

    # Try k=3
    # We might need to run multiple attempts with different random seeds or orderings if it's hard
    # But let's try deterministic first.
    
    # To handle cases where k=3 is hard or impossible, we timebox it.
    # If k=3 doesn't finish in 40s, try k=4.
    
    # Actually, let's just run the solver. If it's slow, we might need to optimize.
    # The number of APs ending at x is small (max ~42).
    # Backtracking depth 128.
    # If k=3 is possible, it might be found quickly. If not, it might run forever.
    # So we MUST timebox.
    
    def try_solve(k, timeout):
        start = time.time()
        # We can modify the global deadline for this specific solve attempt
        # Or pass deadline as argument
        # Let's create a closure or pass deadline
        pass
        
    # Let's restructure to allow timeout per k
    # We'll use a global variable for deadline that we update
    
    best_solution = None
    
    # Try k=3
    deadline_k3 = start_time + 40.0
    # We need a way to stop the backtrack if time runs out.
    # The backtrack function checks `time.time() > deadline`.
    # We need `deadline` to be accessible.
    
    # Let's define a solver class or use nonlocal
    # For simplicity, we'll just run it and hope it's fast or hits the deadline.
    
    # Wait, if k=3 is impossible, backtrack will explore the whole tree (or until deadline).
    # If k=4 is the answer, we need to try k=4.
    
    # Let's try k=3 first.
    # We'll use a simple flag to stop if time exceeds.
    
    # To make it robust, we'll try k=3 for 30s, then k=4 for 30s, then k=5 for 10s.
    
    # We need to be careful with the `deadline` variable in the closure.
    # Let's pass it as an argument to backtrack.
    
    def solve_with_deadline(k, dl):
        colors = [None] * (N + 1)
        colors[1] = 0
        
        def backtrack(x):
            if time.time() > dl:
                return None
            
            if x > N:
                return colors[1:].copy()
            
            for c in range(k):
                valid = True
                for (i, j, k_idx) in aps_ending[x]:
                    ci = colors[i]
                    cj = colors[j]
                    ck = colors[k_idx]
                    if len({ci, cj, ck, c}) < 3:
                        valid = False
                        break
                
                if valid:
                    colors[x] = c
                    res = backtrack(x + 1)
                    if res is not None:
                        return res
                    colors[x] = None
            
            return None
        
        return backtrack(2)

    # Try k=3
    sol = solve_with_deadline(3, start_time + 40.0)
    if sol:
        # Map colors to 0..k-1 to ensure minimal label count representation?
        # The problem says "minimize the number of labels actually used".
        # If we used colors {0,1,2}, count is 3.
        # If we used {0,1,5}, count is 3.
        # The return value is the list of labels.
        # The evaluator counts distinct labels.
        # So we can just return the list as is, or map to 0..max.
        # Mapping to 0..max is cleaner.
        used = sorted(set(sol))
        mapping = {c: i for i, c in enumerate(used)}
        return [mapping[c] for c in sol]
        
    # Try k=4
    sol = solve_with_deadline(4, start_time + 80.0)
    if sol:
        used = sorted(set(sol))
        mapping = {c: i for i, c in enumerate(used)}
        return [mapping[c] for c in sol]
        
    # Fallback: return a valid solution with more colors if found?
    # Or just return the seed solution?
    # The seed solution is valid.
    # We should return the best found.
    # If no solution found, we must return a valid one.
    # The seed program produces a valid one.
    # Let's generate the seed solution as fallback.
    
    # Regenerate seed logic
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color)
                break
    return colors
