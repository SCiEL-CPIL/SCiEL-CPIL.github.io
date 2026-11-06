import time
import sys

# Increase recursion limit just in case, though N=128 is small
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin
    
    N = 128
    
    # Precompute all 4-term arithmetic progressions
    # An AP is (a, d) -> positions a, a+d, a+2d, a+3d
    # 0-indexed: a in [0, N-1], d in [1, ...], a+3d < N
    aps = []
    pos_aps = [[] for _ in range(N)]
    
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(max_a):
            p1 = a
            p2 = a + d
            p3 = a + 2 * d
            p4 = a + 3 * d
            ap_idx = len(aps)
            aps.append((p1, p2, p3, p4))
            pos_aps[p1].append(ap_idx)
            pos_aps[p2].append(ap_idx)
            pos_aps[p3].append(ap_idx)
            pos_aps[p4].append(ap_idx)
    
    total_aps = len(aps)
    
    def solve(k):
        """
        Returns a list of colors of length N using k colors (0..k-1)
        such that every 4-AP has at least 3 distinct colors, or None.
        """
        if k < 3:
            return None
            
        colors = [-1] * N
        # domains[i] is a set of possible colors for position i
        domains = [set(range(k)) for _ in range(N)]
        
        # To speed up, we can pre-allocate or use lists.
        # For backtracking, we need to track state.
        
        # Heuristic: Order of variables.
        # We can just go 0 to N-1, but forward checking is key.
        
        # State for propagation:
        # For each AP, we track the set of colors used so far.
        # If size > 2, it's satisfied (for k=3, size must be 3).
        # If size == 2, the remaining variables MUST be the missing color(s).
        # If size == 1, remaining variables must include at least 2 other colors.
        # If size == 0, no constraint yet.
        
        # Actually, simpler: just check constraints on assignment.
        # With N=128 and ~2000 APs, checking all affected APs is fast.
        
        def is_valid_assignment(idx, col):
            # Check all APs involving idx
            for ap_idx in pos_aps[idx]:
                p1, p2, p3, p4 = aps[ap_idx]
                c1 = colors[p1] if p1 != idx else col
                c2 = colors[p2] if p2 != idx else col
                c3 = colors[p3] if p3 != idx else col
                c4 = colors[p4] if p4 != idx else col
                
                # If any are unassigned (-1), we can't fully validate yet,
                # but we can check for immediate contradictions.
                # Contradiction if:
                # 1. All assigned and distinct count < 3.
                # 2. Some assigned, some unassigned.
                #    Let assigned_colors = set of assigned values.
                #    Let num_unassigned = count of -1.
                #    If len(assigned_colors) + num_unassigned < 3: Impossible.
                #    If len(assigned_colors) == 3: Satisfied (regardless of unassigned).
                #    If len(assigned_colors) == 2: Unassigned MUST be the 3rd color.
                #       If num_unassigned == 1: The unassigned one is forced.
                #       If num_unassigned > 1: They must collectively provide the 3rd color.
                #    If len(assigned_colors) == 1: Unassigned must provide 2 more distinct colors.
                
                assigned = []
                unassigned_count = 0
                for c in (c1, c2, c3, c4):
                    if c == -1:
                        unassigned_count += 1
                    else:
                        assigned.append(c)
                        
                distinct_assigned = set(assigned)
                
                # Check 1: Can we ever reach 3 distinct colors?
                if len(distinct_assigned) + unassigned_count < 3:
                    return False
                
                # Check 2: If all assigned, is it valid?
                if unassigned_count == 0:
                    if len(distinct_assigned) < 3:
                        return False
                    # else valid for this AP
                
                # Check 3: Propagation hints (optional but helpful for pruning)
                # If distinct_assigned has 2 colors and unassigned_count == 1:
                # The last one is forced. We don't enforce it here, but we could.
                # For now, just return True if no contradiction.
                
            return True

        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == N:
                return True
            
            # Get possible colors for idx from domain
            # We iterate through 0..k-1
            # Optimization: Try colors that are most constrained? 
            # For now, just 0..k-1.
            
            # Forward Checking: 
            # Before assigning, we can prune domains of neighbors?
            # Given N=128, simple backtracking with validity check might suffice.
            
            for col in range(k):
                if col not in domains[idx]:
                    continue
                    
                # Tentative assignment
                colors[idx] = col
                domains[idx].discard(col) # Effectively used it
                
                if is_valid_assignment(idx, col):
                    if backtrack(idx + 1):
                        return True
                
                # Undo
                colors[idx] = -1
                domains[idx].add(col)
                
            return False
            
        # Initial check: Is it possible?
        # Just start backtracking
        return backtrack(0) if backtrack(0) else None

    # Try k=3
    # Note: The problem asks to minimize colors.
    # We know k>=3.
    
    # To be safe and fast, we can try to find *any* solution with 3 colors.
    # If it takes too long, we might need a better heuristic.
    # Given the structure, 3 colors might be impossible.
    
    # Let's try 3 colors first.
    # If it fails or times out, try 4.
    
    # However, backtracking can be slow if the first branch is a dead end.
    # A better approach: Randomized restarts or a more sophisticated CSP solver.
    # But let's stick to the structural change: CSP with backtracking.
    
    # To improve performance, we can order variables by "degree" (number of constraints).
    # Positions in the middle have more APs.
    # Let's create an order of indices to assign.
    
    # Actually, sequential 0..N-1 is fine if forward checking is strong.
    # My `is_valid_assignment` is weak (only checks immediate contradiction).
    
    # Let's enhance `is_valid_assignment` to do arc consistency or at least domain reduction.
    
    # Revised Solver with Domain Propagation
    
    def solve_with_propagation(k):
        colors = [-1] * N
        domains = [set(range(k)) for _ in range(N)]
        
        # Queue for propagation
        # We'll implement a simple recursive backtracking that reduces domains of neighbors.
        
        def get_ap_status(ap_idx):
            p1, p2, p3, p4 = aps[ap_idx]
            cs = [colors[p1], colors[p2], colors[p3], colors[p4]]
            assigned = [c for c in cs if c != -1]
            unassigned = [p for p, c in zip((p1,p2,p3,p4), cs) if c == -1]
            distinct = set(assigned)
            return distinct, unassigned, len(assigned)
            
        def propagate(idx, col):
            """
            Assign colors[idx] = col.
            Update domains of neighbors.
            Returns False if contradiction.
            """
            # We don't actually update domains globally in this simple version,
            # but we can check if any neighbor's domain becomes empty.
            
            for ap_idx in pos_aps[idx]:
                p1, p2, p3, p4 = aps[ap_idx]
                # Get current state of this AP
                c1 = colors[p1] if p1 != idx else col
                c2 = colors[p2] if p2 != idx else col
                c3 = colors[p3] if p3 != idx else col
                c4 = colors[p4] if p4 != idx else col
                
                assigned_vals = [c for c in (c1, c2, c3, c4) if c != -1]
                unassigned_pos = [p for p, c in zip((p1,p2,p3,p4), (c1,c2,c3,c4)) if c == -1]
                distinct_assigned = set(assigned_vals)
                
                # Check 1: Impossible to reach 3 distinct
                if len(distinct_assigned) + len(unassigned_pos) < 3:
                    return False, []
                
                # Check 2: If all assigned, must be >= 3
                if not unassigned_pos:
                    if len(distinct_assigned) < 3:
                        return False, []
                    continue
                
                # Check 3: Propagation
                # If distinct_assigned has 2 colors, the unassigned ones MUST be the 3rd color.
                if len(distinct_assigned) == 2:
                    missing_colors = set(range(k)) - distinct_assigned
                    # The unassigned positions can ONLY take colors from missing_colors?
                    # No, they must take *at least* one color from missing_colors to make the set size 3.
                    # Actually, if there is 1 unassigned position, it MUST be one of the missing colors.
                    # If there are >1 unassigned, at least one of them must be a missing color.
                    
                    if len(unassigned_pos) == 1:
                        p = unassigned_pos[0]
                        # p must be in missing_colors
                        # Intersect domain of p with missing_colors
                        new_domain = domains[p].intersection(missing_colors)
                        if not new_domain:
                            return False, []
                        # We should update domains[p]
                        # For simplicity in this recursive step, we just check validity.
                        # True domain propagation requires a queue.
                        pass
                    
                # If distinct_assigned has 1 color, unassigned must provide 2 more.
                # This is harder to prune locally without a full AC-3.
                
            return True, []

        # This simple propagation is weak. Let's rely on the speed of N=128.
        # 3^128 is huge, but constraints are tight.
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == N:
                return True
            
            # Heuristic: Pick the variable with the smallest domain?
            # Here domains are static sets 0..k-1 initially.
            # Let's just use idx.
            
            # Try colors
            # Order of colors: Maybe try the one that appears most often in neighbors?
            # For now, 0..k-1.
            
            for col in range(k):
                colors[idx] = col
                
                # Check constraints
                valid = True
                for ap_idx in pos_aps[idx]:
                    p1, p2, p3, p4 = aps[ap_idx]
                    c1 = colors[p1] if p1 != idx else col
                    c2 = colors[p2] if p2 != idx else col
                    c3 = colors[p3] if p3 != idx else col
                    c4 = colors[p4] if p4 != idx else col
                    
                    assigned = [c for c in (c1, c2, c3, c4) if c != -1]
                    unassigned_count = 4 - len(assigned)
                    distinct = set(assigned)
                    
                    if len(distinct) + unassigned_count < 3:
                        valid = False
                        break
                    if unassigned_count == 0 and len(distinct) < 3:
                        valid = False
                        break
                        
                if valid:
                    if backtrack(idx + 1):
                        return True
                
                colors[idx] = -1
                
            return False

        if backtrack(0):
            return colors
        return None

    # Try 3 colors
    # If it's too slow, we might need to fall back to 4.
    # But 3 is the target.
    
    # To handle potential slowness of 3-color search if it's unsatisfiable,
    # we can add a node limit or time check inside the recursion (already done).
    
    # Let's try 3.
    res = solve_with_propagation(3)
    
    if res is None:
        # Try 4
        res = solve_with_propagation(4)
        
    if res is None:
        # Fallback to seed-like greedy to ensure valid output (though score might be worse)
        # But seed uses up to 128 colors. We want to minimize.
        # If CSP fails, return a valid coloring with more colors.
        # Greedy construction:
        colors = []
        for x in range(1, N + 1):
            # Try colors 0..127
            # This is slow if we do it naively.
            # But we only get here if CSP failed.
            # Let's just return a known valid pattern if available, or fail.
            # Actually, a simple random or greedy might work.
            # For safety, return a 4-coloring if 3 failed?
            # If 4 also failed, something is wrong.
            
            # Let's try a simple deterministic 4-coloring:
            # c_i = i mod 4?
            # AP: a, a+d, a+2d, a+3d.
            # Colors: a%4, (a+d)%4, (a+2d)%4, (a+3d)%4.
            # If d=1: 0,1,2,3 (4 distinct) - OK.
            # If d=2: 0,2,0,2 (2 distinct) - FAIL.
            # So mod 4 doesn't work.
            
            # Let's just return the seed result if all else fails.
            colors = []
            for x in range(1, N + 1):
                for color in range(128):
                    if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                        and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                        colors.append(color);break
            return colors

    return res

# Run construct to test locally? No, just define it.
