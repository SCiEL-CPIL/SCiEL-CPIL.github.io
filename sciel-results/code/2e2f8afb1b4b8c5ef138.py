import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin for 60s limit
    
    n = 128
    # We want to minimize the number of distinct colors.
    # Let's try k=3, then k=4, etc.
    # The seed program uses up to 128 colors, but likely far fewer.
    # Based on "Minimize colors on [1,128]", we start with small k.
    
    # Precompute 4-term progressions
    # A 4-term progression is (a, a+d, a+2d, a+3d) with d > 0
    # Indices are 0-based in our array, but problem says positions 1..128
    # So position i corresponds to index i-1.
    # Progression: i, i+d, i+2d, i+3d
    # Condition: at least 3 distinct colors among these 4 positions.
    
    progressions = []
    for i in range(n):
        for d in range(1, n):
            j = i + d
            k = i + 2*d
            l = i + 3*d
            if l < n:
                progressions.append((i, j, k, l))
    
    # We need to check validity efficiently.
    # For a partial assignment, we can check if any completed progression violates the rule.
    # Also, we can use forward checking: when assigning a color to a position, 
    # check if it forces any future progression to be invalid.
    
    # A simpler but effective approach for n=128 with small k (likely 3 or 4):
    # Backtracking with MRV (Minimum Remaining Values) heuristic is hard to implement 
    # directly on a static graph without dynamic domain filtering.
    
    # Let's use a recursive backtracking that assigns colors to positions 0..n-1.
    # To speed it up, we can use symmetry breaking: 
    # The first position can be color 0.
    # The second position can be color 0 or 1.
    # This reduces the branching factor initially.
    
    # However, with k=3, the branching factor is 3. 3^128 is huge.
    # We need strong pruning.
    
    # Pruning rule:
    # If we have assigned colors to positions 0..i, we can check all progressions 
    # that are fully contained in 0..i. If any has < 3 distinct colors, prune.
    # Additionally, we can check "immediate" violations:
    # For any progression (a, b, c, d) where a,b,c are assigned and d is not,
    # if {color[a], color[b], color[c]} has size 1 or 2, then color[d] MUST be different 
    # from the set if size is 1, or different from one of them if size is 2?
    # Actually, if {A,B,C} has size 1 (all same), D must be different.
    # If {A,B,C} has size 2 (e.g., A=B != C), then D can be anything? 
    # No, the set {A,B,C,D} must have size >= 3.
    # If A=B=C, D != A.
    # If A=B!=C, then {A,C} are the colors. D can be A, B, or C. 
    # If D=A, set is {A,C} (size 2) -> Invalid.
    # If D=C, set is {A,C} (size 2) -> Invalid.
    # So if A=B!=C, D must be a THIRD color.
    # If A!=B!=C and A!=C, set is already size 3. D can be anything.
    
    # This logic allows us to restrict the domain of the next variable or detect conflicts early.
    
    # Let's implement a recursive backtracking with this constraint propagation.
    
    best_result = None
    
    def is_valid_prefix(colors, i):
        # Check all progressions fully within [0, i]
        for (a, b, c, d) in progressions:
            if d <= i:
                cols = {colors[a], colors[b], colors[c], colors[d]}
                if len(cols) < 3:
                    return False
        return True
    
    # To make it faster, we can precompute for each index i, which progressions end at i.
    # progressions_ending_at[i] = list of (a, b, c) such that (a, b, c, i) is a progression.
    progressions_ending_at = [[] for _ in range(n)]
    for (a, b, c, d) in progressions:
        progressions_ending_at[d].append((a, b, c))
        
    # Also progressions where i is the first, second, or third element?
    # For forward checking, when we assign color to i, we look at progressions 
    # where i is a, b, or c, and check if d is already assigned or if constraints on d are violated.
    
    # Let's stick to a simpler check first: 
    # When assigning color to i, check all progressions where i is the LAST element (d).
    # This ensures all completed progressions ending at i are valid.
    # This is necessary but not sufficient for pruning earlier.
    
    # Let's add a stronger check:
    # When assigning color to i, for any progression (a, b, c, d) where i is a, b, or c:
    # If d is already assigned, check validity.
    # If d is not assigned, we can check if the current partial assignment makes it IMPOSSIBLE 
    # for d to take a valid color from the palette.
    
    # Let's try k=3 first. If it fails, try k=4.
    
    def solve(k):
        colors = [-1] * n
        # Symmetry breaking: colors[0] = 0
        colors[0] = 0
        # colors[1] can be 0 or 1. 
        # If k >= 2, we can set colors[1] = 0 or 1. 
        # Actually, if we assume colors are 0..k-1, 
        # we can fix colors[0]=0.
        # For colors[1], if we pick 0, it's symmetric to picking 0. 
        # If we pick 1, it's symmetric to picking 1.
        # Are there other symmetries?
        # Permuting color labels is a symmetry.
        # So we can fix the sequence of first distinct colors.
        # colors[0] = 0.
        # If colors[1] == 0, next new color must be 1.
        # If colors[1] != 0, colors[1] must be 1 (since 0 is used, next new is 1).
        # So we just need to branch on whether colors[1] is 0 or 1.
        
        def backtrack(i):
            if time.time() > deadline:
                return False, None
            
            if i == n:
                # Check full validity (should be guaranteed by incremental checks)
                return True, colors[:]
            
            # Determine available colors for position i
            # Heuristic: Try colors that are most constrained?
            # For now, just try 0..k-1.
            
            for c in range(k):
                # Check if assigning c to i is valid with respect to progressions ending at i
                valid = True
                for (a, b, c_idx) in progressions_ending_at[i]:
                    if colors[a] == -1 or colors[b] == -1 or colors[c_idx] == -1:
                        # Should not happen if we process in order, 
                        # because a,b,c < d=i
                        pass 
                    cols = {colors[a], colors[b], colors[c_idx], c}
                    if len(cols) < 3:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Forward Checking:
                # Check if this assignment makes any future progression unsatisfiable.
                # Look at progressions where i is a, b, or c.
                # We need to know which progressions have i as a, b, or c.
                # This requires precomputation.
                
                # Let's precompute progressions_involving[i] = list of (idx_in_prog, a, b, c, d)
                # where i is one of a,b,c,d.
                
                # Actually, let's just do a quick check for immediate conflicts with already assigned neighbors.
                # A stronger check:
                # For any progression (a, b, c, d) where d > i:
                # If a,b,c are assigned (including i), check if there exists a color for d.
                # If a,b,c are assigned and d is not, the set {A,B,C} must allow some D in 0..k-1 
                # such that {A,B,C,D} has size >= 3.
                # If {A,B,C} has size 1, D must be != A. Requires k >= 2.
                # If {A,B,C} has size 2, D must be a 3rd color. Requires k >= 3.
                # If {A,B,C} has size 3, D can be anything.
                
                # We can precompute for each i, the list of progressions where i is a, b, or c 
                # and d is unassigned.
                # But d depends on i.
                
                # Let's implement a simple forward check:
                # For each progression (a, b, c, d) with d > i:
                # If i is a, b, or c, and the other two of {a,b,c} are assigned:
                #   Let assigned_colors = {colors[x] for x in {a,b,c} if x <= i}
                #   If len(assigned_colors) == 1:
                #       # d must be different. Always possible if k>=2.
                #   If len(assigned_colors) == 2:
                #       # d must be a 3rd color. Always possible if k>=3.
                #   If len(assigned_colors) == 3:
                #       # d can be anything.
                
                # This check is weak. It only checks if a color EXISTS for d.
                # It doesn't check if d's choice will conflict with OTHER progressions.
                
                # Given n=128, a more sophisticated check is needed to be fast.
                # However, with k=3, the constraint is tight.
                # Let's try running this simple backtracking with k=3.
                
                colors[i] = c
                ok, res = backtrack(i + 1)
                if ok:
                    return True, res
                colors[i] = -1
                
            return False, None
        
        # Symmetry breaking for i=1
        # Option 1: colors[1] = 0
        # Option 2: colors[1] = 1 (if k >= 2)
        
        # Try Option 1
        colors[1] = 0
        ok, res = backtrack(2)
        if ok:
            return res
        
        # Try Option 2
        if k >= 2:
            colors[1] = 1
            ok, res = backtrack(2)
            if ok:
                return res
            
        return None
    
    # Precompute progressions_ending_at properly
    progressions_ending_at = [[] for _ in range(n)]
    for (a, b, c, d) in progressions:
        progressions_ending_at[d].append((a, b, c))
        
    # Try k=3
    result = solve(3)
    if result:
        return result
        
    # Try k=4
    result = solve(4)
    if result:
        return result
        
    # If both fail (shouldn't happen if deadline is met), return a valid greedy coloring
    # Fallback to seed logic
    colors=[]
    for x in range(1,129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors

# To make the backtracking faster, we need better pruning.
# The current implementation might be too slow for n=128 with k=3.
# Let's add a time check inside the loop.

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 128
    progressions = []
    for i in range(n):
        for d in range(1, n):
            j = i + d
            k = i + 2*d
            l = i + 3*d
            if l < n:
                progressions.append((i, j, k, l))
                
    progressions_ending_at = [[] for _ in range(n)]
    for (a, b, c, d) in progressions:
        progressions_ending_at[d].append((a, b, c))
        
    def solve(k):
        colors = [-1] * n
        colors[0] = 0
        
        def backtrack(i):
            if time.time() > deadline:
                return False, None
            
            if i == n:
                return True, colors[:]
            
            for c in range(k):
                valid = True
                for (a, b, c_idx) in progressions_ending_at[i]:
                    cols = {colors[a], colors[b], colors[c_idx], c}
                    if len(cols) < 3:
                        valid = False
                        break
                if not valid:
                    continue
                
                colors[i] = c
                ok, res = backtrack(i + 1)
                if ok:
                    return True, res
                colors[i] = -1
                
            return False, None
        
        # Symmetry breaking
        colors[1] = 0
        ok, res = backtrack(2)
        if ok:
            return res
        
        if k >= 2:
            colors[1] = 1
            ok, res = backtrack(2)
            if ok:
                return res
                
        return None

    result = solve(3)
    if result:
        return result
        
    result = solve(4)
    if result:
        return result
        
    # Fallback
    colors=[]
    for x in range(1,129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors

# This simple backtracking is likely too slow.
# We need to use the "fail-first" heuristic or constraint propagation more aggressively.
# Or, we can use a SAT solver approach if available? No, only stdlib/np/scipy.
# Let's try to improve the pruning by checking forward constraints.

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 128
    progressions = []
    for i in range(n):
        for d in range(1, n):
            j = i + d
            k = i + 2*d
            l = i + 3*d
            if l < n:
                progressions.append((i, j, k, l))
                
    # For each index i, store progressions where i is a, b, or c.
    # prog_involving[i] = list of (pos, a, b, c, d) where pos is 0,1,2 for a,b,c
    prog_involving = [[] for _ in range(n)]
    for (a, b, c, d) in progressions:
        prog_involving[a].append((0, a, b, c, d))
        prog_involving[b].append((1, a, b, c, d))
        prog_involving[c].append((2, a, b, c, d))
        
    def solve(k):
        colors = [-1] * n
        colors[0] = 0
        
        def backtrack(i):
            if time.time() > deadline:
                return False, None
            
            if i == n:
                return True, colors[:]
            
            # Heuristic: Try colors that are used less frequently? Or just 0..k-1
            for c in range(k):
                # Check backward constraints (progressions ending at i)
                valid = True
                for (a, b, c_idx) in progressions_ending_at[i]:
                    cols = {colors[a], colors[b], colors[c_idx], c}
                    if len(cols) < 3:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check forward constraints (progressions where i is a, b, or c)
                # For each such progression, if the other elements are assigned,
                # check if a valid color exists for the unassigned element.
                
                # Simplified: Just check if the assignment creates an immediate conflict
                # with any already assigned element in the same progression.
                # This is already covered by backward constraints if we process in order?
                # No, backward constraints only check progressions where i is the LAST element.
                # If i is the FIRST element, the progression is not complete.
                
                # Let's do a light forward check:
                # For each progression (a, b, c, d) where i is a, b, or c:
                # If d is assigned, check validity.
                # If d is not assigned, check if the current partial assignment 
                # restricts d to an empty set.
                
                # To make it fast, we only check if the set of assigned colors in {a,b,c}
                # has size 3. If it does, d can be anything.
                # If size 2, d must be a 3rd color.
                # If size 1, d must be a 2nd color.
                
                # This check is necessary but not sufficient.
                
                # Given the time limit, let's rely on the fact that k=3 is very constraining
                # and the search space might be small enough with symmetry breaking.
                
                colors[i] = c
                ok, res = backtrack(i + 1)
                if ok:
                    return True, res
                colors[i] = -1
                
            return False, None
        
        # Symmetry breaking
        colors[1] = 0
        ok, res = backtrack(2)
        if ok:
            return res
        
        if k >= 2:
            colors[1] = 1
            ok, res = backtrack(2)
            if ok:
                return res
                
        return None

    # Precompute progressions_ending_at
    progressions_ending_at = [[] for _ in range(n)]
    for (a, b, c, d) in progressions:
        progressions_ending_at[d].append((a, b, c))
        
    result = solve(3)
    if result:
        return result
        
    result = solve(4)
    if result:
        return result
        
    # Fallback
    colors=[]
    for x in range(1,129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors

# The above code is still likely too slow.
# Let's try a different approach: Randomized Local Search (Simulated Annealing or WalkSAT style).
# Objective: Minimize number of distinct colors.
# Constraint: Every 4-AP has >= 3 colors.
# We can start with a random coloring using k colors and try to reduce k or fix violations.
# Or start with k=3 and try to find a valid coloring. If not, increase k.

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 128
    progressions = []
    for i in range(n):
        for d in range(1, n):
            j = i + d
            k = i + 2*d
            l = i + 3*d
            if l < n:
                progressions.append((i, j, k, l))
                
    def count_violations(colors):
        v = 0
        for (a, b, c, d) in progressions:
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                v += 1
        return v
        
    def minimize_colors(k, max_iters=100000):
        # Random initialization
        import random
        random.seed(42)
        colors = [random.randint(0, k-1) for _ in range(n)]
        
        best_colors = colors[:]
        best_violations = count_violations(colors)
        
        if best_violations == 0:
            return colors
            
        # Local search: pick a random position and change its color
        # to the one that reduces violations the most.
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            i = random.randint(0, n-1)
            old_c = colors[i]
            
            best_new_c = old_c
            best_delta = 0
            
            # Calculate delta for each possible new color
            # To be fast, we can just pick a random new color and accept if it improves.
            # Or try all k colors.
            
            # Let's try all k colors
            min_violations = best_violations
            for c in range(k):
                if c == old_c:
                    continue
                colors[i] = c
                v = count_violations(colors)
                if v < min_violations:
                    min_violations = v
                    best_new_c = c
            colors[i] = best_new_c
            
            if min_violations < best_violations:
                best_violations = min_violations
                best_colors = colors[:]
                if best_violations == 0:
                    return best_colors
            else:
                # Revert if not improved (greedy local search)
                # Or accept with some probability (simulated annealing)
                # Let's stick to greedy for simplicity
                colors[i] = old_c
                
        return best_colors if best_violations == 0 else None

    # Try k=3
    result = minimize_colors(3)
    if result:
        return result
        
    # Try k=4
    result = minimize_colors(4)
    if result:
        return result
        
    # Fallback to greedy
    colors=[]
    for x in range(1,129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors

# The local search might be slow if count_violations is O(n^2).
# n=128, progressions ~ 128*64 = 8192.
# 100000 iters * 8192 checks = 8e8 operations. Too slow in Python.
# We need to update violations incrementally.

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 128
    progressions = []
    for i in range(n):
        for d in range(1, n):
            j = i + d
            k = i + 2*d
            l = i + 3*d
            if l < n:
                progressions.append((i, j, k, l))
                
    # For each position i, store the list of progressions it belongs to.
    prog_at = [[] for _ in range(n)]
    for idx, (a, b, c, d) in enumerate(progressions):
        prog_at[a].append(idx)
        prog_at[b].append(idx)
        prog_at[c].append(idx)
        prog_at[d].append(idx)
        
    def count_violations(colors):
        v = 0
        for (a, b, c, d) in progressions:
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                v += 1
        return v
        
    def minimize_colors(k):
        import random
        random.seed(42)
        colors = [random.randint(0, k-1) for _ in range(n)]
        violations = count_violations(colors)
        
        if violations == 0:
            return colors
            
        # Incremental update
        # When we change color at i, only progressions containing i are affected.
        
        def get_violation_count_for_prog(a, b, c, d):
            return 0 if len({colors[a], colors[b], colors[c], colors[d]}) >= 3 else 1
            
        best_colors = colors[:]
        best_violations = violations
        
        max_iters = 50000
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            i = random.randint(0, n-1)
            old_c = colors[i]
            
            # Calculate current violations for progressions involving i
            # This is expensive to do from scratch.
            # Let's just do a simple hill climbing with full recalculation for simplicity 
            # but limit iterations.
            
            # Actually, let's try a different heuristic:
            # Pick a random progression that is violated, and change one of its colors.
            
            # Find a violated progression
            violated_progs = []
            for idx, (a, b, c, d) in enumerate(progressions):
                if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                    violated_progs.append(idx)
                    
            if not violated_progs:
                return colors
                
            # Pick one randomly
            p_idx = random.choice(violated_progs)
            a, b, c, d = progressions[p_idx]
            
            # Try changing one of the 4 positions to a different color
            best_improvement = 0
            best_change = None
            
            for pos in [a, b, c, d]:
                old_color = colors[pos]
                for new_color in range(k):
                    if new_color == old_color:
                        continue
                    colors[pos] = new_color
                    # Recalculate violations for this progression and others?
                    # Just check this progression
                    if len({colors[a], colors[b], colors[c], colors[d]}) >= 3:
                        # Check if this change creates new violations elsewhere
                        # This is the hard part.
                        # Let's just accept if it fixes this one and hope it doesn't break others too much.
                        # Or do a full check.
                        pass
                    colors[pos] = old_color
                    
            # This is getting complicated. Let's stick to the simple backtracking 
            # with a very tight time limit and see if it finds a solution for k=3.
            # If not, fallback.
            
        return best_colors if best_violations == 0 else None

    # Try k=3 with backtracking
    # (Reusing the backtracking code from before, but with a stricter time limit)
    
    progressions_ending_at = [[] for _ in range(n)]
    for (a, b, c, d) in progressions:
        progressions_ending_at[d].append((a, b, c))
        
    def solve(k):
        colors = [-1] * n
        colors[0] = 0
        
        def backtrack(i):
            if time.time() > deadline:
                return False, None
            
            if i == n:
                return True, colors[:]
            
            for c in range(k):
                valid = True
                for (a, b, c_idx) in progressions_ending_at[i]:
                    cols = {colors[a], colors[b], colors[c_idx], c}
                    if len(cols) < 3:
                        valid = False
                        break
                if not valid:
                    continue
                
                colors[i] = c
                ok, res = backtrack(i + 1)
                if ok:
                    return True, res
                colors[i] = -1
                
            return False, None
        
        colors[1] = 0
        ok, res = backtrack(2)
        if ok:
            return res
        
        if k >= 2:
            colors[1] = 1
            ok, res = backtrack(2)
            if ok:
                return res
                
        return None

    result = solve(3)
    if result:
        return result
        
    result = solve(4)
    if result:
        return result
        
    # Fallback
    colors=[]
    for x in range(1,129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors

# This is a mess. Let's clean it up and focus on the most promising strategy:
# Backtracking with constraint propagation.

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 128
    progressions = []
    for i in range(n):
        for d in range(1, n):
            j = i + d
            k = i + 2*d
            l = i + 3*d
            if l < n:
                progressions.append((i, j, k, l))
                
    progressions_ending_at = [[] for _ in range(n)]
    for (a, b, c, d) in progressions:
        progressions_ending_at[d].append((a, b, c))
        
    def solve(k):
        colors = [-1] * n
        colors[0] = 0
        
        def backtrack(i):
            if time.time() > deadline:
                return False, None
            
            if i == n:
                return True, colors[:]
            
            for c in range(k):
                valid = True
                for (a, b, c_idx) in progressions_ending_at[i]:
                    cols = {colors[a], colors[b], colors[c_idx], c}
                    if len(cols) < 3:
                        valid = False
                        break
                if not valid:
                    continue
                
                colors[i] = c
                ok, res = backtrack(i + 1)
                if ok:
                    return True, res
                colors[i] = -1
                
            return False, None
        
        # Symmetry breaking
        colors[1] = 0
        ok, res = backtrack(2)
        if ok:
            return res
        
        if k >= 2:
            colors[1] = 1
            ok, res = backtrack(2)
            if ok:
                return res
                
        return None

    result = solve(3)
    if result:
        return result
        
    result = solve(4)
    if result:
        return result
        
    # Fallback
    colors=[]
    for x in range(1,129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors

# The backtracking is likely too slow.
# Let's try the local search with incremental violation counting.

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 128
    progressions = []
    for i in range(n):
        for d in range(1, n):
            j = i + d
            k = i + 2*d
            l = i + 3*d
            if l < n:
                progressions.append((i, j, k, l))
                
    prog_at = [[] for _ in range(n)]
    for idx, (a, b, c, d) in enumerate(progressions):
        prog_at[a].append(idx)
        prog_at[b].append(idx)
        prog_at[c].append(idx)
        prog_at[d].append(idx)
        
    def count_violations(colors):
        v = 0
        for (a, b, c, d) in progressions:
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                v += 1
        return v
        
    def minimize_colors(k):
        import random
        random.seed(42)
        colors = [random.randint(0, k-1) for _ in range(n)]
        violations = count_violations(colors)
        
        if violations == 0:
            return colors
            
        best_colors = colors[:]
        best_violations = violations
        
        # Precompute violation status for each progression
        is_violated = [False] * len(progressions)
        for idx, (a, b, c, d) in enumerate(progressions):
            is_violated[idx] = len({colors[a], colors[b], colors[c], colors[d]}) < 3
            
        for _ in range(100000):
            if time.time() > deadline:
                break
                
            # Pick a random position
            i = random.randint(0, n-1)
            old_c = colors[i]
            
            # Find best new color for i
            best_new_c = old_c
            best_delta = 0
            
            # Calculate delta for each new color
            # Only progressions containing i are affected
            affected_progs = prog_at[i]
            
            # Current violations for affected progs
            current_violations = sum(1 for idx in affected_progs if is_violated[idx])
            
            for c in range(k):
                if c == old_c:
                    continue
                colors[i] = c
                
                new_violations = 0
                for idx in affected_progs:
                    a, b, c_idx, d = progressions[idx]
                    if len({colors[a], colors[b], colors[c_idx], colors[d]}) < 3:
                        new_violations += 1
                        
                delta = new_violations - current_violations
                if delta < best_delta:
                    best_delta = delta
                    best_new_c = c
                    
            colors[i] = best_new_c
            
            # Update is_violated and violations count
            if best_new_c != old_c:
                for idx in affected_progs:
                    a, b, c_idx, d = progressions[idx]
                    old_violated = is_violated[idx]
                    new_violated = len({colors[a], colors[b], colors[c_idx], colors[d]}) < 3
                    if old_violated != new_violated:
                        if new_violated:
                            violations += 1
                        else:
                            violations -= 1
                        is_violated[idx] = new_violated
                        
                if violations < best_violations:
                    best_violations = violations
                    best_colors = colors[:]
                    if best_violations == 0:
                        return best_colors
                        
        return best_colors if best_violations == 0 else None

    result = minimize_colors(3)
    if result:
        return result
        
    result = minimize_colors(4)
    if result:
        return result
        
    # Fallback
    colors=[]
    for x in range(1,129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors

# This local search is more likely to find a solution within the time limit.
# The incremental update makes it fast enough.
