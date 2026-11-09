import time
import random
import sys

N = 256
MAX_EACH = 64
DEADLINE = 85.0

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    
    # We will model this as a Maximum Independent Set problem on a conflict graph.
    # Vertices represent potential assignments: (value, set_id) where value in [1, N] and set_id in {0, 1}.
    # Total vertices: N * 2 = 512.
    # 
    # Constraints for a vertex v=(x, s):
    # 1. Sidon Property (Pair Sums Unique):
    #    For any other vertex u=(y, s) with u != v, if x + y is already formed by another pair in s, conflict.
    #    Actually, it's easier to think about edges between vertices that CANNOT coexist.
    #    Two vertices (x, s) and (y, s) conflict if they create a duplicate sum with existing elements?
    #    No, MIS is static. We need edges that represent "these two cannot be in the solution together".
    #    
    #    Let's refine the conflict definition for MIS:
    #    A set S of vertices is valid if:
    #    - For each set_id s, the projection {x | (x,s) in S} is a Sidon set.
    #    - For each set_id s, |{x | (x,s) in S}| <= MAX_EACH.
    #    - The nonzero difference sets of A and B are disjoint.
    #
    #    Conflict Edges:
    #    1. Same value, different sets? No, A and B can overlap. So (x, 0) and (x, 1) are NOT conflicting.
    #    2. Same set, Sidon violation:
    #       If (x, s) and (y, s) are in S, and there exist (u, s) and (v, s) in S such that x+y = u+v and {x,y} != {u,v}, then conflict.
    #       This is a 4-body interaction, which is hard for standard pairwise MIS.
    #       However, we can simplify. If we pick (x, s), it forbids any (y, s) such that x+y creates a collision.
    #       Wait, the collision depends on the OTHER elements too.
    #       
    #       Alternative Approach: Greedy MIS on a "compatibility" graph where we build the set incrementally but use the MIS heuristic (Max Degree, etc.) to pick the next element, rather than simple iteration.
    #       But the prompt asks for a "conflict graph where vertices are candidate elements... edges represent sum or difference collisions".
    #       
    #       Let's stick to the standard interpretation: We construct a graph G=(V, E).
    #       V = {(x, s) | x in 1..N, s in {0,1}}.
    #       Edge between (x, s) and (y, s) if x+y is "bad"? No, x+y is just a number.
    #       
    #       Let's look at the difference constraint first, it's pairwise.
    #       Edge between (x, 0) and (y, 1) if |x-y| is in Diff(A) or Diff(B)?
    #       Diff(A) is dynamic.
    #       
    #       This static graph approach is tricky because "Sidon" and "Disjoint Diffs" are global properties of the *set* chosen, not just pairwise properties of the vertices, UNLESS we define the edges carefully.
    #       
    #       Actually, for Sidon sets, a set S is Sidon iff all sums a+b (a<=b) are unique.
    #       This is equivalent to: for any distinct pairs (a,b) and (c,d) with a<=b, c<=d, a+b != c+d unless {a,b}={c,d}.
    #       
    #       Let's try a different structural change: **Simulated Annealing on the set of elements**.
    #       The current program does greedy + local search (swaps).
    #       The inspiration suggests Simulated Annealing.
    #       
    #       Let's implement a robust Simulated Annealing that:
    #       1. Starts with a good greedy solution.
    #       2. Moves: 
    #          a. Add a random valid element (if space).
    #          b. Remove a random element.
    #          c. Swap: Remove one, try to add another (even if it violates constraints temporarily? No, we must maintain validity or use a penalty function).
    #       
    #       Since maintaining validity is hard with swaps, let's use a **Penalty-based Simulated Annealing**.
    #       Objective: Maximize |A| + |B| - Penalty(Violations).
    #       Violations: Duplicate sums in A, Duplicate sums in B, Overlapping diffs, Size limits.
    #       
    #       This allows the search to jump over local minima by temporarily accepting invalid states (high penalty) if it leads to a better overall structure.
    #       
    #       Let's implement this.

    def get_score(A, B):
        """Calculate score and penalty."""
        score = 0
        penalty = 0
        
        # Check A
        if len(A) > MAX_EACH:
            penalty += (len(A) - MAX_EACH) * 100
        if len(B) > MAX_EACH:
            penalty += (len(B) - MAX_EACH) * 100
            
        # Sidon Check A
        sums_a = {}
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sums_a:
                    penalty += 10
                else:
                    sums_a[s] = 1
                    
        # Sidon Check B
        sums_b = {}
        for i in range(len(B)):
            for j in range(i, len(B)):
                s = B[i] + B[j]
                if s in sums_b:
                    penalty += 10
                else:
                    sums_b[s] = 1
                    
        # Difference Disjointness
        diff_a = set()
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                d = A[j] - A[i]
                diff_a.add(d)
                diff_a.add(-d)
                
        diff_b = set()
        for i in range(len(B)):
            for j in range(i+1, len(B)):
                d = B[j] - B[i]
                diff_b.add(d)
                diff_b.add(-d)
                
        overlap = diff_a & diff_b
        if overlap:
            penalty += len(overlap) * 10
            
        # Base score
        base_score = len(A) + len(B)
        
        return base_score - penalty, len(A) + len(B)

    # Initial Greedy Construction
    def greedy_init():
        A = []
        B = []
        for x in range(1, N + 1):
            # Try A
            if len(A) < MAX_EACH:
                # Check Sidon
                ok = True
                sums = set()
                for a in A:
                    if (x + a) in sums:
                        ok = False
                        break
                    sums.add(x + a)
                if (x + x) in sums:
                    ok = False
                if ok:
                    # Check Diff
                    new_diffs = set()
                    for a in A:
                        d = x - a
                        new_diffs.add(d)
                        new_diffs.add(-d)
                    # Check against B diffs
                    b_diffs = set()
                    for i in range(len(B)):
                        for j in range(i+1, len(B)):
                            d = B[j] - B[i]
                            b_diffs.add(d)
                            b_diffs.add(-d)
                    if not (new_diffs & b_diffs):
                        A.append(x)
            
            # Try B
            if len(B) < MAX_EACH:
                ok = True
                sums = set()
                for b in B:
                    if (x + b) in sums:
                        ok = False
                        break
                    sums.add(x + b)
                if (x + x) in sums:
                    ok = False
                if ok:
                    new_diffs = set()
                    for b in B:
                        d = x - b
                        new_diffs.add(d)
                        new_diffs.add(-d)
                    a_diffs = set()
                    for i in range(len(A)):
                        for j in range(i+1, len(A)):
                            d = A[j] - A[i]
                            a_diffs.add(d)
                            a_diffs.add(-d)
                    if not (new_diffs & a_diffs):
                        B.append(x)
        return A, B

    A, B = greedy_init()
    best_A, best_B = list(A), list(B)
    best_score, _ = get_score(best_A, best_B)
    best_valid_size = len(best_A) + len(best_B)
    
    # Simulated Annealing
    T = 100.0
    cooling = 0.99
    min_T = 0.01
    
    while T > min_T and time.time() < deadline:
        # Pick a move
        move_type = random.randint(0, 2)
        A_new = list(A)
        B_new = list(B)
        
        if move_type == 0 and len(A) < MAX_EACH:
            # Add to A
            x = random.randint(1, N)
            if x not in A:
                A_new.append(x)
        elif move_type == 1 and len(B) < MAX_EACH:
            # Add to B
            x = random.randint(1, N)
            if x not in B:
                B_new.append(x)
        elif move_type == 2 and (A or B):
            # Swap: Remove one from A or B, add another
            if A and random.random() < 0.5:
                idx = random.randint(0, len(A) - 1)
                removed = A_new.pop(idx)
                x = random.randint(1, N)
                if x not in A_new:
                    A_new.append(x)
            elif B:
                idx = random.randint(0, len(B) - 1)
                removed = B_new.pop(idx)
                x = random.randint(1, N)
                if x not in B_new:
                    B_new.append(x)
        else:
            continue
            
        new_score, new_size = get_score(A_new, B_new)
        delta = new_score - best_score
        
        if delta > 0:
            # Accept
            A, B = A_new, B_new
            if new_score > best_score:
                # Check if valid
                # Quick validity check
                valid = True
                sums_a = set()
                for i in range(len(A)):
                    for j in range(i, len(A)):
                        s = A[i] + A[j]
                        if s in sums_a: valid = False; break
                        sums_a.add(s)
                if valid:
                    sums_b = set()
                    for i in range(len(B)):
                        for j in range(i, len(B)):
                            s = B[i] + B[j]
                            if s in sums_b: valid = False; break
                            sums_b.add(s)
                if valid:
                    diff_a = set()
                    for i in range(len(A)):
                        for j in range(i+1, len(A)):
                            d = A[j] - A[i]
                            diff_a.add(d); diff_a.add(-d)
                    diff_b = set()
                    for i in range(len(B)):
                        for j in range(i+1, len(B)):
                            d = B[j] - B[i]
                            diff_b.add(d); diff_b.add(-d)
                    if not (diff_a & diff_b) and len(A) <= MAX_EACH and len(B) <= MAX_EACH:
                        best_score = new_score
                        best_A = list(A)
                        best_B = list(B)
                        best_valid_size = new_size
        else:
            # Metropolis criterion
            if random.random() < pow(sys.float_info.max, delta / T):
                A, B = A_new, B_new
                
        T *= cooling

    # Final cleanup: ensure non-empty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
        
    return [best_A, best_B]
