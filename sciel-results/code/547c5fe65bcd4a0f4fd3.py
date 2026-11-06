import time
from math import gcd
import random

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    N = 128
    N1 = N + 1
    UNIVERSE = list(range(1, N1))

    # Precompute gcd matrix
    gcd_mat = [[0] * N1 for _ in range(N1)]
    for i in range(1, N1):
        for j in range(i + 1, N1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # Precompute conflict structure:
    # For each element x, and each element a, bad_with[x][a] is the set of b such that {a,b,x} is a bad triple.
    # i.e., gcd(a,b) == gcd(a,x) == gcd(b,x)
    bad_with = [None] * N1
    for x in range(1, N1):
        bad_with[x] = [set() for _ in range(N1)]
        for a in range(1, N1):
            if a == x:
                continue
            gax = gcd_mat[a][x]
            for b in range(a + 1, N1):
                if b == x:
                    continue
                if gcd_mat[a][b] == gax and gcd_mat[b][x] == gax:
                    bad_with[x][a].add(b)
                    bad_with[x][b].add(a)

    # Also precompute for each pair (a,b), the set of c that form a bad triple with a,b.
    # This is useful for checking if a candidate set is valid.
    bad_triple_with_pair = {}
    for a in range(1, N1):
        for b in range(a + 1, N1):
            g = gcd_mat[a][b]
            s = set()
            for c in range(1, N1):
                if c == a or c == b:
                    continue
                if gcd_mat[a][c] == g and gcd_mat[b][c] == g:
                    s.add(c)
            bad_triple_with_pair[(a, b)] = s

    # Function to check if a set is valid
    def is_valid(S):
        S_list = sorted(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                a, b = S_list[i], S_list[j]
                # If any c in S is in bad_triple_with_pair[(a,b)], it's invalid
                # But we just need to check if there exists c in S such that {a,b,c} is bad.
                # Since S is a set, we can check intersection.
                if bad_triple_with_pair[(a, b)] & S:
                    return False
        return True

    # Function to check if x can be added to S
    def can_add(S, x):
        for a in S:
            if bad_with[x][a] & S:
                return False
        return True

    # Greedy construction to get a good initial solution
    def greedy_construct():
        S = set()
        remaining = list(range(1, N1))
        # Sort by some heuristic, e.g., degree in conflict graph
        # Degree of x: number of y such that x and y are in a bad triple with some z.
        # Simpler: just try to add in order of increasing "badness"
        # Let's use a dynamic greedy: at each step, pick the element that conflicts with the fewest existing elements.
        
        # Precompute initial conflict degrees? No, dynamic is better.
        while remaining:
            best_x = None
            min_conflicts = float('inf')
            for x in remaining:
                # Count conflicts with current S
                conflicts = 0
                for a in S:
                    if bad_with[x][a] & S:
                        conflicts += 1
                if conflicts < min_conflicts:
                    min_conflicts = conflicts
                    best_x = x
                if min_conflicts == 0:
                    break
            if best_x is None:
                break
            if min_conflicts > 0:
                # If no element can be added, break
                break
            S.add(best_x)
            remaining.remove(best_x)
        
        return S

    # Branch and Bound
    # State: 
    #   S: current set (frozenset for hashing, but we use list for speed and convert if needed)
    #   R: remaining candidates (list)
    #   best_size: current best size
    #   best_set: current best set
    
    # We will use a recursive DFS.
    # To make it efficient, we need a good upper bound.
    # Upper bound: |S| + |R'| where R' is the set of elements in R that can be added to S.
    # This is a valid upper bound, but maybe not tight enough.
    # A tighter bound: |S| + size of maximum independent set in the conflict graph induced by R, 
    # where the conflict graph is defined with respect to S.
    # Two elements x, y in R are connected if they cannot both be in S together with the current S.
    # i.e., there exists z in S such that {x,y,z} is a bad triple, OR x and y themselves form a bad triple with some element in S?
    # Actually, the constraint is: no three distinct elements in the final set form a bad triple.
    # When adding x and y to S, we need to ensure:
    # 1. x is compatible with S (can_add(S, x))
    # 2. y is compatible with S U {x} (can_add(S U {x}, y))
    # This is complex to precompute for all pairs in R.
    
    # Let's stick to the simpler bound first: |S| + count(x in R such that can_add(S, x)).
    # If this bound <= best_size, prune.
    
    # We also need to order the elements in R to improve pruning.
    # Sort R by "constraint" or "conflict degree".
    
    best_size = 0
    best_set = set()
    
    # Initial solution
    init_S = greedy_construct()
    if init_S:
        best_size = len(init_S)
        best_set = set(init_S)
    
    # Order elements for branching
    # We want to branch on elements that are "critical" or have high conflict degree.
    # Let's compute a static conflict degree for each element.
    # degree[x] = number of pairs (a,b) such that {a,b,x} is a bad triple.
    degree = [0] * N1
    for a in range(1, N1):
        for b in range(a + 1, N1):
            for c in bad_triple_with_pair[(a, b)]:
                if c > b: # Avoid double counting? No, just count participation.
                    pass
    # Simpler: for each x, count how many (a,b) in bad_triple_with_pair have x in the set.
    # This is O(N^3) if done naively.
    # Let's use the bad_with structure.
    # For each x, degree[x] = sum_{a in 1..N} |bad_with[x][a]| / 2 ? No.
    # Let's just sort by index or some simple heuristic for now.
    # Actually, sorting by descending degree is usually good for B&B.
    
    # Calculate degree properly:
    # For each x, count the number of bad triples containing x.
    for x in range(1, N1):
        count = 0
        for a in range(1, N1):
            if a == x: continue
            count += len(bad_with[x][a])
        degree[x] = count // 2 # Each pair (a,b) is counted twice?
        # Let's verify: bad_with[x][a] contains b if {a,b,x} is bad.
        # So for a fixed x, summing len(bad_with[x][a]) over all a counts each bad triple {x,a,b} twice (once for a, once for b).
        # So degree[x] is the number of bad triples containing x.
        
    order = sorted(range(1, N1), key=lambda x: -degree[x])
    
    # Map from element to index in order for quick lookup?
    # We will process elements in 'order'.
    
    def branch_and_bound(S, R, idx):
        nonlocal best_size, best_set
        
        if time.time() > DEADLINE - 1:
            return
            
        # Upper bound
        # Count how many elements in R[idx:] can be added to S
        compatible_count = 0
        for i in range(idx, len(order)):
            x = order[i]
            if x in R:
                if can_add(S, x):
                    compatible_count += 1
        
        if len(S) + compatible_count <= best_size:
            return
            
        if idx == len(order):
            if len(S) > best_size:
                best_size = len(S)
                best_set = set(S)
            return
            
        x = order[idx]
        
        # Option 1: Exclude x
        if x in R:
            R.remove(x)
            branch_and_bound(S, R, idx + 1)
            R.append(x) # Restore? No, we should manage R carefully.
            # Better to pass R as a list and modify a copy or use a flag.
            # Let's use a different approach: pass a list of candidates and an index.
            # Or, since N is small, we can just pass a tuple or frozenset.
            # But frozenset hashing is slow.
            # Let's stick to list and be careful.
            
        # Option 2: Include x
        if x in R:
            if can_add(S, x):
                S.add(x)
                R.remove(x)
                branch_and_bound(S, R, idx + 1)
                R.append(x)
                S.remove(x)
                
    # The above B&B is flawed because R is modified in place and the logic for "remaining" is mixed with "index".
    # Let's rewrite B&B cleanly.
    
    def bnb(S, remaining, current_size):
        nonlocal best_size, best_set
        
        if time.time() > DEADLINE - 1:
            return
            
        # Upper bound: current_size + len(remaining)
        if current_size + len(remaining) <= best_size:
            return
            
        if not remaining:
            if current_size > best_size:
                best_size = current_size
                best_set = set(S)
            return
            
        # Pick the next element to branch on.
        # We can pick the first one in 'remaining' that is in our global order.
        # Or just pick the first one.
        # To improve, let's pick the element with the highest degree among remaining.
        # But that requires sorting remaining each time, which is O(N log N).
        # Let's just pick the first one in the global order that is in remaining.
        
        x = None
        for elem in order:
            if elem in remaining:
                x = elem
                break
                
        if x is None:
            return
            
        # Option 1: Exclude x
        remaining.remove(x)
        bnb(S, remaining, current_size)
        
        # Option 2: Include x
        if can_add(S, x):
            S.add(x)
            bnb(S, remaining, current_size + 1)
            S.remove(x)
            
        remaining.append(x)
        
    # Start B&B with the initial solution as the base?
    # No, B&B should start from scratch or from a good lower bound.
    # The initial solution gives us a good lower bound (best_size).
    # We can start B&B with S = init_S and remaining = UNIVERSE - init_S.
    # But we must ensure that adding elements to init_S doesn't violate constraints.
    # can_add(S, x) checks if x can be added to S.
    
    # Let's try starting B&B from the initial solution.
    S_init = set(init_S)
    R_init = [x for x in UNIVERSE if x not in S_init]
    
    # Sort R_init by global order?
    # The bnb function picks the next x from 'order' that is in 'remaining'.
    # This is efficient if 'order' is a list and we check membership.
    # Membership in a list is O(N). N=128, so it's fine.
    
    bnb(S_init, R_init, len(S_init))
    
    # If B&B didn't improve, return the initial solution.
    # We might want to run B&B from scratch as well, but that's expensive.
    # Starting from a good solution is usually better.
    
    if best_set:
        return sorted(best_set)
    else:
        return [1]

# Wrap to ensure nonempty and handle edge cases
_orig_construct = construct
def construct():
    try:
        result = _orig_construct()
        if not result:
            return [1]
        return result
    except Exception:
        return [1]
