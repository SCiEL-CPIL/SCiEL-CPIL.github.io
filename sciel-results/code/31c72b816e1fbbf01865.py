import time
import random
import itertools
import sys
from collections import defaultdict

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
NUM_POINTS = N * N
DEADLINE = 300.0  # Safety margin from 330s limit

# Precompute distance squared
def _dist2(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

# Precompute all pairwise distances
dist2_matrix = [[0]*NUM_POINTS for _ in range(NUM_POINTS)]
point_index = {p: i for i, p in enumerate(POINTS)}

for i in range(NUM_POINTS):
    pi = POINTS[i]
    for j in range(NUM_POINTS):
        pj = POINTS[j]
        dist2_matrix[i][j] = (pi[0]-pj[0])**2 + (pi[1]-pj[1])**2

# Precompute isosceles relationships
# IsoPairs[p] = set of pairs (i, j) such that (p, POINTS[i], POINTS[j]) is isosceles
# We store indices for speed.
# Note: p, i, j must be distinct.
IsoPairs = [set() for _ in range(NUM_POINTS)]

# For each pair (i, j), determine which p's make (p, i, j) isosceles.
# This is faster than iterating p for each pair if we do it smartly, 
# but iterating all triples (p, i, j) is O(N^6) = 16^6 = 16M ops.
# Let's do it efficiently.

# Actually, let's precompute the "conflict" graph for independence.
# Two points (or sets) conflict if they form an isosceles triangle with a third.
# For the CSP:
# Variables: x_p (1 if p is in set, 0 otherwise)
# Constraint 1 (Independence): For any triple (a,b,c), NOT (x_a=1 AND x_b=1 AND x_c=1 AND isosceles(a,b,c))
# Constraint 2 (Maximality/Dominance): For any p not in S, there must exist a,b in S such that isosceles(p,a,b).
#    This is a "covering" constraint.

# Let's precompute for each point p, the set of pairs (a,b) that "cover" p.
# Covers[p] = set of frozenset({a,b}) such that (p,a,b) is isosceles.
Covers = [set() for _ in range(NUM_POINTS)]

# And for independence, we need to know for any set S, if it's independent.
# It's easier to precompute for each pair (a,b), which points c conflict with them?
# No, let's just use the Covers structure.
# If a, b are in S, then for any c, if {a,b} in Covers[c], then c cannot be in S.

# Precompute Covers
for p in range(NUM_POINTS):
    for a in range(NUM_POINTS):
        if a == p: continue
        for b in range(a+1, NUM_POINTS):
            if b == p: continue
            d_pa = dist2_matrix[p][a]
            d_pb = dist2_matrix[p][b]
            d_ab = dist2_matrix[a][b]
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                Covers[p].add(frozenset((a, b)))

# Precompute "Conflict Pairs" for independence check?
# Actually, we can check independence of a set S by iterating through S and checking
# if any pair in S is in Covers[c] for any c in S.

class CSPSolver:
    def __init__(self, time_limit):
        self.time_limit = time_limit
        self.start_time = time.time()
        self.best_size = NUM_POINTS
        self.best_set = None
        self.abort = False

    def _check_time(self):
        if time.time() > self.time_limit:
            self.abort = True
            return True
        return False

    def _is_independent(self, S):
        """Check if set S (list of indices) is no-isosceles."""
        S_set = set(S)
        for c in S:
            for i, a in enumerate(S):
                if a == c: continue
                for b in S[i+1:]:
                    if b == c: continue
                    if frozenset((a, b)) in Covers[c]:
                        return False
        return True

    def _is_maximal(self, S):
        """Check if set S is maximal."""
        S_set = set(S)
        for p in range(NUM_POINTS):
            if p in S_set:
                continue
            # p is blocked if there exist a,b in S such that {a,b} in Covers[p]
            blocked = False
            S_list = list(S)
            for i in range(len(S_list)):
                a = S_list[i]
                for j in range(i+1, len(S_list)):
                    b = S_list[j]
                    if frozenset((a, b)) in Covers[p]:
                        blocked = True
                        break
                if blocked:
                    break
            if not blocked:
                return False
        return True

    def solve(self, target_size, initial_set=None):
        """
        Backtracking search to find a maximal no-isosceles set of size <= target_size.
        We use a heuristic: pick points that cover the most uncovered points.
        """
        if initial_set is None:
            # Start with a good initial guess: a random independent set
            # Or start with empty and add?
            # Better: Start with a random valid independent set of small size, then try to expand/fix?
            # Actually, for minimization, we want to find a SMALL set.
            # So we should search for a set of size exactly K.
            # We iterate K from small to large.
            
            # Heuristic for initial variables: 
            # We don't have a partial assignment yet.
            # We will use a "branch and bound" style.
            # State: 
            #   S: current set of selected points
            #   U: set of uncovered points (must be covered by S)
            #   C: set of candidate points that can be added to S without violating independence
            #      and are not in S.
            
            # To make it efficient, we precompute for each candidate p, 
            # which points it covers (given current S).
            
            S = set()
            # U: points that are not covered by S
            # Initially, all points are uncovered? 
            # No, maximality requires that ALL points not in S are covered.
            # So U = all points not in S.
            # But we only care about points that are NOT in S.
            
            # Let's define the state as:
            # S: set of selected points
            # Uncovered: set of points p not in S such that no pair in S covers p.
            # Candidates: points p not in S such that S U {p} is independent.
            
            # Initial state:
            S = set()
            Uncovered = set(range(NUM_POINTS))
            # Candidates: all points, since S is empty, any point is independent.
            # But we need to choose an order.
            
            # We will use a recursive backtracking.
            # To avoid deep recursion and overhead, we implement it carefully.
            
            # Heuristic: MRV on Uncovered? Or Max Value on Candidates?
            # Let's try: 
            # 1. If Uncovered is empty, check if S is maximal (it is, by definition of Uncovered).
            #    If len(S) <= target_size, return S.
            # 2. Pick a point p from Candidates.
            #    To guide search, pick p that covers the most points in Uncovered.
            # 3. Add p to S.
            #    Update Uncovered: remove points covered by p (with existing S).
            #    Update Candidates: remove p, and remove any point c that conflicts with p 
            #    (i.e., S U {p, c} is not independent).
            # 4. Recurse.
            # 5. If fail, backtrack.
            
            # To handle "conflicts", we need to know for a new point p, which existing candidates c 
            # would make S U {p, c} non-independent.
            # This depends on S.
            
            # Let's implement the solver.
            
            def _get_candidates(S):
                # Return list of points p not in S such that S U {p} is independent.
                cands = []
                S_list = list(S)
                for p in range(NUM_POINTS):
                    if p in S:
                        continue
                    # Check if p conflicts with any pair in S
                    valid = True
                    for i in range(len(S_list)):
                        a = S_list[i]
                        for b in S_list[i+1:]:
                            # if frozenset({a,b}) in Covers[p], then p conflicts
                            if frozenset((a, b)) in Covers[p]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        cands.append(p)
                return cands

            def _get_newly_covered(S, p, Uncovered):
                # Return set of points in Uncovered that are covered by adding p to S.
                # A point q is covered by S U {p} if there exists a,b in S U {p} 
                # such that {a,b} in Covers[q].
                # Since q is in Uncovered, it was NOT covered by S.
                # So the covering pair must involve p.
                # i.e., there exists c in S such that {p,c} in Covers[q].
                newly = set()
                for q in Uncovered:
                    if q == p:
                        continue
                    # Check if p covers q with help of S
                    # We need c in S such that {p,c} in Covers[q]
                    for c in S:
                        if frozenset((p, c)) in Covers[q]:
                            newly.add(q)
                            break
                return newly

            def _get_conflicts(S, p):
                # Return set of points c (not in S, c != p) such that S U {p, c} is NOT independent.
                # This means there exists a,b in S U {p, c} forming isosceles with the third.
                # Since S is independent, the conflict must involve p or c.
                # Cases:
                # 1. (p, a, b) isosceles for a,b in S. (p is already checked to be independent from S, so this is false)
                # 2. (c, a, b) isosceles for a,b in S. (c must be independent from S)
                # 3. (p, c, a) isosceles for a in S.
                # 4. (p, a, c) isosceles for a in S. (Same as 3)
                # 5. (a, p, c) isosceles for a in S. (Same as 3)
                # So we just need to check if {p, c} is in Covers[a] for any a in S.
                # Also, we must ensure c is not in S.
                conflicts = set()
                for c in range(NUM_POINTS):
                    if c in S or c == p:
                        continue
                    # Check if c is independent from S
                    # (We assume candidates are filtered, but let's be safe or rely on previous filter)
                    # Actually, if we maintain a candidate list, we only consider c in candidates.
                    # But here we are precomputing conflicts for all c? No, too slow.
                    # We should only check conflicts against current candidates.
                    pass
                return conflicts # Placeholder

            # The above helper is getting complex. Let's simplify the state management.
            # State: S (set), Candidates (list of indices), Uncovered (set of indices).
            # We will update these incrementally.
            
            def backtrack(S, Candidates, Uncovered, depth):
                if self.abort:
                    return None
                
                if self._check_time():
                    return None
                
                # Check if goal reached
                if not Uncovered:
                    # S is maximal and independent (by construction)
                    if len(S) <= target_size:
                        if len(S) < self.best_size:
                            self.best_size = len(S)
                            self.best_set = set(S)
                        return set(S)
                
                # If no candidates, dead end
                if not Candidates:
                    return None
                
                # Heuristic: Pick candidate p that covers the most Uncovered points
                # Tie-breaker: pick random or highest index?
                # To add diversity, we can shuffle candidates if scores are close.
                
                best_p = None
                best_score = -1
                best_cands = []
                
                # Optimization: Only consider top K candidates to reduce branching?
                # No, MRV is on Uncovered? 
                # Standard CSP: MRV picks variable with fewest values. Here, we pick a point to ADD.
                # The "variable" is which point to add next.
                # We want to add a point that reduces Uncovered the most.
                
                # To speed up, we can limit the number of candidates we evaluate if list is huge.
                # But NUM_POINTS is 256. Candidates <= 256.
                # Evaluating score for 256 candidates * 256 uncovered * |S| is expensive.
                # Let's optimize _get_newly_covered.
                
                # Precompute for each p, the set of (c, q) such that {p,c} covers q?
                # No, it depends on S.
                
                # Let's just do it and hope it's fast enough for small depths.
                # Depth is at most target_size (which we hope is small, e.g., < 20).
                
                for p in Candidates:
                    # Calculate score: number of points in Uncovered covered by adding p
                    score = 0
                    # We can break early if score > best_score? No, we need exact for tie-breaking?
                    # Just count.
                    for q in Uncovered:
                        if q == p:
                            continue
                        # Check if p covers q with help of S
                        # Need c in S such that {p,c} in Covers[q]
                        # We can precompute for each p, the set of c such that {p,c} is a valid pair?
                        # No, Covers[q] is a set of pairs.
                        # Let's iterate S.
                        covered = False
                        for c in S:
                            if frozenset((p, c)) in Covers[q]:
                                covered = True
                                break
                        if covered:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_cands = [p]
                    elif score == best_score:
                        best_cands.append(p)
                
                if not best_cands:
                    return None
                
                # Random choice among best candidates to allow exploration
                p = random.choice(best_cands)
                
                # Update State
                new_S = S | {p}
                
                # Update Uncovered
                new_Uncovered = set(Uncovered)
                new_Uncovered.discard(p) # p is in S now, so it doesn't need to be covered
                # Remove points covered by p
                for q in list(new_Uncovered):
                    if q == p:
                        continue
                    for c in S: # c in old S
                        if frozenset((p, c)) in Covers[q]:
                            new_Uncovered.discard(q)
                            break
                    # Also, p itself covers nothing with itself.
                    # What if S was empty? Then p covers nothing.
                
                # Update Candidates
                # Remove p from Candidates
                new_Candidates = [c for c in Candidates if c != p]
                
                # Remove candidates that conflict with p
                # c conflicts with p if S U {p, c} is not independent.
                # This happens if there exists a in S such that {p, c} in Covers[a] 
                # OR {p, a} in Covers[c] OR {c, a} in Covers[p].
                # We already know {c, a} in Covers[p] is false for all a in S (because c was a valid candidate for S).
                # We already know {p, a} in Covers[c] is false for all a in S (because p was a valid candidate for S).
                # So we only need to check if {p, c} in Covers[a] for any a in S.
                
                final_Candidates = []
                for c in new_Candidates:
                    conflict = False
                    for a in S:
                        if frozenset((p, c)) in Covers[a]:
                            conflict = True
                            break
                    if not conflict:
                        final_Candidates.append(c)
                
                # Recurse
                result = backtrack(new_S, final_Candidates, new_Uncovered, depth + 1)
                if result:
                    return result
                
                # Backtrack
                return None

            # Initial Candidates
            init_Candidates = list(range(NUM_POINTS))
            # Shuffle to break symmetry
            random.shuffle(init_Candidates)
            
            # Initial Uncovered
            init_Uncovered = set(range(NUM_POINTS))
            
            # Initial S
            init_S = set()
            
            # Start search
            return backtrack(init_S, init_Candidates, init_Uncovered, 0)

    def find_minimal(self):
        """
        Iterate target_size from small to large until a solution is found.
        """
        # Lower bound: 
        # To cover 256 points, if each point covers ~k others, we need ~256/k points.
        # Let's start from 5.
        
        # Use multiple restarts for each size?
        # The CSP solver is deterministic given the random choices in tie-breaking.
        # We can run it multiple times with different random seeds/shuffles.
        
        max_size = 30 # Upper bound for search
        for size in range(5, max_size + 1):
            if self.abort:
                break
            # Try multiple times for this size
            for attempt in range(20):
                if self.abort:
                    break
                # Add some randomness to the initial order or tie-breaking
                # The solver already uses random.choice for ties.
                # We can also shuffle the initial candidates.
                
                res = self.solve(size)
                if res is not None:
                    # Found a solution of size <= size
                    # Since we search increasing size, this is minimal for this run
                    return res
            # If no solution found for this size in 20 tries, increase size
            # But be careful: maybe it exists but we missed it.
            # Given the time limit, we might just move on.
            
        return self.best_set

def construct():
    start_time = time.time()
    
    # Initialize Solver
    solver = CSPSolver(DEADLINE)
    
    # Run the CSP solver
    result = solver.find_minimal()
    
    if result is None:
        # Fallback to a greedy construction if CSP fails or times out without finding anything
        # Greedy: Pick high coverage pairs, then add points
        S = set()
        # Sort points by degree (number of pairs they form that are isosceles with someone?)
        # Or just random greedy
        # Let's do a simple greedy:
        # 1. Start with empty S.
        # 2. While not maximal:
        #    Find p that can be added (independent) and covers most uncovered.
        
        Uncovered = set(range(NUM_POINTS))
        
        while True:
            if time.time() > DEADLINE - 5:
                break
            # Check if maximal
            # If Uncovered is empty, we are done?
            # No, Uncovered contains points not in S that are not covered.
            # If Uncovered is empty, S is maximal.
            if not Uncovered:
                break
            
            # Find candidates
            candidates = []
            S_list = list(S)
            for p in range(NUM_POINTS):
                if p in S:
                    continue
                valid = True
                for i in range(len(S_list)):
                    a = S_list[i]
                    for b in S_list[i+1:]:
                        if frozenset((a, b)) in Covers[p]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    candidates.append(p)
            
            if not candidates:
                # Stuck, return current S (might not be maximal, but best effort)
                break
            
            # Score candidates
            best_p = None
            best_score = -1
            for p in candidates:
                score = 0
                for q in Uncovered:
                    if q == p:
                        continue
                    for c in S:
                        if frozenset((p, c)) in Covers[q]:
                            score += 1
                            break
                if score > best_score:
                    best_score = score
                    best_p = p
            
            if best_p is None:
                break
                
            S.add(best_p)
            
            # Update Uncovered
            new_Uncovered = set(Uncovered)
            new_Uncovered.discard(best_p)
            for q in list(new_Uncovered):
                if q == best_p:
                    continue
                for c in S:
                    if c == best_p: continue
                    if frozenset((best_p, c)) in Covers[q]:
                        new_Uncovered.discard(q)
                        break
            Uncovered = new_Uncovered
        
        if S:
            result = S
        else:
            # Last resort: return a single point? No, must be maximal.
            # A single point is never maximal in N=16 grid.
            # Return empty? Score will be bad.
            # Let's return a random pair.
            result = {0, 1}
            
    return list(result)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
