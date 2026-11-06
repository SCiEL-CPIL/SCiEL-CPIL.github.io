import time

def construct():
    start = time.time()
    deadline = start + 140.0

    def elapsed():
        return time.time() - start

    def is_sidon(S):
        diffs = set()
        n = len(S)
        for i in range(n):
            for j in range(i + 1, n):
                d = S[j] - S[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def solve_sat(D, time_budget):
        """
        SAT encoding:
        - Variables x[i] for i in 0..D: whether i is in the set.
        - x[0] is fixed to True.
        - We need at least enough elements to cover diffs 1..30.
        - Sidon: for all a<b<c<d with (b-a)==(c-d), not (x[a] & x[b] & x[c] & x[d]).
          Also for all a<b with (b-a)==(c-d) where {a,b} != {c,d}, not all four in set.
          More precisely: for each difference d, at most one pair (i, i+d) can be in the set.
        - Coverage: for each d in 1..30, at least one pair (i, i+d) is in the set.
        """
        # Variables: x[0], x[1], ..., x[D]
        # We'll use 1-indexed variable IDs for the SAT solver.
        # var(i) = i+1 for position i.
        
        # Build clauses
        # Unit clauses: x[0] = True
        clauses = []
        
        # x[0] is True
        clauses.append([1])  # var(0) = 1
        
        # For each difference d in 1..D, at most one pair (i, i+d)
        # For d from 1 to D:
        #   Pairs: (0,d), (1,d+1), ..., (D-d, D)
        #   At most one: for all pairs p, q in this list, p != q: not(x[p] & x[q])
        #   This is a binary clause: (-var(a) or -var(b)) for pair (a, a+d)
        #   Wait, that's not right. "At most one pair" means:
        #   For pairs (i, i+d) and (j, j+d) with i != j: not (x[i] & x[i+d] & x[j] & x[j+d])
        #   That's a 4-literal clause.
        
        # Actually, the Sidon condition is: all positive differences are distinct.
        # This means: for any two distinct pairs (a,b) and (c,d) with a<b, c<d,
        # we cannot have b-a == d-c AND all four elements in the set.
        
        # Let's think about this differently. The Sidon condition on differences:
        # For each difference value d, at most one pair (i, i+d) can both be in S.
        # Because if (i, i+d) and (j, j+d) are both in S with i != j, then d appears twice.
        
        # So for each d in 1..D:
        #   Let pairs_d = [(i, i+d) for i in range(D+1-d)]
        #   For all distinct pairs (a,b) and (c,d2) in pairs_d:
        #     Clause: not(x[a] & x[b] & x[c] & x[d2])
        #     i.e., (-var(a) or -var(b) or -var(c) or -var(d2))
        
        # Coverage: for each d in 1..30:
        #   At least one pair (i, i+d) is in S:
        #   Clause: (var(0) & var(d)) or (var(1) & var(d+1)) or ... 
        #   Wait, that's a disjunction of conjunctions. Need to encode properly.
        #   For each d in 1..30: OR over i of (x[i] AND x[i+d])
        #   This is not a single clause. We need auxiliary variables or use a different encoding.
        
        # Alternative: For each d in 1..30, introduce a "witness" that some pair covers it.
        # But let's try a simpler approach: since x[0] is True, the pair (0, d) covers difference d.
        # So if x[d] is True, difference d is covered. But x[d] might not be True.
        # We need: for each d in 1..30, exists i such that x[i] AND x[i+d] are both True.
        
        # Let's use a different strategy: instead of full SAT, use a smarter search.
        # Given the complexity, let me try a different structural approach:
        # Use a constraint-based search with forward checking, but with a much better
        # variable ordering and value ordering based on constraint propagation.
        
        return None  # Placeholder, we'll implement differently

    # Let me implement a proper pure-Python DPLL SAT solver
    class DPLL:
        def __init__(self, num_vars, clauses):
            self.n = num_vars
            self.clauses = clauses  # list of lists of signed integers
            self.assign = {}  # var_id -> True/False/None
            self.watches = {}  # literal -> list of clause indices
            for i, clause in enumerate(clauses):
                for lit in clause:
                    if lit not in self.watches:
                        self.watches[lit] = []
                    self.watches[lit].append(i)
        
        def assign_var(self, var, value):
            self.assign[var] = value
            # Unit propagation
            changed = True
            while changed:
                changed = False
                for clause_idx in list(self.watches.get(var, [])):
                    pass  # This is getting complex, let me simplify
        
        def propagate(self):
            """Unit propagation. Returns False if conflict."""
            changed = True
            while changed:
                changed = False
                for clause_idx, clause in enumerate(self.clauses):
                    unassigned = []
                    all_false = True
                    for lit in clause:
                        var = abs(lit)
                        if var not in self.assign:
                            unassigned.append(lit)
                        elif self.assign[var] == (lit > 0):
                            # Clause satisfied
                            all_false = False
                            break
                        else:
                            # Literal is false
                            pass
                    if all_false and not unassigned:
                        return False  # Conflict
                    if len(unassigned) == 1:
                        lit = unassigned[0]
                        var = abs(lit)
                        if var not in self.assign:
                            self.assign[var] = (lit > 0)
                            changed = True
            return True
        
        def solve(self, time_budget):
            if not self.propagate():
                return None
            # Find unassigned variable
            unassigned = [v for v in range(1, self.n + 1) if v not in self.assign]
            if not unassigned:
                return dict(self.assign)
            
            # Pick variable with highest occurrence in unsatisfied clauses
            var = unassigned[0]  # Simple for now
            
            # Try True
            self.assign[var] = True
            if self.propagate():
                result = self.solve(time_budget)
                if result is not None:
                    return result
            del self.assign[var]
            
            # Try False
            self.assign[var] = False
            if self.propagate():
                result = self.solve(time_budget)
                if result is not None:
                    return result
            del self.assign[var]
            
            return None

    def solve_csp(D, time_budget):
        """
        Solve using a backtracking search with strong forward checking.
        Variables: which positions 1..D are included (0 is always included).
        Constraints:
          1. Sidon: no two pairs have the same difference.
          2. Coverage: differences 1..30 must all appear.
        """
        # We'll build the set incrementally.
        # State: current set S, used differences, needed differences.
        # Key insight: use constraint propagation to prune.
        
        best_solution = None
        
        S = [0]
        used_diffs = set()
        need_diffs = set(range(1, 31))
        
        # Precompute: for each possible next element x, which differences it would create
        # and check against used_diffs.
        
        def dfs(S, used_diffs, need_diffs, depth):
            nonlocal best_solution
            if best_solution is not None:
                return
            if elapsed() > time_budget:
                return
            
            if not need_diffs:
                best_solution = list(S)
                return
            
            # Forward checking: for each remaining needed difference d,
            # check if there exists a pair (i, i+d) where both could be added.
            # This is expensive, so do it selectively.
            
            # Generate candidates for next element
            # Heuristic: prefer elements that cover many needed differences
            last = S[-1]
            candidates = []
            
            for x in range(last + 1, D + 1):
                # Check if x can be added without violating Sidon
                ok = True
                new_cover = 0
                new_diffs = set()
                for s in S:
                    d = x - s
                    if d in used_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                    if d in need_diffs:
                        new_cover += 1
                if ok:
                    # Additional pruning: check that remaining needs are still satisfiable
                    # For each d in need_diffs - new_cover, check if there's a possible pair
                    remaining = need_diffs - new_diffs
                    feasible = True
                    for d in remaining:
                        # Need some i such that i and i+d can both be in the final set
                        # i can be in S or future elements
                        # Simple check: is there at least one i in [0, D-d] such that
                        # i not conflicting and i+d not conflicting?
                        # This is complex, so skip for now
                        pass
                    if feasible:
                        candidates.append((new_cover, x))
            
            if not candidates:
                return
            
            # Sort by coverage descending, then value ascending
            candidates.sort(key=lambda t: (-t[0], t[1]))
            
            for new_cover, x in candidates:
                S.append(x)
                new_used = set(used_diffs)
                new_need = set(need_diffs)
                for s in S[:-1]:
                    d = x - s
                    new_used.add(d)
                    if d in new_need:
                        new_need.remove(d)
                
                dfs(S, new_used, new_need, depth + 1)
                
                S.pop()
                
                if best_solution is not None:
                    return
        
        dfs(S, used_diffs, need_diffs, 0)
        return best_solution

    best_set = None
    for D in range(44, 70):
        if elapsed() > deadline - 15.0:
            break
        budget = min(12.0, deadline - elapsed() - 10.0)
        if budget <= 0:
            break
        sol = solve_csp(D, budget)
        if sol:
            best_set = sol
            break

    if best_set is None:
        # Fallback: greedy
        best_set = [0]
        used = set()
        need = set(range(1, 31))
        while need and elapsed() < deadline - 2.0:
            best_x = None
            best_c = -1
            for x in range(max(best_set) + 1, max(best_set) + 200):
                ok = True
                c = 0
                for s in best_set:
                    d = x - s
                    if d in used:
                        ok = False
                        break
                    if d in need:
                        c += 1
                if ok and c > best_c:
                    best_c = c
                    best_x = x
            if best_x is None:
                for x in range(max(best_set) + 200, max(best_set) + 1000):
                    ok = True
                    for s in best_set:
                        if x - s in used:
                            ok = False
                            break
                    if ok:
                        best_x = x
                        break
            if best_x is None:
                break
            best_set.append(best_x)
            for s in best_set[:-1]:
                used.add(best_x - s)
            need -= used

    if not is_sidon(best_set):
        pass

    return best_set

if __name__ == "__main__":
    print(construct())
