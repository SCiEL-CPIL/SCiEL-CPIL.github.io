# EVOLVE-BLOCK-START
import time
import random

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline: 300s limit, use 295s to be safe
    deadline = time.time() + 295.0

    # Preprocessing
    cvars = []
    csigns = []
    for cl in clauses:
        vs = []
        ss = []
        for l in cl:
            v = abs(l) - 1
            s = 1 if l > 0 else -1
            vs.append(v)
            ss.append(s)
        cvars.append(vs)
        csigns.append(ss)
        
    var_clauses = [[] for _ in range(n)]
    for ci in range(m):
        for v in cvars[ci]:
            var_clauses[v].append(ci)
            
    def check_all(assignment):
        for ci in range(m):
            sat = False
            vs = cvars[ci]
            ss = csigns[ci]
            for k in range(len(vs)):
                v = vs[k]
                s = ss[k]
                if s == 1:
                    if assignment[v]: sat = True; break
                else:
                    if not assignment[v]: sat = True; break
            if not sat:
                return False
        return True

    # DPLL Solver
    class DPLL:
        def __init__(self):
            self.assignment = [None] * n
            self.trail = []
            self.trail_lim = [0]
            self.activity = [0.0] * n
            self.var_inc = 1.0 / (1.0 + n)
            self.var_decay = 0.95
            
            # Watched Literals
            # lit_id = v * 2 + (0 if pos else 1)
            self.watch = [[] for _ in range(2 * n)]
            for ci in range(m):
                if len(cvars[ci]) < 2:
                    continue
                v1, s1 = cvars[ci][0], csigns[ci][0]
                v2, s2 = cvars[ci][1], csigns[ci][1]
                id1 = v1 * 2 + (0 if s1 == 1 else 1)
                id2 = v2 * 2 + (0 if s2 == 1 else 1)
                self.watch[id1].append(ci)
                self.watch[id2].append(ci)
                
        def lit_id(self, v, s):
            return v * 2 + (0 if s == 1 else 1)
            
        def is_false(self, v, s):
            if self.assignment[v] is None: return False
            val = self.assignment[v]
            if s == 1: return not val
            else: return val
            
        def is_true(self, v, s):
            if self.assignment[v] is None: return False
            val = self.assignment[v]
            if s == 1: return val
            else: return not val
            
        def propagate(self):
            # Queue of unit clauses
            # We scan for unit clauses.
            # Optimization: Keep a list of potentially unit clauses.
            # For simplicity, scan all clauses.
            while True:
                if time.time() > deadline:
                    return "timeout"
                
                found_unit = False
                for ci in range(m):
                    vs = cvars[ci]
                    ss = csigns[ci]
                    if len(vs) == 0:
                        return "conflict"
                        
                    unassigned = []
                    false_count = 0
                    for k in range(len(vs)):
                        v = vs[k]
                        s = ss[k]
                        if self.assignment[v] is None:
                            unassigned.append((v, s, k))
                        elif self.is_false(v, s):
                            false_count += 1
                        else:
                            break
                    else:
                        if len(unassigned) == 1:
                            v, s, k = unassigned[0]
                            # Assign
                            self.assignment[v] = (s == 1)
                            self.trail.append(v)
                            found_unit = True
                            
                            # Update watches
                            # The literal (v, s) is now TRUE.
                            # Clauses watching (v, s) are satisfied.
                            # We don't need to move watches for them.
                            # But we must check if any clause becomes unit due to this assignment.
                            # The loop continues, so it will catch it.
                            break
                        elif len(unassigned) == 0 and false_count == len(vs):
                            return "conflict"
                            
                if not found_unit:
                    return "none"
                    
        def select_var(self):
            max_act = -1
            max_var = -1
            for v in range(n):
                if self.assignment[v] is None:
                    if self.activity[v] > max_act:
                        max_act = self.activity[v]
                        max_var = v
            return max_var
            
        def search(self):
            if time.time() > deadline:
                return "timeout"
                
            status = self.propagate()
            if status == "conflict":
                return "unsat"
            if status == "timeout":
                return "timeout"
                
            while True:
                if time.time() > deadline:
                    return "timeout"
                    
                if len(self.trail) == n:
                    return "sat"
                    
                v = self.select_var()
                if v == -1:
                    return "sat"
                    
                # Decide
                self.assignment[v] = True
                self.trail.append(v)
                
                status = self.propagate()
                if status == "sat":
                    return "sat"
                elif status == "conflict":
                    # Backtrack
                    # Simple backtrack: undo last decision
                    # This is inefficient but correct for small backtracks.
                    # For robustness, if we conflict, we restart with a different heuristic or just return timeout.
                    # Let's try a simple restart:
                    # Reset assignment and trail?
                    # No, let's just return "conflict" and let the outer loop handle it (e.g., by restarting DPLL or falling back to WalkSAT).
                    return "conflict"
                elif status == "timeout":
                    return "timeout"

    # Run DPLL
    dpll = DPLL()
    result = dpll.search()
    
    if result == "sat":
        final_assign = [bool(dpll.assignment[v]) for v in range(n)]
        if check_all(final_assign):
            return final_assign
            
    # Fallback to WalkSAT
    def run_walksat():
        assignment = [random.random() < 0.5 for _ in range(n)]
        unsat_count = [0] * m
        total_unsat = 0
        for ci in range(m):
            vs = cvars[ci]
            ss = csigns[ci]
            sat = False
            for k in range(len(vs)):
                v = vs[k]
                s = ss[k]
                if s == 1:
                    if assignment[v]: sat = True; break
                else:
                    if not assignment[v]: sat = True; break
            if not sat:
                unsat_count[ci] = 1
                total_unsat += 1
                
        if total_unsat == 0:
            return assignment
            
        best_unsat = total_unsat
        best_assignment = assignment[:]
        
        while time.time() < deadline - 0.5:
            unsat_clauses = [ci for ci in range(m) if unsat_count[ci] > 0]
            if not unsat_clauses:
                return assignment
                
            if len(unsat_clauses) < best_unsat:
                best_unsat = len(unsat_clauses)
                best_assignment = assignment[:]
                
            ci = random.choice(unsat_clauses)
            vs = cvars[ci]
            ss = csigns[ci]
            
            scores = []
            for v in vs:
                score = 0
                for cj in var_clauses[v]:
                    if cj == ci: continue
                    if unsat_count[cj] == 0: continue
                    vs_cj = cvars[cj]
                    ss_cj = csigns[cj]
                    for k in range(len(vs_cj)):
                        if vs_cj[k] == v:
                            s_cj = ss_cj[k]
                            if s_cj == 1:
                                if not assignment[v]: score += 1
                            else:
                                if assignment[v]: score += 1
                            break
                scores.append(score)
                
            if random.random() < 0.85:
                max_score = max(scores)
                candidates = [i for i, s in enumerate(scores) if s == max_score]
                idx = random.choice(candidates)
            else:
                idx = random.randint(0, len(vs)-1)
                
            v_to_flip = vs[idx]
            old_val = assignment[v_to_flip]
            new_val = not old_val
            assignment[v_to_flip] = new_val
            
            for cj in var_clauses[v_to_flip]:
                vs_cj = cvars[cj]
                ss_cj = csigns[cj]
                was_unsat = unsat_count[cj] > 0
                
                idx_v = -1
                for k in range(len(vs_cj)):
                    if vs_cj[k] == v_to_flip:
                        idx_v = k
                        break
                if idx_v == -1: continue
                
                s_v = ss_cj[idx_v]
                lit_was_true = (old_val and s_v == 1) or (not old_val and s_v == -1)
                lit_now_true = (new_val and s_v == 1) or (not new_val and s_v == -1)
                
                if was_unsat:
                    if lit_now_true:
                        unsat_count[cj] = 0
                        total_unsat -= 1
                else:
                    if lit_was_true and not lit_now_true:
                        any_other_true = False
                        for k in range(len(vs_cj)):
                            if k == idx_v: continue
                            v_k = vs_cj[k]
                            s_k = ss_cj[k]
                            if (assignment[v_k] and s_k == 1) or (not assignment[v_k] and s_k == -1):
                                any_other_true = True
                                break
                        if not any_other_true:
                            unsat_count[cj] = 1
                            total_unsat += 1
                            
            if total_unsat == 0:
                return assignment
                
        return best_assignment

    result = run_walksat()
    if result is not None and check_all(result):
        return result
        
    return [False] * n

# EVOLVE-BLOCK-END
