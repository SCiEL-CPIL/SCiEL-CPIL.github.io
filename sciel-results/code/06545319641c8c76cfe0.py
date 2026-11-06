import time
import random

class CDCLSolver:
    def __init__(self, n, clauses):
        self.n = n
        self.clauses = clauses
        self.assignment = [0] * (n + 1)
        self.trail = []
        self.trail_lim = []
        self.reason = [-1] * (n + 1)
        self.level = [0] * (n + 1)
        self.activity = [0.0] * (n + 1)
        self.var_inc = 1.0
        self.decay = 0.95
        
        self.all_clauses = [list(c) for c in clauses]
        self.watched_lits = [[c[0], c[1]] for c in self.all_clauses]
        self.watches = [[] for _ in range(2 * n)]
        
        for ci in range(len(self.all_clauses)):
            l1, l2 = self.watched_lits[ci]
            self.watches[self.lit_to_idx(l1)].append(ci)
            self.watches[self.lit_to_idx(l2)].append(ci)
            
    def lit_to_idx(self, lit):
        if lit > 0:
            return lit - 1
        else:
            return self.n - lit - 1
            
    def propagate(self):
        qhead = 0
        while qhead < len(self.trail):
            p = self.trail[qhead]
            qhead += 1
            val = self.assignment[p]
            lit = p if val == 1 else -p
            idx = self.lit_to_idx(lit)
            
            new_watches = []
            for ci in self.watches[idx]:
                clause = self.all_clauses[ci]
                w1, w2 = self.watched_lits[ci]
                
                if w1 == lit:
                    w1, w2 = w2, w1
                    self.watched_lits[ci] = [w1, w2]
                
                v2 = abs(w2)
                if self.assignment[v2] == 1 and (w2 > 0) or (self.assignment[v2] == -1 and w2 < 0):
                    new_watches.append(ci)
                elif self.assignment[v2] == -1 and (w2 > 0) or (self.assignment[v2] == 1 and w2 < 0):
                    new_w = None
                    for l in clause:
                        if l == lit or l == w2:
                            continue
                        v = abs(l)
                        if self.assignment[v] == 0:
                            new_w = l
                            break
                        if (self.assignment[v] == 1 and l > 0) or (self.assignment[v] == -1 and l < 0):
                            new_w = l
                            break
                            
                    if new_w is None:
                        sat = False
                        for l in clause:
                            v = abs(l)
                            if (self.assignment[v] == 1 and l > 0) or (self.assignment[v] == -1 and l < 0):
                                sat = True
                                break
                        if not sat:
                            return ci
                        else:
                            new_w = l
                            
                    self.watched_lits[ci] = [w1, new_w]
                    self.watches[self.lit_to_idx(new_w)].append(ci)
                else:
                    new_watches.append(ci)
                    
            self.watches[idx] = new_watches
        return None

    def analyze(self, conflict_ci):
        seen = set()
        new_clause = []
        bt_level = 0
        cur_clause = self.all_clauses[conflict_ci]
        
        while True:
            max_level = -1
            p = None
            for l in cur_clause:
                v = abs(l)
                if v in seen:
                    continue
                if self.level[v] > max_level:
                    max_level = self.level[v]
                    p = l
                    
            if p is None:
                break
                
            seen.add(abs(p))
            
            cnt = 0
            for l in cur_clause:
                if self.level[abs(l)] == self.level[abs(p)]:
                    cnt += 1
            if cnt == 1:
                new_clause.append(p)
                for l in cur_clause:
                    if l == p:
                        continue
                    if self.level[abs(l)] > bt_level:
                        bt_level = self.level[abs(l)]
                break
            
            r_ci = self.reason[abs(p)]
            if r_ci == -1:
                new_clause.append(p)
                for l in cur_clause:
                    if l == p:
                        continue
                    if self.level[abs(l)] > bt_level:
                        bt_level = self.level[abs(l)]
                break
                
            r_clause = self.all_clauses[r_ci]
            for l in r_clause:
                if l == p or abs(l) in seen:
                    continue
                cur_clause.append(l)
                
        self.all_clauses.append(new_clause)
        new_ci = len(self.all_clauses) - 1
        self.watched_lits.append([new_clause[0], new_clause[1]])
        self.watches[self.lit_to_idx(new_clause[0])].append(new_ci)
        self.watches[self.lit_to_idx(new_clause[1])].append(new_ci)
        
        return new_ci, bt_level

    def solve(self, time_limit):
        start = time.time()
        while time.time() - start < time_limit:
            conflict_ci = self.propagate()
            if conflict_ci is not None:
                new_ci, bt_level = self.analyze(conflict_ci)
                while len(self.trail_lim) > bt_level:
                    lim = self.trail_lim.pop()
                    while len(self.trail) > lim:
                        v = self.trail.pop()
                        self.assignment[v] = 0
                        self.reason[v] = -1
                        self.level[v] = 0
                for l in self.all_clauses[new_ci]:
                    self.bump(abs(l))
                self.var_inc /= self.decay
            else:
                unassigned = [v for v in range(1, self.n + 1) if self.assignment[v] == 0]
                if not unassigned:
                    # Check satisfaction
                    for c in self.all_clauses:
                        sat = False
                        for l in c:
                            v = abs(l)
                            if (self.assignment[v] == 1 and l > 0) or (self.assignment[v] == -1 and l < 0):
                                sat = True
                                break
                        if not sat:
                            break
                    else:
                        return self.assignment[1:]
                    return None
                v = self.select_var()
                val = 1 if random.random() < 0.5 else -1
                self.assignment[v] = val
                self.trail.append(v)
                self.level[v] = len(self.trail_lim)
                self.reason[v] = -1
                self.trail_lim.append(len(self.trail))
                self.bump(v)
        return None

    def select_var(self):
        best_v = 0
        best_act = -1
        for v in range(1, self.n + 1):
            if self.assignment[v] == 0:
                if self.activity[v] > best_act:
                    best_act = self.activity[v]
                    best_v = v
        return best_v

    def bump(self, v):
        self.activity[v] += self.var_inc
        if self.activity[v] > 1e10:
            for i in range(1, self.n + 1):
                self.activity[i] *= 1e-10
            self.var_inc *= 1e-10

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    start_time = time.time()
    time_limit = 290.0
    
    cdcl = CDCLSolver(n, clauses)
    sol = cdcl.solve(180.0)
    if sol:
        return sol
        
    # WalkSAT Fallback
    # ...
