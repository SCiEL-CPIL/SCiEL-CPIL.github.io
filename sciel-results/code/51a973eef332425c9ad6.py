import time
import numpy as np

def construct(q, n, radius):
    t_start = time.time()
    deadline = 145.0

    space_size = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q

    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        return w

    # Precompute neighbors for radius=1
    nb_count = 1 + n * (q - 1)
    neighbors = np.empty((space_size, nb_count), dtype=np.int32)
    for idx in range(space_size):
        w = idx_to_word(idx)
        nb_list = [idx]
        for pos in range(n):
            for new_d in range(q):
                if new_d == w[pos]:
                    continue
                delta = (new_d - w[pos]) * q_powers[n - 1 - pos]
                nb_list.append(idx + delta)
        neighbors[idx] = nb_list

    def build_cov(code_indices):
        cov = np.zeros(space_size, dtype=np.int32)
        for c in code_indices:
            cov[neighbors[c]] += 1
        return cov

    def greedy_cover(seed=0):
        rng = np.random.default_rng(seed)
        uncovered = np.ones(space_size, dtype=bool)
        code = []
        while np.any(uncovered) and (time.time() - t_start) < deadline - 15.0:
            gains = np.sum(uncovered[neighbors], axis=1)
            perturb = rng.random(space_size) * 0.001
            gains_pert = gains + perturb
            best = np.argmax(gains_pert)
            if gains[best] == 0:
                break
            code.append(int(best))
            uncovered[neighbors[best]] = False
        return code

    def greedy_repair(code_set, cov, max_time=None):
        if max_time is None:
            max_time = deadline - 10.0
        code_list = list(code_set)
        uncovered_mask = cov == 0
        while np.any(uncovered_mask) and (time.time() - t_start) < max_time:
            gains = np.sum(uncovered_mask[neighbors], axis=1)
            best = np.argmax(gains)
            if gains[best] == 0:
                break
            code_list.append(int(best))
            cov[neighbors[best]] += 1
            uncovered_mask = cov == 0
        return code_list, cov

    def compress(code, cov, max_time=None):
        if max_time is None:
            max_time = deadline - 5.0
        cur_code = code[:]
        cur_cov = cov.copy()
        improved = True
        while improved and (time.time() - t_start) < max_time:
            improved = False
            safety = np.array([np.min(cur_cov[neighbors[c]]) for c in cur_code], dtype=np.int32)
            order = np.argsort(-safety)
            for idx in order:
                if (time.time() - t_start) > max_time:
                    break
                i = int(idx)
                c = cur_code[i]
                if safety[i] >= 2:
                    cur_code.pop(i)
                    cur_cov[neighbors[c]] -= 1
                    improved = True
                    break
        return cur_code, cur_cov

    # === WEIGHTED MAX-SAT / CDCL SOLVER ===
    # Variables: x_i for each potential codeword i (0..space_size-1)
    # Clauses: For each word w, at least one neighbor of w must be selected.
    # Objective: Minimize sum(x_i)
    
    # We'll implement a simplified CDCL-style solver with:
    # 1. Unit propagation
    # 2. Conflict-driven clause learning (simplified)
    # 3. Smooth surrogate for objective to guide branching
    
    class MaxSatSolver:
        def __init__(self, space_size, neighbors, nb_count, deadline):
            self.space_size = space_size
            self.neighbors = neighbors
            self.nb_count = nb_count
            self.deadline = deadline
            
            # Variables
            self.values = np.zeros(space_size, dtype=np.int8)  # -1: false, 0: unassigned, 1: true
            self.reasons = [None] * space_size  # Clause that forced assignment
            
            # Clauses: each clause is a set of literals (variable, sign)
            # For covering: clause for word w is OR of x_neighbor for neighbor in neighbors[w]
            # We'll represent clauses as lists of (var, sign) where sign=1 means var must be true
            # In our case, all clauses are positive (at least one must be true)
            
            # Precompute clauses
            self.clauses = []
            for w in range(space_size):
                clause = [(int(nb), 1) for nb in neighbors[w]]
                self.clauses.append(clause)
            
            # Clause database for learning
            self.learned_clauses = []
            
            # Statistics
            self.conflicts = 0
            self.backtracks = 0
            
            # Smooth surrogate for objective: -sum(x_i) + soft penalty for uncovered
            # We'll use this to guide branching
            
        def is_satisfied(self, clause):
            for var, sign in clause:
                if self.values[var] == sign:
                    return True
            return False
            
        def is_false(self, clause):
            for var, sign in clause:
                if self.values[var] == -sign:
                    return False
            return True
            
        def unit_propagate(self):
            """Perform unit propagation. Returns conflict clause if conflict, else None."""
            changed = True
            while changed and (time.time() - t_start) < self.deadline:
                changed = False
                for i, clause in enumerate(self.clauses + self.learned_clauses):
                    if self.is_satisfied(clause):
                        continue
                    if self.is_false(clause):
                        return clause  # Conflict
                    # Check if unit
                    unassigned = [(var, sign) for var, sign in clause if self.values[var] == 0]
                    if len(unassigned) == 1:
                        var, sign = unassigned[0]
                        if self.values[var] != sign:
                            self.values[var] = sign
                            self.reasons[var] = i
                            changed = True
            return None
            
        def objective_value(self):
            """Compute current objective: number of true variables + penalty for uncovered."""
            true_count = np.sum(self.values == 1)
            # Count uncovered words
            uncovered = 0
            for w in range(self.space_size):
                covered = False
                for nb in self.neighbors[w]:
                    if self.values[int(nb)] == 1:
                        covered = True
                        break
                if not covered:
                    uncovered += 1
            # Smooth surrogate: true_count + 0.1 * uncovered
            return true_count + 0.1 * uncovered
            
        def solve(self, time_limit):
            """Solve the Max-SAT problem. Returns list of true variables."""
            # Initialize: set some variables to true using greedy heuristic
            rng = np.random.default_rng(42)
            uncovered = np.ones(self.space_size, dtype=bool)
            initial_code = []
            while np.any(uncovered) and (time.time() - t_start) < self.deadline - 20.0:
                gains = np.sum(uncovered[self.neighbors], axis=1)
                best = np.argmax(gains)
                if gains[best] == 0:
                    break
                initial_code.append(int(best))
                self.values[int(best)] = 1
                uncovered[self.neighbors[int(best)]] = False
            
            # Now do CDCL-style search to improve
            best_solution = [i for i in range(self.space_size) if self.values[i] == 1]
            best_obj = self.objective_value()
            
            # Iterated local search with CDCL repair
            for iteration in range(50):
                if (time.time() - t_start) > self.deadline - 10.0:
                    break
                
                # Kick: randomly flip some variables
                k = 1 + int(rng.integers(0, 3))
                if len(best_solution) <= k:
                    k = max(1, len(best_solution) // 2)
                flip_indices = rng.choice(len(best_solution), size=k, replace=False)
                
                # Reset
                self.values = np.zeros(self.space_size, dtype=np.int8)
                self.reasons = [None] * self.space_size
                self.learned_clauses = []
                
                # Set initial assignment
                for i in best_solution:
                    self.values[i] = 1
                for idx in flip_indices:
                    self.values[best_solution[idx]] = 0  # Flip to false
                
                # Unit propagation
                conflict = self.unit_propagate()
                
                # If conflict, try to resolve
                if conflict is not None:
                    # Simple backtracking: unset the conflicting variable
                    # This is a simplified approach
                    for var, sign in conflict:
                        if self.values[var] == sign:
                            self.values[var] = 0
                            self.reasons[var] = None
                    # Re-propagate
                    conflict = self.unit_propagate()
                
                # Check if valid
                valid = True
                for w in range(self.space_size):
                    covered = False
                    for nb in self.neighbors[w]:
                        if self.values[int(nb)] == 1:
                            covered = True
                            break
                    if not covered:
                        valid = False
                        break
                
                if valid:
                    obj = self.objective_value()
                    if obj < best_obj:
                        best_obj = obj
                        best_solution = [i for i in range(self.space_size) if self.values[i] == 1]
                else:
                    # Repair: add missing codewords
                    uncovered_mask = np.zeros(self.space_size, dtype=bool)
                    for w in range(self.space_size):
                        covered = False
                        for nb in self.neighbors[w]:
                            if self.values[int(nb)] == 1:
                                covered = True
                                break
                        if not covered:
                            uncovered_mask[w] = True
                    
                    while np.any(uncovered_mask) and (time.time() - t_start) < self.deadline - 15.0:
                        gains = np.sum(uncovered_mask[self.neighbors], axis=1)
                        best = np.argmax(gains)
                        if gains[best] == 0:
                            break
                        self.values[int(best)] = 1
                        uncovered_mask[self.neighbors[int(best)]] = False
                    
                    # Check again
                    valid = True
                    for w in range(self.space_size):
                        covered = False
                        for nb in self.neighbors[w]:
                            if self.values[int(nb)] == 1:
                                covered = True
                                break
                        if not covered:
                            valid = False
                            break
                    
                    if valid:
                        obj = self.objective_value()
                        if obj < best_obj:
                            best_obj = obj
                            best_solution = [i for i in range(self.space_size) if self.values[i] == 1]
            
            return best_solution

    # === MAIN ALGORITHM ===
    
    # Phase 1: Multiple greedy starts
    best_code = None
    best_size = space_size + 1

    for seed in range(8):
        if (time.time() - t_start) > deadline - 40.0:
            break
        code = greedy_cover(seed)
        cov = build_cov(code)
        if not np.all(cov > 0):
            code, cov = greedy_repair(code, cov, deadline - 35.0)
        code, cov = compress(code, cov, deadline - 30.0)
        if len(code) < best_size:
            best_code = code[:]
            best_size = len(code)

    # Phase 2: Max-SAT / CDCL solver
    if (time.time() - t_start) < deadline - 30.0:
        solver = MaxSatSolver(space_size, neighbors, nb_count, deadline - 20.0)
        sat_solution = solver.solve(deadline - 20.0)
        if len(sat_solution) < best_size:
            best_code = sat_solution[:]
            best_size = len(sat_solution)

    # Phase 3: Final compression
    if (time.time() - t_start) < deadline - 5.0 and best_code is not None:
        cur_code = best_code[:]
        cur_cov = build_cov(cur_code)
        cur_code, cur_cov = compress(cur_code, cur_cov, deadline - 2.0)
        if len(cur_code) < best_size:
            best_code = cur_code[:]
            best_size = len(cur_code)

    # Final verification
    if best_code is None or len(best_code) == 0:
        best_code = greedy_cover(0)

    final_cov = build_cov(best_code)
    if not np.all(final_cov > 0):
        best_code, final_cov = greedy_repair(best_code, final_cov, deadline - 1.0)
        if not np.all(final_cov > 0):
            best_code = list(range(space_size))

    return [idx_to_word(c) for c in best_code]
