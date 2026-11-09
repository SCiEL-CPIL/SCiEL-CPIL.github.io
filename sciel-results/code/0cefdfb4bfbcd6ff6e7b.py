import time
import random
import math
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    start_time = time.time()
    deadline = start_time + 285

    # Precomputation
    var_clauses = [[] for _ in range(n)]
    clauses_store = []
    
    for i, clause in enumerate(clauses):
        cl_store = []
        for l in clause:
            v = abs(l) - 1
            sign = 1 if l > 0 else -1
            cl_store.append((v, sign))
            var_clauses[v].append((i, sign))
        clauses_store.append(cl_store)
    
    m = len(clauses)
    
    def is_satisfied(assignment, clause_idx):
        for v, sign in clauses_store[clause_idx]:
            if assignment[v] == (sign == 1):
                return True
        return False
        
    def count_unsat(assignment):
        cnt = 0
        for i in range(m):
            if not is_satisfied(assignment, i):
                cnt += 1
        return cnt

    # Spectral Initialization
    initial_assignment = None
    try:
        B = np.zeros((n, m), dtype=np.float32)
        for c_idx, clause in enumerate(clauses):
            for v, sign in clauses_store[c_idx]:
                B[v, c_idx] = sign
                
        cov = np.dot(B.T, B)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        principal_vec = eigenvectors[:, -1]
        initial_assignment = [x > 0 for x in principal_vec]
    except Exception:
        initial_assignment = None

    def walk_sat_fast(rng, max_steps=50000, start_assign=None, temp=5.0):
        if start_assign is None:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        else:
            assignment = start_assign.copy()
            
        is_unsat = [False] * m
        unsat_count = 0
        for c in range(m):
            if not is_satisfied(assignment, c):
                is_unsat[c] = True
                unsat_count += 1
                
        if unsat_count == 0:
            return assignment
            
        best_unsat = unsat_count
        best_assignment = assignment.copy()
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if unsat_count == 0:
                return assignment
                
            # Pick a random unsatisfied clause
            # Simple rejection sampling
            while True:
                cl_idx = rng.randint(0, m - 1)
                if is_unsat[cl_idx]:
                    break
                
            # Pick variable from clause
            v_to_flip = clauses_store[cl_idx][rng.randint(0, 2)][0]
            
            # Flip
            assignment[v_to_flip] = not assignment[v_to_flip]
            
            # Update unsat status for affected clauses
            delta = 0
            for c_idx, c_sign in var_clauses[v_to_flip]:
                was_unsat = is_unsat[c_idx]
                is_sat_now = False
                for v2, s2 in clauses_store[c_idx]:
                    if assignment[v2] == (s2 == 1):
                        is_sat_now = True
                        break
                
                if is_sat_now:
                    if was_unsat:
                        delta -= 1
                        is_unsat[c_idx] = False
                else:
                    if not was_unsat:
                        delta += 1
                        is_unsat[c_idx] = True
                        
            new_unsat = unsat_count + delta
            
            # Acceptance (Simulated Annealing)
            if delta <= 0 or (temp > 0.1 and rng.random() < math.exp(-delta / temp)):
                unsat_count = new_unsat
                if unsat_count < best_unsat:
                    best_unsat = unsat_count
                    best_assignment = assignment.copy()
            else:
                # Revert
                assignment[v_to_flip] = not assignment[v_to_flip]
                
            temp *= 0.999
            if temp < 0.01:
                temp = 0.01
                
        if best_unsat == 0:
            return best_assignment
        return None

    # Strategy 1: Spectral Init + Fast WalkSAT
    rng = random.Random(42)
    sol = walk_sat_fast(rng, max_steps=50000, start_assign=initial_assignment, temp=5.0)
    if sol is not None and count_unsat(sol) == 0:
        return sol
        
    # Strategy 2: Multiple random restarts with Fast WalkSAT
    for r in range(200):
        if time.time() > deadline:
            break
        sol = walk_sat_fast(rng, max_steps=20000, start_assign=None, temp=5.0 + r * 0.1)
        if sol is not None and count_unsat(sol) == 0:
            return sol
            
    # Fallback
    return [rng.random() < 0.5 for _ in range(n)]
