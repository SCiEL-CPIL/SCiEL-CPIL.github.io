"""Synthetic planted 3-SAT solver (150 variables)."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 290  # safety margin before 300s limit

    # Precompute incidence matrix A
    # A[c, v] = 1 if literal is positive, -1 if negative, 0 if not present
    A = np.zeros((m, n), dtype=np.int8)
    for ci, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            A[ci, v] = sign

    def compute_sums(x):
        # x is vector of 1/-1
        # S = A @ x
        return A.dot(x)

    def count_unsat(S):
        # Clause is unsat if sum < 1 (i.e., -3 or -1)
        # Clause is sat if sum >= 1 (i.e., 1 or 3)
        return np.sum(S < 1)

    def get_deltas(x, S):
        """
        Compute the change in unsat count for flipping each variable.
        Returns a vector D of length n, where D[v] is the new unsat count 
        if we flip variable v (relative to current state, or absolute? 
        Let's compute absolute new unsat counts for all v).
        """
        # For each v, S_new = S - 2 * A[:, v] * x[v]
        # We want to count how many elements in S_new are < 1.
        # Instead of looping v, we can do this vectorized?
        # S is (m,), A is (m, n), x is (n,)
        # We want for each v: sum_{c} (S[c] - 2*A[c,v]*x[v] < 1)
        
        # Let's try a different vectorization.
        # The condition S[c] - 2*A[c,v]*x[v] < 1 is equivalent to:
        # If A[c,v]*x[v] == 1 (literal satisfied), then S[c] - 2 < 1 => S[c] < 3.
        #   Since max S is 3, S[c] < 3 means S[c] is 1 or -1 or -3.
        #   If S[c] was 3 (all sat), flipping breaks it.
        #   If S[c] was 1 (one sat, two unsat), flipping makes it -1 (unsat).
        #   So if literal is sat, flipping makes clause unsat ONLY IF S[c] >= 1? 
        #   Wait. If S[c]=1, literal is sat. New sum = -1. Unsat.
        #   If S[c]=3, literal is sat. New sum = 1. Sat.
        #   So if literal is sat, it becomes unsat iff S[c] == 1.
        
        # If A[c,v]*x[v] == -1 (literal unsat), then S[c] + 2 < 1 => S[c] < -1.
        #   Since min S is -3, S[c] < -1 means S[c] == -3.
        #   If S[c]=-3, literal is unsat. New sum = -1. Unsat.
        #   If S[c]=-1, literal is unsat. New sum = 1. Sat.
        #   So if literal is unsat, it becomes sat iff S[c] == -1.
        #   It remains unsat if S[c] == -3.

        # So, Delta Unsat for flipping v:
        # +1 for each clause c where A[c,v]*x[v] == 1 AND S[c] == 1
        # -1 for each clause c where A[c,v]*x[v] == -1 AND S[c] == -1
        
        # Let's compute this vectorized.
        # L = A * x  (m, n) -- literal satisfaction status (1 if sat, -1 if unsat)
        L = A * x[np.newaxis, :]
        
        # We need for each v:
        # count(L[:, v] == 1 AND S == 1) - count(L[:, v] == -1 AND S == -1)
        
        # Create masks
        # mask_sat_1: (m, n) where L==1 and S==1
        # mask_unsat_neg1: (m, n) where L==-1 and S==-1
        
        # This might be memory heavy (m*n booleans) but m=450, n=150 -> 67k booleans. Fine.
        
        mask_sat_1 = (L == 1) & (S[:, np.newaxis] == 1)
        mask_unsat_neg1 = (L == -1) & (S[:, np.newaxis] == -1)
        
        # Sum over clauses (axis 0) to get per-variable delta
        # D[v] = sum(mask_sat_1[:, v]) - sum(mask_unsat_neg1[:, v])
        D = np.sum(mask_sat_1, axis=0) - np.sum(mask_unsat_neg1, axis=0)
        
        return D

    def walksat_vectorized(x, S, rng, steps, deadline):
        """
        Vectorized WalkSAT / SA hybrid.
        """
        for _ in range(steps):
            if time.time() > deadline:
                return False
            
            unsat = count_unsat(S)
            if unsat == 0:
                return True

            # Get deltas for all variables
            D = get_deltas(x, S)
            
            # Heuristic: Pick a variable with minimal delta (most improvement)
            # But to allow exploration, pick from top K or use randomization
            # Let's pick the variable with the minimum D.
            # If there are ties, pick randomly among them.
            min_d = np.min(D)
            candidates = np.where(D == min_d)[0]
            
            if len(candidates) > 0:
                v = rng.choice(candidates)
            else:
                v = rng.randrange(n)
                
            # Flip v
            x[v] = -x[v]
            
            # Update S
            # S = S - 2 * A[:, v] * x_old
            # x_old = -x[v] (since we just flipped)
            # So S = S + 2 * A[:, v] * x[v]
            S += 2 * A[:, v] * x[v]
            
            # Random walk probability to escape local minima
            if rng.random() < 0.1:
                v_rand = rng.randrange(n)
                x[v_rand] = -x[v_rand]
                S += 2 * A[:, v_rand] * x[v_rand]

        return False

    def sa_vectorized(x, S, rng, steps, deadline):
        """
        Simulated Annealing with vectorized delta calculation.
        """
        T = 1.0
        cooling_rate = 0.9999
        
        for _ in range(steps):
            if time.time() > deadline:
                return False
            
            unsat = count_unsat(S)
            if unsat == 0:
                return True

            # Pick random variable
            v = rng.randrange(n)
            
            # Calculate delta for this specific v
            # We can reuse the logic from get_deltas but just for one v
            # L_v = A[:, v] * x[v]
            L_v = A[:, v] * x[v]
            
            # Delta = count(L_v == 1 and S == 1) - count(L_v == -1 and S == -1)
            mask_sat_1 = (L_v == 1) & (S == 1)
            mask_unsat_neg1 = (L_v == -1) & (S == -1)
            delta = np.sum(mask_sat_1) - np.sum(mask_unsat_neg1)
            
            # Accept or reject
            if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
                x[v] = -x[v]
                S += 2 * A[:, v] * x[v]
            
            T *= cooling_rate
            if T < 0.01:
                T = 0.01

        return False

    # Main loop with restarts
    best_unsat = m + 1
    best_x = None
    seed = 0
    restart_count = 0
    
    while time.time() < deadline:
        rng = random.Random(seed)
        
        # Initialize x with random +/-1
        x = np.random.choice([-1, 1], size=n, p=[0.5, 0.5])
        S = compute_sums(x)
        
        cur_unsat = count_unsat(S)
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_x = x.copy()
            
        if cur_unsat == 0:
            break

        # Alternate strategies
        strategy = restart_count % 2
        if strategy == 0:
            # Vectorized Greedy/WalkSAT
            solved = walksat_vectorized(x, S, rng, 5000, deadline)
        else:
            # Vectorized SA
            solved = sa_vectorized(x, S, rng, 5000, deadline)
        
        if solved:
            break

        cur_unsat = count_unsat(S)
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_x = x.copy()

        restart_count += 1
        seed += 1

    # Convert x (1/-1) to booleans (True/False)
    # If x[v] == 1, True; if x[v] == -1, False
    # Original literals: +v is True if x[v]==1, -v is True if x[v]==-1
    # The problem asks for a list of booleans for variables 1..n
    # Usually, variable i being True means literal +i is True.
    # So if x[i-1] == 1, return True.
    
    if best_x is not None:
        return [bool(v == 1) for v in best_x]
    return [False] * n

# EVOLVE-BLOCK-END
