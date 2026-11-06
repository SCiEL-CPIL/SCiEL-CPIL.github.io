import random
import time
import math
import numpy as np

N = 64

# Precompute all grid points as a list of (x, y) tuples
ALL_POINTS = [(x, y) for x in range(N) for y in range(N)]
ALL_POINTS_ARRAY = np.array(ALL_POINTS, dtype=np.int32)

def construct():
    start_time = time.time()
    deadline = start_time + 285.0  # Safety margin

    best_set = []
    best_size = 0

    # Vectorized validity check: can point P be added to set S?
    def is_valid_addition(P, S):
        """Check if point P can be added to set S without forming an isosceles triangle."""
        if not S:
            return True
        if len(S) == 1:
            return True

        P_arr = np.array(P, dtype=np.int64)
        S_arr = np.array(S, dtype=np.int64)

        # Distances from P to all points in S (squared)
        diffs_PA = S_arr - P_arr
        d2_PA = np.sum(diffs_PA ** 2, axis=1)

        # Check 1: all distances from P to points in S must be distinct
        if len(set(d2_PA.tolist())) != len(d2_PA):
            return False

        # Check 2: for each A in S, d_PA must not equal d_AB for any B in S, B != A
        # Compute all pairwise squared distances within S
        n = len(S)
        # d2_AB[i][j] = squared distance between S[i] and S[j]
        diffs_AB = S_arr[:, np.newaxis, :] - S_arr[np.newaxis, :, :]
        d2_AB = np.sum(diffs_AB ** 2, axis=2)

        # For each A (row i), check if d2_PA[i] appears in d2_AB[i][j] for j != i
        for i in range(n):
            d_pa_i = d2_PA[i]
            # Get all distances from S[i] to other points in S
            row = d2_AB[i].copy()
            row[i] = -1  # exclude self
            if d_pa_i in row:
                return False

        return True

    # Greedy construction with a given order
    def greedy_construction(order):
        """Build a set by greedily adding points in the given order."""
        S = []
        for P in order:
            if is_valid_addition(P, S):
                S.append(P)
        return S

    # Local search: try to improve by adding points
    def local_search(S, time_limit):
        """Try to improve set S by adding more points."""
        S = list(S)
        S_set = set(S)
        
        # Repeatedly try to add a point
        added_any = True
        while added_any and time.time() < time_limit:
            added_any = False
            candidates = [p for p in ALL_POINTS if p not in S_set]
            random.shuffle(candidates)
            for P in candidates:
                if time.time() >= time_limit:
                    break
                if is_valid_addition(P, S):
                    S.append(P)
                    S_set.add(P)
                    added_any = True
                    break  # restart after adding
        
        return S

    # Structured randomization: create different ordering strategies
    def create_orderings(num_orders):
        """Create various random/structured orderings."""
        orderings = []
        
        # Pure random
        n_random = max(1, num_orders // 3)
        for _ in range(n_random):
            order = ALL_POINTS[:]
            random.shuffle(order)
            orderings.append(order)
        
        # Row-major with random row permutation
        n_row = max(1, num_orders // 3)
        for _ in range(n_row):
            rows = list(range(N))
            random.shuffle(rows)
            order = []
            for r in rows:
                cols = list(range(N))
                random.shuffle(cols)
                for c in cols:
                    order.append((c, r))
            orderings.append(order)
        
        # Diagonal-based ordering
        n_diag = max(1, num_orders // 3)
        for _ in range(n_diag):
            # Order by (x+y) mod N, then random within each diagonal
            diag_order = {}
            for x in range(N):
                for y in range(N):
                    d = (x + y) % N
                    if d not in diag_order:
                        diag_order[d] = []
                    diag_order[d].append((x, y))
            
            diags = list(range(N))
            random.shuffle(diags)
            order = []
            for d in diags:
                pts = diag_order.get(d, [])
                random.shuffle(pts)
                order.extend(pts)
            orderings.append(order)
        
        return orderings

    # Main search loop
    num_trials = 0
    while time.time() < deadline:
        # Create batch of orderings
        remaining = deadline - time.time()
        if remaining < 5:
            break
        batch_size = min(5, max(1, int(remaining / 30)))
        orderings = create_orderings(batch_size)
        
        for order in orderings:
            if time.time() >= deadline - 10:
                break
            
            S = greedy_construction(order)
            
            # Apply local search if we have time
            if time.time() < deadline - 15:
                S = local_search(S, deadline - 10)
            
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
            
            num_trials += 1
    
    return best_set
