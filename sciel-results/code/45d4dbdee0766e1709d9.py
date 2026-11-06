"""Combined: bitmask coprime checks + heuristic local search + MILP polish."""
import math
import random
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0  # safety margin before 150s kill
    N = 512

    # Precompute prime factor bitmasks for all numbers 1..511
    is_prime = [True] * N
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, N, i):
                is_prime[j] = False

    primes = [i for i in range(2, N) if is_prime[i]]
    prime_index = {p: i for i, p in enumerate(primes)}

    prime_factors_mask = [0] * N
    for a in range(2, N):
        x = a
        mask = 0
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                mask |= (1 << prime_index[p])
                while x % p == 0:
                    x //= p
        if x > 1:
            mask |= (1 << prime_index[x])
        prime_factors_mask[a] = mask
    prime_factors_mask[1] = 0

    def score(arr):
        return sum(1.0 / (N - a) for a in arr)

    # Conflict graph: two numbers conflict if not coprime
    # For MWIS: we want max-weight subset with no conflicts
    # Vertex a (index 0..511, value 1..511) has weight w[a] = 1/(512-a)
    weights = [0.0] * N
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)

    # Conflict adjacency: conflict_mask[a] is a bitmask of which other numbers conflict with a
    # Since we have 512 vertices, we use Python integers as bitsets
    conflict_mask = [0] * N
    for a in range(1, N):
        for b in range(a + 1, N):
            if (prime_factors_mask[a] & prime_factors_mask[b]) != 0:
                # They share a prime factor, so not coprime -> conflict
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)

    # Greedy from largest to smallest (baseline)
    def greedy_desc():
        chosen = []
        used_mask = 0
        for a in range(511, 0, -1):
            if (prime_factors_mask[a] & used_mask) == 0:
                chosen.append(a)
                used_mask |= prime_factors_mask[a]
        return chosen

    best = greedy_desc()
    best_score = score(best)

    # Local search for quick improvement before MILP
    def local_search(chosen, max_iters=50):
        chosen_set = set(chosen)
        current_score = sum(1.0 / (N - a) for a in chosen)
        used_mask = 0
        for a in chosen:
            used_mask |= prime_factors_mask[a]

        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() > deadline - 10:
                break
            improved = False
            iters += 1

            # Try adding a new element
            for a in range(511, 0, -1):
                if a in chosen_set:
                    continue
                if (prime_factors_mask[a] & used_mask) == 0:
                    chosen.append(a)
                    chosen_set.add(a)
                    used_mask |= prime_factors_mask[a]
                    current_score += 1.0 / (N - a)
                    improved = True
                    break

            if improved:
                continue

            # Try single removal + greedy refill
            sorted_chosen = sorted(chosen, key=lambda x: 1.0 / (N - x))
            for a in sorted_chosen:
                if time.time() > deadline - 10:
                    break
                remaining = [x for x in chosen if x != a]
                rem_set = set(remaining)
                rem_mask = 0
                for x in remaining:
                    rem_mask |= prime_factors_mask[x]

                new_chosen = list(remaining)
                new_set = set(remaining)
                new_mask = rem_mask
                new_score = sum(1.0 / (N - x) for x in new_chosen)

                for c in range(511, 0, -1):
                    if c in new_set:
                        continue
                    if (prime_factors_mask[c] & new_mask) == 0:
                        new_chosen.append(c)
                        new_set.add(c)
                        new_mask |= prime_factors_mask[c]
                        new_score += 1.0 / (N - c)

                if new_score > current_score + 1e-12:
                    chosen = new_chosen
                    chosen_set = new_set
                    used_mask = new_mask
                    current_score = new_score
                    improved = True
                    break

        return chosen, current_score

    best, best_score = local_search(best)

    # MILP Formulation
    # Variables: x_i for i in 1..511
    # Maximize sum(w_i * x_i)
    # Subject to: x_i + x_j <= 1 for all (i,j) where gcd(i,j) > 1
    # x_i in {0,1}
    
    n_vars = 511
    c = np.array([-weights[1:512]]) # Minimize -sum(w_i x_i)
    
    # Build constraint matrix A for x_i + x_j <= 1
    # A is sparse, shape (num_constraints, n_vars)
    # We only need to add constraints for conflicting pairs.
    # Number of conflicts is roughly N^2 / (log N)^2, which is manageable.
    
    rows = []
    cols = []
    data = []
    
    for i in range(1, 512):
        mask = conflict_mask[i]
        # Iterate through set bits in mask that are > i to avoid duplicates? 
        # Actually, constraint x_i + x_j <= 1 is symmetric.
        # If we add it for i < j, it covers the pair.
        # We can just iterate all j where bit is set.
        # To be efficient, we only add constraint if i < j? 
        # The matrix A doesn't care about order, but adding duplicate constraints is wasteful.
        # Let's just add for all j in mask.
        # Wait, if I add x_i + x_j <= 1 and x_j + x_i <= 1, it's redundant.
        # Let's restrict to j > i.
        m = mask
        while m:
            lsb = m & (-m)
            j = lsb.bit_length()
            if j > i:
                rows.append(len(rows))
                cols.append(i - 1)
                data.append(1.0)
                cols.append(j - 1)
                data.append(1.0)
            m &= m - 1
            
    if not rows:
        return best
        
    from scipy.sparse import csr_matrix
    A = csr_matrix((data, (rows, cols)), shape=(len(rows), n_vars))
    
    lb = np.zeros(len(rows))
    ub = np.ones(len(rows))
    
    constraints = LinearConstraint(A, lb, ub)
    integrality = np.ones(n_vars)
    bounds = Bounds(0, 1)
    
    start_time = time.time()
    
    try:
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds, options={'time_limit': deadline - time.time() - 5})
        
        if res.status == 0: # Optimal
            # res.x contains the optimal solution
            x = res.x
            new_chosen = []
            for i in range(n_vars):
                if x[i] > 0.5:
                    new_chosen.append(i + 1)
            
            new_score = score(new_chosen)
            if new_score > best_score:
                best = new_chosen
                best_score = new_score
                
    except Exception as e:
        # If MILP fails or times out, we keep the heuristic result
        pass

    # Final safety check
    if not best:
        best = [1]

    return best

# EVOLVE-BLOCK-END
