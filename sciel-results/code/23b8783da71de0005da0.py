<<<<<<< SEARCH
def _random_greedy(seed, deadline):
    """Randomized: build terms with random support patterns, then verify."""
    rng = random.Random(seed)
    # Try multiple target ranks
    for R in [30, 35, 40, 45, 50]:
        if time.time() > deadline:
            break
        U = [[0] * R for _ in range(NM)]
        V = [[0] * R for _ in range(MP)]
        W = [[0] * R for _ in range(PN)]
        
        # Strategy: for each term r, assign a random subset of (i,j,k) slots
        # that it will contribute to
        for r in range(R):
            # Randomly decide which (i,j) pairs this term affects in U
            u_support = set()
            for i in range(N):
                for j in range(M):
                    if rng.random() < 0.3:  # 30% chance to include
                        u_support.add((i, j))
            
            # Randomly decide which (j,k) pairs this term affects in V
            v_support = set()
            for j in range(M):
                for k in range(P):
                    if rng.random() < 0.3:
                        v_support.add((j, k))
            
            # Randomly decide which (k,i) pairs this term affects in W
            w_support = set()
            for k in range(P):
                for i in range(N):
                    if rng.random() < 0.3:
                        w_support.add((k, i))
            
            # For each (i,j,k) slot, if all three supports include it,
            # this term can contribute
            # Assign random weights
            for i in range(N):
                for j in range(M):
                    if (i, j) in u_support:
                        U[i * M + j][r] = rng.choice([1, -1, 0.5, -0.5])
            
            for j in range(M):
                for k in range(P):
                    if (j, k) in v_support:
                        V[j * P + k][r] = rng.choice([1, -1, 0.5, -0.5])
            
            for k in range(P):
                for i in range(N):
                    if (k, i) in w_support:
                        W[k * N + i][r] = rng.choice([1, -1, 0.5, -0.5])
        
        cand = {"U": U, "V": V, "W": W}
        if _is_valid(cand["U"], cand["V"], cand["W"]):
            U, V, W = _drop_terms(cand["U"], cand["V"], cand["W"], deadline)
            return {"U": U, "V": V, "W": W}
    
    return None
=======
def _random_greedy(seed, deadline):
    """Randomized: build terms with random support patterns, then verify."""
    rng = random.Random(seed)
    # Try multiple target ranks
    for R in [30, 35, 40, 45, 50]:
        if time.time() > deadline:
            break
        U = [[0] * R for _ in range(NM)]
        V = [[0] * R for _ in range(MP)]
        W = [[0] * R for _ in range(PN)]
        
        # Strategy: for each term r, assign a random subset of (i,j,k) slots
        for r in range(R):
            u_support = set()
            for i in range(N):
                for j in range(M):
                    if rng.random() < 0.3:
                        u_support.add((i, j))
            
            v_support = set()
            for j in range(M):
                for k in range(P):
                    if rng.random() < 0.3:
                        v_support.add((j, k))
            
            w_support = set()
            for k in range(P):
                for i in range(N):
                    if rng.random() < 0.3:
                        w_support.add((k, i))
            
            for i in range(N):
                for j in range(M):
                    if (i, j) in u_support:
                        U[i * M + j][r] = rng.choice([1, -1, 0.5, -0.5])
            
            for j in range(M):
                for k in range(P):
                    if (j, k) in v_support:
                        V[j * P + k][r] = rng.choice([1, -1, 0.5, -0.5])
            
            for k in range(P):
                for i in range(N):
                    if (k, i) in w_support:
                        W[k * N + i][r] = rng.choice([1, -1, 0.5, -0.5])
        
        cand = {"U": U, "V": V, "W": W}
        if _is_valid(cand["U"], cand["V"], cand["W"]):
            U, V, W = _drop_terms(cand["U"], cand["V"], cand["W"], deadline)
            return {"U": U, "V": V, "W": W}
    
    return None
>>>>>>> REPLACE

