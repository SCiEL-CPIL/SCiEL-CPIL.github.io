<<<<<<< SEARCH
def _optimize_parameters(start_time, deadline):
    """Optimize construction parameters directly against the objective."""
    best_score = -1.0
    best_U = [0]
    
    # Try various construction strategies with optimized parameters
    
    # Strategy 1: Golomb rulers of various sizes
    for n in range(3, 25):
        if time.time() > deadline:
            break
        gr = _construct_golomb(n)
        s = _score_set(gr)
        if s > best_score:
            best_score = s
            best_U = gr[:]
    
    # Strategy 2: Sidon sequences
    for n in range(3, 20):
        if time.time() > deadline:
            break
        ss = _construct_sidon_sequence(n, max_val=n * n * 5)
        if len(ss) >= n:
            s = _score_set(ss)
            if s > best_score:
                best_score = s
                best_U = ss[:]
    
    # Strategy 3: Parametric family U = {0, a, 2a, ..., (n-1)a} with perturbations
    for n in range(3, 15):
        if time.time() > deadline:
            break
        for a in range(1, 10):
            base = list(range(n)) * a  # {0, a, 2a, ..., (n-1)a}
            # Add small perturbations
            for pert in [0, 1, -1, 2]:
                if pert == 0:
                    candidate = base[:]
                else:
                    candidate = []
                    for i, v in enumerate(base):
                        if i == 0:
                            candidate.append(0)
                        elif i == n - 1:
                            candidate.append(v + pert)
                        else:
                            candidate.append(v + (pert if i % 2 == 0 else -pert))
                    candidate = sorted(set(x for x in candidate if x >= 0))
                    if 0 not in candidate:
                        continue
                if len(candidate) < 2:
                    continue
                s = _score_set(candidate)
                if s > best_score:
                    best_score = s
                    best_U = candidate[:]
    
    # Strategy 4: Power-of-prime constructions
    for p in [2, 3, 5, 7]:
        if time.time() > deadline:
            break
        for n in range(3, 8):
            candidate = [0]
            val = 1
            for i in range(n - 1):
                val *= p
                candidate.append(val)
            s = _score_set(candidate)
            if s > best_score:
                best_score = s
                best_U = candidate[:]
    
    # Strategy 5: Quadratic-like constructions
    for n in range(3, 12):
        if time.time() > deadline:
            break
        for c in range(1, 6):
            candidate = sorted(set(i * i * c for i in range(n)))
            if 0 not in candidate:
                candidate = [0] + [x for x in candidate if x > 0]
            if len(candidate) >= 2:
                s = _score_set(candidate)
                if s > best_score:
                    best_score = s
                    best_U = candidate[:]
    
    return best_U, best_score
=======
def _optimize_parameters(start_time, deadline):
    """Optimize construction parameters directly against the objective."""
    best_score = -1.0
    best_U = [0]
    
    # Strategy 1: Golomb rulers of various sizes
    for n in range(3, 25):
        if time.time() > deadline:
            break
        gr = _construct_golomb(n)
        s = _score_set(gr)
        if s > best_score:
            best_score = s
            best_U = gr[:]
    
    # Strategy 2: Symmetric core constructions
    for M in range(10, 100, 5):
        if time.time() > deadline:
            break
        for k in range(2, min(10, M // 4)):
            # Try different S sets
            S_candidates = []
            # Arithmetic progression
            S_candidates.append(list(range(1, k + 1)))
            # Geometric-like
            S_candidates.append([1, 2, 4, 8][:k])
            # Random
            random.seed(M * 100 + k)
            S_candidates.append(sorted(random.sample(range(1, M - 1), k)))
            
            for S in S_candidates:
                if time.time() > deadline:
                    break
                U = _construct_symmetric(M, S)
                s = _score_set_fast(U)
                if s > best_score:
                    best_score = s
                    best_U = U[:]
    
    # Strategy 3: Parametric family with perturbations
    for n in range(3, 15):
        if time.time() > deadline:
            break
        for a in range(1, 10):
            base = list(range(n)) * a
            for pert in [0, 1, -1, 2]:
                if pert == 0:
                    candidate = base[:]
                else:
                    candidate = []
                    for i, v in enumerate(base):
                        if i == 0:
                            candidate.append(0)
                        elif i == n - 1:
                            candidate.append(v + pert)
                        else:
                            candidate.append(v + (pert if i % 2 == 0 else -pert))
                    candidate = sorted(set(x for x in candidate if x >= 0))
                    if 0 not in candidate:
                        continue
                if len(candidate) < 2:
                    continue
                s = _score_set(candidate)
                if s > best_score:
                    best_score = s
                    best_U = candidate[:]
    
    # Strategy 4: Power-of-prime constructions
    for p in [2, 3, 5, 7]:
        if time.time() > deadline:
            break
        for n in range(3, 8):
            candidate = [0]
            val = 1
            for i in range(n - 1):
                val *= p
                candidate.append(val)
            s = _score_set(candidate)
            if s > best_score:
                best_score = s
                best_U = candidate[:]
    
    # Strategy 5: Quadratic-like constructions
    for n in range(3, 12):
        if time.time() > deadline:
            break
        for c in range(1, 6):
            candidate = sorted(set(i * i * c for i in range(n)))
            if 0 not in candidate:
                candidate = [0] + [x for x in candidate if x > 0]
            if len(candidate) >= 2:
                s = _score_set(candidate)
                if s > best_score:
                    best_score = s
                    best_U = candidate[:]
    
    return best_U, best_score
>>>>>>> REPLACE

