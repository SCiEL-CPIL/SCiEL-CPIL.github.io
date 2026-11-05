<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin (evaluator kills at 150s)
    
    best_n = 0
    best_edges = []
    
    # Try n from 40 down to 25
    for n in range(40, 24, -1):
        if time.time() > deadline - 5:
            break
        
        found = False
        
        # Strategy 1: Systematic search with varying densities
        densities = [0.2, 0.25, 0.3, 0.35, 0.4]
        if n <= 32:
            densities += [0.15, 0.45, 0.5]
        
        for density in densities:
            if time.time() > deadline - 3:
                break
            
            # Multiple random restarts for this density
            for restart in range(30):
                if time.time() > deadline - 2:
                    break
                
                S = _greedy_circulant(n, target_density=density, time_limit=0.5)
                
                if len(S) < 4:
                    continue
                
                valid, alpha = _verify_circulant(n, S)
                if valid:
                    edges = _build_edges(n, S)
                    best_n = n
                    best_edges = edges
                    found = True
                    break
            
            if found:
                break
        
        if found:
            break
    
    # Fallback: Known small graphs
    if best_n < 19:
        n0 = 26
        best_edges = [[i, j] for i in range(n0) for j in range(i + 1, n0) if (j - i) % n0 % 3 == 1]
        best_n = n0
    
    return {'n': best_n, 'edges': best_edges}
=======
def _try_known_constructions(n, deadline):
    """Try known graph constructions for specific n values."""
    # For n = 39, try specific shift sets known to work
    # The (3,10)-Ramsey problem: known constructions exist for n up to 39
    
    # Try various structured shift sets
    half = n // 2
    
    # Strategy: try arithmetic progressions and other structured sets
    structured_sets = []
    
    # Arithmetic progression based
    for start in range(1, half + 1):
        for step in range(1, half + 1):
            S = set()
            x = start
            while x <= half:
                S.add(x)
                x += step
            if 2 <= len(S) <= half:
                structured_sets.append(S)
    
    # Complementary sets
    for k in range(3, min(half + 1, 10)):
        for start in range(1, half - k + 2):
            S = set(range(start, start + k))
            structured_sets.append(S)
    
    # Random sets of various sizes
    for k in range(4, min(half + 1, 12)):
        for _ in range(20):
            S = set(random.sample(range(1, half + 1), k))
            structured_sets.append(S)
    
    random.shuffle(structured_sets)
    
    for S in structured_sets:
        if time.time() > deadline - 3:
            return None
        
        # Make symmetric
        S_sym = set()
        for x in S:
            S_sym.add(x)
            S_sym.add((-x) % n)
        S_sym.discard(0)
        
        if len(S_sym) < 2:
            continue
        
        valid, alpha = _verify_circulant(n, S_sym)
        if valid:
            return S_sym
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin (evaluator kills at 150s)
    
    best_n = 0
    best_edges = []
    
    # Try n from 40 down to 25
    for n in range(40, 24, -1):
        if time.time() > deadline - 5:
            break
        
        found = False
        
        # Strategy 1: Known constructions
        if time.time() < deadline - 10:
            S_known = _try_known_constructions(n, deadline)
            if S_known is not None:
                edges = _build_edges(n, S_known)
                best_n = n
                best_edges = edges
                found = True
        
        if not found:
            # Strategy 2: Systematic search with varying densities
            densities = [0.2, 0.25, 0.3, 0.35, 0.4]
            if n <= 32:
                densities += [0.15, 0.45, 0.5]
            
            for density in densities:
                if time.time() > deadline - 3:
                    break
                
                # Multiple random restarts for this density
                for restart in range(30):
                    if time.time() > deadline - 2:
                        break
                    
                    S = _greedy_circulant(n, target_density=density, time_limit=0.5)
                    
                    if len(S) < 4:
                        continue
                    
                    valid, alpha = _verify_circulant(n, S)
                    if valid:
                        edges = _build_edges(n, S)
                        best_n = n
                        best_edges = edges
                        found = True
                        break
                
                if found:
                    break
        
        if found:
            break
    
    # Fallback: Known small graphs
    if best_n < 19:
        n0 = 26
        best_edges = [[i, j] for i in range(n0) for j in range(i + 1, n0) if (j - i) % n0 % 3 == 1]
        best_n = n0
    
    return {'n': best_n, 'edges': best_edges}
>>>>>>> REPLACE
