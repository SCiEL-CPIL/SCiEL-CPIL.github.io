<<<<<<< SEARCH
def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 115
    
    if k != 3:
        q = k
        n, edges = _luw_construct(q, g)
        return n, edges
    
    # Moore bound for k=3, girth g
    if g % 2 == 0:
        moore = 2 * (2**(g//2) - 1)
    else:
        moore = 2 * (2**(g//2 - 1) - 1) + 2
    
    best_result = None
    best_n = float('inf')
    
    # Get base graphs
    base_graphs = _get_base_graphs()
    
    # Strategy 1: Voltage lifts - try many m values with multiple strategies
    for base_edges, n_base in base_graphs:
        if _time.time() > deadline - 25:
            break
        m_min = max(1, (moore + n_base - 1) // n_base)
        m_max = m_min + 500
        
        for m in range(m_min, m_max):
            if _time.time() > deadline - 20:
                break
            n_lift = n_base * m
            if n_lift < moore:
                continue
            if n_lift >= best_n:
                break
            
            # Try deterministic voltage first
            result = _deterministic_voltage_lift(base_edges, n_base, m, g)
            if result is not None:
                n_r, edges_r = result
                if n_r < best_n:
                    best_n = n_r
                    best_result = (n_r, edges_r)
                break
            
            # Try structured voltage
            result = _structured_voltage_lift(base_edges, n_base, m, g, seed=m * 7919 + n_base * 104729)
            if result is not None:
                n_r, edges_r = result
                if n_r < best_n:
                    best_n = n_r
                    best_result = (n_r, edges_r)
                break
            
            # Try random voltage assignments
            result = _voltage_lift_search(base_edges, n_base, m, g, seed=m * 100000 + n_base, n_attempts=50)
            if result is not None:
                n_r, edges_r = result
                if n_r < best_n:
                    best_n = n_r
                    best_result = (n_r, edges_r)
                break
    
    # Strategy 2: LUW construction as fallback
    if _time.time() < deadline - 20:
        n_luw, edges_luw = _luw_construct(3, g)
        if n_luw < best_n:
            best_n = n_luw
            best_result = (n_luw, edges_luw)
    
    # Strategy 3: Random 3-regular graphs (less promising but cheap to try)
    if _time.time() < deadline - 15 and best_result is not None:
        n_try = moore
        if n_try % 2 != 0:
            n_try += 1
        while n_try < best_n and _time.time() < deadline - 10:
            for attempt in range(10):
                if _time.time() > deadline - 10:
                    break
                result = _random_3reg(n_try, g, seed=attempt * 104729 + n_try)
                if result is not None:
                    n_r, edges_r = result
                    if n_r < best_n:
                        best_n = n_r
                        best_result = (n_r, edges_r)
                    break
            n_try += 32
    
    # Fallback: LUW construction
    if best_result is None:
        n_luw, edges_luw = _luw_construct(3, g)
        best_result = (n_luw, edges_luw)
    
    return best_result
=======
def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 110
    
    if k != 3:
        q = k
        n, edges = _luw_construct(q, g)
        return n, edges
    
    # Moore bound for k=3, girth g
    if g % 2 == 0:
        moore = 2 * (2**(g//2) - 1)
    else:
        moore = 2 * (2**(g//2 - 1) - 1) + 2
    
    best_result = None
    best_n = float('inf')
    
    # Get base graphs
    base_graphs = _get_base_graphs()
    
    # Strategy 1: Voltage lifts - try many m values with multiple strategies
    # Use both base graphs and interleave m values
    all_ms = set()
    for base_edges, n_base in base_graphs:
        m_min = max(1, (moore + n_base - 1) // n_base)
        for m in range(m_min, m_min + 200):
            n_lift = n_base * m
            if n_lift < moore:
                continue
            all_ms.add((m, base_edges, n_base))
    
    # Sort by resulting n (smaller first)
    all_ms = sorted(all_ms, key=lambda x: x[1] and x[2] * x[0])
    
    for m, base_edges, n_base in all_ms:
        if _time.time() > deadline - 15:
            break
        n_lift = n_base * m
        if n_lift < moore:
            continue
        if n_lift >= best_n:
            break
        
        # Try deterministic voltage first (fast)
        result = _deterministic_voltage_lift(base_edges, n_base, m, g, deadline)
        if result is not None:
            n_r, edges_r = result
            if n_r < best_n:
                best_n = n_r
                best_result = (n_r, edges_r)
            continue
        
        # Try structured voltage (medium)
        result = _structured_voltage_lift(base_edges, n_base, m, g, seed=m * 7919 + n_base * 104729, deadline=deadline)
        if result is not None:
            n_r, edges_r = result
            if n_r < best_n:
                best_n = n_r
                best_result = (n_r, edges_r)
            continue
        
        # Try random voltage assignments (slower, fewer attempts)
        n_attempts = 30 if n_lift < 500 else 15
        result = _voltage_lift_search(base_edges, n_base, m, g, seed=m * 100000 + n_base, n_attempts=n_attempts, deadline=deadline)
        if result is not None:
            n_r, edges_r = result
            if n_r < best_n:
                best_n = n_r
                best_result = (n_r, edges_r)
    
    # Strategy 2: LUW construction as fallback
    if _time.time() < deadline - 10:
        n_luw, edges_luw = _luw_construct(3, g)
        if n_luw < best_n:
            best_n = n_luw
            best_result = (n_luw, edges_luw)
    
    # Fallback: LUW construction
    if best_result is None:
        n_luw, edges_luw = _luw_construct(3, g)
        best_result = (n_luw, edges_luw)
    
    return best_result
>>>>>>> REPLACE
