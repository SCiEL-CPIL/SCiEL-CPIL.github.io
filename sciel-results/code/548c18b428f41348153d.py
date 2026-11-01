<<<<<<< SEARCH
    # 1. Try voltage graph construction first
    if time.time() < deadline - 60:
        result = search_voltage_graph(k, g, time_limit=50.0)
        if result is not None:
            n, edges = result
            if n < best_n:
                best_n = n
                best_edges = edges
=======
    # 1. Try voltage graph construction first
    if time.time() < deadline - 60:
        result = search_voltage_graph(k, g, time_limit=50.0)
        if result is not None:
            n, edges = result
            if n < best_n:
                best_n = n
                best_edges = edges
    
    # 2. Try systematic voltage search with girth pre-check
    if best_edges is None or best_n > 320:
        if time.time() < deadline - 30:
            # Systematic search for voltage assignments
            for m in range(64, 100):  # 5*m = 320 to 500
                if time.time() > deadline - 10:
                    break
                
                # Try structured voltage assignments
                # Use a primitive root or similar structure
                if m >= 3:
                    # Try voltages based on a generator
                    # For K5, assign voltages w(u,v) = g^(u*v) mod m or similar
                    for gen in range(2, min(m, 20)):
                        if time.time() > deadline - 10:
                            break
                        
                        voltages = {}
                        valid = True
                        for u in range(5):
                            for v in range(5):
                                if u != v:
                                    # Use a deterministic formula
                                    w = (gen * (u + 1) * (v + 1)) % m
                                    if w == 0:
                                        w = 1
                                    voltages[(u, v)] = w
                        
                        # Check 2-cycles: w(u,v) + w(v,u) != 0 mod m
                        valid = True
                        for u in range(5):
                            for v in range(u + 1, 5):
                                if (voltages[(u, v)] + voltages[(v, u)]) % m == 0:
                                    valid = False
                                    break
                            if not valid:
                                break
                        
                        if not valid:
                            continue
                        
                        # Check girth using pre-check
                        if check_girth_voltage(k, g, m, voltages):
                            # Build and verify
                            n_lift = 5 * m
                            result = build_voltage_graph(5, m, voltages, k, g)
                            if result:
                                n, edges = result
                                if n < best_n:
                                    if verify_graph(n, edges, k, g, time_limit=10.0):
                                        best_n = n
                                        best_edges = edges
                                        if n <= 320:
                                            break
                    if best_n <= 320:
                        break
>>>>>>> REPLACE

