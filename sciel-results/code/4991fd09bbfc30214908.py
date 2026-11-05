    for p in primes_to_try:
        alpha = find_primitive_root(p)
        if alpha is None:
            continue
            
        # Precompute powers of alpha
        pow_alpha = [pow(alpha, i, p) for i in range(p-1)]
        
        found = False
        best_voltages = None
        max_trials = 50000
        
        for trial in range(max_trials):
            if trial % 20 == 0:
                # Structured: Linear exponents
                a = (trial // 20) % (p-1)
                b = (trial // 40) % (p-1)
                exponents = np.array([(a*i + b) % (p-1) for i in range(num_edges)])
            else:
                # Random exponents
                exponents = np.random.randint(0, p-1, size=num_edges)
            
            voltages = [pow_alpha[e] for e in exponents]
            
            if check_voltages(voltages, p):
                found = True
                best_voltages = voltages
                break
        
        if found:
            return build_graph(best_voltages, p)
