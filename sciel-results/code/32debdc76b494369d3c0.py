    for L in range(L_min, min(L_min + 100, 257)):
        if time.time() > deadline:
            break
            
        if not is_valid_interval(L, n):
            continue
            
        # Precompute valid x for base interval
        valid_xs = []
        for x in range(2, L):
            if can_add_to_interval(x, L, n):
                valid_xs.append(x)
                
        # Sort decreasing
        valid_xs.sort(reverse=True)
        
        base = set(range(L, n + 1))
        small_elements = set()
        
        for x in valid_xs:
            if time.time() > deadline:
                break
            # Check against small_elements
            # We can reuse the logic from can_add_to_set but skip the interval part
            s_list = list(small_elements)
            
            ok = True
            # Case: b, c in small_elements
            for i, b in enumerate(s_list):
                for k in range(i, len(s_list)):
                    c = s_list[k]
                    if (b + c) % x == 0:
                        ok = False
                        break
                if not ok:
                    break
                    
            if ok:
                # Case: b in small_elements, c in [L, n]
                for b in s_list:
                    k_min = (L + b + x - 1) // x
                    k_max = (n + b) // x
                    k_min2 = (2 * b + x - 1) // x
                    k_min = max(k_min, k_min2)
                    
                    for k in range(k_min, k_max + 1):
                        c = k * x - b
                        if L <= c <= n:
                            ok = False
                            break
                    if not ok:
                        break
                        
            if ok:
                small_elements.add(x)
                
        current_set = base.union(small_elements)
        if len(current_set) > len(best_set):
            best_set = current_set
