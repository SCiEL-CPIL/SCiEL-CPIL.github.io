<<<<<<< SEARCH
    squares = [L1]
    
    start_total = time.time()
    total_limit = 85.0 # Safety margin
    
    # Dynamic time allocation
    # L2 is usually fast.
    # If L2 is found, try L3 with remaining time.
    
    # Try L2
    t_start = time.time()
    L2 = find_orthogonal(squares, 5.0)
    t_elapsed = time.time() - t_start
    if L2 is not None:
        squares.append(L2)
    
    # Try L3
    elapsed_total = time.time() - start_total
    if len(squares) == 2 and elapsed_total < total_limit - 5.0:
        # Give L3 a significant chunk of the remaining time
        # If L2 was fast, we have lots of time.
        # Let's say we reserve 5s for potential L4, L5...
        # But L3 is the hardest bottleneck usually.
        budget_L3 = min(40.0, total_limit - elapsed_total - 10.0)
        if budget_L3 > 2.0:
            L3 = find_orthogonal(squares, budget_L3)
            if L3 is not None:
                squares.append(L3)
                
                # Try L4
                elapsed_total = time.time() - start_total
                if len(squares) == 3 and elapsed_total < total_limit - 5.0:
                    budget_L4 = min(20.0, total_limit - elapsed_total - 5.0)
                    if budget_L4 > 1.0:
                        L4 = find_orthogonal(squares, budget_L4)
                        if L4 is not None:
                            squares.append(L4)
                            
                            # Try L5
                            elapsed_total = time.time() - start_total
                            if len(squares) == 4 and elapsed_total < total_limit - 2.0:
                                budget_L5 = min(10.0, total_limit - elapsed_total - 1.0)
                                if budget_L5 > 0.5:
                                    L5 = find_orthogonal(squares, budget_L5)
                                    if L5 is not None:
                                        squares.append(L5)

    # Ensure types
    final_result = []
    for sq in squares:
        final_sq = []
        for row in sq:
            final_row = [int(x) for x in row]
            final_sq.append(final_row)
        final_result.append(final_sq)
        
    return final_result
=======
    squares = [L1, L2]
    
    start_total = time.time()
    total_limit = 85.0 # Safety margin
    
    # Now use backtracking to find L3, L4, L5, ...
    # We have ~85s total, L1 and L2 are free, so we have ~85s for search
    
    # Try L3
    elapsed_total = time.time() - start_total
    if elapsed_total < total_limit - 5.0:
        budget_L3 = min(45.0, total_limit - elapsed_total - 10.0)
        if budget_L3 > 2.0:
            L3 = find_orthogonal(squares, budget_L3)
            if L3 is not None:
                squares.append(L3)
                
                # Try L4
                elapsed_total = time.time() - start_total
                if len(squares) == 3 and elapsed_total < total_limit - 5.0:
                    budget_L4 = min(25.0, total_limit - elapsed_total - 5.0)
                    if budget_L4 > 1.0:
                        L4 = find_orthogonal(squares, budget_L4)
                        if L4 is not None:
                            squares.append(L4)
                            
                            # Try L5
                            elapsed_total = time.time() - start_total
                            if len(squares) == 4 and elapsed_total < total_limit - 2.0:
                                budget_L5 = min(12.0, total_limit - elapsed_total - 1.0)
                                if budget_L5 > 0.5:
                                    L5 = find_orthogonal(squares, budget_L5)
                                    if L5 is not None:
                                        squares.append(L5)

    # Ensure types
    final_result = []
    for sq in squares:
        final_sq = []
        for row in sq:
            final_row = [int(x) for x in row]
            final_sq.append(final_row)
        final_result.append(final_sq)
        
    return final_result
>>>>>>> REPLACE
