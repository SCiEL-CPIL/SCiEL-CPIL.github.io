import random
def construct():
        best_A = []
        best_B = []
        best_size = 0
        
        # Different strategies for candidate ordering
        strategies = [
            "random",
            "sorted",
            "reversed",
            "stride_2",
            "stride_3",
            "stride_4"
        ]
        
        for strat in strategies:
            # Run multiple restarts for this strategy
            for _ in range(2000):
                A = []
                B = []
                sums_A = set()
                sums_B = set()
                diffs_A = set()
                diffs_B = set()
                
                if strat == "random":
                    candidates = list(range(1, 257))
                    random.shuffle(candidates)
                elif strat == "sorted":
                    candidates = list(range(1, 257))
                elif strat == "reversed":
                    candidates = list(range(256, 0, -1))
                elif strat.startswith("stride_"):
                    step = int(strat.split("_")[1])
                    # Create a list with stride, then maybe fill gaps?
                    # Or just shuffle the strided list?
                    # Let's just use a random permutation of the strided order?
                    # Actually, a fixed stride order might be too rigid.
                    # Let's do: base order is stride, then randomize within groups?
                    # Simpler: Just random shuffle is usually best.
                    # Let's stick to random, sorted, reversed.
                    pass # Skip stride for now, focus on random quality.
                
                # Optimized Greedy Loop
                for x in candidates:
                    can_add_A = True
                    can_add_B = True
                    
                    # Check A
                    for a in A:
                        if a + x in sums_A or 2*x in sums_A: # 2*x check can be outside loop? No, depends on x.
                             # Actually 2*x is just one check.
                            pass
                    # Let's do it properly.
                    
                    # Check Sidon A
                    if 2*x in sums_A:
                        can_add_A = False
                    else:
                        for a in A:
                            if a + x in sums_A:
                                can_add_A = False
                                break
                    
                    # Check Diff A vs B
                    if can_add_A:
                        for a in A:
                            if (x - a) in diffs_B or (a - x) in diffs_B:
                                can_add_A = False
                                break

                    # Check Sidon B
                    if can_add_B:
                        if 2*x in sums_B:
                            can_add_B = False
                        else:
                            for b in B:
                                if b + x in sums_B:
                                    can_add_B = False
                                    break
                    
                    # Check Diff B vs A
                    if can_add_B:
                        for b in B:
                            if (x - b) in diffs_A or (b - x) in diffs_A:
                                can_add_B = False
                                break

                    # Decision
                    if can_add_A and can_add_B:
                        # Heuristic: Balance
                        if len(A) <= len(B):
                            # Add to A
                            A.append(x)
                            for a in A[:-1]:
                                sums_A.add(a + x)
                                diffs_A.add(x - a)
                                diffs_A.add(a - x)
                            sums_A.add(2 * x)
                        else:
                            # Add to B
                            B.append(x)
                            for b in B[:-1]:
                                sums_B.add(b + x)
                                diffs_B.add(x - b)
                                diffs_B.add(b - x)
                            sums_B.add(2 * x)
                    elif can_add_A:
                        A.append(x)
                        for a in A[:-1]:
                            sums_A.add(a + x)
                            diffs_A.add(x - a)
                            diffs_A.add(a - x)
                        sums_A.add(2 * x)
                    elif can_add_B:
                        B.append(x)
                        for b in B[:-1]:
                            sums_B.add(b + x)
                            diffs_B.add(x - b)
                            diffs_B.add(b - x)
                        sums_B.add(2 * x)
                
                current_size = len(A) + len(B)
                if current_size > best_size:
                    best_size = current_size
                    best_A = A[:]
                    best_B = B[:]
                    
        # Ensure non-empty
        if not best_A:
            best_A = [1]
        if not best_B:
            best_B = [2]
            
        return [best_A, best_B]
