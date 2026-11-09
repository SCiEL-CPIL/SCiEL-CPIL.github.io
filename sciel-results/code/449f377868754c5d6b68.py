    def polish_sets(A, B, sums_A, sums_B, diffs_A, diffs_B):
        in_A = set(A)
        in_B = set(B)
        all_elements = list(range(1, 257))
        
        improved = True
        while improved and elapsed() < deadline - 2.0:
            improved = False
            
            # 1. Try adding new elements
            for x in all_elements:
                if x in in_A or x in in_B:
                    continue
                
                # Try adding to A
                if len(A) < 64:
                    can_add_A = True
                    if (2*x) in sums_A:
                        can_add_A = False
                    else:
                        for s in A:
                            if (x + s) in sums_A:
                                can_add_A = False
                                break
                            if abs(x - s) in diffs_B:
                                can_add_A = False
                                break
                    
                    if can_add_A:
                        A.append(x)
                        in_A.add(x)
                        for s in A[:-1]:
                            sums_A.add(x + s)
                            diffs_A.add(abs(x - s))
                        sums_A.add(2*x)
                        improved = True
                        continue

                # Try adding to B
                if len(B) < 64:
                    can_add_B = True
                    if (2*x) in sums_B:
                        can_add_B = False
                    else:
                        for s in B:
                            if (x + s) in sums_B:
                                can_add_B = False
                                break
                            if abs(x - s) in diffs_A:
                                can_add_B = False
                                break
                    
                    if can_add_B:
                        B.append(x)
                        in_B.add(x)
                        for s in B[:-1]:
                            sums_B.add(x + s)
                            diffs_B.add(abs(x - s))
                        sums_B.add(2*x)
                        improved = True
            
            # 2. Try swapping elements
            for x in list(A):
                if elapsed() > deadline - 2.0:
                    break
                
                # Remove x from A
                A.remove(x)
                in_A.discard(x)
                
                # Update sums_A and diffs_A by removing x's contributions
                sums_A.discard(2*x)
                for s in A:
                    sums_A.discard(x + s)
                    diffs_A.discard(abs(x - s))
                
                # Try adding x to B
                can_add_B = True
                if len(B) < 64:
                    if (2*x) in sums_B:
                        can_add_B = False
                    else:
                        for s in B:
                            if (x + s) in sums_B:
                                can_add_B = False
                                break
                            if abs(x - s) in diffs_A: # Note: diffs_A is now updated
                                can_add_B = False
                                break
                else:
                    can_add_B = False

                if can_add_B:
                    # Move x to B
                    B.append(x)
                    in_B.add(x)
                    for s in B[:-1]:
                        sums_B.add(x + s)
                        diffs_B.add(abs(x - s))
                    sums_B.add(2*x)
                    improved = True
                else:
                    # Restore x to A
                    A.append(x)
                    in_A.add(x)
                    for s in A[:-1]:
                        sums_A.add(x + s)
                        diffs_A.add(abs(x - s))
                    sums_A.add(2*x)

            for x in list(B):
                if elapsed() > deadline - 2.0:
                    break
                
                # Remove x from B
                B.remove(x)
                in_B.discard(x)
                
                sums_B.discard(2*x)
                for s in B:
                    sums_B.discard(x + s)
                    diffs_B.discard(abs(x - s))
                
                # Try adding x to A
                can_add_A = True
                if len(A) < 64:
                    if (2*x) in sums_A:
                        can_add_A = False
                    else:
                        for s in A:
                            if (x + s) in sums_A:
                                can_add_A = False
                                break
                            if abs(x - s) in diffs_B: # Note: diffs_B is now updated
                                can_add_A = False
                                break
                else:
                    can_add_A = False

                if can_add_A:
                    # Move x to A
                    A.append(x)
                    in_A.add(x)
                    for s in A[:-1]:
                        sums_A.add(x + s)
                        diffs_A.add(abs(x - s))
                    sums_A.add(2*x)
                    improved = True
                else:
                    # Restore x to B
                    B.append(x)
                    in_B.add(x)
                    for s in B[:-1]:
                        sums_B.add(x + s)
                        diffs_B.add(abs(x - s))
                    sums_B.add(2*x)

        return A, B, sums_A, sums_B, diffs_A, diffs_B
