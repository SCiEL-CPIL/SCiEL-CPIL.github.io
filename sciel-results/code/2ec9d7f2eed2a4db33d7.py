    # 3. Local Polish: Try to insert missing numbers into best_seq
    if time.time() < deadline - 5:
        current_seq = list(best_seq)
        current_prods = set(best_prods)
        current_set = set(current_seq)
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for x in range(2, 129):
                if x in current_set:
                    continue
                    
                # Try inserting x at every position
                for i in range(len(current_seq) + 1):
                    # Check bounds: seq[i-1] < x < seq[i]
                    if i > 0 and current_seq[i-1] >= x:
                        continue
                    if i < len(current_seq) and current_seq[i] <= x:
                        continue
                        
                    # Compute new products
                    new_prods_set = set()
                    new_prods_set.add(x)
                    
                    # Right side: x * suffixes of current_seq[i:]
                    p = x
                    for j in range(i, len(current_seq)):
                        new_prods_set.add(p)
                        p *= current_seq[j]
                        
                    # Left side: prefixes of current_seq[:i] * x
                    p = x
                    for j in range(i-1, -1, -1):
                        new_prods_set.add(p)
                        p *= current_seq[j]
                    
                    # Check collisions with existing
                    if not new_prods_set.isdisjoint(current_prods):
                        continue
                        
                    # Check internal collisions
                    if len(new_prods_set) != len(current_seq) + 1:
                        continue
                        
                    # Valid
                    current_seq = current_seq[:i] + [x] + current_seq[i:]
                    current_prods = current_prods.union(new_prods_set)
                    current_set.add(x)
                    improved = True
                    break # Restart x loop
                if improved:
                    break

    return current_seq
