# EVOLVE-BLOCK-START

def construct():
    """
    Construct a set of distinct integers in 1..256 maximizing size such that
    no two subsets of different cardinalities have the same sum.
    """
    # sumsets[c] holds the set of sums for subsets of cardinality c
    # Initially, only the empty subset exists.
    sumsets = [set() for _ in range(1)]
    sumsets[0] = {0}
    current_size = 0
    current_elements = []
    
    # We iterate through candidates. 
    # Descending order is a strong heuristic for minimizing collisions.
    # We can also try ascending, but descending usually allows larger numbers 
    # to be placed, which have fewer combinatorial interactions with smaller numbers 
    # in terms of sum overlap for different cardinalities.
    
    for x in range(256, 0, -1):
        # Tentatively add x
        # new_sumsets[c] = sumsets[c] U {s + x | s in sumsets[c-1]}
        
        # We need to check if any new sum s+x creates a collision with an existing sum 
        # in a different cardinality bucket, or if two new sums collide? 
        # No, sums within the same cardinality are allowed to repeat (though they are sets, so they merge).
        # The constraint is: sumsets[i] and sumsets[j] must be disjoint for i != j.
        
        # Let's compute the new sets.
        # To be efficient, we only need to check the NEW elements added to each bucket.
        
        new_elements_per_card = [set() for _ in range(current_size + 2)]
        
        for c in range(current_size, -1, -1):
            if not sumsets[c]:
                continue
            for s in sumsets[c]:
                new_sum = s + x
                new_elements_per_card[c + 1].add(new_sum)
                
        # Now check for collisions involving these new elements.
        # A collision occurs if a new element in bucket k is in an existing bucket j (j != k)
        # OR if a new element in bucket k is in a new element in bucket l (l != k)?
        # If a new element in k is in a new element in l, it means the same sum is achievable by two different subsets of different sizes (one using x, one using x? No, both use x? 
        # Wait. 
        # Subset A of size k uses x. Sum = S_A.
        # Subset B of size l uses x. Sum = S_B.
        # If S_A == S_B and k != l, then we have a collision.
        # Also, Subset C of size m does NOT use x. Sum = S_C.
        # If S_A == S_C and k != m, collision.
        
        # So we need to check:
        # 1. For each k, new_elements_per_card[k] intersect (sumsets[j] for all j != k)
        # 2. For each k, l (k != l), new_elements_per_card[k] intersect new_elements_per_card[l]
        
        valid = True
        
        # Check 1: New elements vs Old elements (different cardinality)
        for k in range(1, current_size + 2):
            if not new_elements_per_card[k]:
                continue
            for j in range(current_size + 1): # j is old cardinality
                if j == k:
                    continue
                if sumsets[j] and (new_elements_per_card[k] & sumsets[j]):
                    valid = False
                    break
            if not valid:
                break
                
        if valid:
            # Check 2: New elements vs New elements (different cardinality)
            # This is O(K^2 * SetSize)
            for k in range(1, current_size + 2):
                if not new_elements_per_card[k]:
                    continue
                for l in range(k + 1, current_size + 2):
                    if not new_elements_per_card[l]:
                        continue
                    if new_elements_per_card[k] & new_elements_per_card[l]:
                        valid = False
                        break
                if not valid:
                    break
                    
        if valid:
            # Update sumsets
            for k in range(1, current_size + 2):
                sumsets[k].update(new_elements_per_card[k])
            # Ensure we have the right number of lists
            if len(sumsets) < current_size + 2:
                sumsets.extend([set() for _ in range(len(sumsets), current_size + 2)])
                
            current_elements.append(x)
            current_size += 1
            
    return current_elements

# EVOLVE-BLOCK-END
