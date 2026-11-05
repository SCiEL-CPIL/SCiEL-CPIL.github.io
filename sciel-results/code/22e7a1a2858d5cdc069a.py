import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    # --- Utility Functions ---

    def get_product_set(seq):
        """Compute all consecutive block products for a sequence."""
        products = set()
        n = len(seq)
        if n == 0:
            return products
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                products.add(prod)
        return products

    def is_valid(seq):
        """
        Check if sequence is strictly increasing and all block products are distinct.
        """
        if len(seq) == 0:
            return True
        for i in range(len(seq) - 1):
            if seq[i] >= seq[i+1]:
                return False
        if any(x < 2 or x > 128 for x in seq):
            return False
        
        n = len(seq)
        expected_count = n * (n + 1) // 2
        products = get_product_set(seq)
        return len(products) == expected_count

    def run_greedy():
        """
        Fast greedy heuristic to generate a strong initial sequence.
        Returns (sequence, product_set).
        """
        values = []
        products = set()
        for x in range(2, 129):
            # Calculate new tail products if x is appended
            tails = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            
            # Check for collisions
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values, products

    # --- Phase 1: Beam Search with Efficient Incremental Tracking ---
    # The core structural idea from Inspiration 1: iterate over values 2..128,
    # maintaining a beam of states (seq, prods). This is more efficient than
    # expanding all candidates per beam because we process each value once.
    # We also include the "skip" option to allow gaps.

    def beam_search_phase():
        W = 200  # Beam width
        # Initialize beam with empty sequence
        # State: (seq_tuple, product_set)
        beam = {(): set()}
        
        best_seq = []
        best_len = 0

        for x in range(2, 129):
            if time.time() > deadline - 10:
                break

            next_beam = {}
            
            for seq, prods in beam.items():
                # Option 1: Skip x
                # The state remains the same
                if seq not in next_beam:
                    next_beam[seq] = prods
                
                # Option 2: Include x
                # Calculate new tail products
                tails = [x]
                prod = x
                for a in reversed(seq):
                    prod *= a
                    tails.append(prod)
                
                # Check for collisions
                has_collision = False
                for t in tails:
                    if t in prods:
                        has_collision = True
                        break
                
                if not has_collision:
                    new_seq = seq + (x,)
                    new_prods = set(prods)
                    new_prods.update(tails)
                    
                    # Add to next_beam
                    # Note: new_seq is unique for this step if generated from a unique parent
                    # But could two parents generate the same new_seq? No, because x is fixed and appended.
                    # Parent is determined by new_seq minus x.
                    next_beam[new_seq] = new_prods
            
            # Prune beam
            # Sort by length desc, then by sum asc (smaller sum leaves more room for larger numbers later? 
            # Actually, smaller sum usually means we picked smaller numbers, which might be better for 
            # avoiding overflow or collisions with large products? Or does it mean we are "lighter"?
            # Let's stick with length desc, then sum asc as a heuristic for flexibility.)
            sorted_seqs = sorted(next_beam.keys(), key=lambda s: (-len(s), sum(s) if s else 0))
            
            # Keep top W
            beam = {s: next_beam[s] for s in sorted_seqs[:W]}
            
            # Update best
            if sorted_seqs:
                top_seq = sorted_seqs[0]
                if len(top_seq) > best_len:
                    best_len = len(top_seq)
                    best_seq = list(top_seq)
                elif len(top_seq) == best_len:
                    # Tie-breaker: prefer smaller sum?
                    if sum(top_seq) < sum(best_seq):
                        best_seq = list(top_seq)

        return best_seq, best_len

    # --- Phase 2: Local Search / Polish ---
    # Refinement machinery from Inspiration 1/2: Try to insert missing numbers.
    # Also try simple swaps if stuck, though insertion is primary for length maximization.

    def local_search(seq, deadline):
        current_seq = seq[:]
        current_set = set(current_seq)
        all_numbers = list(range(2, 129))
        missing = [x for x in all_numbers if x not in current_set]
        
        improved = True
        # Limit iterations to avoid infinite loops or excessive time
        max_iterations = 50
        
        while improved and time.time() < deadline - 2 and max_iterations > 0:
            improved = False
            max_iterations -= 1
            
            # Try adding each missing element
            for m in missing[:]:  # Iterate over a copy
                if time.time() > deadline - 2:
                    break
                
                # Find insertion position
                pos = 0
                while pos < len(current_seq) and current_seq[pos] < m:
                    pos += 1
                if pos < len(current_seq) and current_seq[pos] == m:
                    continue # Already there? Shouldn't happen as m is missing.
                
                candidate = current_seq[:pos] + [m] + current_seq[pos:]
                if is_valid(candidate):
                    current_seq = candidate
                    current_set.add(m)
                    missing.remove(m)
                    improved = True
                    break # Restart loop with new missing list
            
            # If we couldn't add anything, try a swap?
            # Removing one element and adding another of the same "slot" size?
            # Actually, if we can't add anything, the sequence is likely maximal 
            # with respect to single insertions. 
            # A swap (remove e, add m) keeps length same. It doesn't help maximize length 
            # unless it enables a future insertion. 
            # Given time constraints, simple insertion is the primary driver for length.
            # We could try: Remove e, Add m, then try to Add k.
            # This is a 2-move lookahead.
            
            if not improved and time.time() < deadline - 5:
                # Try 2-move: Remove one, Add one, then try to Add another
                found_improvement = False
                for idx in range(len(current_seq)):
                    if time.time() > deadline - 5:
                        break
                    e = current_seq[idx]
                    without_e = current_seq[:idx] + current_seq[idx+1:]
                    
                    # Try adding a missing number m to without_e
                    for m in missing[:]:
                        if time.time() > deadline - 5:
                            break
                        pos = 0
                        while pos < len(without_e) and without_e[pos] < m:
                            pos += 1
                        if pos < len(without_e) and without_e[pos] == m:
                            continue
                        
                        candidate1 = without_e[:pos] + [m] + without_e[pos:]
                        if not is_valid(candidate1):
                            continue
                        
                        # Now try to add another missing number k to candidate1
                        # This is expensive, so limit k checks
                        for k in missing[:10]: # Limit lookahead
                            if k == m:
                                continue
                            pos_k = 0
                            while pos_k < len(candidate1) and candidate1[pos_k] < k:
                                pos_k += 1
                            if pos_k < len(candidate1) and candidate1[pos_k] == k:
                                continue
                            
                            candidate2 = candidate1[:pos_k] + [k] + candidate1[pos_k:]
                            if is_valid(candidate2):
                                # We found a 2-step improvement: removed e, added m, added k
                                # Net gain: +1 length
                                current_seq = candidate2
                                current_set = set(current_seq)
                                # Update missing list
                                missing = [x for x in all_numbers if x not in current_set]
                                found_improvement = True
                                break
                        if found_improvement:
                            break
                    if found_improvement:
                        break
                
                if found_improvement:
                    improved = True

        return current_seq

    # --- Main Logic ---
    
    # 1. Run Beam Search
    best_seq, best_len = beam_search_phase()
    
    # 2. Fallback to Greedy if Beam Search failed or returned nothing
    if best_len == 0:
        best_seq, _ = run_greedy()
        best_len = len(best_seq)
    
    # 3. Local Search Polish
    if time.time() < deadline - 5:
        refined_seq = local_search(best_seq, deadline)
        if len(refined_seq) > best_len:
            best_seq = refined_seq
            best_len = len(best_seq)
    
    # Final validation
    if not is_valid(best_seq):
        # Fallback to greedy if something went wrong
        best_seq, _ = run_greedy()

    return best_seq
