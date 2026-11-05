import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit

    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

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

    def get_product_set(seq):
        """Compute all consecutive block products for a sequence."""
        products = set()
        n = len(seq)
        if n == 0:
            return products
        # Use prefix products for efficiency? 
        # For n up to ~60, O(n^2) is fine.
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

    # 1. Initial Greedy Solution
    best_seq, best_prods = run_greedy()
    best_len = len(best_seq)
    
    # 2. Beam Search
    # State: (seq_tuple, product_set)
    # We initialize the beam with the greedy result to ensure a high starting point.
    # However, beam search usually starts from empty and explores. 
    # To leverage the greedy result, we can just start the beam search from scratch 
    # but with a wide beam, OR seed the beam with the greedy result.
    # Let's seed the beam with the greedy result and also allow exploration from scratch?
    # No, seeding with the greedy result restricts the beam to subsets/supersets?
    # Actually, beam search from scratch with W=500 is very powerful for this size.
    # Let's run a beam search from scratch to find potentially better structures than pure greedy.
    
    beam = {(): set()}  # Start with empty sequence
    
    # We will iterate x from 2 to 128.
    # At each step, we expand the beam.
    
    W = 100  # Beam width. We can increase this if time permits.
    
    for x in range(2, 129):
        if time.time() > deadline:
            break
            
        next_beam = {}
        
        for seq, prods in beam.items():
            # Option 1: Skip x
            # The state (seq, prods) remains valid.
            # We add it to next_beam.
            # If multiple paths lead to the same seq, prods is identical.
            if seq not in next_beam:
                next_beam[seq] = prods
            
            # Option 2: Include x
            # Since we iterate x in increasing order, any seq in the beam has elements < x.
            # So appending x maintains strictly increasing order.
            
            # Calculate new tails
            tails = [x]
            prod = x
            for a in reversed(seq):
                prod *= a
                tails.append(prod)
                
            # Check collision
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
                # If new_seq is already in next_beam, it must be from a different path?
                # No, x is unique in this step. If new_seq is in next_beam, it was added by "Skip" from a state that was already new_seq?
                # No, "Skip" keeps the old seq. "Include" creates new_seq.
                # Can two different parent states produce the same new_seq?
                # Parent state is determined by the set of elements < x.
                # If parent1 and parent2 both produce new_seq, then parent1 = parent2 = new_seq without x.
                # So new_seq is unique for this x step.
                next_beam[new_seq] = new_prods
        
        # Prune beam
        # Sort by length desc, then by sum asc (smaller sum is generally better for flexibility)
        sorted_seqs = sorted(next_beam.keys(), key=lambda s: (-len(s), sum(s) if s else 0))
        
        # Keep top W
        # If we have fewer than W, keep all
        beam = {s: next_beam[s] for s in sorted_seqs[:W]}
        
        # Update best
        if len(sorted_seqs) > 0:
            top_seq = sorted_seqs[0]
            if len(top_seq) > best_len:
                best_len = len(top_seq)
                best_seq = list(top_seq)
                best_prods = next_beam[top_seq]
            elif len(top_seq) == best_len:
                # Tie-breaker: prefer smaller sum? Or just keep existing?
                # Let's keep the one with smaller sum for potential future improvements
                if sum(top_seq) < sum(best_seq):
                    best_seq = list(top_seq)
                    best_prods = next_beam[top_seq]

    # 3. Local Search / Polish on the best sequence found
    # Try to insert any missing numbers into the best sequence.
    # Since the sequence is sorted, inserting a number $m$ is only possible if it fits in the sorted order 
    # and doesn't cause product collisions.
    # Note: The beam search already considers all subsets. 
    # However, the beam width W might have pruned a path that leads to a longer sequence later.
    # A local search can try to "repair" or "extend" the best sequence found.
    
    if time.time() < deadline - 5:
        current_best = best_seq[:]
        current_best_set = set(current_best)
        missing = [x for x in range(2, 129) if x not in current_best_set]
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try adding each missing element
            for m in missing:
                if time.time() > deadline - 2:
                    break
                # Find insertion position
                pos = 0
                while pos < len(current_best) and current_best[pos] < m:
                    pos += 1
                if pos < len(current_best) and current_best[pos] == m:
                    continue # Already there? Shouldn't happen as m is missing.
                
                candidate = current_best[:pos] + [m] + current_best[pos:]
                if is_valid(candidate):
                    current_best = candidate
                    current_best_set.add(m)
                    missing.remove(m)
                    improved = True
                    best_len = len(current_best)
                    best_seq = current_best[:]
                    best_prods = get_product_set(current_best)
                    break # Restart loop with new missing list
            
            # If we couldn't add anything, try swapping?
            # Swapping is hard because order is fixed by value.
            # "Swap" in this context means: Remove one element e, Add one element m.
            # This is equivalent to finding a sequence of same length but different composition.
            # Does same length help? No, we maximize length.
            # So we only care about adding.
            
            # If we are stuck, maybe we should try a different seed from the beam?
            # The beam search already did a global search.
            # Let's just stick with the best found.
    
    # Final check
    if not is_valid(best_seq):
        # Fallback to greedy
        best_seq, _ = run_greedy()

    return best_seq
