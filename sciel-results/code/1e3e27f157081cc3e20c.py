import time
import math

def construct():
    start_time = time.time()
    deadline = 85.0  # 5 seconds margin
    
    # 1. Generate all primes in [2, 128]
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
        
    primes = [x for x in range(2, 129) if is_prime(x)]
    composites = [x for x in range(2, 129) if not is_prime(x)]
    
    # 2. Initialize sequence with primes
    seq = primes
    candidates = composites
    
    # Helper to check if a sequence is valid and return its products
    def get_products(sequence):
        prods = set()
        n = len(sequence)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= sequence[j]
                if prod in prods:
                    return None  # Collision found
                prods.add(prod)
        return prods
        
    # Initial products
    current_prods = get_products(seq)
    if current_prods is None:
        return seq # Should not happen for primes
    
    # 3. Iterative insertion of composites
    # We will try to insert composites one by one.
    # To maximize length, we should try to fit as many as possible.
    # Strategy: In each pass, try to insert each remaining candidate at the best available spot.
    # If a candidate fits, insert it and remove from candidates.
    # Repeat until no candidates can be inserted.
    
    while True:
        if time.time() > deadline:
            break
            
        improved = False
        # Sort candidates by value? Or maybe by "difficulty"?
        # Let's just iterate through the current list of candidates.
        # We need to be careful: inserting one candidate changes the validity for others.
        # So we process one candidate at a time.
        
        for i in range(len(candidates)):
            if time.time() > deadline:
                break
                
            c = candidates[i]
            best_pos = -1
            best_prods = None
            
            # Try inserting c at every position in seq
            for pos in range(len(seq) + 1):
                new_seq = seq[:pos] + [c] + seq[pos:]
                prods = get_products(new_seq)
                if prods is not None:
                    # Valid insertion found.
                    # We pick the first valid position for simplicity.
                    # A smarter choice might be to pick the position that
                    # minimizes "crowding" or similar, but first valid is a good heuristic.
                    best_pos = pos
                    best_prods = prods
                    break
            
            if best_pos != -1:
                # Perform insertion
                seq = seq[:best_pos] + [c] + seq[best_pos:]
                current_prods = best_prods
                # Remove candidate from list
                candidates.pop(i)
                improved = True
                # Restart the candidate loop because indices shifted and seq changed
                # Breaking the inner loop to restart the outer 'for i' logic is tricky with indices.
                # It's easier to just break and restart the whole 'while' loop?
                # No, that's inefficient if we have many candidates.
                # Let's just break the inner 'for pos' loop and continue to next candidate i.
                # But wait, if we modify 'seq', the positions for subsequent candidates change.
                # The current structure:
                # while True:
                #   for i in range(len(candidates)):
                #      ... insert ...
                #      if inserted:
                #         candidates.pop(i)
                #         improved = True
                #         # Since we popped i, the next candidate is now at i.
                #         # So we should NOT do i+=1.
                #         # But the for loop increments i automatically.
                #         # So we need to decrement i or use a while loop for candidates.
                #         pass
                
        if not improved:
            break
            
    return seq
