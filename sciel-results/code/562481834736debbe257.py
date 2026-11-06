import time

def construct():
    # Generate primes <= 1000
    primes = []
    for i in range(2, 1001):
        is_prime = True
        for j in range(2, int(i ** 0.5) + 1):
            if i % j == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
    
    n_primes = len(primes)
    LIMIT = 3000  # Increased lookahead
    
    # counts[x] = number of primes covering x
    counts = [0] * (LIMIT + 1)
    residues = [0] * n_primes
    
    current_L = 0
    
    for i, p in enumerate(primes):
        best_r = 0
        best_score_tuple = (-1, -1, -1) # (new_L, num_single_in_frontier, -num_zero_in_frontier)
        
        for r in range(p):
            # Update counts
            start = r if r > 0 else p
            if start <= LIMIT:
                for x in range(start, LIMIT + 1, p):
                    counts[x] += 1
            
            # Calculate new_L
            new_L = current_L
            x = current_L + 1
            while x <= LIMIT and counts[x] >= 2:
                new_L = x
                x += 1
            
            # Calculate frontier stats
            # We look at the region [new_L + 1, new_L + 50]
            # Maximizing '1's here means we are creating candidates for the next extension
            # Minimizing '0's means we are filling gaps
            frontier_start = new_L + 1
            frontier_end = min(LIMIT, new_L + 50)
            
            num_ones = 0
            num_zeros = 0
            for x in range(frontier_start, frontier_end + 1):
                if counts[x] == 1:
                    num_ones += 1
                elif counts[x] == 0:
                    num_zeros += 1
            
            score_tuple = (new_L, num_ones, -num_zeros)
            
            if score_tuple > best_score_tuple:
                best_score_tuple = score_tuple
                best_r = r
            
            # Revert counts
            if start <= LIMIT:
                for x in range(start, LIMIT + 1, p):
                    counts[x] -= 1
        
        # Apply the best residue
        residues[i] = best_r
        start = best_r if best_r > 0 else p
        if start <= LIMIT:
            for x in range(start, LIMIT + 1, p):
                counts[x] += 1
        
        # Update current_L
        x = current_L + 1
        while x <= LIMIT and counts[x] >= 2:
            current_L = x
            x += 1

    return residues
