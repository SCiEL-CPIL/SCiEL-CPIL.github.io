import time
import random

def construct():
    t_start = time.time()
    DEADLINE = t_start + 55.0  # Safety margin before 60s limit
    N = 4096
    MAX_SIZE = 80
    
    def can_add(x, S, triple_sums):
        """
        Check if x can be added to S without creating duplicate triple sums.
        S is a list of selected numbers.
        triple_sums is a set of all sums of unordered triples from S.
        """
        new_sums = set()
        
        # Triple (x, x, x)
        s = 3 * x
        if s in triple_sums:
            return False
        new_sums.add(s)
        
        # Triples (x, x, y) for y in S
        for y in S:
            s = 2 * x + y
            if s in triple_sums:
                return False
            if s in new_sums:
                return False
            new_sums.add(s)
            
        # Triples (x, y, z) for y, z in S, y <= z
        # Note: y and z are from S, so y != x is guaranteed if x not in S yet.
        # We must ensure y <= z to represent unordered triples correctly 
        # (since (x,y,z) and (x,z,y) are the same sum).
        # Also, we must consider y,z from S.
        # Since S is a list, we iterate indices.
        n = len(S)
        for i in range(n):
            y = S[i]
            for j in range(i, n):
                z = S[j]
                s = x + y + z
                if s in triple_sums:
                    return False
                if s in new_sums:
                    return False
                new_sums.add(s)
                
        return True

    def greedy_random():
        """
        Perform a greedy construction with a random order of candidates.
        Returns the resulting set.
        """
        S = []
        triple_sums = set()
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if len(S) >= MAX_SIZE:
                break
            if time.time() > DEADLINE:
                break
                
            if can_add(x, S, triple_sums):
                S.append(x)
                # Update triple_sums
                # Add 3x
                triple_sums.add(3 * x)
                # Add 2x + y
                for y in S[:-1]: # S[:-1] is S before adding x
                    triple_sums.add(2 * x + y)
                # Add x + y + z
                n_old = len(S) - 1
                for i in range(n_old):
                    y = S[i]
                    for j in range(i, n_old):
                        z = S[j]
                        triple_sums.add(x + y + z)
                        
        return S

    best_set = []
    
    # Run multiple iterations
    while time.time() < DEADLINE:
        current_set = greedy_random()
        if len(current_set) > len(best_set):
            best_set = current_set
            if len(best_set) == MAX_SIZE:
                break
                
    return best_set
