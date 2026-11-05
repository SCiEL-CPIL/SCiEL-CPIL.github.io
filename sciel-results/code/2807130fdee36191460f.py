import time
import random

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    best_set = []
    
    def greedy_optimized():
        S = []
        # R[d] is a set of residues r such that (x % d == r) implies conflict with d
        # Specifically, if x % d == r, then there exist a, b in S \ {d} such that x + a + b is divisible by d.
        # r = -(a+b) % d
        R = {}
        # TripleSums is a list of sums of 3 distinct elements in S
        triple_sums = []
        
        candidates = list(range(1, 193))
        random.shuffle(candidates)
        
        for x in candidates:
            # Check 1: Does x divide any triple sum in S?
            if any(s % x == 0 for s in triple_sums):
                continue
            
            # Check 2: For any d in S, does x violate the residue constraint?
            valid = True
            for d in S:
                if x % d in R[d]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Add x to S
            S.append(x)
            
            # Update triple_sums: add x + a + b for all a, b in S \ {x}
            # Note: S now contains x. We need pairs from S \ {x}.
            old_S = S[:-1]
            for i in range(len(old_S)):
                a = old_S[i]
                for j in range(i+1, len(old_S)):
                    b = old_S[j]
                    triple_sums.append(x + a + b)
            
            # Initialize R[x]
            R[x] = set()
            for i in range(len(old_S)):
                a = old_S[i]
                for j in range(i+1, len(old_S)):
                    b = old_S[j]
                    R[x].add(-(a + b) % x)
            
            # Update R[d] for all existing d in old_S
            for d in old_S:
                # New pairs involving x and b in old_S \ {d}
                for b in old_S:
                    if b == d:
                        continue
                    # The pair is (x, b). Residue is -(x + b) % d
                    R[d].add(-(x + b) % d)
                    
        return S

    # Run multiple trials with different random seeds/orders
    for _ in range(1000):
        if time.time() - start_time > time_limit:
            break
        
        S = greedy_optimized()
        if len(S) > len(best_set):
            best_set = S
            if len(best_set) == 48:
                break

    return best_set
