import time
import random

def construct():
    start_time = time.time()
    limit = 55.0  # Safety margin
    
    def is_valid(seq):
        """
        Check if the sequence has distinct products for all non-empty consecutive blocks.
        seq must be sorted.
        """
        n = len(seq)
        if n == 0:
            return True
        products = set()
        # Use a list of lists or just compute on the fly?
        # n is small (< 130), so O(n^2) products is fine.
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in products:
                    return False
                products.add(prod)
        return True

    # 1. Greedy Initialization
    S = []
    for x in range(2, 129):
        # Check if adding x to S keeps it valid
        # Since S is sorted, x is always appended at the end
        if is_valid(S + [x]):
            S.append(x)
    
    best_S = S[:]
    best_len = len(S)
    
    # 2. Local Search / Hill Climbing
    # We try to improve the length.
    
    while time.time() - start_time < limit:
        improved = False
        
        # Step A: Try to insert any single missing number
        missing = [x for x in range(2, 129) if x not in S]
        random.shuffle(missing)
        
        for x in missing:
            if time.time() - start_time > limit:
                break
            # Find position to insert
            # Since we check validity of the whole list, we can just create the new list
            # But we can optimize by checking if x can be inserted at its sorted position
            new_S = S[:]
            # Insert x in sorted order
            # Python's bisect is good, but list insert is O(N)
            # N is small, so just find index
            idx = 0
            for i, val in enumerate(new_S):
                if val > x:
                    idx = i
                    break
            else:
                idx = len(new_S)
            
            new_S.insert(idx, x)
            
            if is_valid(new_S):
                S = new_S
                improved = True
                if len(S) > best_len:
                    best_S = S[:]
                    best_len = len(S)
                break # Restart the missing list scan with the new S
        
        if time.time() - start_time > limit:
            break
            
        if not improved:
            # Step B: Try 1-out, 2-in moves to escape local optima
            # This increases length by 1
            if len(S) < 127: # Can't exceed total numbers
                moves_tried = 0
                while moves_tried < 50 and time.time() - start_time < limit:
                    moves_tried += 1
                    # Pick a random element to remove
                    y = random.choice(S)
                    # Pick two random elements to add
                    # Ensure they are distinct and not y
                    candidates = [x for x in range(2, 129) if x not in S and x != y]
                    if len(candidates) < 2:
                        continue
                    
                    # Try random pairs
                    for _ in range(10): # Try a few random pairs for this y
                        if time.time() - start_time > limit:
                            break
                        x1, x2 = random.sample(candidates, 2)
                        if x1 > x2:
                            x1, x2 = x2, x1
                        
                        new_S = [v for v in S if v != y]
                        new_S.append(x1)
                        new_S.append(x2)
                        new_S.sort()
                        
                        if is_valid(new_S):
                            S = new_S
                            improved = True
                            if len(S) > best_len:
                                best_S = S[:]
                                best_len = len(S)
                            break
                    
                    if improved:
                        break
                        
    return best_S
