import math
import itertools
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 110.0  # safety margin before 120s limit
    
    def elapsed():
        return time.time() - start_time
    
    def build_paley(p):
        """Build Paley tournament on p vertices (p must be prime ≡ 3 mod 4)."""
        squares = set()
        for x in range(1, p):
            squares.add((x*x) % p)
        
        orientation = []
        for i in range(p):
            for j in range(i+1, p):
                diff = (j - i) % p
                if diff in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        
        return p, orientation

    def build_in_nb(p, orientation):
        """Build in-neighborhood bitsets."""
        adj = np.zeros((p, p), dtype=np.uint8)
        idx = 0
        for i in range(p):
            for j in range(i+1, p):
                if orientation[idx] == 0:
                    adj[i, j] = 1
                else:
                    adj[j, i] = 1
                idx += 1
        
        in_nb = [0] * p
        for v in range(p):
            for u in range(p):
                if adj[v, u]:
                    in_nb[u] |= (1 << v)
        return in_nb, adj

    def find_reducible_vertex(p, in_nb):
        """
        Check if there is a vertex w that can be removed such that the remaining
        tournament is still valid.
        w can be removed if there is no 4-set S in V\{w} such that D(S) = {w}.
        i.e., w is not the unique dominator of any 4-set S that does not contain w.
        """
        critical = [False] * p
        
        # Iterate over all 4-sets
        for a in range(p - 3):
            if elapsed() > DEADLINE:
                return -1
            for b in range(a + 1, p - 2):
                for c in range(b + 1, p - 1):
                    abc = in_nb[a] & in_nb[b] & in_nb[c]
                    if abc == 0:
                        continue
                    for d in range(c + 1, p):
                        D = abc & in_nb[d]
                        if D == 0:
                            return -1 # Base tournament invalid (should not happen for Paley 67)
                        
                        # Check if D is a singleton
                        # D & (D - 1) == 0 means singleton
                        if D & (D - 1) == 0:
                            # Find the index
                            w = D.bit_length() - 1
                            # w is the unique dominator of {a,b,c,d}
                            # {a,b,c,d} does not contain w (since w beats them)
                            # So if we remove w, this 4-set has no dominator.
                            critical[w] = True
        
        # If any vertex is not critical, we can remove it
        for w in range(p):
            if not critical[w]:
                return w
        
        return -1

    def remove_vertex(p, orientation, w):
        """Remove vertex w and return new n and orientation."""
        new_n = p - 1
        new_orientation = []
        
        # Build adj matrix
        adj = np.zeros((p, p), dtype=np.uint8)
        idx = 0
        for i in range(p):
            for j in range(i+1, p):
                if orientation[idx] == 0:
                    adj[i, j] = 1
                else:
                    adj[j, i] = 1
                idx += 1
        
        # Remove row/col w
        mask = [i for i in range(p) if i != w]
        # Create new adj
        new_adj = np.zeros((new_n, new_n), dtype=np.uint8)
        for i_new, i_old in enumerate(mask):
            for j_new, j_old in enumerate(mask):
                if i_new < j_new:
                    new_adj[i_new, j_new] = adj[i_old, j_old]
                    new_adj[j_new, i_new] = 1 - adj[i_old, j_old]
        
        # Flatten to orientation list
        for i in range(new_n):
            for j in range(i+1, new_n):
                if new_adj[i, j] == 1:
                    new_orientation.append(0) # i -> j
                else:
                    new_orientation.append(1) # j -> i
        
        return new_n, new_orientation

    # Try primes
    candidate_primes = [31, 43, 59, 67, 79, 83]
    
    best_result = None
    
    for p in candidate_primes:
        if elapsed() > DEADLINE - 20:
            break
            
        _, orientation = build_paley(p)
        
        # Quick check: is it valid?
        # We'll use the in_nb method which also helps for reduction
        in_nb, adj = build_in_nb(p, orientation)
        
        # Check validity and find reducible vertex
        # We can combine validity check and critical marking
        valid = True
        critical = [False] * p
        
        for a in range(p - 3):
            if elapsed() > DEADLINE - 5:
                break
            for b in range(a + 1, p - 2):
                for c in range(b + 1, p - 1):
                    abc = in_nb[a] & in_nb[b] & in_nb[c]
                    if abc == 0:
                        continue
                    for d in range(c + 1, p):
                        D = abc & in_nb[d]
                        if D == 0:
                            valid = False
                            break
                        if D & (D - 1) == 0:
                            w = D.bit_length() - 1
                            critical[w] = True
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            continue
            
        # If valid, check if we can reduce
        reduced_w = -1
        for w in range(p):
            if not critical[w]:
                reduced_w = w
                break
        
        if reduced_w != -1:
            new_n, new_orientation = remove_vertex(p, orientation, reduced_w)
            # Try to remove more vertices greedily
            current_n = new_n
            current_orientation = new_orientation
            current_in_nb, current_adj = build_in_nb(current_n, current_orientation)
            
            # Keep trying to remove non-critical vertices
            while elapsed() < DEADLINE - 10:
                crit = [False] * current_n
                still_valid = True
                for a in range(current_n - 3):
                    if elapsed() > DEADLINE - 10:
                        break
                    for b in range(a + 1, current_n - 2):
                        for c in range(b + 1, current_n - 1):
                            abc = current_in_nb[a] & current_in_nb[b] & current_in_nb[c]
                            if abc == 0:
                                continue
                            for d in range(c + 1, current_n):
                                D = abc & current_in_nb[d]
                                if D == 0:
                                    still_valid = False
                                    break
                                if D & (D - 1) == 0:
                                    w = D.bit_length() - 1
                                    crit[w] = True
                            if not still_valid:
                                break
                        if not still_valid:
                            break
                    if not still_valid:
                        break
                
                if not still_valid:
                    break
                    
                # Find a non-critical vertex
                removable = -1
                for w in range(current_n):
                    if not crit[w]:
                        removable = w
                        break
                
                if removable == -1:
                    break
                    
                current_n, current_orientation = remove_vertex(current_n, current_orientation, removable)
                current_in_nb, current_adj = build_in_nb(current_n, current_orientation)
            
            if current_n < p:
                return {'n': current_n, 'orientation': current_orientation}
            else:
                return {'n': p, 'orientation': orientation}
        else:
            # Cannot reduce by 1, return p
            return {'n': p, 'orientation': orientation}
            
    # Fallback
    p = 67
    _, orientation = build_paley(p)
    return {'n': p, 'orientation': orientation}

# EVOLVE-BLOCK-END
