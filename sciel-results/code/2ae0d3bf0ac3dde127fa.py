import time
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    best_seq = []
    best_len = 0
    
    # State for DFS
    # current_seq: list of ints
    # products_set: set of all subarray products
    
    def dfs(start_val, current_seq, products_set):
        nonlocal best_seq, best_len
        
        # Time check
        if time.time() > deadline:
            return
        
        # Update best
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = current_seq[:]
            
        # Pruning: If we can't possibly beat best_len
        # Max possible length is 128 - start_val + 1 (if we took all)
        # This is a very loose bound, but better than nothing.
        # Actually, if best_len is already high, we might prune deeper branches.
        # But DFS explores one path fully.
        
        # Try adding next elements
        for x in range(start_val, 129):
            if time.time() > deadline:
                return
            
            # Check if x can be appended
            # Calculate new suffix products
            new_prods = []
            curr_prod = x
            valid = True
            
            # Q_0 = x
            if curr_prod in products_set:
                valid = False
            else:
                new_prods.append(curr_prod)
            
            # Q_k for k >= 1
            # Iterate backwards through current_seq
            # S = [s_0, ..., s_{m-1}]
            # We need s_{m-1}*x, s_{m-2}*s_{m-1}*x, etc.
            
            if valid:
                # We need to check collisions with existing products
                # And also, the new products must not collide with each other?
                # They are distinct because x >= 2 and prefix products are distinct.
                # So we only check against products_set.
                
                # Wait, we must check all new prods against products_set
                # before committing.
                # We can do this in the loop.
                
                pass # Logic restructured below
            
            # Let's restructure the check for efficiency
            # We compute all new prods first, check if any in products_set.
            # If any is in products_set, it's invalid.
            
            new_prods = []
            curr_prod = x
            # Check x
            if x in products_set:
                continue
            new_prods.append(x)
            
            # Check suffixes
            # We traverse current_seq from end to start
            # prod = x * S[-1]
            # prod = x * S[-1] * S[-2]
            # ...
            
            # To avoid creating a huge list if it fails early, we can check as we go?
            # No, we need to add them all to the set if valid.
            # But we can check validity first.
            
            # Optimization:
            # The new products are $x \cdot P(i, m)$ for $i=1..m$ and $x$.
            # $P(i, m)$ are the suffix products of the current sequence.
            # Let $S_{suf} = [P(1, m), P(2, m), \dots, P(m, m)]$.
            # Note $P(m, m) = S[m-1]$.
            # The new products are $x, x \cdot S[m-1], x \cdot S[m-2] \cdot S[m-1], \dots$
            
            # We can maintain a list of suffix products of the current sequence to speed this up?
            # Or just calculate on the fly.
            # Since m is small (< 20), calculating on the fly is fine.
            
            suffix_prod = 1
            valid = True
            new_prods = [x] # Start with x
            
            # Check x against set
            if x in products_set:
                continue
                
            for i in range(len(current_seq) - 1, -1, -1):
                suffix_prod *= current_seq[i]
                new_prod = suffix_prod * x
                if new_prod in products_set:
                    valid = False
                    break
                new_prods.append(new_prod)
                
            if not valid:
                continue
                
            # If valid, recurse
            # Create new set
            # Set union is O(size). Size is ~ N^2/2. N is small.
            new_products_set = products_set.union(new_prods)
            
            dfs(x + 1, current_seq + [x], new_products_set)
            
            # Backtracking is implicit in recursion (immutable structures)
            
    # Initial call
    # Start with empty sequence?
    # Or start with first element?
    # The problem asks for sequence in [2, 128].
    # We can start the loop from 2.
    
    # To ensure we find the *longest*, we should explore all possibilities.
    # However, DFS with "smaller first" tends to find long sequences early.
    # We can also try to prune if current length + max possible remaining <= best_len.
    
    # Let's run DFS starting from empty sequence, trying 2..128 as first element.
    # Actually, the loop in dfs handles the choice of the next element.
    # So we call dfs(2, [], set())
    
    dfs(2, [], set())
    
    return best_seq

if __name__ == "__main__":
    res = construct()
    print(res)
