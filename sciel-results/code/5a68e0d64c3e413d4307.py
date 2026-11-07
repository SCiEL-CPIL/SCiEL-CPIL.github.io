import time
import random

def construct():
    start_time = time.time()
    time_limit = 50.0  # Safety margin
    
    max_val = 256
    max_size = 60
    
    best_size = 0
    best_set = []
    
    # Precompute masks or just use ints
    
    def is_valid(A):
        n = len(A) - 1
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if A[i] & A[j]:
                    return False
        return True

    # To speed up is_valid, we can maintain a global "used" bitset?
    # No, because a sum can be used in A_k and A_m is invalid, but A_k and A_l might be ok?
    # No, "any two subsets with different cardinalities their sums must differ".
    # So A_k and A_m must be disjoint for ALL k != m.
    # This means the union of all A_k must have no overlaps.
    # We can maintain a single bitset `used_sums`?
    # No, because a sum $s$ can be in $A_k$. If we add $x$, $s$ might move to $A_{k+1}$?
    # No, $s$ is a sum of a specific subset. The subset doesn't change.
    # The sets $A_k$ are additive.
    # If $s \in A_k$, it stays in $A_k$ (as a subset of the larger set).
    # New sums are added to $A_{k+1}$.
    # So we just need to ensure that the NEW sums added to $A_{k+1}$ do not intersect with ANY $A_j$ ($j \neq k+1$).
    # And the OLD sums in $A_{k+1}$ (which are subsets of size $k+1$ from old S) are already valid.
    # Wait.
    # $A'_{k+1} = A_{k+1} \cup (A_k \ll x)$.
    # We know $A_{k+1}$ is disjoint from all other $A_j$.
    # We need to check if $(A_k \ll x)$ intersects with:
    # 1. $A_{k+1}$ (old sums of size $k+1$)
    # 2. Any $A_j$ for $j \neq k+1$ and $j \neq k$?
    #    Actually, $A_k \ll x$ are sums of size $k+1$.
    #    They must not intersect with ANY $A_j$ where $j \neq k+1$.
    #    This includes $A_{k+1}$ (old) and all other $A_j$.
    
    # So for each candidate $x$, we can check:
    # For each $k$, let $NewSums = A[k] << x$.
    # Check if $NewSums$ intersects with $A[j]$ for any $j \neq k+1$.
    # AND check if $NewSums$ intersects with $A[k+1]$ (old).
    # If no intersection, it's valid.
    
    # This is much faster than checking all pairs of the new A.
    
    def can_add(A, x):
        n = len(A) - 1
        # A[k] is bitset for size k
        # We are adding x.
        # New sums for size k+1 are (A[k] << x).
        # These new sums must not conflict with any existing sum in A[j] where j != k+1.
        # Also, they must not conflict with the EXISTING sums in A[k+1].
        
        # We can combine all A[j] for j != k+1 into one bitset?
        # Or just check against each.
        
        # Optimization:
        # Let's check for each k from 0 to n:
        # new_bits = A[k] << x
        # For each j from 0 to n:
        #   if j != k+1:
        #     if new_bits & A[j]: return False
        #   # Note: if j == k+1, we also need to check?
        #   # Yes, new_bits are added to A[k+1]. They must not conflict with OLD A[k+1].
        #   # So actually, new_bits must not conflict with ANY A[j] for j != k+1, 
        #   # AND new_bits must not conflict with A[k+1] (old).
        #   # So new_bits must not conflict with ANY A[j] for j != k+1?
        #   # Wait. The new sums are IN A[k+1] (new).
        #   # They must be disjoint from A[j] for j != k+1.
        #   # They must be disjoint from A[k+1] (old) ?
        #   # Yes, because A[k+1] (new) = A[k+1] (old) U new_bits.
        #   # If new_bits intersects A[k+1] (old), that's fine?
        #   # NO. The condition is that sums in DIFFERENT cardinalities are disjoint.
        #   # Sums within the SAME cardinality can overlap.
        #   # So new_bits intersecting A[k+1] (old) is ALLOWED.
        #   # new_bits intersecting A[j] for j != k+1 is FORBIDDEN.
        
        # So:
        # For each k:
        #   new_bits = A[k] << x
        #   For each j != k+1:
        #     if new_bits & A[j]: return False
        
        # This is O(n^2) bit checks per candidate.
        # n <= 60. 3600 checks. Fast.
        
        for k in range(n + 1):
            new_bits = A[k] << x
            if new_bits == 0: continue
            for j in range(n + 1):
                if j == k + 1:
                    continue
                if new_bits & A[j]:
                    return False
        return True

    # Search
    while time.time() - start_time < time_limit:
        S = []
        A = [1]  # A[0] = 1
        
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if len(S) >= max_size:
                break
            
            if can_add(A, x):
                # Update A
                new_A = A[:]
                n = len(S)
                # Update in reverse order to avoid using x multiple times?
                # No, we are computing A_new from A_old.
                # A_new[k] = A_old[k] | (A_old[k-1] << x)
                # We can update A in place if we go from high k to low k?
                # A[k] depends on A[k-1].
                # If we update A[k+1] first, then A[k] is still old.
                # So:
                for k in range(n, -1, -1):
                    A[k+1] |= (A[k] << x)
                # Note: A is a list. A[k+1] might not exist?
                # A has size n+1. We need to append 0?
                # A[0]..A[n].
                # New size n+1. A[0]..A[n+1].
                # We should extend A.
                
                # Let's do it cleanly:
                # new_A = A + [0]
                # for k in range(n, -1, -1):
                #    new_A[k+1] |= (A[k] << x)
                # A = new_A
                
                # To save memory/time, let's just create new list.
                new_A = A + [0]
                for k in range(n, -1, -1):
                    new_A[k+1] |= (A[k] << x)
                A = new_A
                
                S.append(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
        if best_size == max_size:
            break

    return best_set
