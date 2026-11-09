"""Rewritten program: maximize subset of [1,128] with no equal-gcd triples."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # 1. Precompute bad triples and conflicts
    # Indices 0 to 127 correspond to numbers 1 to 128
    N = 128
    nums = list(range(1, N + 1))
    
    # bad_pairs_for_u[u] contains list of (v, w) such that {u, v, w} is a bad triple
    # We store v < w to avoid duplicates in the pair itself, but u can be anything
    bad_pairs_for_u = [[] for _ in range(N)]
    
    # Precompute gcds for all pairs to speed up triple checking?
    # gcd_matrix[i][j] = gcd(nums[i], nums[j])
    gcd_matrix = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(i, N):
            g = gcd(nums[i], nums[j])
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g
    
    # Identify bad triples
    # A triple (i, j, k) is bad if gcd(i,j) == gcd(j,k) == gcd(k,i)
    # We iterate i < j < k
    for i in range(N):
        for j in range(i + 1, N):
            g_ij = gcd_matrix[i][j]
            for k in range(j + 1, N):
                g_jk = gcd_matrix[j][k]
                if g_ij == g_jk:
                    g_ki = gcd_matrix[k][i]
                    if g_ij == g_ki:
                        # This triple is bad
                        # Add pairs to conflict lists
                        # For u=i, conflict pair is (j, k)
                        bad_pairs_for_u[i].append((j, k))
                        # For u=j, conflict pair is (i, k)
                        bad_pairs_for_u[j].append((i, k))
                        # For u=k, conflict pair is (i, j)
                        bad_pairs_for_u[k].append((i, j))
    
    # Convert to sets for faster lookup if needed, but list iteration is fine if small
    # Actually, for is_valid_add, we just iterate the list and check membership in S
    
    def is_valid_add(u, S):
        """Check if adding u to set S creates any bad triple."""
        # S is a set of indices
        for v, w in bad_pairs_for_u[u]:
            if v in S and w in S:
                return False
        return True

    def greedy_init(order):
        S = set()
        for u in order:
            if is_valid_add(u, S):
                S.add(u)
        return S

    # Initial greedy solution
    # Heuristic: sort by number of conflicts (degree in conflict graph)
    # Elements with fewer conflicts are "safer" to add? Or "harder" to exclude?
    # Let's try sorting by number of conflicts ascending (add easy ones first)
    # Or descending?
    # Let's try multiple random orders and keep the best
    
    best_size = 0
    best_set = set()
    
    # Try a few random greedy starts
    for _ in range(10):
        if time.time() > deadline:
            break
        random_order = nums[:]
        random.shuffle(random_order)
        # Map values to indices for the greedy function
        # Our indices are 0..127, values 1..128
        # random_order contains values 1..128
        # We need to convert to indices 0..127
        idx_order = [x - 1 for x in random_order]
        S = greedy_init(idx_order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    # Local Search on the best greedy solution
    S = best_set
    current_size = len(S)
    
    # To speed up, maintain a list of elements in S for iteration
    # But set is fine for membership checks
    
    # We will try to add elements. If stuck, try swaps.
    # To avoid getting stuck in a local minimum where no single add works,
    # we can try "k-exchanges" but 1-exchange (add) and 2-exchange (swap) is usually good.
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try to add any element
        # We can iterate through all u not in S
        # To make it faster, we can check only a subset or randomize
        # But N is small (128), so checking all is fast
        
        # Optimization: Precompute list of candidates not in S
        # But S changes.
        
        # Let's try to add.
        # We pick a random order of candidates to avoid bias
        candidates = [u for u in range(N) if u not in S]
        random.shuffle(candidates)
        
        added_any = False
        for u in candidates:
            if is_valid_add(u, S):
                S.add(u)
                current_size += 1
                improved = True
                added_any = True
                break # Add one at a time, then re-evaluate
        
        if added_any:
            continue
            
        # 2. If no add possible, try swaps
        # Pick random v in S, random u not in S
        # Check if (S - {v}) + {u} is valid
        # This requires checking if adding u to (S - {v}) is valid
        
        # To be efficient, we can try a limited number of swaps per iteration
        swaps_attempted = 0
        swap_success = False
        while swaps_attempted < 50:
            swaps_attempted += 1
            if time.time() > deadline:
                break
                
            v = random.choice(list(S))
            u = random.choice(candidates) # candidates was computed before, might contain v? No, v in S, u not in S.
            # But candidates list is static from previous loop. It's fine.
            
            # Check validity of S' = S - {v} + {u}
            # Equivalent to: Is u valid to add to S - {v}?
            # We can check is_valid_add(u, S - {v})
            # But S is a set. S - {v} creates a new set.
            # We can temporarily remove v, check, and re-add if failed.
            
            S.remove(v)
            valid = is_valid_add(u, S)
            if valid:
                S.add(u)
                swap_success = True
                improved = True
                break
            else:
                S.add(v) # Revert
                
        if swap_success:
            continue
            
        # 3. If no swaps work, we are stuck.
        # Try to remove an element to open up space?
        # Or just restart local search with a different random seed?
        # Since we have time, let's try a "kick": remove 1 random element, then try to add 1-2 elements.
        
        # Kick: remove random v
        v = random.choice(list(S))
        S.remove(v)
        current_size -= 1
        
        # Now try to add as many as possible from the new state
        # This might recover the size or improve it
        
        # Re-run the add phase immediately
        # We loop back to top, so it will try to add.
        
        # To prevent infinite loops of removing the same element, 
        # the randomization in add/swap should handle it.
        
    return sorted(list(S))

# EVOLVE-BLOCK-END
