"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute GCD table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def is_bad_triple(a, b, c):
        return gcd_table[a][b] == gcd_table[b][c] == gcd_table[c][a]

    def can_add(x, S):
        # Check if x can be added to set S without creating a bad triple
        # A bad triple is formed if there exist a, b in S such that is_bad_triple(a, b, x)
        for a, b in itertools.combinations(S, 2):
            if is_bad_triple(a, b, x):
                return False
        return True

    # 1. Initial Greedy Construction
    # Sort numbers. We can try different orders.
    # A good heuristic: numbers with higher "constraint" first?
    # Or just 1..N. Let's try 1..N first, then maybe reverse, or by number of divisors.
    # Let's stick to 1..N for the initial seed, as it's simple.
    S = set()
    for x in numbers:
        if time.time() > deadline:
            break
        if can_add(x, S):
            S.add(x)
    
    R = set(numbers) - S
    
    best_S = set(S)
    best_size = len(S)

    # 2. Local Search
    # We try to improve the set by swapping elements or adding new ones.
    # Strategy:
    # Iterate through R. For each x in R:
    #   If x can be added, add it.
    #   Else, find conflicts. Try to remove one element v from S to resolve conflicts.
    #   If removing v resolves conflicts, remove v, add x.
    #   Then try to add any y from R to the new S.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Sort R to have a deterministic order, maybe by value
        for x in sorted(R):
            if time.time() > deadline:
                break
            
            # Check if x can be added directly
            if can_add(x, S):
                S.add(x)
                R.remove(x)
                improved = True
                break # Restart loop to check other elements with new S
            
            # Find conflicts
            conflicts = []
            involved = set()
            for a, b in itertools.combinations(S, 2):
                if is_bad_triple(a, b, x):
                    conflicts.append((a, b))
                    involved.add(a)
                    involved.add(b)
            
            if not conflicts:
                # Should have been caught above, but just in case
                S.add(x)
                R.remove(x)
                improved = True
                break

            # Try removing 1 element from S to resolve all conflicts
            # The element must be in 'involved'
            for v in involved:
                # Check if removing v resolves all conflicts
                # i.e., for all (a,b) in conflicts, a==v or b==v
                resolved = True
                for a, b in conflicts:
                    if a != v and b != v:
                        resolved = False
                        break
                
                if resolved:
                    # Remove v, add x
                    S.remove(v)
                    S.add(x)
                    R.remove(x)
                    R.add(v)
                    
                    # Try to add any other element from R to increase size
                    added_more = False
                    for y in sorted(R):
                        if time.time() > deadline:
                            break
                        if can_add(y, S):
                            S.add(y)
                            R.remove(y)
                            added_more = True
                            break
                    
                    if added_more:
                        improved = True
                        break # Restart outer loop
                    
                    # If we didn't add more, we just swapped. 
                    # This might enable future additions, so we consider it an improvement for the search structure, 
                    # but not for size. To avoid infinite loops of swaps, we should only mark improved if size increased 
                    # OR if we want to explore the space. 
                    # However, simple swaps can lead to cycles. 
                    # Let's only mark improved if size increased. 
                    # If size didn't increase, we might get stuck. 
                    # To handle this, we can limit swaps or use a randomization. 
                    # For now, let's just try to add more. If we can't, we revert? 
                    # No, the new state might be better. 
                    # Let's track the best size.
                    if len(S) > best_size:
                        best_S = set(S)
                        best_size = len(S)
                        improved = True
                        break
                    else:
                        # Size same. We made a swap. 
                        # To prevent cycling, we might need a history or randomization. 
                        # Given the small N, let's just continue the loop. 
                        # But if we don't mark improved, the while loop ends. 
                        # So we should probably only do swaps if they lead to immediate gain or are part of a multi-step plan.
                        # Let's try a 2-swap logic later if needed.
                        # For now, if no gain, revert?
                        S.remove(x)
                        S.add(v)
                        R.add(x)
                        R.remove(v)
                        # Try next v or next x?
                        # Let's just continue to next v
                        pass
            
            if improved:
                break

    # Update best if current S is better
    if len(S) > best_size:
        best_S = set(S)
        best_size = len(S)

    return list(best_S)

# EVOLVE-BLOCK-END
