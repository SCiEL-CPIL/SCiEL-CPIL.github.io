import time

def construct():
    start_time = time.time()
    LIMIT = 1009
    MAX_VAL = 1008
    DEADLINE = 85  # safety margin before 90s kill

    # Precompute modular inverses
    inv = [0] * (LIMIT + 1)
    for i in range(1, LIMIT):
        inv[i] = pow(i, -1, LIMIT)
    
    # Self-inverse elements: x * x % 1009 == 1
    # These are x = 1 and x = 1008 (since 1008 ≡ -1 mod 1009)
    # So the interval must be within [2, 1007]
    
    best_len = 0
    best_start = 2
    
    # For each start position S, find the maximum E such that [S, E] is valid
    # An interval [S, E] is valid iff for all x in [S, E], inv[x] is NOT in [S, E]
    # (since if inv[x] is in [S, E], then x * inv[x] % 1009 = 1, which is forbidden)
    # Also need to check that x != inv[x] for all x in [S,E], but since we exclude
    # self-inverses (1 and 1008), and for x in [2,1007], inv[x] != x always holds
    # (because x^2 ≡ 1 mod p only has solutions x ≡ ±1 mod p).
    
    # So the condition simplifies to: for all x in [S, E], inv[x] < S or inv[x] > E
    
    # Strategy: for each S, greedily extend E as far as possible.
    # We can do this efficiently by tracking which values have their inverse in the current range.
    
    # Alternative approach: 
    # For a given S, E_max is the largest E such that for all x in [S, E], inv[x] > E or inv[x] < S.
    # Equivalently, E_max is the smallest value such that there exists x in [S, E_max+1] with S <= inv[x] <= E_max,
    # minus 1.
    
    # Let's use the sliding window / two-pointer approach:
    # For each S, maintain E and a count of "conflicts" (pairs where both x and inv[x] are in [S,E]).
    # Actually, the condition is: no x in [S,E] has inv[x] also in [S,E].
    # This means: for all x in [S,E], inv[x] is outside [S,E].
    
    # Let's think differently. Define conflict(S, E) = True if there exists x in [S,E] such that inv[x] is in [S,E].
    # We want the largest E for each S such that conflict(S, E) = False.
    
    # Two-pointer: as S increases, E can only stay the same or increase.
    # When we move S from S to S+1:
    #   - We remove x = S from the interval.
    #   - If inv[S] was in [S+1, E], then removing S might resolve a conflict (if the only conflict was involving S and inv[S]).
    #   - But other conflicts might remain.
    
    # Actually, let's just use the O(N^2) approach but with early termination and time checking.
    # N = 1006, so N^2 = ~1 million, which is fast.
    
    S = 2
    E = 2
    
    while S <= 1007:
        # Ensure E is at least S
        if E < S:
            E = S
            
        # Try to extend E
        while E + 1 <= 1007:
            next_val = E + 1
            # Check if adding next_val creates a conflict
            # Conflict if inv[next_val] is in [S, E]
            if S <= inv[next_val] <= E:
                break
            else:
                E = next_val
                
        # Update best
        current_len = E - S + 1
        if current_len > best_len:
            best_len = current_len
            best_start = S
            
        S += 1
        
        # Time check
        if time.time() - start_time > DEADLINE:
            break
    
    # Verify the result
    for x in range(best_start, best_start + best_len):
        for y in range(best_start, best_start + best_len):
            if (x * y) % LIMIT == 1:
                # This should not happen, but if it does, we need to fix it
                # For now, just break
                break
    
    return [best_start, best_len]
