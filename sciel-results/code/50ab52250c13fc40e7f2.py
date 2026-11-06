"""Model-and-optimize: parametrize interval by start s, fit maximal valid end e."""
import time

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 82.0  # explicit safety margin before the 90 s kill

    MOD = 1009  # prime
    N = MOD - 1  # residues 1..1008

    # --- Model construction -------------------------------------------------
    # Modular inverse of each residue (1009 is prime => inv exists for all).
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)

    # Self-inverse residues: x where inv[x] == x, i.e., x^2 ≡ 1 (mod 1009)
    self_inverse = set()
    for x in range(1, MOD):
        if inv[x] == x:
            self_inverse.add(x)

    best_start = 1
    best_len = 0
    
    # Two-pointer (sliding window) approach
    # We want to find max length L such that there exists start S where [S, S+L-1] is valid.
    # An interval [S, E] is valid iff for all x in [S, E], inv[x] is NOT in [S, E].
    # This is equivalent to:
    #   1. No x in [S, E] is self-inverse (inv[x] == x).
    #   2. For all x in [S, E], if inv[x] >= S, then inv[x] > E.
    #      Equivalently, if inv[x] <= E, then inv[x] < S.
    
    # Let's maintain a window [L, R].
    # We expand R. If the window becomes invalid, we shrink L.
    # Condition for validity of [L, R]:
    #   For all x in [L, R]:
    #     - x not in self_inverse
    #     - inv[x] < L or inv[x] > R
    
    # A faster check for validity when expanding R to R_new:
    # The new element is R_new.
    # 1. R_new must not be self-inverse.
    # 2. inv[R_new] must not be in [L, R_new-1].
    #    If inv[R_new] >= L and inv[R_new] <= R_new-1, conflict.
    #    Note: if inv[R_new] > R_new-1, it's fine (it's outside to the right).
    #    If inv[R_new] < L, it's fine (it's outside to the left).
    
    # However, adding R_new might also invalidate the window if R_new is the inverse of some x in [L, R_new-1].
    # i.e., if there exists x in [L, R_new-1] such that inv[x] == R_new.
    # Since inv is a bijection, there is exactly one such x: x = inv[R_new].
    # So the condition is: inv[R_new] must not be in [L, R_new-1].
    # This is the same condition as above!
    
    # So the only check needed when expanding R to R_new is:
    #   1. R_new not in self_inverse
    #   2. inv[R_new] < L  OR  inv[R_new] > R_new (which is always true if inv[R_new] != R_new)
    #      Wait, inv[R_new] > R_new-1 is equivalent to inv[R_new] >= R_new.
    #      Since inv[R_new] != R_new (checked by 1), inv[R_new] > R_new-1 means inv[R_new] >= R_new.
    #      But we need inv[R_new] NOT in [L, R_new].
    #      So if inv[R_new] >= R_new, it is outside [L, R_new] only if inv[R_new] > R_new? No.
    #      The interval is [L, R_new].
    #      If inv[R_new] is in [L, R_new-1], it's a conflict.
    #      If inv[R_new] == R_new, it's a conflict (self-inverse).
    #      If inv[R_new] > R_new-1, i.e., inv[R_new] >= R_new.
    #         If inv[R_new] == R_new, conflict.
    #         If inv[R_new] > R_new, then inv[R_new] is NOT in [L, R_new]. So it's safe.
    
    # So the condition for R_new to be added to [L, R_new-1] is:
    #   R_new not in self_inverse AND inv[R_new] < L OR inv[R_new] > R_new.
    #   Actually, if inv[R_new] > R_new, it's safe. If inv[R_new] < L, it's safe.
    #   If L <= inv[R_new] < R_new, it's unsafe.
    
    # If unsafe, we must increment L until the window is valid again.
    # When we increment L, do we need to re-check other elements?
    # No, because the only new potential conflict introduced by increasing L is... wait.
    # Increasing L removes elements from the left. It cannot create new conflicts.
    # The conflicts are defined by pairs (x, inv[x]) both being in the window.
    # If we remove x from the window, the pair (x, inv[x]) is broken.
    # So shrinking L always maintains validity (or restores it).
    
    # Algorithm:
    # L = 1
    # For R in 1..N:
    #   While window [L, R] is invalid:
    #      L += 1
    #   Update best_len
    
    # How to check if [L, R] is invalid efficiently?
    # We know [L, R-1] was valid.
    # The only way [L, R] becomes invalid is if R conflicts with some x in [L, R-1] or R is self-inverse.
    # Conflict with x in [L, R-1] means inv[x] = R, i.e., x = inv[R].
    # So [L, R] is invalid iff:
    #   1. R in self_inverse
    #   2. inv[R] in [L, R-1]
    
    # If invalid, we need to increase L.
    # If R in self_inverse, we must set L = R + 1 (since R cannot be in any valid interval).
    # If inv[R] in [L, R-1], we must set L = inv[R] + 1 (to exclude inv[R] from the window).
    #   Note: Setting L = inv[R] + 1 might still leave the window invalid if there are other conflicts?
    #   No, because [L_old, R-1] was valid. The only new element is R.
    #   The only conflict involving R is with inv[R].
    #   By setting L > inv[R], we ensure inv[R] is not in [L, R].
    #   Are there any other conflicts? No, because all other pairs were valid in [L_old, R-1] and removing elements from the left doesn't create new pairs.
    
    # So the "While" loop will actually execute at most once per R?
    # Let's check.
    # If R is self-inverse, L becomes R+1. Window is empty (or [R+1, R]).
    # Next R+1. Check validity.
    # If inv[R] in [L, R-1], L becomes inv[R]+1.
    # Is it possible that after setting L = inv[R]+1, the window [L, R] is still invalid?
    # Only if R is self-inverse (handled) or if there is another conflict.
    # But as argued, only R can introduce a new conflict.
    # So yes, at most one increment of L is needed per R?
    # Wait. What if L was already > inv[R]? Then no conflict.
    # What if L <= inv[R] < R? Then conflict. Set L = inv[R] + 1.
    # Now is [L, R] valid?
    # We removed [Old_L, inv[R]] from the window.
    # The remaining window is [inv[R]+1, R].
    # Does this window contain any conflict pair?
    # Any pair (x, y) in [inv[R]+1, R] with x*y=1.
    # If x < R, then y = inv[x].
    # Since [Old_L, R-1] was valid, and [inv[R]+1, R-1] is a subset of [Old_L, R-1], it is valid.
    # The only new element is R.
    # We ensured inv[R] < L (since L = inv[R]+1).
    # And R is not self-inverse (if it were, L would be R+1).
    # So yes, [L, R] is valid.
    
    # So we can do this in O(N) total.

    L = 1
    for R in range(1, MOD):
        if time.time() > deadline:
            break
        
        # Check if R causes a conflict in [L, R]
        # Conflict 1: R is self-inverse
        # Conflict 2: inv[R] is in [L, R-1]
        
        if R in self_inverse:
            # R cannot be in the interval.
            # Move L past R.
            L = R + 1
        else:
            ix = inv[R]
            # If ix is in [L, R-1], we have a conflict.
            # Since R is not self-inverse, ix != R.
            # So ix < R.
            if ix >= L:
                # ix is in [L, R-1]
                # We must exclude ix.
                L = ix + 1
        
        # Now [L, R] is valid (assuming L <= R)
        if L <= R:
            length = R - L + 1
            if length > best_len:
                best_len = length
                best_start = L

    # Fallback
    if best_start is None or best_len == 0:
        # Find any valid singleton
        for s in range(1, MOD):
            if s not in self_inverse:
                best_start = s
                best_len = 1
                break
        if best_start is None:
            best_start = 2
            best_len = 1

    return [best_start, best_len]

# EVOLVE-BLOCK-END
