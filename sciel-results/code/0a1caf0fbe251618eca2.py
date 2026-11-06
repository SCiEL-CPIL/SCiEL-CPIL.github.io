"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    MOD = 1009
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Precompute modular inverses for 1..1008
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # Self-inverse elements: x^2 = 1 mod 1009 -> x=1 or x=1008
    # These cannot appear in any valid interval
    
    # For each position, precompute the minimum right boundary where a conflict occurs
    # An interval [s, e] is valid iff for all x in [s,e], inv[x] is not in [s,e]
    # and x is not self-inverse (1 or 1008)
    
    # Strategy: for each start s, find the maximum end e such that [s,e] is valid
    # We can do this more efficiently by tracking conflicts
    
    # For each element x, its "conflict" with another element y in an interval means
    # y = inv[x]. So if x is in the interval, inv[x] cannot be in the interval.
    
    # Let's think of it differently: for a fixed start s, as we extend e,
    # we need to ensure that for every x in [s,e], inv[x] is not in [s,e].
    # This means inv[x] < s or inv[x] > e.
    
    # For a fixed s, the constraint on e is: e < min(inv[x] for x in [s,e] if inv[x] >= s)
    # But this is circular. Let's use a different approach.
    
    # Precompute for each x, the "inverse partner" inv[x].
    # An interval [s, e] is invalid if there exists x in [s,e] such that inv[x] in [s,e].
    # Equivalently, for each pair (x, inv[x]) with x != inv[x], both cannot be in the interval.
    # Also x=1 and x=1008 are self-inverse, so they can't be in any interval.
    
    # Let's create a list of "forbidden pairs": for each x in 2..1007, the pair (x, inv[x])
    # where x != inv[x]. For x where x == inv[x] (only 1 and 1008 in this range), they're excluded.
    
    # For a given start s, the interval [s, e] is valid iff:
    # 1. No self-inverse element is in [s, e] (i.e., s > 1 and e < 1008)
    # 2. For no x in [s, e] is inv[x] also in [s, e]
    
    # Condition 2 can be rephrased: for all x in [s, e], inv[x] < s or inv[x] > e.
    # This means e < min{inv[x] : x in [s, e], inv[x] >= s}
    
    # Let's compute for each s the maximum valid e using a sweep approach.
    
    # Alternative: for each s from 2 to 1007, find the smallest e >= s such that
    # [s, e] is invalid. The answer for start s is e-1.
    
    # An interval [s, e] becomes invalid when we include some x such that inv[x] is also in [s, e].
    # For a fixed s, as e increases from s, the first conflict occurs when:
    # - e reaches some x where inv[x] >= s and inv[x] <= e
    #   i.e., e >= max(x, inv[x]) where x >= s and inv[x] >= s
    
    # So for start s, the max valid e is: min over all x >= s of max(x, inv[x]) - 1,
    # but only considering x where both x and inv[x] >= s, and x != inv[x].
    # Also need e < 1008 (since 1008 is self-inverse).
    
    # Wait, that's not quite right either. Let me think again.
    
    # For start s, we want the largest e such that for all x in [s, e], inv[x] is not in [s, e].
    # This is equivalent to: for all x in [s, e], inv[x] < s or inv[x] > e.
    
    # The binding constraint is: e < min{inv[x] : x in [s, e], inv[x] >= s}
    # But this depends on e itself.
    
    # Let's just do the direct scan but optimize it.
    
    best_start = 2
    best_len = 1  # at least length 1 is always valid (single element, no pair with itself unless self-inverse)
    
    # Actually, for a single element x, we need x*x % MOD != 1, so x != 1 and x != 1008.
    # So any x in 2..1007 gives a valid interval of length 1.
    
    for s in range(2, MOD - 1):
        if time.time() > deadline:
            break
        
        # Find maximum e such that [s, e] is valid
        e = s
        while e < MOD - 1:  # e can go up to 1007 (1008 is self-inverse)
            # Check if adding e+1 would cause a conflict
            next_e = e + 1
            if next_e == 1 or next_e == 1008:
                break
            inv_next = inv[next_e]
            # Conflict if inv_next is in [s, e+1]
            if s <= inv_next <= next_e:
                break
            e = next_e
        
        length = e - s + 1
        if length > best_len:
            best_len = length
            best_start = s
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
