import time
import itertools

def construct():
    """
    Returns a list of 101 lists. Each list contains distinct denominators in 1..10
    such that the sum of their modular inverses modulo 101 equals the index.
    Minimizes the largest row length.
    """
    MOD = 101
    MAX_DENOM = 10
    denominators = tuple(range(1, MAX_DENOM + 1))
    
    # Precompute modular inverses
    inverses = {d: pow(d, -1, MOD) for d in denominators}
    
    # For each residue, find the minimum-length subset and store the combo.
    # Since denominators can recur across rows (no cross-row constraint),
    # each residue is solved independently: find shortest subset with sum == r mod 101.
    
    best_combo = [None] * MOD
    min_len = [MAX_DENOM + 1] * MOD
    
    # Iterate subsets by increasing length; first hit for a residue is optimal.
    for length in range(1, MAX_DENOM + 1):
        for combo in itertools.combinations(denominators, length):
            s = 0
            for d in combo:
                s += inverses[d]
            s %= MOD
            if min_len[s] > length:
                min_len[s] = length
                best_combo[s] = list(combo)
        # Early termination: if all residues found, stop
        if all(l <= MAX_DENOM for l in min_len):
            break
    
    # Fallback for any uncovered residue (should not happen)
    for r in range(MOD):
        if best_combo[r] is None:
            # Try all subsets again
            for length in range(1, MAX_DENOM + 1):
                for combo in itertools.combinations(denominators, length):
                    s = sum(inverses[d] for d in combo) % MOD
                    if s == r:
                        best_combo[r] = list(combo)
                        min_len[r] = length
                        break
                if best_combo[r] is not None:
                    break
            if best_combo[r] is None:
                best_combo[r] = [1]
    
    # Verify
    for r in range(MOD):
        total = sum(inverses[d] for d in best_combo[r]) % MOD
        assert total == r, f"Residue {r}: got {total}"
        assert len(set(best_combo[r])) == len(best_combo[r])
    
    return best_combo
